// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator

   KOKKOS implementation of compute align/template/atom.

   This implementation targets ACA case 0 (cluster-template alignment).
   It deliberately keeps the alignment state local to one atom and uses one
   Kokkos work item per atom.  The expensive alignment loop therefore runs on
   the selected Kokkos execution space while the LAMMPS neighbor list and atom
   coordinates are supplied through Kokkos views.
------------------------------------------------------------------------- */
#include "compute_align_template_atom_kokkos.h"
#include "align_engine_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor_kokkos.h"
#include "update.h"
#include "comm.h"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <limits>
#include <Kokkos_Random.hpp>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
template<class DeviceType>
ComputeAlignTemplateAtomKokkos<DeviceType>::ComputeAlignTemplateAtomKokkos(
    LAMMPS *lmp, int narg, char **arg) : ComputeAlignTemplateAtom(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | TYPE_MASK | MASK_MASK | TAG_MASK;
  datamask_modify = EMPTY_MASK;
  base_seed = options.seed;
  cutoff_sq = cutoff * cutoff;
}

/* ---------------------------------------------------------------------- */
template<class DeviceType>
ComputeAlignTemplateAtomKokkos<DeviceType>::~ComputeAlignTemplateAtomKokkos()
{
  if (copymode) return;

  if (score_allocated) {
    memoryKK->destroy_kokkos(k_score, score);
    score = nullptr;
    vector_atom = nullptr;
    score_allocated = false;
  }
}

/* ---------------------------------------------------------------------- */
template<class DeviceType>
void ComputeAlignTemplateAtomKokkos<DeviceType>::init()
{
  ComputeAlignTemplateAtom::init();

  if (template_cluster.size() > MAX_TEMPLATE)
    error->all(FLERR, "align/template/atom/kk template contains too many atoms for the "
                     "KOKKOS implementation");

  if (options.mirror)
    error->all(FLERR, "align/template/atom/kk does not yet support mirror");

  ntemplate = template_cluster.size();
  max_cluster = static_cast<int>(1.5 * ntemplate);
  if (max_cluster < ntemplate) max_cluster = ntemplate;
  if (max_cluster > MAX_CLUSTER) max_cluster = MAX_CLUSTER;

  d_template_pos = Kokkos::View<double **, DeviceType>(
      "align/template/atom/template_pos", ntemplate, 3);
  d_template_type = Kokkos::View<int *, DeviceType>(
      "align/template/atom/template_type", ntemplate);

  auto h_pos = Kokkos::create_mirror_view(d_template_pos);
  auto h_type = Kokkos::create_mirror_view(d_template_type);

  for (int i = 0; i < ntemplate; ++i) {
    h_pos(i, 0) = template_cluster.atoms[i].x;
    h_pos(i, 1) = template_cluster.atoms[i].y;
    h_pos(i, 2) = template_cluster.atoms[i].z;
    h_type(i) = template_cluster.atoms[i].type;
  }

  Kokkos::deep_copy(d_template_pos, h_pos);
  Kokkos::deep_copy(d_template_type, h_type);

  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType, LMPHostType> &&
                           !std::is_same_v<DeviceType, LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);
}

/* ---------------------------------------------------------------------- */
template<class DeviceType>
void ComputeAlignTemplateAtomKokkos<DeviceType>::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  //atomKK->sync(Device, X_MASK | MASK_MASK);
  atomKK->sync(Device, X_MASK | TYPE_MASK | MASK_MASK);

  nlocal = atom->nlocal;

  if (atom->nmax > nmax) {
    nmax = atom->nmax;

    memoryKK->destroy_kokkos(k_score, vector_atom);
    memoryKK->create_kokkos(k_score, vector_atom, nmax,
                            "align/template/atom/kk:score");
  }

  // ---- NEW: access only neighbor counts ----

  neighbor->build_one(list);
  NeighListKokkos<DeviceType> *k_list =
      static_cast<NeighListKokkos<DeviceType> *>(list);

  d_ilist = k_list->d_ilist;
  d_numneigh = k_list->d_numneigh;
  inum = list->inum;

  d_neighbors = k_list->d_neighbors;

  auto d_x_local = atomKK->k_x.d_view;
  auto d_neighbors_local = d_neighbors;
  auto d_score_local = k_score.d_view;
  auto d_mask = atomKK->k_mask.d_view;
  auto d_type_local = atomKK->k_type.d_view;
  
  auto d_ilist_local = d_ilist;
  auto d_numneigh_local = d_numneigh;
  
  const int groupbit_local = groupbit;
  const int inum_local = inum;
  const int nall_local = atom->nlocal + atom->nghost;
  const double cutsq_local = cutoff_sq;
  const int max_cluster_local = max_cluster;
  auto d_template_pos_local = d_template_pos;
  auto d_template_type_local = d_template_type;
  
  const int ntemplate_local = ntemplate;
  const bool element_match_local = options.element;
  const bool options_translate_local = options.translate;
  const int nrandom_local =
    (options.random_rotations > 0) ?
    options.random_rotations : 20000;
  const int nconv_local =
    (options.random_time > 100) ?
    options.random_time : 100;
  const int nresize_local = options.resize;
  
  using RandomPool = Kokkos::Random_XorShift64_Pool<DeviceType>;

  const uint64_t seed = 12345;
  RandomPool random_pool(seed);
  Kokkos::parallel_for(
    Kokkos::RangePolicy<DeviceType>(0, inum_local),
    KOKKOS_LAMBDA(const int ii) {

      const int i = d_ilist_local(ii);

      if (!(d_mask(i) & groupbit_local)) {
        d_score_local(i) = 0.0;
        return;
      }

      // Local cluster storage (single precision)
      float cx[MAX_CLUSTER];
      float cy[MAX_CLUSTER];
      float cz[MAX_CLUSTER];
      float bond2[MAX_CLUSTER];
      int ctype[MAX_CLUSTER];

      // Central atom
      int count = 1;

      cx[0] = 0.0f;
      cy[0] = 0.0f;
      cz[0] = 0.0f;
      bond2[0] = 0.0f;
      ctype[0] = d_type_local(i);

      const double xi = d_x_local(i, 0);
      const double yi = d_x_local(i, 1);
      const double zi = d_x_local(i, 2);

      const int jnum = d_numneigh_local(i);

      for (int jj = 0; jj < jnum; ++jj) {

        const int j = d_neighbors_local(i, jj) & NEIGHMASK;

        if (j < 0 || j >= nall_local) continue;

        const double dx = d_x_local(j, 0) - xi;
        const double dy = d_x_local(j, 1) - yi;
        const double dz = d_x_local(j, 2) - zi;

        const double rsq = dx*dx + dy*dy + dz*dz;

        if (rsq > cutsq_local) continue; 

        // Fixed-size storage: do not write beyond MAX_CLUSTER
        if (count >= MAX_CLUSTER) continue;

        cx[count] = static_cast<float>(dx);
        cy[count] = static_cast<float>(dy);
        cz[count] = static_cast<float>(dz);
        bond2[count] = static_cast<float>(rsq);
        ctype[count] = d_type_local(j);

        ++count;
      }
      // Number of atoms to use in alignment
      const int nselected =
          (count < max_cluster_local) ? count : max_cluster_local;
      
      // Sort by distance using selection sort.
      // Index 0 is the central atom and must remain fixed.
      for (int k = 1; k < nselected; ++k) {
      
        int nearest = k;
      
        for (int j = k + 1; j < count; ++j) {
          if (bond2[j] < bond2[nearest])
            nearest = j;
        }
      
        if (nearest != k) {
          float tmp;
      
          tmp = cx[k]; cx[k] = cx[nearest]; cx[nearest] = tmp;
          tmp = cy[k]; cy[k] = cy[nearest]; cy[nearest] = tmp;
          tmp = cz[k]; cz[k] = cz[nearest]; cz[nearest] = tmp;
          tmp = bond2[k]; bond2[k] = bond2[nearest]; bond2[nearest] = tmp;
      
          int itmp = ctype[k];
          ctype[k] = ctype[nearest];
          ctype[nearest] = itmp;
        }
      }

      // Obtain a random-number generator
      auto rng = random_pool.get_state();

      const float score_final =
          ALIGN_KK::align_resize<MAX_CLUSTER, MAX_TEMPLATE>(
              cx, cy, cz, ctype, nselected,
              d_template_pos_local,
              d_template_type_local,
              ntemplate_local,
              element_match_local,
              options_translate_local,
              nrandom_local,
              nconv_local,
              nresize_local,
              rng);
      
      random_pool.free_state(rng); 
      d_score_local(i) = static_cast<double>(score_final);
    });
  k_score.modify_device();
  Kokkos::fence();
  k_score.modify_device();
  k_score.sync_host();
}

namespace LAMMPS_NS {
template class ComputeAlignTemplateAtomKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ComputeAlignTemplateAtomKokkos<LMPHostType>;
#endif
}  // namespace LAMMPS_NS
