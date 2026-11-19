/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator

   KOKKOS implementation of compute align/template/atom.
------------------------------------------------------------------------- */
#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(align/template/atom/kk,ComputeAlignTemplateAtomKokkos<LMPDeviceType>);
ComputeStyle(align/template/atom/kk/device,ComputeAlignTemplateAtomKokkos<LMPDeviceType>);
ComputeStyle(align/template/atom/kk/host,ComputeAlignTemplateAtomKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_COMPUTE_ALIGN_TEMPLATE_ATOM_KOKKOS_H
#define LMP_COMPUTE_ALIGN_TEMPLATE_ATOM_KOKKOS_H

#include "compute_align_template_atom.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<class DeviceType>
class ComputeAlignTemplateAtomKokkos : public ComputeAlignTemplateAtom {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  ComputeAlignTemplateAtomKokkos(class LAMMPS *, int, char **);
  ~ComputeAlignTemplateAtomKokkos() override;

  void init() override;
  void compute_peratom() override;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &) const;

 private:
  static constexpr int MAX_CLUSTER = 64;
  static constexpr int MAX_TEMPLATE = 64;

  int inum;
  int nlocal;
  int ntemplate;
  int max_cluster;

  typename AT::t_x_array_randomread x;
  typename AT::t_int_1d_randomread type;
  typename AT::t_int_1d_randomread mask;
  typename AT::t_tagint_1d_randomread tag;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

  Kokkos::View<double**, DeviceType> d_template_pos;
  Kokkos::View<int*, DeviceType> d_template_type;

  DAT::tdual_float_1d k_score;
  typename AT::t_float_1d d_score;
  bool score_allocated = false;

  unsigned long long base_seed;
  double cutoff_sq;

};

}  // namespace LAMMPS_NS

#endif
#endif
