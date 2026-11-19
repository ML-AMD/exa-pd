#include "align_cluster_builder.h"

#include "atom.h"
#include "neigh_list.h"
#include "neighbor.h"

using namespace LAMMPS_NS;

AlignClusterBuilder::AlignClusterBuilder(Atom *atom, NeighList *list, double cutoff) :
    atom_(atom), list_(list), cutsq_(cutoff * cutoff)
{}

void AlignClusterBuilder::build(int i, ALIGN::Cluster &cluster) const
{
  cluster = ALIGN::Cluster();
  cluster.atoms.emplace_back();
  cluster.atoms[0].type = atom_->type[i];
  cluster.atoms[0].x = 0.0;
  cluster.atoms[0].y = 0.0;
  cluster.atoms[0].z = 0.0;

  double **x = atom_->x;
  const int nneigh = list_->numneigh[i];
  int *jlist = list_->firstneigh[i];

  for (int jj = 0; jj < nneigh; ++jj) {
    const int j = jlist[jj] & NEIGHMASK;
    if (j < 0 || j >= atom_->nlocal + atom_->nghost) continue;

    // For a LAMMPS neighbor list, ghost coordinates already represent the
    // appropriate periodic image, so no additional minimum-image operation
    // is needed here.
    const double dx = x[j][0] - x[i][0];
    const double dy = x[j][1] - x[i][1];
    const double dz = x[j][2] - x[i][2];
    const double rsq = dx*dx + dy*dy + dz*dz;
    if (rsq > cutsq_) continue;

    ALIGN::Atom a;
    a.type = atom_->type[j];
    a.x = dx;
    a.y = dy;
    a.z = dz;
    a.bond2 = rsq;
    a.pair = -1;
    cluster.atoms.push_back(a);
  }

  cluster.bond_sum();
}
