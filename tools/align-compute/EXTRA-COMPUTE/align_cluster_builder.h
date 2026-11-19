#ifndef LMP_ALIGN_CLUSTER_BUILDER_H
#define LMP_ALIGN_CLUSTER_BUILDER_H

#include "align_engine.h"

namespace LAMMPS_NS {

class Atom;
class NeighList;

// LAMMPS-specific adapter: converts atom i and its neighbor list into the
// generic ALIGN::Cluster representation used by the alignment engine.
class AlignClusterBuilder {
 public:
  AlignClusterBuilder(Atom *atom, NeighList *list, double cutoff);

  void build(int i, ALIGN::Cluster &cluster) const;

 private:
  Atom *atom_;
  NeighList *list_;
  double cutsq_;
};

}  // namespace LAMMPS_NS

#endif
