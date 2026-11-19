#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(align/template/atom,ComputeAlignTemplateAtom);
// clang-format on
#else

#ifndef LMP_COMPUTE_ALIGN_TEMPLATE_ATOM_H
#define LMP_COMPUTE_ALIGN_TEMPLATE_ATOM_H

#include "compute.h"
#include "align_engine.h"

#include <string>

namespace LAMMPS_NS {

class ComputeAlignTemplateAtom : public Compute {
 public:
  ComputeAlignTemplateAtom(class LAMMPS *, int, char **);
  ~ComputeAlignTemplateAtom() override;

  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;
  double memory_usage() override;

 protected:
  int nmax;
  double *score;
  double cutoff;
  std::string template_file;
  ALIGN::Cluster template_cluster;
  ALIGN::Options options;
  unsigned long long base_seed;
  class NeighList *list;

  void read_template();
  void grow_arrays();
};

}  // namespace LAMMPS_NS

#endif
#endif
