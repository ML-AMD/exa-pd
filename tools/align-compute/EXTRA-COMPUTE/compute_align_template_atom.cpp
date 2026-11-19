/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator

   Per-atom local-cluster/template alignment using the shared ACA
   alignment engine.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(align/template/atom,ComputeAlignTemplateAtom)
// clang-format on
#else

#include "compute_align_template_atom.h"
#include "align_cluster_builder.h"

#include "atom.h"
#include "error.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "update.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <algorithm>
#include <sstream>
#include <string>

using namespace LAMMPS_NS;

namespace {

long parse_long(const char *text, const char *what, LAMMPS_NS::LAMMPS *lmp)
{
  char *end = nullptr;
  errno = 0;
  const long value = std::strtol(text, &end, 10);
  if (errno || end == text || *end != '\0')
    lmp->error->all(FLERR, "Invalid {} value '{}' in compute align/template/atom", what, text);
  return value;
}

double parse_double(const char *text, const char *what, LAMMPS_NS::LAMMPS *lmp)
{
  char *end = nullptr;
  errno = 0;
  const double value = std::strtod(text, &end);
  if (errno || end == text || *end != '\0')
    lmp->error->all(FLERR, "Invalid {} value '{}' in compute align/template/atom", what, text);
  return value;
}

unsigned long long parse_seed(const char *text, LAMMPS_NS::LAMMPS *lmp)
{
  char *end = nullptr;
  errno = 0;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno || end == text || *end != '\0')
    lmp->error->all(FLERR, "Invalid seed value '{}' in compute align/template/atom", text);
  return value;
}

bool parse_bool(const char *text, const char *what, LAMMPS_NS::LAMMPS *lmp)
{
  const long value = parse_long(text, what, lmp);
  if (value != 0 && value != 1)
    lmp->error->all(FLERR, "{} must be 0 or 1 in compute align/template/atom", what);
  return value != 0;
}

}  // namespace

ComputeAlignTemplateAtom::ComputeAlignTemplateAtom(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), nmax(0), score(nullptr), cutoff(0.0), list(nullptr),
    base_seed(13579ULL)
{
  if (narg < 5)
    error->all(FLERR, "Illegal compute align/template/atom command");

  peratom_flag = 1;
  size_peratom_cols = 0;

  template_file = arg[3];
  cutoff = parse_double(arg[4], "cutoff", lmp);
  if (cutoff <= 0.0)
    error->all(FLERR, "align/template/atom cutoff must be > 0");

  // Defaults correspond to the supplied ACA case-0 implementation.
  options.translate = true;
  options.resize = 0;
  options.mirror = false;
  options.element = false;
  options.random_time = 200;
  options.random_rotations = 20000;
  options.seed = base_seed;

  int iarg = 5;
  while (iarg < narg) {
    const std::string keyword(arg[iarg]);
    if (keyword == "translate") {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal compute align/template/atom command");
      options.translate = parse_bool(arg[iarg+1], "translate", lmp);
      iarg += 2;
    } else if (keyword == "resize") {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal compute align/template/atom command");
      options.resize = static_cast<int>(parse_long(arg[iarg+1], "resize", lmp));
      if (options.resize < 0) error->all(FLERR, "align/template/atom resize must be >= 0");
      iarg += 2;
    } else if (keyword == "mirror") {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal compute align/template/atom command");
      options.mirror = parse_bool(arg[iarg+1], "mirror", lmp);
      iarg += 2;
    } else if (keyword == "element") {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal compute align/template/atom command");
      options.element = parse_bool(arg[iarg+1], "element", lmp);
      iarg += 2;
    } else if (keyword == "random_time") {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal compute align/template/atom command");
      options.random_time = static_cast<int>(parse_long(arg[iarg+1], "random_time", lmp));
      if (options.random_time < 0) error->all(FLERR, "align/template/atom random_time must be >= 0");
      iarg += 2;
    } else if (keyword == "random_rotations") {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal compute align/template/atom command");
      options.random_rotations = static_cast<int>(parse_long(arg[iarg+1], "random_rotations", lmp));
      if (options.random_rotations <= 0) error->all(FLERR, "align/template/atom random_rotations must be > 0");
      iarg += 2;
    } else if (keyword == "seed") {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal compute align/template/atom command");
      base_seed = parse_seed(arg[iarg+1], lmp);
      options.seed = base_seed;
      iarg += 2;
    } else {
      error->all(FLERR, "Unknown keyword '{}' in compute align/template/atom", keyword);
    }
  }
}

ComputeAlignTemplateAtom::~ComputeAlignTemplateAtom()
{
  memory->destroy(score);
}

void ComputeAlignTemplateAtom::init()
{
  read_template();

  // The score is evaluated only when a consumer asks for the per-atom
  // vector.  Request an occasional full neighbor list with the alignment cutoff.
  auto *request = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
  request->set_cutoff(cutoff);
}

void ComputeAlignTemplateAtom::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

void ComputeAlignTemplateAtom::read_template()
{
  std::ifstream in(template_file);
  if (!in)
    error->all(FLERR, "Cannot open alignment template file '{}'", template_file);

  int n = 0;
  if (!(in >> n) || n < 1)
    error->all(FLERR, "Invalid alignment template atom count in '{}'", template_file);

  std::string comment;
  std::getline(in, comment);
  std::getline(in, comment);

  template_cluster = ALIGN::Cluster(n);
  template_cluster.comment = comment;

  for (int i = 0; i < n; ++i) {
    int type;
    double x, y, z;
    if (!(in >> type >> x >> y >> z))
      error->all(FLERR, "Invalid alignment template atom {} in '{}'", i+1, template_file);
    template_cluster.atoms[i].type = type;
    template_cluster.atoms[i].x = x;
    template_cluster.atoms[i].y = y;
    template_cluster.atoms[i].z = z;
    template_cluster.atoms[i].pair = -1;
  }

  template_cluster.recenter();
  template_cluster.bond_sum();

  if (template_cluster.size() < 2)
    error->all(FLERR, "alignment template must contain at least two atoms");

  double max_radius = 0.0;
  for (const auto &a : template_cluster.atoms)
    max_radius = std::max(max_radius, std::sqrt(a.bond2));

  if (cutoff + 1.0e-12 < max_radius)
    error->warning(FLERR,
                   "align/template/atom cutoff {} is smaller than the largest template radius {}; "
                   "template matches may be impossible", cutoff, max_radius);
}

void ComputeAlignTemplateAtom::grow_arrays()
{
  if (atom->nmax <= nmax) return;
  memory->destroy(score);
  nmax = atom->nmax;
  memory->create(score, nmax, "align/template/atom:score");
  vector_atom = score;
}

void ComputeAlignTemplateAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  grow_arrays();

  if (!list)
    error->all(FLERR, "align/template/atom neighbor list is not available");

  // This is an occasional list; explicitly build it when the compute is invoked.
  neighbor->build_one(list);

  const int nlocal = atom->nlocal;
  const int *mask = atom->mask;
  AlignClusterBuilder builder(atom, list, cutoff);

  for (int i = 0; i < nlocal; ++i) {
    score[i] = 9999.0;
    if (!(mask[i] & groupbit)) continue;

    ALIGN::Cluster cluster;
    builder.build(i, cluster);

    if (cluster.size() < template_cluster.size()) {
      score[i] = 9999.0;
      continue;
    }

    cluster.bond_sum();

    // Make each atom's random sequence independent of MPI domain
    // decomposition.  The global atom ID is used when available.
    const tagint id = atom->tag[i];
    const unsigned long long seed =
        options.seed ^ (static_cast<unsigned long long>(id) * 0x9e3779b97f4a7c15ULL);

    score[i] = ALIGN::align(cluster, template_cluster, options, seed);
  }

  for (int i = nlocal; i < nmax; ++i) score[i] = 0.0;
}

double ComputeAlignTemplateAtom::memory_usage()
{
  return static_cast<double>(nmax) * sizeof(double);
}

#endif
