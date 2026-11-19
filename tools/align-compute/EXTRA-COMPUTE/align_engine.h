#ifndef LMP_ALIGN_ENGINE_H
#define LMP_ALIGN_ENGINE_H

#include <string>
#include <vector>
#include <random>

namespace ALIGN {

struct Atom {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double bond2 = 0.0;
  int type = 0;
  int pair = -1;
  double iscore = 999.0;
};

class Cluster {
 public:
  Cluster() = default;
  explicit Cluster(int n) : atoms(n) {}

  std::vector<Atom> atoms;
  std::string comment;
  double bonds2 = 0.0;

  int size() const { return static_cast<int>(atoms.size()); }
  void bond_sum();
  void recenter();
  void rotate(double ux, double uy, double uz, double theta_deg);
  void translate(double ux, double uy, double uz, double distance);
  void resize(double scale);
  void mirror();

  double score(const Cluster &temp, bool element_match);
  double paired_score(const Cluster &temp, double *force, double *torque,
                      bool element_match);
  double bond2_compare(const Cluster &temp) const;
};

struct Options {
  bool translate = true;
  int resize = 0;
  bool mirror = false;
  bool element = false;
  int random_time = 200;
  int random_rotations = 20000;
  unsigned long long seed = 13579ULL;
};

// Shared ACA alignment engine.  The template compute uses this to compare
// a local atomic cluster against a reference template.  Future alignment
// styles (e.g. align/pair) can reuse the same Cluster and align() machinery.
double align(Cluster cluster, const Cluster &template_cluster,
             const Options &options, unsigned long long seed);

void random_rotate(Cluster &cluster, std::mt19937_64 &rng);
double local_rotate(Cluster &cluster, const Cluster &temp, double score,
                    bool element_match);
double translate(Cluster &cluster, const Cluster &temp, double score,
                 bool element_match);

}  // namespace ALIGN

#endif
