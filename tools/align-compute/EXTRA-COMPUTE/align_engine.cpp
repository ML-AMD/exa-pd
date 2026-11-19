#include "align_engine.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace ALIGN {
namespace para {
constexpr int nran = 20000;
constexpr int nran_conv = 500;
constexpr int nlo = 1000;
constexpr int nmn = 100;
constexpr double rdeg = 5.0;
constexpr double ratio = 0.7;
constexpr double LSTEPSTART = 0.1;
constexpr int LMAX = 200;
constexpr double converge = 1e-4;
constexpr double LEPSO = 1e-8;
constexpr double opt_delta = 0.02;
}

void Cluster::bond_sum()
{
  bonds2 = 0.0;
  for (auto &a : atoms) {
    a.bond2 = a.x*a.x + a.y*a.y + a.z*a.z;
    bonds2 += a.bond2;
  }
}

void Cluster::recenter()
{
  if (atoms.empty()) return;
  const double x = atoms[0].x;
  const double y = atoms[0].y;
  const double z = atoms[0].z;
  for (auto &a : atoms) {
    a.x -= x;
    a.y -= y;
    a.z -= z;
  }
}

void Cluster::rotate(double ux, double uy, double uz, double theta_deg)
{
  const double u = std::sqrt(ux*ux + uy*uy + uz*uz);
  if (u == 0.0) return;
  ux /= u; uy /= u; uz /= u;

  const double theta = theta_deg * M_PI / 180.0;
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  const double omc = 1.0 - c;

  const double R[3][3] = {
    {c + ux*ux*omc, ux*uy*omc - uz*s, ux*uz*omc + uy*s},
    {uy*ux*omc + uz*s, c + uy*uy*omc, uy*uz*omc - ux*s},
    {uz*ux*omc - uy*s, uz*uy*omc + ux*s, c + uz*uz*omc}
  };

  for (auto &a : atoms) {
    const double x = a.x*R[0][0] + a.y*R[0][1] + a.z*R[0][2];
    const double y = a.x*R[1][0] + a.y*R[1][1] + a.z*R[1][2];
    const double z = a.x*R[2][0] + a.y*R[2][1] + a.z*R[2][2];
    a.x = x; a.y = y; a.z = z;
  }
}

void Cluster::translate(double ux, double uy, double uz, double distance)
{
  const double u = std::sqrt(ux*ux + uy*uy + uz*uz);
  if (u == 0.0) return;
  ux /= u; uy /= u; uz /= u;
  for (auto &a : atoms) {
    a.x += ux*distance;
    a.y += uy*distance;
    a.z += uz*distance;
  }
}

void Cluster::resize(double scale)
{
  for (auto &a : atoms) {
    a.x *= scale; a.y *= scale; a.z *= scale;
    a.bond2 *= scale*scale;
  }
  bonds2 *= scale*scale;
}

void Cluster::mirror()
{
  for (auto &a : atoms) a.x *= -1.0;
}

double Cluster::score(const Cluster &temp, bool element_match)
{
  if (atoms.size() < temp.atoms.size()) return 9999.0;

  for (auto &a : atoms) {
    a.iscore = 999.0;
    a.pair = -999;
  }
  atoms[0].pair = 0;

  const int cal_size = std::min<int>(atoms.size(),
                                     static_cast<int>(1.5*temp.atoms.size()));
  const int nt = temp.size();
  std::vector<double> sc(static_cast<size_t>(cal_size)*nt,
                         std::numeric_limits<double>::max());

  for (int i = 1; i < cal_size; ++i) {
    for (int j = 1; j < nt; ++j) {
      const double dx = atoms[i].x - temp.atoms[j].x;
      const double dy = atoms[i].y - temp.atoms[j].y;
      const double dz = atoms[i].z - temp.atoms[j].z;
      double value = dx*dx + dy*dy + dz*dz;
      if (element_match && atoms[i].type != temp.atoms[j].type) value += 0.10;
      sc[static_cast<size_t>(i)*nt + j] = value;
    }
  }

  std::vector<bool> select_C(cal_size, false);
  std::vector<bool> select_T(nt, false);
  double sct = 0.0;

  for (int status = 0; status < nt-1; ++status) {
    double scmin = 9999.0;
    int minT = 0, minC = 0;
    for (int j = 1; j < nt; ++j) {
      if (select_T[j]) continue;
      for (int i = 1; i < cal_size; ++i) {
        if (select_C[i]) continue;
        const double value = sc[static_cast<size_t>(i)*nt + j];
        if (value < scmin) {
          scmin = value;
          minC = i;
          minT = j;
        }
      }
    }
    if (minC == 0 || minT == 0 || !std::isfinite(scmin)) return 9999.0;
    select_C[minC] = true;
    select_T[minT] = true;
    atoms[minC].pair = minT;
    atoms[minC].iscore = scmin;
    sct += scmin;
  }

  const double dx = atoms[0].x - temp.atoms[0].x;
  const double dy = atoms[0].y - temp.atoms[0].y;
  const double dz = atoms[0].z - temp.atoms[0].z;
  atoms[0].iscore = dx*dx + dy*dy + dz*dz;
  atoms[0].pair = 0;
  sct += atoms[0].iscore;
  return std::sqrt(sct / nt);
}

double Cluster::paired_score(const Cluster &temp, double *force,
                              double *torque, bool element_match)
{
  const int cal_size = std::min<int>(atoms.size(),
                                     static_cast<int>(1.5*temp.atoms.size()));
  double sc = 0.0;
  for (int k = 0; k < 3; ++k) { force[k] = 0.0; torque[k] = 0.0; }

  int count = 0;
  for (int i = 0; i < cal_size; ++i) {
    if (atoms[i].pair < 0) continue;
    ++count;
    const int j = atoms[i].pair;
    const double dx = atoms[i].x - temp.atoms[j].x;
    const double dy = atoms[i].y - temp.atoms[j].y;
    const double dz = atoms[i].z - temp.atoms[j].z;
    double isc = dx*dx + dy*dy + dz*dz;
    if (element_match && atoms[i].type != temp.atoms[j].type) isc += 0.10;
    sc += isc;
    atoms[i].iscore = isc;
    const double fa[3] = {-dx, -dy, -dz};
    for (int k = 0; k < 3; ++k) force[k] += fa[k];
    torque[0] += atoms[i].y*fa[2] - atoms[i].z*fa[1];
    torque[1] += atoms[i].z*fa[0] - atoms[i].x*fa[2];
    torque[2] += atoms[i].x*fa[1] - atoms[i].y*fa[0];
  }
  if (count != temp.size()) return 9999.0;
  return std::sqrt(sc / temp.size());
}

double Cluster::bond2_compare(const Cluster &temp) const
{
  double bonds2 = 0.0;
  for (int i = 0; i < temp.size(); ++i) bonds2 += atoms[i].bond2;
  if (bonds2 == 0.0) return 1.0;
  return temp.bonds2 / bonds2;
}

void random_rotate(Cluster &cluster, std::mt19937_64 &rng)
{
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  const double ux = 0.5 - u01(rng);
  const double uy = 0.5 - u01(rng);
  const double uz = 0.5 - u01(rng);
  const double degree = 360.0*u01(rng);
  cluster.rotate(ux, uy, uz, degree);
}

double local_rotate(Cluster &A, const Cluster &B, double sc,
                    bool element_match)
{
  double degree = para::rdeg;
  double vfor[3], vtor[3];
  A.paired_score(B, vfor, vtor, element_match);

  for (int j = 0; j < para::nlo; ++j) {
    Cluster backup = A;
    A.rotate(vtor[0], vtor[1], vtor[2], degree);
    double cfor[3], ctor[3];
    const double sc_now = A.paired_score(B, cfor, ctor, element_match);

    if (2.0*std::abs(sc_now-sc) <
        para::converge*(std::abs(sc)+std::abs(sc_now)+para::LEPSO)) {
      sc = sc_now;
      break;
    }
    if (sc_now < sc) {
      sc = sc_now;
      for (int m = 0; m < 3; ++m) vtor[m] = ctor[m];
    } else {
      degree *= para::ratio;
      A = backup;
    }
  }
  return sc;
}

double translate(Cluster &A, const Cluster &B, double sc,
                 bool element_match)
{
  double vfor[3], vtor[3];
  A.paired_score(B, vfor, vtor, element_match);
  double step = para::LSTEPSTART;

  for (int j = 0; j < para::LMAX; ++j) {
    Cluster backup = A;
    A.translate(vfor[0], vfor[1], vfor[2], step);
    double cfor[3], ctor[3];
    const double sc_now = A.paired_score(B, cfor, ctor, element_match);

    if (2.0*std::abs(sc_now-sc) <
        para::converge*(std::abs(sc)+std::abs(sc_now)+para::LEPSO)) {
      sc = sc_now;
      break;
    }
    if (sc_now < sc) {
      sc = sc_now;
      for (int m = 0; m < 3; ++m) vfor[m] = cfor[m];
    } else {
      step *= para::ratio;
      A = backup;
    }
  }
  return sc;
}

double align(Cluster cluster, const Cluster &template_cluster,
             const Options &options, unsigned long long seed)
{
  // The original source comments that the large-cluster shortcut requires
  // bond sorting.  Sort non-central atoms by distance from the center so the
  // 1.5*N template truncation uses the closest atoms.
  const int max_cluster = static_cast<int>(1.5*template_cluster.size());
  if (cluster.size() > max_cluster) {
    std::sort(cluster.atoms.begin() + 1, cluster.atoms.end(),
              [](const Atom &a, const Atom &b) { return a.bond2 < b.bond2; });
    cluster.atoms.resize(max_cluster);
  }
  cluster.bond_sum();

  // The original case-0 code creates scaled and/or mirrored copies of the
  // template.  RESIZE=0 therefore corresponds to one template copy.
  const int nopt = options.resize > 0
      ? (options.mirror ? 2*options.resize : options.resize)
      : (options.mirror ? 2 : 1);
  std::vector<Cluster> templates(nopt);

  const double delta = para::opt_delta;
  const double start = 1.0 - delta*static_cast<int>(options.resize/2);

  if (options.resize == 0) {
    templates[0] = template_cluster;
    if (options.mirror) {
      templates[1] = template_cluster;
      templates[1].mirror();
    }
  } else {
    double fac = 1.0/std::sqrt(cluster.bond2_compare(template_cluster));
    for (int i = 0; i < options.resize; ++i) {
      templates[i] = template_cluster;
      templates[i].resize(fac*(start + delta*i));
      if (options.mirror) {
        templates[2*i+1] = templates[2*i];
        templates[2*i+1].mirror();
      }
    }
  }

  std::mt19937_64 rng(seed);
  const int nconv = std::max(options.random_time, para::nran_conv);
  const int nrandom = options.random_rotations > 0
      ? options.random_rotations : para::nran;

  double final_score = 9999.0;
  int best_opt = 0;
  Cluster best_cluster = cluster;

  for (int iopt = 0; iopt < nopt; ++iopt) {
    double score_random = 9999.0;
    int iconv = 0;
    Cluster random_min = cluster;

    for (int i = 0; i < nrandom; ++i) {
      Cluster current = cluster;
      random_rotate(current, rng);
      double score_min = current.score(templates[iopt], options.element);

      score_min = local_rotate(current, templates[iopt], score_min,
                                      options.element);
      if (options.translate) {
        score_min = translate(current, templates[iopt], 
			score_min, options.element);
        }

      if (score_min < score_random) {
        score_random = score_min;
        random_min = current;
        iconv = 0;
      } else {
        ++iconv;
      }
      if (iconv >= nconv) break;
    }

    if (score_random < final_score) {
      final_score = score_random;
      best_opt = iopt;
      best_cluster = random_min;
    }
  }

  (void)best_opt;
  (void)best_cluster;
  return final_score;
}

}  // namespace ALIGN
