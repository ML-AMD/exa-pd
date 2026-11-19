#ifndef LMP_ALIGN_ENGINE_KOKKOS_H
#define LMP_ALIGN_ENGINE_KOKKOS_H

#include "kokkos_type.h"
#include <cmath>
#include <Kokkos_Random.hpp>

namespace LAMMPS_NS {
namespace ALIGN_KK {

// Fixed-orientation greedy matching, equivalent in logic to
// ALIGN::Cluster::score(). The center is paired with the template center.
// Coordinates are float; template views can retain double storage.
// nc is the selected cluster size, including its center.
// TemplatePosView and TemplateTypeView must be device-accessible Kokkos views.
template<int MAX_CLUSTER, int MAX_TEMPLATE,
         class TemplatePosView, class TemplateTypeView>
KOKKOS_INLINE_FUNCTION
float score(const float *cx, const float *cy, const float *cz,
            const int *ctype, const int nc,
            const TemplatePosView &tpos, const TemplateTypeView &ttype,
            const int nt, const bool element_match, int *pair)
{
  for (int c = 0; c < nc; ++c)
    pair[c] = -1;
  if (nt < 1 || nt > MAX_TEMPLATE || nc < nt || nc > MAX_CLUSTER)
    return 9999.0f;

  pair[0] = 0;

  bool used_cluster[MAX_CLUSTER] = {};
  bool used_template[MAX_TEMPLATE] = {};

  const float dx0 = cx[0] - static_cast<float>(tpos(0, 0));
  const float dy0 = cy[0] - static_cast<float>(tpos(0, 1));
  const float dz0 = cz[0] - static_cast<float>(tpos(0, 2));
  float sum = dx0*dx0 + dy0*dy0 + dz0*dz0;

  used_cluster[0] = true;
  used_template[0] = true;

  for (int p = 1; p < nt; ++p) {
    float best = 9999.0f;
    int best_c = -1;
    int best_t = -1;

    // Match CPU loop order and strict comparison for tie behavior.
    for (int t = 1; t < nt; ++t) {
      if (used_template[t]) continue;
      const float tx = static_cast<float>(tpos(t, 0));
      const float ty = static_cast<float>(tpos(t, 1));
      const float tz = static_cast<float>(tpos(t, 2));
      for (int c = 1; c < nc; ++c) {
        if (used_cluster[c]) continue;
        const float dx = cx[c] - tx;
        const float dy = cy[c] - ty;
        const float dz = cz[c] - tz;
        float value = dx*dx + dy*dy + dz*dz;
        if (element_match && ctype[c] != ttype(t)) value += 0.10f;
        if (value < best) {
          best = value;
          best_c = c;
          best_t = t;
        }
      }
    }
    if (best_c < 0 || best_t < 0) return 9999.0f;
    used_cluster[best_c] = true;
    used_template[best_t] = true;
    pair[best_c] = best_t;
    sum += best;
  }
  return sqrtf(sum / static_cast<float>(nt));
}
KOKKOS_INLINE_FUNCTION
void rotate(float *cx, float *cy, float *cz,
            int n,
            float ux, float uy, float uz,
            float theta_deg)
{
  const float u = sqrtf(ux*ux + uy*uy + uz*uz);
  if (u == 0.0f) return;

  ux /= u;
  uy /= u;
  uz /= u;

  constexpr float DEG_TO_RAD =
      3.14159265358979323846f / 180.0f;

  const float theta = theta_deg * DEG_TO_RAD;
  const float c = cosf(theta);
  const float s = sinf(theta);
  const float omc = 1.0f - c;

  const float r00 = c + ux*ux*omc;
  const float r01 = ux*uy*omc - uz*s;
  const float r02 = ux*uz*omc + uy*s;

  const float r10 = uy*ux*omc + uz*s;
  const float r11 = c + uy*uy*omc;
  const float r12 = uy*uz*omc - ux*s;

  const float r20 = uz*ux*omc - uy*s;
  const float r21 = uz*uy*omc + ux*s;
  const float r22 = c + uz*uz*omc;

  for (int i = 0; i < n; ++i) {
    const float x = cx[i];
    const float y = cy[i];
    const float z = cz[i];

    cx[i] = r00*x + r01*y + r02*z;
    cy[i] = r10*x + r11*y + r12*z;
    cz[i] = r20*x + r21*y + r22*z;
  }
}

template<class TemplatePosView, class TemplateTypeView>
KOKKOS_INLINE_FUNCTION
float paired_score(
    const float *cx, const float *cy, const float *cz,
    const int *ctype, const int nc,
    const TemplatePosView &tpos,
    const TemplateTypeView &ttype,
    const int nt,
    const int *pair,
    const bool element_match,
    float *force, float *torque)
{
    force[0] = force[1] = force[2] = 0.0f;
    torque[0] = torque[1] = torque[2] = 0.0f;

    float sum = 0.0f;
    int matched = 0;

    for (int c = 0; c < nc; ++c) {
        const int t = pair[c];
        if (t < 0 || t >= nt) continue;

        const float dx = static_cast<float>(tpos(t,0)) - cx[c];
        const float dy = static_cast<float>(tpos(t,1)) - cy[c];
        const float dz = static_cast<float>(tpos(t,2)) - cz[c];

        float value = dx*dx + dy*dy + dz*dz;

        if (element_match && ctype[c] != ttype(t))
            value += 0.10f;

        sum += value;

        force[0] += dx;
        force[1] += dy;
        force[2] += dz;

        torque[0] += cy[c]*dz - cz[c]*dy;
        torque[1] += cz[c]*dx - cx[c]*dz;
        torque[2] += cx[c]*dy - cy[c]*dx;

        ++matched;
    }

    if (matched != nt) return 9999.0f;

    return sqrtf(sum / static_cast<float>(nt));
}

//template<class TemplatePos>
//KOKKOS_INLINE_FUNCTION
//float paired_score(
//    const float *cx, const float *cy, const float *cz,
//    int nselected,
//    const TemplatePos &tpos,
//    int ntemplate,
//    const int *pair,
//    float *force,
//    float *torque)
//{
//  force[0] = force[1] = force[2] = 0.0f;
//  torque[0] = torque[1] = torque[2] = 0.0f;
//
//  float sum = 0.0f;
//  int matched = 0;
//
//  for (int c = 0; c < nselected; ++c) {
//    const int t = pair[c];
//
//    if (t < 0 || t >= ntemplate) continue;
//
//    const float dx = static_cast<float>(tpos(t,0)) - cx[c];
//    const float dy = static_cast<float>(tpos(t,1)) - cy[c];
//    const float dz = static_cast<float>(tpos(t,2)) - cz[c];
//
//    sum += dx*dx + dy*dy + dz*dz;
//
//    force[0] += dx;
//    force[1] += dy;
//    force[2] += dz;
//
//    torque[0] += cy[c]*dz - cz[c]*dy;
//    torque[1] += cz[c]*dx - cx[c]*dz;
//    torque[2] += cx[c]*dy - cy[c]*dx;
//
//    ++matched;
//  }
//
//  if (matched != ntemplate)
//    return 9999.0f;
//
//  return sqrtf(sum / static_cast<float>(ntemplate));
//}

template<class TemplatePosView, class TemplateTypeView>
KOKKOS_INLINE_FUNCTION
float local_rotate(
    float *cx, float *cy, float *cz,
    const int *ctype, const int nc,
    const TemplatePosView &tpos,
    const TemplateTypeView &ttype,
    const int nt,
    const int *pair,
    const bool element_match,
    float sc)
{
    constexpr float CONVERGE = 1.0e-4f;
    constexpr float LEPSO = 1.0e-8f;
    constexpr float RATIO = 0.7f;
    constexpr int NLO = 1000;

    float degree = 5.0f;
    float force[3], torque[3];

    paired_score(cx, cy, cz, ctype, nc,
                 tpos, ttype, nt, pair, element_match,
                 force, torque);

    for (int j = 0; j < NLO; ++j) {

        const float ux = torque[0];
        const float uy = torque[1];
        const float uz = torque[2];

        rotate(cx, cy, cz, nc, ux, uy, uz, degree);

        float cforce[3], ctorque[3];

        const float sc_now = paired_score(
            cx, cy, cz, ctype, nc,
            tpos, ttype, nt, pair, element_match,
            cforce, ctorque);

        if (2.0f*fabsf(sc_now-sc) <
            CONVERGE*(fabsf(sc)+fabsf(sc_now)+LEPSO)) {
            sc = sc_now;
            break;
        }

        if (sc_now < sc) {
            sc = sc_now;

            for (int m = 0; m < 3; ++m)
                torque[m] = ctorque[m];

        } else {
            rotate(cx, cy, cz, nc, ux, uy, uz, -degree);
            degree *= RATIO;
        }
    }

    return sc;
}

//template<class TemplatePosView>
//KOKKOS_INLINE_FUNCTION
//float local_rotate(
//    float *cx, float *cy, float *cz,
//    const int nc,
//    const TemplatePosView &tpos,
//    const int nt,
//    const int *pair)
//{
//    float force[3], torque[3];
//
//    float score = paired_score(
//        cx, cy, cz, nc, tpos, nt, pair, force, torque);
//
//    float angle = 5.0f;
//
//    constexpr int MAX_ITER = 1000;
//    constexpr float ANGLE_FACTOR = 0.7f;
//    constexpr float MIN_ANGLE = 1.0e-5f;
//    constexpr float MIN_TORQUE = 1.0e-8f;
//
//    for (int iter = 0; iter < MAX_ITER; ++iter) {
//
//        const float tnorm = sqrtf(
//            torque[0]*torque[0] +
//            torque[1]*torque[1] +
//            torque[2]*torque[2]);
//
//        if (tnorm < MIN_TORQUE || angle < MIN_ANGLE)
//            break;
//
//        // Save the current torque direction.
//        const float ux = torque[0] / tnorm;
//        const float uy = torque[1] / tnorm;
//        const float uz = torque[2] / tnorm;
//
//        // Trial rotation.
//        rotate(cx, cy, cz, nc, ux, uy, uz, angle);
//
//        float trial_force[3], trial_torque[3];
//
//        const float trial_score = paired_score(
//            cx, cy, cz, nc, tpos, nt, pair,
//            trial_force, trial_torque);
//
//        if (trial_score < score) {
//
//            // Accept the rotation.
//            score = trial_score;
//
//            force[0] = trial_force[0];
//            force[1] = trial_force[1];
//            force[2] = trial_force[2];
//
//            torque[0] = trial_torque[0];
//            torque[1] = trial_torque[1];
//            torque[2] = trial_torque[2];
//
//        } else {
//
//            // Reject the trial and undo the rotation.
//            rotate(cx, cy, cz, nc, ux, uy, uz, -angle);
//
//            angle *= ANGLE_FACTOR;
//        }
//    }
//
//    return score;
//}

template<class TemplatePosView, class TemplateTypeView>
KOKKOS_INLINE_FUNCTION
float local_translate(
    float *cx, float *cy, float *cz,
    const int *ctype, const int nc,
    const TemplatePosView &tpos,
    const TemplateTypeView &ttype,
    const int nt,
    const int *pair,
    const bool element_match,
    float sc)
{
    constexpr float CONVERGE = 1.0e-4f;
    constexpr float LEPSO = 1.0e-8f;
    constexpr float RATIO = 0.7f;
    constexpr int LMAX = 200;

    float step = 0.1f;
    float force[3], torque[3];

    paired_score(cx, cy, cz, ctype, nc,
                 tpos, ttype, nt, pair, element_match,
                 force, torque);

    for (int j = 0; j < LMAX; ++j) {

        const float fnorm = sqrtf(
            force[0]*force[0] +
            force[1]*force[1] +
            force[2]*force[2]);

        if (fnorm == 0.0f) break;

        const float dx = step * force[0] / fnorm;
        const float dy = step * force[1] / fnorm;
        const float dz = step * force[2] / fnorm;

        for (int c = 0; c < nc; ++c) {
            cx[c] += dx;
            cy[c] += dy;
            cz[c] += dz;
        }

        float cforce[3], ctorque[3];

        const float sc_now = paired_score(
            cx, cy, cz, ctype, nc,
            tpos, ttype, nt, pair, element_match,
            cforce, ctorque);

        if (2.0f*fabsf(sc_now-sc) <
            CONVERGE*(fabsf(sc)+fabsf(sc_now)+LEPSO)) {
            sc = sc_now;
            break;
        }

        if (sc_now < sc) {
            sc = sc_now;

            for (int m = 0; m < 3; ++m)
                force[m] = cforce[m];

        } else {
            for (int c = 0; c < nc; ++c) {
                cx[c] -= dx;
                cy[c] -= dy;
                cz[c] -= dz;
            }
            step *= RATIO;
        }
    }

    return sc;
}

//template<class TemplatePosView>
//KOKKOS_INLINE_FUNCTION
//float local_translate(
//    float *cx, float *cy, float *cz,
//    const int nc,
//    const TemplatePosView &tpos,
//    const int nt,
//    const int *pair)
//{
//    float force[3], torque[3];
//
//    float score = paired_score(
//        cx, cy, cz, nc, tpos, nt, pair,
//        force, torque);
//
//    float step = 0.1f;
//
//    constexpr int MAX_ITER = 200;
//    constexpr float STEP_FACTOR = 0.7f;
//    constexpr float MIN_STEP = 1.0e-6f;
//    constexpr float MIN_FORCE = 1.0e-8f;
//
//    for (int iter = 0; iter < MAX_ITER; ++iter) {
//
//        const float fnorm = sqrtf(
//            force[0]*force[0] +
//            force[1]*force[1] +
//            force[2]*force[2]);
//
//        if (fnorm < MIN_FORCE || step < MIN_STEP)
//            break;
//
//        // Unit vector along the force direction
//        const float ux = force[0] / fnorm;
//        const float uy = force[1] / fnorm;
//        const float uz = force[2] / fnorm;
//
//        const float dx = step * ux;
//        const float dy = step * uy;
//        const float dz = step * uz;
//
//        // Trial translation
//        for (int c = 0; c < nc; ++c) {
//            cx[c] += dx;
//            cy[c] += dy;
//            cz[c] += dz;
//        }
//
//        float trial_force[3], trial_torque[3];
//
//        const float trial_score = paired_score(
//            cx, cy, cz, nc, tpos, nt, pair,
//            trial_force, trial_torque);
//
//        if (trial_score < score) {
//
//            // Accept translation
//            score = trial_score;
//
//            force[0] = trial_force[0];
//            force[1] = trial_force[1];
//            force[2] = trial_force[2];
//
//            torque[0] = trial_torque[0];
//            torque[1] = trial_torque[1];
//            torque[2] = trial_torque[2];
//
//        } else {
//
//            // Reject translation and restore coordinates
//            for (int c = 0; c < nc; ++c) {
//                cx[c] -= dx;
//                cy[c] -= dy;
//                cz[c] -= dz;
//            }
//
//            step *= STEP_FACTOR;
//        }
//    }
//
//    return score;
//}

template<int MAX_CLUSTER, int MAX_TEMPLATE,
         class TemplatePosView, class TemplateTypeView>
KOKKOS_INLINE_FUNCTION
float align_single(
    float *cx, float *cy, float *cz,
    const int *ctype, const int nc,
    const TemplatePosView &tpos,
    const TemplateTypeView &ttype,
    const int nt,
    const bool element_match,
    const bool enable_translation)
{
    if (nc < nt || nt <= 0)
        return 9999.0f;

    int pair[MAX_CLUSTER];

    // Greedy pairing: performed once per starting configuration.
    float sc = score<MAX_CLUSTER, MAX_TEMPLATE>(
    cx, cy, cz, ctype, nc,
    tpos, ttype, nt, element_match, pair);

    sc = local_rotate(
        cx, cy, cz, ctype, nc,
        tpos, ttype, nt, pair, element_match, sc);
    
    if (enable_translation) {
        sc = local_translate(
            cx, cy, cz, ctype, nc,
            tpos, ttype, nt, pair, element_match, sc);
}

return sc;
}

template<class RandomGenerator>
KOKKOS_INLINE_FUNCTION
void random_rotate(
    float *cx, float *cy, float *cz,
    const int nc,
    RandomGenerator &rng)
{
    const float ux = 0.5f - rng.frand();
    const float uy = 0.5f - rng.frand();
    const float uz = 0.5f - rng.frand();
    const float degree = 360.0f * rng.frand();

    rotate(cx, cy, cz, nc, ux, uy, uz, degree);
}

template<class TemplatePosView>
struct ScaledTemplateView {
    TemplatePosView pos;
    float scale;

    KOKKOS_INLINE_FUNCTION
    float operator()(const int i, const int j) const
    {
        return scale * static_cast<float>(pos(i,j));
    }
};

template<int MAX_CLUSTER, int MAX_TEMPLATE,
         class TemplatePosView, class TemplateTypeView,
         class RandomGenerator>
KOKKOS_INLINE_FUNCTION
float align_random(
    const float *ox, const float *oy, const float *oz,
    const int *ctype, const int nc,
    const TemplatePosView &tpos,
    const TemplateTypeView &ttype,
    const int nt,
    const bool element_match,
    const bool enable_translation,
    const int nrandom,
    const int nconv,
    RandomGenerator &rng)
{
    if (nc < nt) return 9999.0f;

    float cx[MAX_CLUSTER];
    float cy[MAX_CLUSTER];
    float cz[MAX_CLUSTER];

    float best_score = 9999.0f;
    int iconv = 0;

    for (int ir = 0; ir < nrandom; ++ir) {

        // Restore original cluster before each random trial.
        for (int k = 0; k < nc; ++k) {
            cx[k] = ox[k];
            cy[k] = oy[k];
            cz[k] = oz[k];
        }

        // Random orientation, using the same sampling scheme as CPU.
        random_rotate(cx, cy, cz, nc, rng);

        // Greedy pairing, iterative rotation, optional translation.
        const float sc =
            align_single<MAX_CLUSTER, MAX_TEMPLATE>(
                cx, cy, cz, ctype, nc,
                tpos, ttype, nt,
                element_match, enable_translation);

        if (sc < best_score) {
            best_score = sc;
            iconv = 0;
        } else {
            ++iconv;
        }

        if (iconv >= nconv) break;
    }

    return best_score;
}

template<int MAX_CLUSTER, int MAX_TEMPLATE,
         class TemplatePosView, class TemplateTypeView,
         class RandomGenerator>
KOKKOS_INLINE_FUNCTION
float align_resize(
    const float *cx, const float *cy, const float *cz,
    const int *ctype, const int nc,
    const TemplatePosView &tpos,
    const TemplateTypeView &ttype,
    const int nt,
    const bool element_match,
    const bool enable_translation,
    const int nrandom,
    const int nconv,
    const int nresize,
    RandomGenerator &rng)
{
    // Preserve original behavior when resize is disabled.
    if (nresize == 0) {
        return align_random<MAX_CLUSTER, MAX_TEMPLATE>(
            cx, cy, cz, ctype, nc,
            tpos, ttype, nt,
            element_match, enable_translation,
            nrandom, nconv, rng);
    }

    if (nc < nt) return 9999.0f;

    // CPU bond2_compare() uses the first nt cluster atoms.
    float cluster_bond2 = 0.0f;
    float template_bond2 = 0.0f;

    for (int i = 0; i < nt; ++i) {
        cluster_bond2 +=
            cx[i]*cx[i] + cy[i]*cy[i] + cz[i]*cz[i];

        const float tx = static_cast<float>(tpos(i,0));
        const float ty = static_cast<float>(tpos(i,1));
        const float tz = static_cast<float>(tpos(i,2));

        template_bond2 += tx*tx + ty*ty + tz*tz;
    }

    // Match the CPU's bond2_compare() zero-denominator behavior.
    const float fac = (cluster_bond2 == 0.0f ||
                       template_bond2 == 0.0f)
                    ? 1.0f
                    : sqrtf(cluster_bond2 / template_bond2);

    constexpr float delta = 0.02f;
    const float start = 1.0f - delta * (nresize / 2);

    float final_score = 9999.0f;

    for (int k = 0; k < nresize; ++k) {
        const float scale = fac * (start + delta*k);

        ScaledTemplateView<TemplatePosView> scaled{tpos, scale};

        const float sc =
            align_random<MAX_CLUSTER, MAX_TEMPLATE>(
                cx, cy, cz, ctype, nc,
                scaled, ttype, nt,
                element_match, enable_translation,
                nrandom, nconv, rng);

        if (sc < final_score)
            final_score = sc;
    }

    return final_score;
}

}  // namespace ALIGN_KK
}  // namespace LAMMPS_NS

#endif
