#include "nloc_opt.h"

#include <arm_sve.h>
#include <chrono>
#include <cstring>
#include <omp.h>

#include "chefsi_layout.h"

namespace nloc_opt {

constexpr int NLOC_B_TILE = 16;

constexpr uint32_t BACK_IR_BITS = 23;
constexpr uint32_t BACK_IR_MASK = (1u << BACK_IR_BITS) - 1;
// constexpr uint32_t BACK_IP_BITS = 32u - BACK_IR_BITS;
// constexpr size_t BACK_IP_MAX = 1u << BACK_IP_BITS;
// constexpr size_t BACK_IR_MAX = 1u << BACK_IR_BITS;

inline size_t nloc_chi_vector_size(const size_t M, const size_t n_chi) {
  return (M / NLOC_B_TILE) * n_chi * NLOC_B_TILE;
}

template <typename F>
double time_ms(F&& fn) {
  const auto t0 = std::chrono::steady_clock::now();
  fn();
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

inline bool projector_valid(const NlocProjector& P) {
  return P.nrow > 0 && P.ncol > 0;
}

inline uint32_t back_pack_entry(const size_t ip, const size_t ir) {
  return (static_cast<uint32_t>(ip) << BACK_IR_BITS) |
         static_cast<uint32_t>(ir);
}

inline size_t back_unpack_ip(const uint32_t entry) {
  return static_cast<size_t>(entry >> BACK_IR_BITS);
}

inline size_t back_unpack_ir(const uint32_t entry) {
  return static_cast<size_t>(entry & BACK_IR_MASK);
}

void project_atom_group(const size_t psi_bt_base, const size_t chi_bt_base,
                        const NlocAtomGroup& group,
                        const NlocProjector* projectors,
                        double* __restrict__ chi_vector,
                        const double* __restrict__ psi_packed,
                        const svbool_t ptrue, const svfloat64_t dv_vec) {
  for (size_t ip = group.first_proj; ip < group.end_proj; ++ip) {
    const NlocProjector& P = projectors[ip];
    if (!projector_valid(P)) {
      continue;
    }
    const size_t chi_off = P.chi_offset;
    const size_t nrow = P.nrow;

    for (size_t icol = 0; icol < P.ncol; ++icol) {
      svfloat64_t acc0 = svdup_f64(0.0);
      svfloat64_t acc1 = svdup_f64(0.0);
      const double* chi_col = P.chi + icol * nrow;

      for (size_t ir = 0; ir < nrow; ++ir) {
        const double* src =
            psi_packed + psi_bt_base + P.index[ir] * NLOC_B_TILE;
        const svfloat64_t c = svdup_f64(chi_col[ir]);
        acc0 = svmla_x(ptrue, acc0, c, svld1(ptrue, src));
        acc1 = svmla_x(ptrue, acc1, c, svld1(ptrue, src + 8));
      }

      double* dst =
          chi_vector + chi_bt_base + (icol + chi_off) * NLOC_B_TILE;
      const svfloat64_t old0 = svld1(ptrue, dst);
      const svfloat64_t old1 = svld1(ptrue, dst + 8);
      const svfloat64_t inc0 = svmul_x(ptrue, acc0, dv_vec);
      const svfloat64_t inc1 = svmul_x(ptrue, acc1, dv_vec);
      svst1(ptrue, dst, svadd_x(ptrue, old0, inc0));
      svst1(ptrue, dst + 8, svadd_x(ptrue, old1, inc1));
    }
  }
}

void scatter_add_tile_16(double* out, const svfloat64_t inc0,
                         const svfloat64_t inc1, const svbool_t ptrue) {
  const svfloat64_t old0 = svld1(ptrue, out);
  const svfloat64_t old1 = svld1(ptrue, out + 8);
  svst1(ptrue, out, svadd_x(ptrue, old0, inc0));
  svst1(ptrue, out + 8, svadd_x(ptrue, old1, inc1));
}

}  // namespace nloc_opt

void nloc_plan_build(NlocAtomGroup* atom_groups, size_t* n_atom_groups,
                     uint32_t* back_offsets, uint32_t* back_entries,
                     size_t* back_entries_count, const NlocProjector* projectors,
                     const size_t n_proj, const size_t K, uint32_t* scratch_k) {
  *n_atom_groups = 0;
  *back_entries_count = 0;
  if (back_offsets != nullptr) {
    back_offsets[0] = 0;
  }

  if (n_proj == 0 || projectors == nullptr || K == 0 || atom_groups == nullptr ||
      n_atom_groups == nullptr || back_entries_count == nullptr ||
      scratch_k == nullptr) {
    return;
  }

  size_t start = 0;
  size_t n_groups = 0;
  for (size_t ip = 1; ip <= n_proj; ++ip) {
    if (ip == n_proj ||
        projectors[ip].atom_index != projectors[ip - 1].atom_index) {
      NlocAtomGroup group;
      group.first_proj = start;
      group.end_proj = ip;
      group.chi_offset = projectors[start].chi_offset;
      group.gamma_proj = NLOC_GAMMA_PROJ_NONE;
      for (size_t jp = start; jp < ip; ++jp) {
        if (nloc_opt::projector_valid(projectors[jp])) {
          group.gamma_proj = jp;
          break;
        }
      }
      atom_groups[n_groups++] = group;
      start = ip;
    }
  }
  *n_atom_groups = n_groups;

  std::memset(scratch_k, 0, K * sizeof(uint32_t));
  for (size_t ip = 0; ip < n_proj; ++ip) {
    const NlocProjector& P = projectors[ip];
    if (!nloc_opt::projector_valid(P)) {
      continue;
    }
    for (size_t ir = 0; ir < P.nrow; ++ir) {
      const uint32_t g = P.index[ir];
      ++scratch_k[g];
    }
  }

  if (back_offsets == nullptr) {
    uint32_t total = 0;
    for (size_t g = 0; g < K; ++g) {
      total += scratch_k[g];
    }
    *back_entries_count = total;
    return;
  }

  back_offsets[0] = 0;
  for (size_t g = 0; g < K; ++g) {
    back_offsets[g + 1] = back_offsets[g] + scratch_k[g];
  }
  *back_entries_count = static_cast<size_t>(back_offsets[K]);

  if (back_entries == nullptr) {
    return;
  }

  for (size_t g = 0; g < K; ++g) {
    scratch_k[g] = back_offsets[g];
  }

  for (size_t ip = 0; ip < n_proj; ++ip) {
    const NlocProjector& P = projectors[ip];
    if (!nloc_opt::projector_valid(P)) {
      continue;
    }
    for (size_t ir = 0; ir < P.nrow; ++ir) {
      const uint32_t g = P.index[ir];
      const uint32_t pos = scratch_k[g]++;
      back_entries[pos] = nloc_opt::back_pack_entry(ip, ir);
    }
  }
}

void nloc_project(const NlocPlan& plan, const size_t K, const size_t M,
                  const size_t n_chi, const NlocProjector* projectors,
                  double* __restrict__ chi_vector,
                  const double* __restrict__ psi_packed, const double dv,
                  NlocOptStats* stats) {
  const size_t k_out = chefsi_layout::k_out(K);
  const size_t band_tiles = M / nloc_opt::NLOC_B_TILE;
  const svbool_t ptrue = svptrue_b64();
  const svfloat64_t dv_vec = svdup_f64(dv);

  auto run = [&]() {
#pragma omp parallel for
    for (size_t ic = 0; ic < nloc_opt::nloc_chi_vector_size(M, n_chi); ic++) {
      chi_vector[ic] = 0.0;
    }

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t bt = 0; bt < band_tiles; ++bt) {
      for (size_t ia = 0; ia < plan.n_atom_groups; ++ia) {
        const size_t psi_bt_base = bt * k_out * nloc_opt::NLOC_B_TILE;
        const size_t chi_bt_base = bt * n_chi * nloc_opt::NLOC_B_TILE;
        nloc_opt::project_atom_group(psi_bt_base, chi_bt_base,
                                     plan.atom_groups[ia], projectors,
                                     chi_vector, psi_packed, ptrue, dv_vec);
      }
    }
  };

  if (stats != nullptr) {
    stats->project_ms = nloc_opt::time_ms(run);
  } else {
    run();
  }
}

void nloc_apply_gamma(const NlocPlan& plan, const size_t M, const size_t n_chi,
                      const NlocProjector* projectors,
                      double* __restrict__ chi_vector,
                      NlocOptStats* stats) {
  if (plan.n_atom_groups == 0) {
    return;
  }

  const size_t band_tiles = M / nloc_opt::NLOC_B_TILE;

  auto run = [&]() {
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t bt = 0; bt < band_tiles; ++bt) {
      for (size_t ia = 0; ia < plan.n_atom_groups; ++ia) {
        const NlocAtomGroup& group = plan.atom_groups[ia];
        if (group.gamma_proj == NLOC_GAMMA_PROJ_NONE) {
          continue;
        }
        const NlocProjector& P = projectors[group.gamma_proj];
        double* chi_bt = chi_vector + bt * n_chi * nloc_opt::NLOC_B_TILE;
        for (size_t icol = 0; icol < P.ncol; ++icol) {
          double* dst = chi_bt + (P.chi_offset + icol) * nloc_opt::NLOC_B_TILE;
          const double g = P.gamma[icol];
          for (size_t b = 0; b < nloc_opt::NLOC_B_TILE; ++b) {
            dst[b] *= g;
          }
        }
      }
    }
  };

  if (stats != nullptr) {
    stats->gamma_ms = nloc_opt::time_ms(run);
  } else {
    run();
  }
}

namespace nloc_back_opt {

constexpr int NLOC_B_TILE = 16;

inline size_t nloc_K_pad(const size_t K) { return K + ((NT - K % NT) % NT); }

inline size_t nloc_KLOCAL(const size_t K) { return nloc_K_pad(K) / NT; }

template <int OutTile>
inline double* out_ptr(double* base, const size_t g, const size_t bt,
                       const size_t grids, const size_t nb,
                       const size_t KLOCAL) {
  if constexpr (OutTile == 16) {
    const size_t k_out = chefsi_layout::k_out(grids);
    return base + bt * k_out * NLOC_B_TILE + g * NLOC_B_TILE;
  } else if constexpr (OutTile == -16) {
    const size_t tid = g % NT;
    const size_t ploc = g / NT;
    return base + tid * (nb / 16) * KLOCAL * 16 + bt * KLOCAL * 16 + ploc * 16;
  } else {
    const size_t tid = g % NT;
    const size_t ploc = g / NT;
    const size_t bo = bt / 2;
    const size_t half = bt & 1;
    return base + tid * (nb / 32) * KLOCAL * 32 + bo * KLOCAL * 32 + ploc * 32 +
           half * 16;
  }
}

template <int OutTile>
void nloc_backproject_impl(const NlocPlan& plan, const size_t K, const size_t M,
                           const size_t n_chi, const NlocProjector* projectors,
                           double* __restrict__ h_packed,
                           const double* __restrict__ chi_vector,
                           NlocOptStats* stats) {
  const size_t grids = K;
  const size_t band_tiles = M / NLOC_B_TILE;
  const size_t KLOCAL = nloc_KLOCAL(K);
  const size_t ntasks = band_tiles * K;
  const svbool_t ptrue = svptrue_b64();

  auto run = [&]() {
#pragma omp parallel for schedule(static)
    for (size_t task = 0; task < ntasks; ++task) {
      const size_t bt = task / K;
      const size_t g = task % K;

      const uint32_t begin = plan.back_offsets[g];
      const uint32_t end = plan.back_offsets[g + 1];
      if (begin == end) {
        continue;
      }

      const double* chi_bt = chi_vector + bt * n_chi * NLOC_B_TILE;
      svfloat64_t sum0 = svdup_f64(0.0);
      svfloat64_t sum1 = svdup_f64(0.0);

      for (uint32_t q = begin; q < end; ++q) {
        const uint32_t entry = plan.back_entries[q];
        const size_t ip = nloc_opt::back_unpack_ip(entry);
        const size_t ir = nloc_opt::back_unpack_ir(entry);
        const NlocProjector& P = projectors[ip];
        const size_t nrow = P.nrow;
        const double* chi_proj = chi_bt + P.chi_offset * NLOC_B_TILE;
        const double* chi_ir = P.chi + ir;

        for (size_t icol = 0; icol < P.ncol; ++icol) {
          const svfloat64_t c = svdup_f64(chi_ir[icol * nrow]);
          const double* chi_row = chi_proj + icol * NLOC_B_TILE;
          sum0 = svmla_x(ptrue, sum0, c, svld1(ptrue, chi_row));
          sum1 = svmla_x(ptrue, sum1, c, svld1(ptrue, chi_row + 8));
        }
      }

      double* out = out_ptr<OutTile>(h_packed, g, bt, grids, M, KLOCAL);
      nloc_opt::scatter_add_tile_16(out, sum0, sum1, ptrue);
    }
  };

  if (stats != nullptr) {
    stats->back_compute_ms = nloc_opt::time_ms(run);
    stats->back_reduce_ms = 0.0;
  } else {
    run();
  }
}

}  // namespace nloc_back_opt

template <int OutTile>
void nloc_backproject(const NlocPlan& plan, const size_t K, const size_t M,
                      const size_t n_chi, const NlocProjector* projectors,
                      double* __restrict__ h_packed,
                      const double* __restrict__ chi_vector,
                      NlocOptStats* stats) {
  nloc_back_opt::nloc_backproject_impl<OutTile>(
      plan, K, M, n_chi, projectors, h_packed, chi_vector, stats);
}

template void nloc_backproject<16>(const NlocPlan& plan, const size_t K,
                                   const size_t M, const size_t n_chi,
                                   const NlocProjector* projectors,
                                   double* __restrict__ h_packed,
                                   const double* __restrict__ chi_vector,
                                   NlocOptStats* stats);

template void nloc_backproject<-16>(const NlocPlan& plan, const size_t K,
                                     const size_t M, const size_t n_chi,
                                     const NlocProjector* projectors,
                                     double* __restrict__ h_packed,
                                     const double* __restrict__ chi_vector,
                                     NlocOptStats* stats);

template void nloc_backproject<-32>(const NlocPlan& plan, const size_t K,
                                     const size_t M, const size_t n_chi,
                                     const NlocProjector* projectors,
                                     double* __restrict__ h_packed,
                                     const double* __restrict__ chi_vector,
                                     NlocOptStats* stats);
