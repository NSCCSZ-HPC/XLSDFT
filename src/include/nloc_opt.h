#pragma once

#include <cstddef>
#include <cstdint>

#ifndef NCOL
#define NCOL 4
#endif

struct NlocProjector {
  size_t nrow;
  size_t ncol;
  size_t chi_offset;
  int atom_index;
  uint32_t const* index;
  double const* chi;
  double const* gamma;
};

constexpr size_t NLOC_GAMMA_PROJ_NONE = static_cast<size_t>(-1);

struct NlocAtomGroup {
  size_t first_proj = 0;
  size_t end_proj = 0;  // exclusive
  size_t chi_offset = 0;
  size_t gamma_proj = NLOC_GAMMA_PROJ_NONE;
};

struct NlocPlan {
  size_t n_atom_groups = 0;
  const NlocAtomGroup* atom_groups = nullptr;
  size_t K = 0;
  const uint32_t* back_offsets = nullptr;  // size K + 1
  const uint32_t* back_entries = nullptr;
  size_t back_entries_count = 0;
};

struct NlocOptStats {
  double project_ms = 0.0;
  double gamma_ms = 0.0;
  double back_compute_ms = 0.0;
  double back_reduce_ms = 0.0;
};

template <typename Pool>
inline void* pool_alloc_bytes(Pool& pool, const size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  using T = typename Pool::value_type;
  const size_t n_elems =
      (bytes + sizeof(T) - 1) / sizeof(T);
  if (n_elems == 0) {
    return nullptr;
  }
  return static_cast<void*>(pool.allocate(n_elems));
}

template <typename U, typename Pool>
inline U* pool_alloc(Pool& pool, const size_t n) {
  if (n == 0) {
    return nullptr;
  }
  return static_cast<U*>(pool_alloc_bytes(pool, n * sizeof(U)));
}

void nloc_plan_build(NlocAtomGroup* atom_groups, size_t* n_atom_groups,
                     uint32_t* back_offsets, uint32_t* back_entries,
                     size_t* back_entries_count, const NlocProjector* projectors,
                     const size_t n_proj, const size_t K, uint32_t* scratch_k);

void nloc_project(const NlocPlan& plan, const size_t K, const size_t M,
                  const size_t n_chi, const NlocProjector* projectors,
                  double* __restrict__ chi_vector,
                  const double* __restrict__ psi_packed, const double dv,
                  NlocOptStats* stats = nullptr);

void nloc_apply_gamma(const NlocPlan& plan, const size_t M, const size_t n_chi,
                      const NlocProjector* projectors,
                      double* __restrict__ chi_vector,
                      NlocOptStats* stats = nullptr);

template <int OutTile>
void nloc_backproject(const NlocPlan& plan, const size_t K, const size_t M,
                      const size_t n_chi, const NlocProjector* projectors,
                      double* __restrict__ h_packed,
                      const double* __restrict__ chi_vector,
                      NlocOptStats* stats = nullptr);
