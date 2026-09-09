#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <mpi.h>

#include "chefsi_layout.h"
#include "effective_potential_nloc.h"

namespace nloc_fixture_dump {

inline int& current_element() {
  static int value = -1;
  return value;
}

inline bool write_exact(FILE* fp, const void* src, const size_t bytes) {
  return std::fwrite(src, 1, bytes, fp) == bytes;
}

template<typename T>
inline bool dump_effective_potential_nloc(
    const Effective_potential_nloc<T>& vnloc, const double* psi_packed,
    const size_t K, const size_t M, const double dv, const char* path) {
  if (path == nullptr || path[0] == '\0') {
    return false;
  }
  if (vnloc.offsets.empty()) {
    std::fprintf(stderr, "NLOC_DUMP: empty vnloc offsets\n");
    return false;
  }

  const uint64_t version = 1;
  const uint64_t n_chi = static_cast<uint64_t>(vnloc.offsets.back());
  const uint64_t n_proj = static_cast<uint64_t>(vnloc.nloc_projectors.size());
  const uint64_t K_u = static_cast<uint64_t>(K);
  const uint64_t M_u = static_cast<uint64_t>(M);

  FILE* fp = std::fopen(path, "wb");
  if (fp == nullptr) {
    std::perror("NLOC_DUMP fopen");
    return false;
  }

  bool ok = true;
  ok = ok && write_exact(fp, "NLOCFIX1", 8);
  ok = ok && write_exact(fp, &version, sizeof(version));
  ok = ok && write_exact(fp, &K_u, sizeof(K_u));
  ok = ok && write_exact(fp, &M_u, sizeof(M_u));
  ok = ok && write_exact(fp, &n_chi, sizeof(n_chi));
  ok = ok && write_exact(fp, &n_proj, sizeof(n_proj));
  ok = ok && write_exact(fp, &dv, sizeof(dv));

  for (size_t ip = 0; ip < vnloc.nloc_projectors.size(); ++ip) {
    const auto& proj = vnloc.nloc_projectors[ip];
    const uint64_t nrow = static_cast<uint64_t>(proj.nrow);
    const uint64_t ncol = static_cast<uint64_t>(proj.ncol);
    const uint64_t chi_offset = static_cast<uint64_t>(vnloc.offsets[ip]);
    const int32_t atom_index = static_cast<int32_t>(proj.atom_index);

    ok = ok && write_exact(fp, &nrow, sizeof(nrow));
    ok = ok && write_exact(fp, &ncol, sizeof(ncol));
    ok = ok && write_exact(fp, &chi_offset, sizeof(chi_offset));
    ok = ok && write_exact(fp, &atom_index, sizeof(atom_index));
    for (size_t ir = 0; ir < static_cast<size_t>(nrow); ++ir) {
      const size_t grid_index = static_cast<size_t>(proj.index_data()[ir]);
      ok = ok && write_exact(fp, &grid_index, sizeof(grid_index));
    }
    ok = ok &&
         write_exact(fp, proj.chi.data,
                     static_cast<size_t>(nrow) * static_cast<size_t>(ncol) * sizeof(T));
    ok = ok && write_exact(fp, proj.gamma.data, static_cast<size_t>(ncol) * sizeof(T));
  }

  const size_t wf_elems = chefsi_layout::wf_stride(K, M);
  ok = ok && write_exact(fp, psi_packed, wf_elems * sizeof(double));
  std::fclose(fp);

  if (!ok) {
    std::fprintf(stderr, "NLOC_DUMP: fwrite failed for %s\n", path);
    return false;
  }

  std::printf(
      "NLOC_DUMP wrote %s K=%zu M=%zu n_chi=%llu n_proj=%llu dv=%.9f elem=%d\n",
      path, K, M, static_cast<unsigned long long>(n_chi),
      static_cast<unsigned long long>(n_proj), dv, current_element());
  return true;
}

template<typename T>
inline void maybe_dump_after_nloc_init(
    const Effective_potential_nloc<T>& vnloc, const double* psi_packed,
    const size_t K, const size_t M, const double dv, const int /*comm_rank*/,
    MPI_Comm /*comm*/) {
  const char* path = std::getenv("NLOC_DUMP_PATH");
  if (path == nullptr || path[0] == '\0') {
    return;
  }

  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Barrier(MPI_COMM_WORLD);

  const int dump_rank =
      std::getenv("NLOC_DUMP_RANK") != nullptr
          ? std::atoi(std::getenv("NLOC_DUMP_RANK"))
          : 0;
  const int dump_element =
      std::getenv("NLOC_DUMP_ELEMENT") != nullptr
          ? std::atoi(std::getenv("NLOC_DUMP_ELEMENT"))
          : 0;

  int exit_flag = 0;
  if (world_rank == dump_rank && current_element() == dump_element) {
    static bool already_dumped = false;
    if (!already_dumped) {
      if (dump_effective_potential_nloc(vnloc, psi_packed, K, M, dv, path)) {
        already_dumped = true;
        exit_flag = 1;
      }
    }
  }

  MPI_Bcast(&exit_flag, 1, MPI_INT, dump_rank, MPI_COMM_WORLD);
  if (exit_flag != 0 && std::getenv("NLOC_DUMP_EXIT") != nullptr) {
    if (world_rank == dump_rank) {
      std::printf("NLOC_DUMP_EXIT=1 — exiting after fixture dump\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    std::exit(0);
  }
}

}  // namespace nloc_fixture_dump
