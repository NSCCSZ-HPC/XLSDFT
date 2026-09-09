#include "chefsi.h"

#include "axpy_opt.h"
#include "chefsi_layout.h"
#include "laplacian_opt.h"
#include "nloc_opt.h"
#include "sp_gemm_opt.h"
#include "sp_syrk_opt.h"
#include "sr_gemm_opt.h"
#include "tools.h"
#include "transpose.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

#ifndef NT
#define NT 38
#endif

namespace {

struct NlocOptContext {
  NlocProjector* projectors = nullptr;
  size_t n_proj = 0;
  NlocAtomGroup* atom_groups = nullptr;
  size_t n_atom_groups = 0;
  uint32_t* back_offsets = nullptr;
  uint32_t* back_entries = nullptr;
  size_t back_entries_count = 0;
  NlocPlan plan{};
};

template <typename Pool>
static void build_nloc_opt(const Effective_potential_nloc<double>& vnloc,
                           const size_t K, Pool& pool_fast,
                           NlocOptContext& ctx) {
  ctx.n_proj = vnloc.nloc_projectors.size();
  if (ctx.n_proj == 0 || K == 0) {
    ctx.plan = {};
    return;
  }

  ctx.projectors = pool_alloc<NlocProjector>(pool_fast, ctx.n_proj);
  for (size_t i = 0; i < ctx.n_proj; ++i) {
    const auto& src = vnloc.nloc_projectors[i];
    ctx.projectors[i].nrow = src.nrow;
    ctx.projectors[i].ncol = src.ncol;
    ctx.projectors[i].chi_offset = vnloc.offsets[i];
    ctx.projectors[i].atom_index = src.atom_index;
    if (src.nrow > 0) {
      uint32_t* index_hbm = pool_alloc<uint32_t>(pool_fast, src.nrow);
      std::memcpy(index_hbm, src.index_data(), src.nrow * sizeof(uint32_t));
      ctx.projectors[i].index = index_hbm;
    } else {
      ctx.projectors[i].index = nullptr;
    }
    ctx.projectors[i].chi = src.chi.data;
    ctx.projectors[i].gamma = src.gamma.data;
  }

  uint32_t* scratch_k = pool_alloc<uint32_t>(pool_fast, K);
  ctx.atom_groups = pool_alloc<NlocAtomGroup>(pool_fast, ctx.n_proj);
  ctx.back_offsets = pool_alloc<uint32_t>(pool_fast, K + 1);

  nloc_plan_build(ctx.atom_groups, &ctx.n_atom_groups, ctx.back_offsets, nullptr,
                  &ctx.back_entries_count, ctx.projectors, ctx.n_proj, K,
                  scratch_k);

  ctx.back_entries =
      pool_alloc<uint32_t>(pool_fast, ctx.back_entries_count);

  nloc_plan_build(ctx.atom_groups, &ctx.n_atom_groups, ctx.back_offsets,
                  ctx.back_entries, &ctx.back_entries_count, ctx.projectors,
                  ctx.n_proj, K, scratch_k);

  ctx.plan.n_atom_groups = ctx.n_atom_groups;
  ctx.plan.atom_groups = ctx.atom_groups;
  ctx.plan.K = K;
  ctx.plan.back_offsets = ctx.back_offsets;
  ctx.plan.back_entries = ctx.back_entries;
  ctx.plan.back_entries_count = ctx.back_entries_count;
}

static size_t nloc_total_chi(const Effective_potential_nloc<double>& vnloc) {
  if (vnloc.offsets.empty()) {
    return 0;
  }
  return vnloc.offsets.back();
}

static void apply_nloc_opt(const size_t K, const size_t M, const size_t n_chi,
                           NlocOptContext& ctx, const double* psi_packed,
                           double* h_packed, const int out_tile,
                           double* chi_scratch, const double dv,
                           NlocOptStats* stats = nullptr) {
  if (ctx.n_proj == 0) {
    return;
  }
  const NlocProjector* projectors = ctx.projectors;
  nloc_project(ctx.plan, K, M, n_chi, projectors, chi_scratch, psi_packed, dv,
               stats);
  nloc_apply_gamma(ctx.plan, M, n_chi, projectors, chi_scratch, stats);
  if (out_tile == 16) {
    nloc_backproject<16>(ctx.plan, K, M, n_chi, projectors, h_packed,
                         chi_scratch, stats);
  } else if (out_tile == -16) {
    nloc_backproject<-16>(ctx.plan, K, M, n_chi, projectors, h_packed,
                          chi_scratch, stats);
  } else {
    nloc_backproject<-32>(ctx.plan, K, M, n_chi, projectors, h_packed,
                          chi_scratch, stats);
  }
}

}  // namespace

template <typename T>
void Chefsi<T>::chebyshev_filtering_column_wise_mp_opt(
    double*& psi, double*& psi_buf, double*& psi_m1, T const* const Vloc,
    const Effective_potential_nloc<T>& Vnloc, const bool print_flag,
    Memory_pool<T, Fast_memory>& pool_fast,
    Memory_pool<T, Capacity_memory>& pool_cap) {
  (void)pool_cap;
  Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);

#ifdef ENABLE_CHEFSI_TIMER
  this->chefsi_timer.filter.start();
#endif

  const Vertices_3D& local_vertices_3d =
      this->domain_vertices.get_3D_local_vertices();
  const uint K = local_vertices_3d.get_size();
  const uint nb = this->domain_vertices.get_4D_local_vertices().nb;
  const uint ni = local_vertices_3d.ni;
  const uint nj = local_vertices_3d.nj;
  const uint nk = local_vertices_3d.nk;

  if (K == 0 || nb == 0 || this->chefsi_control.chebyshev_filter_degree == 0) {
    return;
  }

  const double e = 0.5 * (this->lanczos.eig_max - this->lanczos.lambda_cutoff);
  const double c = 0.5 * (this->lanczos.eig_max + this->lanczos.lambda_cutoff);

  Stencil<T> stencil_temp;
  stencil_temp.deepcopy_mp(this->stencil, pool_fast);
  stencil_temp.coeffs_scale_self(-0.5);
  stencil_temp.shift_D2_coeffs(-c);
  const double coef0 = stencil_temp.get_D2_coef0();

  double sigma = e / (this->lanczos.eig_min - c);
  const double sigma1 = sigma;
  const double gamma = 2.0 / sigma1;

  NlocOptContext nloc_ctx;
  build_nloc_opt(Vnloc, K, pool_fast, nloc_ctx);
  const size_t n_chi = nloc_total_chi(Vnloc);
  double* chi_scratch = nullptr;
  if (n_chi > 0) {
    chi_scratch = pool_fast.allocate((nb / 16) * n_chi * 16);
  }

#ifdef ENABLE_CHEFSI_TIMER
  this->chefsi_timer.filter_lap.start();
#endif
  laplacian_4d<16>(ni, nj, nk, nb, psi_buf, psi, Vloc,
                   stencil_temp.get_D2_coeffs_x(),
                   stencil_temp.get_D2_coeffs_y(),
                   stencil_temp.get_D2_coeffs_z(), coef0);
#ifdef ENABLE_CHEFSI_TIMER
  this->chefsi_timer.filter_lap.stop();
  this->chefsi_timer.filter_nloc.start();
#endif
  apply_nloc_opt(K, nb, n_chi, nloc_ctx, psi, psi_buf, 16, chi_scratch,
                 this->mesh_control.delta_V);
#ifdef ENABLE_CHEFSI_TIMER
  this->chefsi_timer.filter_nloc.stop();
  this->chefsi_timer.filter_product.start();
#endif
  chefsi_layout::scale_tile16_sr(psi_buf, sigma1 / e, nb, K);
#ifdef ENABLE_CHEFSI_TIMER
  this->chefsi_timer.filter_product.stop();
#endif

  if (this->chefsi_control.chebyshev_filter_degree == 1) {
    std::swap(psi, psi_buf);
    stencil_temp.destructor_mp();
#ifdef ENABLE_CHEFSI_TIMER
    this->chefsi_timer.filter.stop();
#endif
    (void)print_flag;
    return;
  }

  std::swap(psi_m1, psi);
  std::swap(psi, psi_buf);

  for (uint j = 2; j <= this->chefsi_control.chebyshev_filter_degree; ++j) {
    const double sigma2 = 1.0 / (gamma - sigma);

#ifdef ENABLE_CHEFSI_TIMER
    this->chefsi_timer.filter_lap.start();
#endif
    laplacian_4d<16>(ni, nj, nk, nb, psi_buf, psi, Vloc,
                     stencil_temp.get_D2_coeffs_x(),
                     stencil_temp.get_D2_coeffs_y(),
                     stencil_temp.get_D2_coeffs_z(), coef0);
#ifdef ENABLE_CHEFSI_TIMER
    this->chefsi_timer.filter_lap.stop();
    this->chefsi_timer.filter_nloc.start();
#endif
    apply_nloc_opt(K, nb, n_chi, nloc_ctx, psi, psi_buf, 16, chi_scratch,
                   this->mesh_control.delta_V);
#ifdef ENABLE_CHEFSI_TIMER
    this->chefsi_timer.filter_nloc.stop();
    this->chefsi_timer.filter_product.start();
#endif
    const double vscal = 2.0 * sigma2 / e;
    const double vscal2 = sigma * sigma2;
    cheb_rec_axpy<16>(psi_buf, vscal, psi_m1, vscal2, nb, K);
#ifdef ENABLE_CHEFSI_TIMER
    this->chefsi_timer.filter_product.stop();
#endif

    std::swap(psi_m1, psi);
    std::swap(psi, psi_buf);
    sigma = sigma2;
  }

  stencil_temp.destructor_mp();
#ifdef ENABLE_CHEFSI_TIMER
  this->chefsi_timer.filter.stop();
#endif
  (void)print_flag;
}

template <typename T>
void Chefsi<T>::hamiltonian_product_mp_opt(
    double* h_packed, const double* psi_tile16, double* psi_tile16t_out,
    T const* const Vloc, const Effective_potential_nloc<T>& Vnloc,
    const bool print_flag, Memory_pool<T, Fast_memory>& pool_fast,
    Memory_pool<T, Capacity_memory>& pool_cap) {
  (void)pool_cap;
  Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);

#ifdef ENABLE_CHEFSI_TIMER
  this->chefsi_timer.H_psi.start();
#endif

  const Vertices_3D& local_vertices_3d =
      this->domain_vertices.get_3D_local_vertices();
  const uint K = local_vertices_3d.get_size();
  const uint nb = this->domain_vertices.get_4D_local_vertices().nb;
  const uint ni = local_vertices_3d.ni;
  const uint nj = local_vertices_3d.nj;
  const uint nk = local_vertices_3d.nk;

  Stencil<T> stencil_temp;
  stencil_temp.deepcopy_mp(this->stencil, pool_fast);
  stencil_temp.coeffs_scale_self(-0.5);
  const double coef0 = stencil_temp.get_D2_coef0();

  laplacian_4d<-32>(ni, nj, nk, nb, h_packed, psi_tile16, Vloc,
                    stencil_temp.get_D2_coeffs_x(),
                    stencil_temp.get_D2_coeffs_y(),
                    stencil_temp.get_D2_coeffs_z(), coef0, psi_tile16t_out);

  NlocOptContext nloc_ctx;
  build_nloc_opt(Vnloc, K, pool_fast, nloc_ctx);
  if (nloc_ctx.n_proj != 0) {
    const size_t n_chi_h = nloc_total_chi(Vnloc);
    double* chi_scratch = pool_fast.allocate((nb / 16) * n_chi_h * 16);
    apply_nloc_opt(K, nb, n_chi_h, nloc_ctx, psi_tile16, h_packed, -32,
                   chi_scratch, this->mesh_control.delta_V);
  }
  if (psi_tile16t_out != nullptr) {
    chefsi_layout::zero_sp_padding_tile16t(nb, K, psi_tile16t_out);
  }
  chefsi_layout::zero_sp_padding_tile32t(nb, K, h_packed);

  stencil_temp.destructor_mp();

#ifdef ENABLE_CHEFSI_TIMER
  this->chefsi_timer.H_psi.stop();
#endif
  (void)print_flag;
}

template <typename T>
void Chefsi<T>::project_hamiltonian_mp_opt(const double* psi_tile16t,
                                           const double* h_tile32t, double* hp,
                                           double* mp, const bool print_flag,
                                           Memory_pool<T, Fast_memory>& pool_fast,
                                           Memory_pool<T, Capacity_memory>& pool_cap) {
  (void)pool_cap;
  Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);

  const uint nb = this->dp_domain_vertices.shared_vertices.get_nb();
  const uint K = this->dp_domain_vertices.local_vertices.Vertices_3D::get_size();
  const int K_IN = static_cast<int>(chefsi_layout::k_in(K));

  double* proj_nt = pool_fast.allocate(NT * nb * nb);

  chefsi_layout::zero_sp_padding_tile16t(nb, K, const_cast<double*>(psi_tile16t));
  chefsi_layout::zero_sp_padding_tile32t(nb, K, const_cast<double*>(h_tile32t));

#ifdef ENABLE_CHEFSI_TIMER
  this->chefsi_timer.projection_gemm.start();
#endif
  sp_gemm(static_cast<int>(nb), K_IN, hp, proj_nt,
          const_cast<double*>(psi_tile16t), const_cast<double*>(h_tile32t));
#ifdef ENABLE_CHEFSI_TIMER
  this->chefsi_timer.projection_gemm.stop();
  this->chefsi_timer.projection_syrk.start();
#endif
  sp_syrk(static_cast<int>(nb), K_IN, mp, proj_nt,
          const_cast<double*>(psi_tile16t));
#ifdef ENABLE_CHEFSI_TIMER
  this->chefsi_timer.projection_syrk.stop();
#endif

  (void)print_flag;
}

template <typename T>
void Chefsi<T>::subspace_rotation_mp_opt(double* psi_tile16_out,
                                         const double* psi_tile16t_in,
                                         double* transpose_scratch, double* qp,
                                         const bool print_flag,
                                         Memory_pool<T, Fast_memory>& pool_fast,
                                         Memory_pool<T, Capacity_memory>& pool_cap) {
  (void)pool_cap;
  Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);

  const uint nb = this->dp_domain_vertices.shared_vertices.get_nb();
  const uint K = this->dp_domain_vertices.local_vertices.Vertices_3D::get_size();
  const int K_OUT = static_cast<int>(chefsi_layout::k_out(K));

  transpose_16t_to_16(nb, K, transpose_scratch, psi_tile16t_in);
  transpose_square_colmajor_to_rowmajor(qp, nb);

#ifdef ENABLE_CHEFSI_TIMER
  this->chefsi_timer.rotation.start();
#endif
  sr_gemm(static_cast<int>(nb), K_OUT, psi_tile16_out, transpose_scratch, qp);

#ifdef ENABLE_CHEFSI_TIMER
  this->chefsi_timer.rotation.stop();
#endif
  (void)print_flag;
}

template <typename T>
void Chefsi<T>::run_mp_opt(T*& eigen_vectors_in, T*& eigen_vectors_out,
                           T* const eigen_values, T const* const Vloc,
                           const Effective_potential_nloc<T>& Vnloc,
                           const bool print_flag,
                           Memory_pool<T, Fast_memory>& pool_fast,
                           Memory_pool<T, Capacity_memory>& pool_cap) {
  Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
  Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);

  if (this->domain_vertices.get_4D_shared_vertices().get_size() == 0) {
    return;
  }

#ifdef ENABLE_CHEFSI_TIMER
  this->chefsi_timer.reset();
  this->chefsi_timer.chefsi.start();
#endif

  const Vertices_3D& local_vertices_3d =
      this->domain_vertices.get_3D_local_vertices();
  const uint K = local_vertices_3d.get_size();
  const uint nb = this->domain_vertices.get_4D_local_vertices().nb;
  const uint nstates_shared = this->domain_vertices.get_4D_shared_vertices().nb;
  const size_t wf_elems = chefsi_layout::wf_stride(K, nb);

  double* psi = reinterpret_cast<double*>(eigen_vectors_in);
  double* psi_buf = reinterpret_cast<double*>(eigen_vectors_out);
  double* psi_m1 = this->opt_third_panel;
  if (psi_m1 != nullptr) {
    assert(this->opt_third_panel_elems >= wf_elems);
    assert(psi_m1 != psi && psi_m1 != psi_buf && psi != psi_buf);
  } else {
    psi_m1 = pool_fast.allocate(wf_elems);
  }
  std::memset(psi_m1, 0, wf_elems * sizeof(double));

  const uint niter = unlikely(this->is_very_first)
                         ? this->chefsi_control.rho_trigger
                         : this->chefsi_control.max_iter;

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  for (uint iter = 0; iter < niter; ++iter) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast2(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap2(pool_cap);

#ifdef ENABLE_CHEFSI_TIMER
    this->chefsi_timer.lanczos.start();
#endif
    const bool if_calculate = this->is_very_first && iter == 0;
    this->lanczos.run_mp(
        this->domain_vertices.comm, Vloc, local_vertices_3d, Vnloc, this->stencil,
        T(this->mesh_control.delta_V), this->single_band_exarr_mpi_package,
        if_calculate, eigen_values[nstates_shared - 1], eigen_values[0],
        print_flag, pool_fast, pool_cap);
#ifdef ENABLE_CHEFSI_TIMER
    this->chefsi_timer.lanczos.stop();
#endif

    this->chebyshev_filtering_column_wise_mp_opt(
        psi, psi_buf, psi_m1, Vloc, Vnloc, print_flag, pool_fast, pool_cap);
    if (psi == psi_buf) {
      std::swap(psi_buf, psi_m1);
    }

    double* h_packed = psi_buf;
    double* psi_tile16t = psi_m1;
    this->hamiltonian_product_mp_opt(h_packed, psi, psi_tile16t, Vloc, Vnloc,
                                     print_flag, pool_fast, pool_cap);

    T* hp = pool_fast.allocate(nstates_shared * nstates_shared);
    T* mp = pool_fast.allocate(nstates_shared * nstates_shared);
    this->project_hamiltonian_mp_opt(psi_tile16t, h_packed,
                                     reinterpret_cast<double*>(hp),
                                     reinterpret_cast<double*>(mp), print_flag,
                                     pool_fast, pool_cap);

    this->subspace_diagonalization_mp_opt(hp, mp, eigen_values, print_flag,
                                          pool_fast, pool_cap);

    // psi is dead after final H / projection; reuse as SR transpose scratch.
    this->subspace_rotation_mp_opt(psi_buf, psi_tile16t, psi,
                                   reinterpret_cast<double*>(hp), print_flag,
                                   pool_fast, pool_cap);

    if (iter + 1 < niter) {
      std::swap(psi, psi_buf);
    }
  }

  if (unlikely(this->is_very_first)) {
    this->is_very_first = false;
  }

  eigen_vectors_in = reinterpret_cast<T*>(psi_buf);
  eigen_vectors_out = reinterpret_cast<T*>(psi);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
    std::cout << "The Chefsi run_mp_opt took " << Tools::time_cost(begin, end)
              << "." << std::endl;
  }

#ifdef ENABLE_CHEFSI_TIMER
  this->chefsi_timer.chefsi.stop();
  if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
    this->chefsi_timer.show();
  }
  this->chefsi_flop_counter.set_data(
      this->domain_vertices.shared_vertices, Vnloc, niter,
      this->chefsi_control.chebyshev_filter_degree);
  this->chefsi_performance.update();
#endif
}

template void Chefsi<double>::chebyshev_filtering_column_wise_mp_opt(
    double*& psi, double*& psi_buf, double*& psi_m1, double const* const Vloc,
    const Effective_potential_nloc<double>& Vnloc, const bool print_flag,
    Memory_pool<double, Fast_memory>& pool_fast,
    Memory_pool<double, Capacity_memory>& pool_cap);

template void Chefsi<double>::hamiltonian_product_mp_opt(
    double* h_packed, const double* psi_tile16, double* psi_tile16t_out,
    double const* const Vloc, const Effective_potential_nloc<double>& Vnloc,
    const bool print_flag, Memory_pool<double, Fast_memory>& pool_fast,
    Memory_pool<double, Capacity_memory>& pool_cap);

template void Chefsi<double>::project_hamiltonian_mp_opt(
    const double* psi_tile16t, const double* h_tile32t, double* hp, double* mp,
    const bool print_flag, Memory_pool<double, Fast_memory>& pool_fast,
    Memory_pool<double, Capacity_memory>& pool_cap);

template void Chefsi<double>::subspace_rotation_mp_opt(
    double* psi_tile16_out, const double* psi_tile16t_in,
    double* transpose_scratch, double* qp, const bool print_flag,
    Memory_pool<double, Fast_memory>& pool_fast,
    Memory_pool<double, Capacity_memory>& pool_cap);

template void Chefsi<double>::run_mp_opt(
    double*& eigen_vectors_in, double*& eigen_vectors_out, double* const eigen_values,
    double const* const Vloc, const Effective_potential_nloc<double>& Vnloc,
    const bool print_flag, Memory_pool<double, Fast_memory>& pool_fast,
    Memory_pool<double, Capacity_memory>& pool_cap);
