#ifndef XLSDFT_CHEFSI_LVTX_HPP_
#define XLSDFT_CHEFSI_LVTX_HPP_

#include "xlsdft_backend.h"

#include "linalg.h"

#include <cstdint>
#include <new>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace Xlsdft_chefsi_lvtx {

inline bool communicator_is_self(const MPI_Comm comm) {
    if (comm == MPI_COMM_NULL) {
        return false;
    }
    int size = 0;
    int relation = MPI_UNEQUAL;
    MPI_Comm_size(comm, &size);
    MPI_Comm_compare(comm, MPI_COMM_SELF, &relation);
    return size == 1 &&
           (relation == MPI_IDENT || relation == MPI_CONGRUENT);
}

template<typename T>
inline bool projection_contract_matches(const uint m, const uint k) {
    return std::is_same_v<T, double> &&
           m == static_cast<uint>(Xlsdft_backend::poj_m) &&
           k == static_cast<uint>(Xlsdft_backend::poj_k);
}

template<typename T>
inline bool rotation_contract_matches(const uint m, const uint n) {
    return std::is_same_v<T, double> &&
           m == static_cast<uint>(Xlsdft_backend::rot_m) &&
           n == static_cast<uint>(Xlsdft_backend::rot_n);
}

template<typename T>
int project_hamiltonian(T const* const eigen_vectors,
                        T const* const h_eigen_vectors, T* const hp,
                        T* const mp, double* const workspace,
                        const std::size_t workspace_doubles) {
    if constexpr (std::is_same_v<T, double>) {
        const int poj_status = Xlsdft_backend::dgemm_poj(
            eigen_vectors, h_eigen_vectors, hp, workspace, workspace_doubles);
        if (poj_status != 0) {
            return poj_status;
        }
        return Xlsdft_backend::dsyrk_upper(eigen_vectors, mp, workspace,
                                         workspace_doubles);
    }
    const uint m = static_cast<uint>(Xlsdft_backend::poj_m);
    const uint k = static_cast<uint>(Xlsdft_backend::poj_k);
    Linalg::matrix_product(eigen_vectors, 0, h_eigen_vectors, 1, hp, 1, m, m,
                           k);
    Linalg::matrix_product(eigen_vectors, 0, eigen_vectors, 1, mp, 1, m, m,
                           k);
    return 0;
}

template<typename T>
void project_hamiltonian_fallback(const uint m, const uint k,
                                  T const* const eigen_vectors,
                                  T const* const h_eigen_vectors,
                                  T* const hp, T* const mp) {
    Linalg::matrix_product(eigen_vectors, 0, h_eigen_vectors, 1, hp, 1, m, m,
                           k);
    Linalg::matrix_product(eigen_vectors, 0, eigen_vectors, 1, mp, 1, m, m, k);
}

template<typename T>
int dsygvd_upper(const int n, T* const hp, T* const mp, T* const eigen_values,
                 double* const workspace,
                 const std::size_t workspace_doubles) {
    if constexpr (std::is_same_v<T, double>) {
        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        if (nthreads > Xlsdft_backend::dsygvd_max_threads) {
            nthreads = Xlsdft_backend::dsygvd_max_threads;
        }
        if (nthreads < 1) {
            nthreads = 1;
        }
        return Xlsdft_backend::dsygvd_upper(
            n, hp, n, mp, n, eigen_values, workspace, workspace_doubles,
            nthreads);
    }
    const std::size_t matrix_count = static_cast<std::size_t>(n) * n;
    double* hp_f64 = new (std::align_val_t(64)) double[matrix_count];
    double* mp_f64 = new (std::align_val_t(64)) double[matrix_count];
    double* eigen_values_f64 = new (std::align_val_t(64)) double[n];
    for (std::size_t i = 0; i < matrix_count; ++i) {
        hp_f64[i] = static_cast<double>(hp[i]);
        mp_f64[i] = static_cast<double>(mp[i]);
    }
    const std::size_t ws =
        Xlsdft_backend::dsygvd_workspace_doubles(n);
    double* ws_buf = new (std::align_val_t(64)) double[ws];
    const int info = dsygvd_upper<double>(n, hp_f64, mp_f64, eigen_values_f64,
                                          ws_buf, ws);
    if (info == 0) {
        for (std::size_t i = 0; i < matrix_count; ++i) {
            hp[i] = static_cast<T>(hp_f64[i]);
        }
        for (int i = 0; i < n; ++i) {
            eigen_values[i] = static_cast<T>(eigen_values_f64[i]);
        }
    }
    ::operator delete[](ws_buf, std::align_val_t(64));
    ::operator delete[](eigen_values_f64, std::align_val_t(64));
    ::operator delete[](mp_f64, std::align_val_t(64));
    ::operator delete[](hp_f64, std::align_val_t(64));
    return info;
}

template<typename T>
int dsygvd_upper_opt(const int n, T* const hp, T* const mp, T* const eigen_values,
                     double* const workspace,
                     const std::size_t workspace_doubles) {
    if constexpr (std::is_same_v<T, double>) {
        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        if (nthreads > Xlsdft_backend::dsygvd_max_threads) {
            nthreads = Xlsdft_backend::dsygvd_max_threads;
        }
        if (nthreads < 1) {
            nthreads = 1;
        }
        return Xlsdft_backend::dsygvd_upper_opt(
            n, hp, n, mp, n, eigen_values, workspace, workspace_doubles,
            nthreads);
    }
    const std::size_t matrix_count = static_cast<std::size_t>(n) * n;
    double* hp_f64 = new (std::align_val_t(64)) double[matrix_count];
    double* mp_f64 = new (std::align_val_t(64)) double[matrix_count];
    double* eigen_values_f64 = new (std::align_val_t(64)) double[n];
    for (std::size_t i = 0; i < matrix_count; ++i) {
        hp_f64[i] = static_cast<double>(hp[i]);
        mp_f64[i] = static_cast<double>(mp[i]);
    }
    const std::size_t ws =
        Xlsdft_backend::dsygvd_workspace_doubles(n);
    double* ws_buf = new (std::align_val_t(64)) double[ws];
    const int info = dsygvd_upper_opt<double>(n, hp_f64, mp_f64, eigen_values_f64,
                                              ws_buf, ws);
    if (info == 0) {
        for (std::size_t i = 0; i < matrix_count; ++i) {
            hp[i] = static_cast<T>(hp_f64[i]);
        }
        for (int i = 0; i < n; ++i) {
            eigen_values[i] = static_cast<T>(eigen_values_f64[i]);
        }
    }
    ::operator delete[](ws_buf, std::align_val_t(64));
    ::operator delete[](eigen_values_f64, std::align_val_t(64));
    ::operator delete[](mp_f64, std::align_val_t(64));
    ::operator delete[](hp_f64, std::align_val_t(64));
    return info;
}

template<typename T>
int rotate_subspace(T const* const eigen_vectors_in, T const* const qp,
                    T* const eigen_vectors_out, double* const workspace,
                    const std::size_t workspace_doubles) {
    if constexpr (std::is_same_v<T, double>) {
        return Xlsdft_backend::rotate_raw(eigen_vectors_in, qp,
                                          eigen_vectors_out, workspace,
                                          workspace_doubles);
    }
    const uint m = static_cast<uint>(Xlsdft_backend::rot_m);
    const uint n = static_cast<uint>(Xlsdft_backend::rot_n);
    Linalg::matrix_product(eigen_vectors_in, 1, qp, 1, eigen_vectors_out, 1, m,
                           n, n);
    return 0;
}

template<typename T>
void rotate_subspace_fallback(const uint m, const uint n,
                              T const* const eigen_vectors_in,
                              T const* const qp,
                              T* const eigen_vectors_out) {
    Linalg::matrix_product(eigen_vectors_in, 1, qp, 1, eigen_vectors_out, 1, m,
                           n, n);
}

}  // namespace Xlsdft_chefsi_lvtx

#endif  // XLSDFT_CHEFSI_LVTX_HPP_
