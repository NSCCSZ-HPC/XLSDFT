#include "stencil_kernel.h"

#ifndef __ARM_FEATURE_SVE
#pragma message("Building stencil_default.cpp.")

template<class T>
struct default_traits {
    static_assert(sizeof(T) == 0, "default_traits is not implemented for this type");
};

template<>
struct default_traits<float> {
    static constexpr int64_t tile_size_k = 8;
    static constexpr int64_t tile_size_j = 8;
    static constexpr int64_t tile_size_i = 128;
};

template<>
struct default_traits<double> {
    static constexpr int64_t tile_size_k = 8;
    static constexpr int64_t tile_size_j = 8;
    static constexpr int64_t tile_size_i = 64;
};

template<typename T>
void Stencil3D<6, T>::lap_3d_c2_o0(T const* const __restrict__ s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                                    T* const __restrict__ d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                                    const int64_t ni, const int64_t nj, const int64_t nk,
                                    T const* const __restrict__ d2_coeffs_x,
                                    T const* const __restrict__ d2_coeffs_y,
                                    T const* const __restrict__ d2_coeffs_z,
                                    T const coef_0,
                                    T const* const __restrict__ A_ptr) {
    // constexpr int Radius = 6;
    using traits = default_traits<T>;
    constexpr int64_t tile_size_j = traits::tile_size_j;
    constexpr int64_t tile_size_k = traits::tile_size_k;
    constexpr int64_t tile_size_i = traits::tile_size_i;
    // const T coef_0 = d2_coeffs_x[0] + d2_coeffs_y[0] + d2_coeffs_z[0];
    #ifdef USE_OPENMP
    #pragma omp for schedule(static) collapse(2) nowait
    #endif //USE_OPENMP
    for (int64_t kk = 0; kk < nk; kk += tile_size_k) {
        for (int64_t jj = 0; jj < nj; jj += tile_size_j) {
            for (int64_t ii = 0; ii < ni; ii += tile_size_i) {
                for (int64_t k = kk; k < std::min(kk + tile_size_k, nk); k++) {
                    for (int64_t j = jj; j < std::min(jj + tile_size_j, nj); j++) {
                        for (int64_t i = ii; i < std::min(ii + tile_size_i, ni); i++) {
                            // T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z;
                            // T* const __restrict__ d_ptr_ijk = d_ptr + i + j * d_stride_y + k * d_stride_z;
                            // T res = (coef_0 + A_ptr[i + j * d_stride_y + k * d_stride_z]) * s_ptr_ijk[0];
                            // for (int64_t r = 1; r <= Radius; r++) {
                            //     const int64_t s_stride_y_r = r * s_stride_y;
                            //     const int64_t s_stride_z_r = r * s_stride_z;
                            //     T res_x = (*(s_ptr_ijk+r) + *(s_ptr_ijk-r)) * d2_coeffs_x[r];
                            //     T res_y = (*(s_ptr_ijk+s_stride_y_r) + *(s_ptr_ijk-s_stride_y_r)) * d2_coeffs_y[r];
                            //     T res_z = (*(s_ptr_ijk+s_stride_z_r) + *(s_ptr_ijk-s_stride_z_r)) * d2_coeffs_z[r];
                            //     res += res_x + res_y + res_z;
                            // }
                            // d_ptr_ijk[0] = res;
                            T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z;
                            const int64_t offset = i + j * d_stride_y + k * d_stride_z;
                            d_ptr[offset] = (coef_0 + A_ptr[offset]) * s_ptr_ijk[0]
                                          + (s_ptr_ijk[1] + s_ptr_ijk[-1]) * d2_coeffs_x[1]
                                          + (s_ptr_ijk[2] + s_ptr_ijk[-2]) * d2_coeffs_x[2]
                                          + (s_ptr_ijk[3] + s_ptr_ijk[-3]) * d2_coeffs_x[3]
                                          + (s_ptr_ijk[4] + s_ptr_ijk[-4]) * d2_coeffs_x[4]
                                          + (s_ptr_ijk[5] + s_ptr_ijk[-5]) * d2_coeffs_x[5]
                                          + (s_ptr_ijk[6] + s_ptr_ijk[-6]) * d2_coeffs_x[6]
                                          + (s_ptr_ijk[1 * s_stride_y] + s_ptr_ijk[-1 * s_stride_y]) * d2_coeffs_y[1]
                                          + (s_ptr_ijk[2 * s_stride_y] + s_ptr_ijk[-2 * s_stride_y]) * d2_coeffs_y[2]
                                          + (s_ptr_ijk[3 * s_stride_y] + s_ptr_ijk[-3 * s_stride_y]) * d2_coeffs_y[3]
                                          + (s_ptr_ijk[4 * s_stride_y] + s_ptr_ijk[-4 * s_stride_y]) * d2_coeffs_y[4]
                                          + (s_ptr_ijk[5 * s_stride_y] + s_ptr_ijk[-5 * s_stride_y]) * d2_coeffs_y[5]
                                          + (s_ptr_ijk[6 * s_stride_y] + s_ptr_ijk[-6 * s_stride_y]) * d2_coeffs_y[6]
                                          + (s_ptr_ijk[1 * s_stride_z] + s_ptr_ijk[-1 * s_stride_z]) * d2_coeffs_z[1]
                                          + (s_ptr_ijk[2 * s_stride_z] + s_ptr_ijk[-2 * s_stride_z]) * d2_coeffs_z[2]
                                          + (s_ptr_ijk[3 * s_stride_z] + s_ptr_ijk[-3 * s_stride_z]) * d2_coeffs_z[3]
                                          + (s_ptr_ijk[4 * s_stride_z] + s_ptr_ijk[-4 * s_stride_z]) * d2_coeffs_z[4]
                                          + (s_ptr_ijk[5 * s_stride_z] + s_ptr_ijk[-5 * s_stride_z]) * d2_coeffs_z[5]
                                          + (s_ptr_ijk[6 * s_stride_z] + s_ptr_ijk[-6 * s_stride_z]) * d2_coeffs_z[6];
                        }
                    }
                }
            }
        }
    }
    return;
}

template<int Radius, typename T>
void laplacian_3d_c2_o0_kernel(T const* const __restrict__ s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                            T* const __restrict__ d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                            const int64_t ni, const int64_t nj, const int64_t nk,
                            T const* const __restrict__ d2_coeffs_x,
                            T const* const __restrict__ d2_coeffs_y,
                            T const* const __restrict__ d2_coeffs_z,
                            T const coef_0,
                            T const* const __restrict__ A_ptr)
{
    using traits = default_traits<T>;
    constexpr int64_t tile_size_j = traits::tile_size_j;
    constexpr int64_t tile_size_k = traits::tile_size_k;
    constexpr int64_t tile_size_i = traits::tile_size_i;
    // const T coef_0 = d2_coeffs_x[0] + d2_coeffs_y[0] + d2_coeffs_z[0];
    #ifdef USE_OPENMP
    #pragma omp for schedule(static) collapse(2) nowait
    #endif //USE_OPENMP
    for (int64_t kk = 0; kk < nk; kk += tile_size_k) {
        for (int64_t jj = 0; jj < nj; jj += tile_size_j) {
            for (int64_t ii = 0; ii < ni; ii += tile_size_i) {
                for (int64_t k = kk; k < std::min(kk + tile_size_k, nk); k++) {
                    for (int64_t j = jj; j < std::min(jj + tile_size_j, nj); j++) {
                        for (int64_t i = ii; i < std::min(ii + tile_size_i, ni); i++) {
                            T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z;
                            T* const __restrict__ d_ptr_ijk = d_ptr + i + j * d_stride_y + k * d_stride_z;
                            T res = (coef_0 + A_ptr[i + j * d_stride_y + k * d_stride_z]) * s_ptr_ijk[0];
                            for (int r = 1; r <= Radius; r++) {
                                const int64_t s_stride_y_r = r * s_stride_y;
                                const int64_t s_stride_z_r = r * s_stride_z;
                                T res_x = (*(s_ptr_ijk+r) + *(s_ptr_ijk-r)) * d2_coeffs_x[r];
                                T res_y = (*(s_ptr_ijk+s_stride_y_r) + *(s_ptr_ijk-s_stride_y_r)) * d2_coeffs_y[r];
                                T res_z = (*(s_ptr_ijk+s_stride_z_r) + *(s_ptr_ijk-s_stride_z_r)) * d2_coeffs_z[r];
                                res += res_x + res_y + res_z;
                            }
                            d_ptr_ijk[0] = res;
                        }
                    }
                }
            }
        }
    }
    return;
}

template<typename T>
void laplacian_3d_c2_o0_kernel(T const* const __restrict__ s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                            T* const __restrict__ d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                            const int64_t ni, const int64_t nj, const int64_t nk,
                            const int radius,
                            T const* const __restrict__ d2_coeffs_x,
                            T const* const __restrict__ d2_coeffs_y,
                            T const* const __restrict__ d2_coeffs_z,
                            T const coef_0,
                            T const* const __restrict__ A_ptr) {
    using traits = default_traits<T>;
    constexpr int64_t tile_size_j = traits::tile_size_j;
    constexpr int64_t tile_size_k = traits::tile_size_k;
    constexpr int64_t tile_size_i = traits::tile_size_i;
    // const T coef_0 = d2_coeffs_x[0] + d2_coeffs_y[0] + d2_coeffs_z[0];
    #ifdef USE_OPENMP
    #pragma omp for schedule(static) collapse(2) nowait
    #endif //USE_OPENMP
    for (int64_t kk = 0; kk < nk; kk += tile_size_k) {
        for (int64_t jj = 0; jj < nj; jj += tile_size_j) {
            for (int64_t ii = 0; ii < ni; ii += tile_size_i) {
                for (int64_t k = kk; k < std::min(kk + tile_size_k, nk); k++) {
                    for (int64_t j = jj; j < std::min(jj + tile_size_j, nj); j++) {
                        for (int64_t i = ii; i < std::min(ii + tile_size_i, ni); i++) {
                            T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z;
                            T* const __restrict__ d_ptr_ijk = d_ptr + i + j * d_stride_y + k * d_stride_z;
                            T res = (coef_0 + A_ptr[i + j * d_stride_y + k * d_stride_z]) * s_ptr_ijk[0];
                            for (int r = 1; r <= radius; r++) {
                                const int64_t s_stride_y_r = r * s_stride_y;
                                const int64_t s_stride_z_r = r * s_stride_z;
                                T res_x = (*(s_ptr_ijk+r) + *(s_ptr_ijk-r)) * d2_coeffs_x[r];
                                T res_y = (*(s_ptr_ijk+s_stride_y_r) + *(s_ptr_ijk-s_stride_y_r)) * d2_coeffs_y[r];
                                T res_z = (*(s_ptr_ijk+s_stride_z_r) + *(s_ptr_ijk-s_stride_z_r)) * d2_coeffs_z[r];
                                res += res_x + res_y + res_z;
                            }
                            d_ptr_ijk[0] = res;
                        }
                    }
                }
            }
        }
    }
    return;
}

template<typename T>
void laplacian_3d_c2_o0(T const* const __restrict__ s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                    T* const __restrict__ d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                    const int64_t ni, const int64_t nj, const int64_t nk,
                    const int radius,
                    T const* const __restrict__ d2_coeffs_x,
                    T const* const __restrict__ d2_coeffs_y,
                    T const* const __restrict__ d2_coeffs_z,
                    T const coef_0,
                    T const* const __restrict__ A_ptr)
{
    switch (radius) {
        case 7:
            laplacian_3d_c2_o0_kernel<7, T>(s_ptr, s_stride_y, s_stride_z,
                                      d_ptr, d_stride_y, d_stride_z,
                                      ni, nj, nk,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        case 6:
            Stencil3D<6, T>::lap_3d_c2_o0(s_ptr, s_stride_y, s_stride_z,
                                      d_ptr, d_stride_y, d_stride_z,
                                      ni, nj, nk,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        case 5:
            laplacian_3d_c2_o0_kernel<5, T>(s_ptr, s_stride_y, s_stride_z,
                                      d_ptr, d_stride_y, d_stride_z,
                                      ni, nj, nk,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        case 4:
            laplacian_3d_c2_o0_kernel<4, T>(s_ptr, s_stride_y, s_stride_z,
                                      d_ptr, d_stride_y, d_stride_z,
                                      ni, nj, nk,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        case 3:
            laplacian_3d_c2_o0_kernel<3, T>(s_ptr, s_stride_y, s_stride_z,
                                      d_ptr, d_stride_y, d_stride_z,
                                      ni, nj, nk,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        case 2:
            laplacian_3d_c2_o0_kernel<2, T>(s_ptr, s_stride_y, s_stride_z,
                                      d_ptr, d_stride_y, d_stride_z,
                                      ni, nj, nk,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        case 1:
            laplacian_3d_c2_o0_kernel<1, T>(s_ptr, s_stride_y, s_stride_z,
                                      d_ptr, d_stride_y, d_stride_z,
                                      ni, nj, nk,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        default:
            laplacian_3d_c2_o0_kernel<T>(s_ptr, s_stride_y, s_stride_z,
                                      d_ptr, d_stride_y, d_stride_z,
                                      ni, nj, nk,
                                      radius,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
    }
}
template void laplacian_3d_c2_o0<float>(float const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                                float* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                                const int64_t ni, const int64_t nj, const int64_t nk,
                                const int radius,
                                float const* const d2_coeffs_x,
                                float const* const d2_coeffs_y,
                                float const* const d2_coeffs_z,
                                float const coef_0,
                                float const* const A_ptr);
template void laplacian_3d_c2_o0<double>(double const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                                double* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                                const int64_t ni, const int64_t nj, const int64_t nk,
                                const int radius,
                                double const* const d2_coeffs_x,
                                double const* const d2_coeffs_y,
                                double const* const d2_coeffs_z,
                                double const coef_0,
                                double const* const A_ptr);

template<typename T>
void Stencil3D<6, T>::lap_3d_c2_o0_boundary_safe(T const* const __restrict__ s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                                                T* const __restrict__ d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                                                const int64_t ni, const int64_t nj, const int64_t nk,
                                                const int64_t i_l, const int64_t j_l, const int64_t k_l,
                                                const int64_t i_r, const int64_t j_r, const int64_t k_r,
                                                T const* const __restrict__ d2_coeffs_x,
                                                T const* const __restrict__ d2_coeffs_y,
                                                T const* const __restrict__ d2_coeffs_z,
                                                T const coef_0,
                                                T const* const __restrict__ A_ptr) {
    // constexpr int Radius = 6;
    using traits = default_traits<T>;
    constexpr int64_t tile_size_j = traits::tile_size_j;
    constexpr int64_t tile_size_k = traits::tile_size_k;
    constexpr int64_t tile_size_i = traits::tile_size_i;
    // const T coef_0 = d2_coeffs_x[0] + d2_coeffs_y[0] + d2_coeffs_z[0];
    #ifdef USE_OPENMP
    #pragma omp for schedule(static) collapse(2) nowait
    #endif //USE_OPENMP
    for (int64_t kk = 0; kk < nk; kk += tile_size_k) {
        for (int64_t jj = 0; jj < nj; jj += tile_size_j) {
            for (int64_t ii = 0; ii < ni; ii += tile_size_i) {
                for (int64_t k = kk; k < std::min(kk + tile_size_k, nk); k++) {
                    for (int64_t j = jj; j < std::min(jj + tile_size_j, nj); j++) {
                        for (int64_t i = ii; i < std::min(ii + tile_size_i, ni); i++) {
                            T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z;
                            const int64_t offset = i + j * d_stride_y + k * d_stride_z;
                            d_ptr[offset] = (coef_0 + A_ptr[offset]) * s_ptr_ijk[0]
                                          + (((i + 1 > i_r) ? T(0) : s_ptr_ijk[1]) + ((i - 1 < i_l) ? T(0) : s_ptr_ijk[-1])) * d2_coeffs_x[1]
                                          + (((i + 2 > i_r) ? T(0) : s_ptr_ijk[2]) + ((i - 2 < i_l) ? T(0) : s_ptr_ijk[-2])) * d2_coeffs_x[2]
                                          + (((i + 3 > i_r) ? T(0) : s_ptr_ijk[3]) + ((i - 3 < i_l) ? T(0) : s_ptr_ijk[-3])) * d2_coeffs_x[3]
                                          + (((i + 4 > i_r) ? T(0) : s_ptr_ijk[4]) + ((i - 4 < i_l) ? T(0) : s_ptr_ijk[-4])) * d2_coeffs_x[4]
                                          + (((i + 5 > i_r) ? T(0) : s_ptr_ijk[5]) + ((i - 5 < i_l) ? T(0) : s_ptr_ijk[-5])) * d2_coeffs_x[5]
                                          + (((i + 6 > i_r) ? T(0) : s_ptr_ijk[6]) + ((i - 6 < i_l) ? T(0) : s_ptr_ijk[-6])) * d2_coeffs_x[6]
                                          + (((j + 1 > j_r) ? T(0) : s_ptr_ijk[1 * s_stride_y]) + ((j - 1 < j_l) ? T(0) : s_ptr_ijk[-1 * s_stride_y])) * d2_coeffs_y[1]
                                          + (((j + 2 > j_r) ? T(0) : s_ptr_ijk[2 * s_stride_y]) + ((j - 2 < j_l) ? T(0) : s_ptr_ijk[-2 * s_stride_y])) * d2_coeffs_y[2]
                                          + (((j + 3 > j_r) ? T(0) : s_ptr_ijk[3 * s_stride_y]) + ((j - 3 < j_l) ? T(0) : s_ptr_ijk[-3 * s_stride_y])) * d2_coeffs_y[3]
                                          + (((j + 4 > j_r) ? T(0) : s_ptr_ijk[4 * s_stride_y]) + ((j - 4 < j_l) ? T(0) : s_ptr_ijk[-4 * s_stride_y])) * d2_coeffs_y[4]
                                          + (((j + 5 > j_r) ? T(0) : s_ptr_ijk[5 * s_stride_y]) + ((j - 5 < j_l) ? T(0) : s_ptr_ijk[-5 * s_stride_y])) * d2_coeffs_y[5]
                                          + (((j + 6 > j_r) ? T(0) : s_ptr_ijk[6 * s_stride_y]) + ((j - 6 < j_l) ? T(0) : s_ptr_ijk[-6 * s_stride_y])) * d2_coeffs_y[6]
                                          + (((k + 1 > k_r) ? T(0) : s_ptr_ijk[1 * s_stride_z]) + ((k - 1 < k_l) ? T(0) : s_ptr_ijk[-1 * s_stride_z])) * d2_coeffs_z[1]
                                          + (((k + 2 > k_r) ? T(0) : s_ptr_ijk[2 * s_stride_z]) + ((k - 2 < k_l) ? T(0) : s_ptr_ijk[-2 * s_stride_z])) * d2_coeffs_z[2]
                                          + (((k + 3 > k_r) ? T(0) : s_ptr_ijk[3 * s_stride_z]) + ((k - 3 < k_l) ? T(0) : s_ptr_ijk[-3 * s_stride_z])) * d2_coeffs_z[3]
                                          + (((k + 4 > k_r) ? T(0) : s_ptr_ijk[4 * s_stride_z]) + ((k - 4 < k_l) ? T(0) : s_ptr_ijk[-4 * s_stride_z])) * d2_coeffs_z[4]
                                          + (((k + 5 > k_r) ? T(0) : s_ptr_ijk[5 * s_stride_z]) + ((k - 5 < k_l) ? T(0) : s_ptr_ijk[-5 * s_stride_z])) * d2_coeffs_z[5]
                                          + (((k + 6 > k_r) ? T(0) : s_ptr_ijk[6 * s_stride_z]) + ((k - 6 < k_l) ? T(0) : s_ptr_ijk[-6 * s_stride_z])) * d2_coeffs_z[6];
                        }
                    }
                }
            }
        }
    }
    return;
}

template<int Radius, typename T>
void laplacian_3d_c2_o0_kernel_boundary_safe(
                            T const* const __restrict__ s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                            T* const __restrict__ d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                            const int64_t ni, const int64_t nj, const int64_t nk,
                            const int64_t i_l, const int64_t j_l, const int64_t k_l,
                            const int64_t i_r, const int64_t j_r, const int64_t k_r,
                            T const* const __restrict__ d2_coeffs_x,
                            T const* const __restrict__ d2_coeffs_y,
                            T const* const __restrict__ d2_coeffs_z,
                            T const coef_0,
                            T const* const __restrict__ A_ptr)
{
    using traits = default_traits<T>;
    constexpr int64_t tile_size_j = traits::tile_size_j;
    constexpr int64_t tile_size_k = traits::tile_size_k;
    constexpr int64_t tile_size_i = traits::tile_size_i;
    // const T coef_0 = d2_coeffs_x[0] + d2_coeffs_y[0] + d2_coeffs_z[0];
    #ifdef USE_OPENMP
    #pragma omp for schedule(static) collapse(2) nowait
    #endif //USE_OPENMP
    for (int64_t kk = 0; kk < nk; kk += tile_size_k) {
        for (int64_t jj = 0; jj < nj; jj += tile_size_j) {
            for (int64_t ii = 0; ii < ni; ii += tile_size_i) {
                for (int64_t k = kk; k < std::min(kk + tile_size_k, nk); k++) {
                    for (int64_t j = jj; j < std::min(jj + tile_size_j, nj); j++) {
                        for (int64_t i = ii; i < std::min(ii + tile_size_i, ni); i++) {
                            T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z;
                            T* const __restrict__ d_ptr_ijk = d_ptr + i + j * d_stride_y + k * d_stride_z;
                            T res = (coef_0 + A_ptr[i + j * d_stride_y + k * d_stride_z]) * s_ptr_ijk[0];
                            for (int r = 1; r <= Radius; r++) {
                                const int64_t s_stride_y_r = r * s_stride_y;
                                const int64_t s_stride_z_r = r * s_stride_z;
                                T res_x = (((i + r > i_r)? T(0) : *(s_ptr_ijk + r)) + ((i - r < i_l)? T(0) : *(s_ptr_ijk - r))) * d2_coeffs_x[r];
                                T res_y = (((j + r > j_r)? T(0) : *(s_ptr_ijk + s_stride_y_r)) + ((j - r < j_l)? T(0) : *(s_ptr_ijk - s_stride_y_r))) * d2_coeffs_y[r];
                                T res_z = (((k + r > k_r)? T(0) : *(s_ptr_ijk + s_stride_z_r)) + ((k - r < k_l)? T(0) : *(s_ptr_ijk - s_stride_z_r))) * d2_coeffs_z[r];
                                res += res_x + res_y + res_z;
                            }
                            d_ptr_ijk[0] = res;
                        }
                    }
                }
            }
        }
    }
    return;
}

template<typename T>
void laplacian_3d_c2_o0_kernel_boundary_safe(
                            T const* const __restrict__ s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                            T* const __restrict__ d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                            const int64_t ni, const int64_t nj, const int64_t nk,
                            const int64_t i_l, const int64_t j_l, const int64_t k_l,
                            const int64_t i_r, const int64_t j_r, const int64_t k_r,
                            const int radius,
                            T const* const __restrict__ d2_coeffs_x,
                            T const* const __restrict__ d2_coeffs_y,
                            T const* const __restrict__ d2_coeffs_z,
                            T const coef_0,
                            T const* const __restrict__ A_ptr) {
    using traits = default_traits<T>;
    constexpr int64_t tile_size_j = traits::tile_size_j;
    constexpr int64_t tile_size_k = traits::tile_size_k;
    constexpr int64_t tile_size_i = traits::tile_size_i;
    // const T coef_0 = d2_coeffs_x[0] + d2_coeffs_y[0] + d2_coeffs_z[0];
    #ifdef USE_OPENMP
    #pragma omp for schedule(static) collapse(2) nowait
    #endif //USE_OPENMP
    for (int64_t kk = 0; kk < nk; kk += tile_size_k) {
        for (int64_t jj = 0; jj < nj; jj += tile_size_j) {
            for (int64_t ii = 0; ii < ni; ii += tile_size_i) {
                for (int64_t k = kk; k < std::min(kk + tile_size_k, nk); k++) {
                    for (int64_t j = jj; j < std::min(jj + tile_size_j, nj); j++) {
                        for (int64_t i = ii; i < std::min(ii + tile_size_i, ni); i++) {
                            T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z;
                            T* const __restrict__ d_ptr_ijk = d_ptr + i + j * d_stride_y + k * d_stride_z;
                            T res = (coef_0 + A_ptr[i + j * d_stride_y + k * d_stride_z]) * s_ptr_ijk[0];
                            for (int r = 1; r <= radius; r++) {
                                const int64_t s_stride_y_r = r * s_stride_y;
                                const int64_t s_stride_z_r = r * s_stride_z;
                                T res_x = (((i + r > i_r)? T(0) : *(s_ptr_ijk + r)) + ((i - r < i_l)? T(0) : *(s_ptr_ijk - r))) * d2_coeffs_x[r];
                                T res_y = (((j + r > j_r)? T(0) : *(s_ptr_ijk + s_stride_y_r)) + ((j - r < j_l)? T(0) : *(s_ptr_ijk - s_stride_y_r))) * d2_coeffs_y[r];
                                T res_z = (((k + r > k_r)? T(0) : *(s_ptr_ijk + s_stride_z_r)) + ((k - r < k_l)? T(0) : *(s_ptr_ijk - s_stride_z_r))) * d2_coeffs_z[r];
                                res += res_x + res_y + res_z;
                            }
                            d_ptr_ijk[0] = res;
                        }
                    }
                }
            }
        }
    }
    return;
}

template<typename T>
void laplacian_3d_c2_o0_boundary_safe(T const* const __restrict__ s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                                        T* const __restrict__ d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                                        const int64_t ni, const int64_t nj, const int64_t nk,
                                        const int64_t i_l, const int64_t j_l, const int64_t k_l,
                                        const int64_t i_r, const int64_t j_r, const int64_t k_r,
                                        const int radius,
                                        T const* const __restrict__ d2_coeffs_x,
                                        T const* const __restrict__ d2_coeffs_y,
                                        T const* const __restrict__ d2_coeffs_z,
                                        T const coef_0,
                                        T const* const __restrict__ A_ptr)
{
    switch (radius) {
        case 7:
            laplacian_3d_c2_o0_kernel_boundary_safe<7, T>(s_ptr, s_stride_y, s_stride_z,
                                      d_ptr, d_stride_y, d_stride_z,
                                      ni, nj, nk,
                                      i_l, j_l, k_l,
                                      i_r, j_r, k_r,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        case 6:
            Stencil3D<6, T>::lap_3d_c2_o0_boundary_safe(s_ptr, s_stride_y, s_stride_z,
            // laplacian_3d_c2_o0_kernel_boundary_safe<6, T>(s_ptr, s_stride_y, s_stride_z,
                                      d_ptr, d_stride_y, d_stride_z,
                                      ni, nj, nk,
                                      i_l, j_l, k_l,
                                      i_r, j_r, k_r,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        case 5:
            laplacian_3d_c2_o0_kernel_boundary_safe<5, T>(s_ptr, s_stride_y, s_stride_z,
                                      d_ptr, d_stride_y, d_stride_z,
                                      ni, nj, nk,
                                      i_l, j_l, k_l,
                                      i_r, j_r, k_r,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        case 4:
            laplacian_3d_c2_o0_kernel_boundary_safe<4, T>(s_ptr, s_stride_y, s_stride_z,
                                      d_ptr, d_stride_y, d_stride_z,
                                      ni, nj, nk,
                                      i_l, j_l, k_l,
                                      i_r, j_r, k_r,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        case 3:
            laplacian_3d_c2_o0_kernel_boundary_safe<3, T>(s_ptr, s_stride_y, s_stride_z,
                                      d_ptr, d_stride_y, d_stride_z,
                                      ni, nj, nk,
                                      i_l, j_l, k_l,
                                      i_r, j_r, k_r,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        case 2:
            laplacian_3d_c2_o0_kernel_boundary_safe<7, T>(s_ptr, s_stride_y, s_stride_z,
                                      d_ptr, d_stride_y, d_stride_z,
                                      ni, nj, nk,
                                      i_l, j_l, k_l,
                                      i_r, j_r, k_r,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        case 1:
            laplacian_3d_c2_o0_kernel_boundary_safe<1, T>(s_ptr, s_stride_y, s_stride_z,
                                      d_ptr, d_stride_y, d_stride_z,
                                      ni, nj, nk,
                                      i_l, j_l, k_l,
                                      i_r, j_r, k_r,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        default:
            laplacian_3d_c2_o0_kernel_boundary_safe<T>(s_ptr, s_stride_y, s_stride_z,
                                      d_ptr, d_stride_y, d_stride_z,
                                      ni, nj, nk,
                                      i_l, j_l, k_l,
                                      i_r, j_r, k_r,
                                      radius,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
    }
}
template void laplacian_3d_c2_o0_boundary_safe<float>(float const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                                                    float* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                                                    const int64_t ni, const int64_t nj, const int64_t nk,
                                                    const int64_t i_l, const int64_t j_l, const int64_t k_l,
                                                    const int64_t i_r, const int64_t j_r, const int64_t k_r,
                                                    const int radius,
                                                    float const* const d2_coeffs_x,
                                                    float const* const d2_coeffs_y,
                                                    float const* const d2_coeffs_z,
                                                    float const coef_0,
                                                    float const* const A_ptr);
template void laplacian_3d_c2_o0_boundary_safe<double>(double const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                                                    double* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                                                    const int64_t ni, const int64_t nj, const int64_t nk,
                                                    const int64_t i_l, const int64_t j_l, const int64_t k_l,
                                                    const int64_t i_r, const int64_t j_r, const int64_t k_r,
                                                    const int radius,
                                                    double const* const d2_coeffs_x,
                                                    double const* const d2_coeffs_y,
                                                    double const* const d2_coeffs_z,
                                                    double const coef_0,
                                                    double const* const A_ptr);

template<typename T>
void Stencil3D<6, T>::lap_3d_c2_o0_boundary_safe(T const* const __restrict__ s_ptr, const int64_t s_stride_y, const int64_t s_stride_z, const int64_t s_stride_b,
                                                T* const __restrict__ d_ptr, const int64_t d_stride_y, const int64_t d_stride_z, const int64_t d_stride_b,
                                                const int64_t ni, const int64_t nj, const int64_t nk, const int64_t nb,
                                                const int64_t i_l, const int64_t j_l, const int64_t k_l,
                                                const int64_t i_r, const int64_t j_r, const int64_t k_r,
                                                T const* const __restrict__ d2_coeffs_x,
                                                T const* const __restrict__ d2_coeffs_y,
                                                T const* const __restrict__ d2_coeffs_z,
                                                T const coef_0,
                                                T const* const __restrict__ A_ptr) {
    // constexpr int Radius = 6;
    using traits = default_traits<T>;
    constexpr int64_t tile_size_j = traits::tile_size_j;
    constexpr int64_t tile_size_k = traits::tile_size_k;
    constexpr int64_t tile_size_i = traits::tile_size_i;
    // const T coef_0 = d2_coeffs_x[0] + d2_coeffs_y[0] + d2_coeffs_z[0];
    #ifdef USE_OPENMP
    #pragma omp for schedule(static) nowait
    #endif //USE_OPENMP
    for (int64_t b = 0; b < nb; b++) {
        for (int64_t kk = 0; kk < nk; kk += tile_size_k) {
            for (int64_t jj = 0; jj < nj; jj += tile_size_j) {
                for (int64_t ii = 0; ii < ni; ii += tile_size_i) {
                    for (int64_t k = kk; k < std::min(kk + tile_size_k, nk); k++) {
                        for (int64_t j = jj; j < std::min(jj + tile_size_j, nj); j++) {
                            for (int64_t i = ii; i < std::min(ii + tile_size_i, ni); i++) {
                                T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z + b * s_stride_b;
                                d_ptr[i + j * d_stride_y + k * d_stride_z + b * d_stride_b] = (coef_0 + A_ptr[i + j * d_stride_y + k * d_stride_z]) * s_ptr_ijk[0]
                                            + (((i + 1 > i_r) ? T(0) : s_ptr_ijk[1]) + ((i - 1 < i_l) ? T(0) : s_ptr_ijk[-1])) * d2_coeffs_x[1]
                                            + (((i + 2 > i_r) ? T(0) : s_ptr_ijk[2]) + ((i - 2 < i_l) ? T(0) : s_ptr_ijk[-2])) * d2_coeffs_x[2]
                                            + (((i + 3 > i_r) ? T(0) : s_ptr_ijk[3]) + ((i - 3 < i_l) ? T(0) : s_ptr_ijk[-3])) * d2_coeffs_x[3]
                                            + (((i + 4 > i_r) ? T(0) : s_ptr_ijk[4]) + ((i - 4 < i_l) ? T(0) : s_ptr_ijk[-4])) * d2_coeffs_x[4]
                                            + (((i + 5 > i_r) ? T(0) : s_ptr_ijk[5]) + ((i - 5 < i_l) ? T(0) : s_ptr_ijk[-5])) * d2_coeffs_x[5]
                                            + (((i + 6 > i_r) ? T(0) : s_ptr_ijk[6]) + ((i - 6 < i_l) ? T(0) : s_ptr_ijk[-6])) * d2_coeffs_x[6]
                                            + (((j + 1 > j_r) ? T(0) : s_ptr_ijk[1 * s_stride_y]) + ((j - 1 < j_l) ? T(0) : s_ptr_ijk[-1 * s_stride_y])) * d2_coeffs_y[1]
                                            + (((j + 2 > j_r) ? T(0) : s_ptr_ijk[2 * s_stride_y]) + ((j - 2 < j_l) ? T(0) : s_ptr_ijk[-2 * s_stride_y])) * d2_coeffs_y[2]
                                            + (((j + 3 > j_r) ? T(0) : s_ptr_ijk[3 * s_stride_y]) + ((j - 3 < j_l) ? T(0) : s_ptr_ijk[-3 * s_stride_y])) * d2_coeffs_y[3]
                                            + (((j + 4 > j_r) ? T(0) : s_ptr_ijk[4 * s_stride_y]) + ((j - 4 < j_l) ? T(0) : s_ptr_ijk[-4 * s_stride_y])) * d2_coeffs_y[4]
                                            + (((j + 5 > j_r) ? T(0) : s_ptr_ijk[5 * s_stride_y]) + ((j - 5 < j_l) ? T(0) : s_ptr_ijk[-5 * s_stride_y])) * d2_coeffs_y[5]
                                            + (((j + 6 > j_r) ? T(0) : s_ptr_ijk[6 * s_stride_y]) + ((j - 6 < j_l) ? T(0) : s_ptr_ijk[-6 * s_stride_y])) * d2_coeffs_y[6]
                                            + (((k + 1 > k_r) ? T(0) : s_ptr_ijk[1 * s_stride_z]) + ((k - 1 < k_l) ? T(0) : s_ptr_ijk[-1 * s_stride_z])) * d2_coeffs_z[1]
                                            + (((k + 2 > k_r) ? T(0) : s_ptr_ijk[2 * s_stride_z]) + ((k - 2 < k_l) ? T(0) : s_ptr_ijk[-2 * s_stride_z])) * d2_coeffs_z[2]
                                            + (((k + 3 > k_r) ? T(0) : s_ptr_ijk[3 * s_stride_z]) + ((k - 3 < k_l) ? T(0) : s_ptr_ijk[-3 * s_stride_z])) * d2_coeffs_z[3]
                                            + (((k + 4 > k_r) ? T(0) : s_ptr_ijk[4 * s_stride_z]) + ((k - 4 < k_l) ? T(0) : s_ptr_ijk[-4 * s_stride_z])) * d2_coeffs_z[4]
                                            + (((k + 5 > k_r) ? T(0) : s_ptr_ijk[5 * s_stride_z]) + ((k - 5 < k_l) ? T(0) : s_ptr_ijk[-5 * s_stride_z])) * d2_coeffs_z[5]
                                            + (((k + 6 > k_r) ? T(0) : s_ptr_ijk[6 * s_stride_z]) + ((k - 6 < k_l) ? T(0) : s_ptr_ijk[-6 * s_stride_z])) * d2_coeffs_z[6];
                            }
                        }
                    }
                }
            }
        }
    }
    return;
}

template<int Radius, typename T>
void laplacian_3d_c2_o0_kernel_boundary_safe(
                            T const* const __restrict__ s_ptr, const int64_t s_stride_y, const int64_t s_stride_z, const int64_t s_stride_b,
                            T* const __restrict__ d_ptr, const int64_t d_stride_y, const int64_t d_stride_z, const int64_t d_stride_b,
                            const int64_t ni, const int64_t nj, const int64_t nk, const int64_t nb,
                            const int64_t i_l, const int64_t j_l, const int64_t k_l,
                            const int64_t i_r, const int64_t j_r, const int64_t k_r,
                            T const* const __restrict__ d2_coeffs_x,
                            T const* const __restrict__ d2_coeffs_y,
                            T const* const __restrict__ d2_coeffs_z,
                            T const coef_0,
                            T const* const __restrict__ A_ptr)
{
    using traits = default_traits<T>;
    constexpr int64_t tile_size_j = traits::tile_size_j;
    constexpr int64_t tile_size_k = traits::tile_size_k;
    constexpr int64_t tile_size_i = traits::tile_size_i;
    // const T coef_0 = d2_coeffs_x[0] + d2_coeffs_y[0] + d2_coeffs_z[0];
    #ifdef USE_OPENMP
    #pragma omp for schedule(static) nowait
    #endif //USE_OPENMP
    for (int64_t b = 0; b < nb; b++) {
        for (int64_t kk = 0; kk < nk; kk += tile_size_k) {
            for (int64_t jj = 0; jj < nj; jj += tile_size_j) {
                for (int64_t ii = 0; ii < ni; ii += tile_size_i) {
                    for (int64_t k = kk; k < std::min(kk + tile_size_k, nk); k++) {
                        for (int64_t j = jj; j < std::min(jj + tile_size_j, nj); j++) {
                            for (int64_t i = ii; i < std::min(ii + tile_size_i, ni); i++) {
                                T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z + b * s_stride_b;
                                T* const __restrict__ d_ptr_ijk = d_ptr + i + j * d_stride_y + k * d_stride_z + b * d_stride_b;
                                T res = (coef_0 + A_ptr[i + j * d_stride_y + k * d_stride_z]) * s_ptr_ijk[0];
                                for (int r = 1; r <= Radius; r++) {
                                    const int64_t s_stride_y_r = r * s_stride_y;
                                    const int64_t s_stride_z_r = r * s_stride_z;
                                    T res_x = (((i + r > i_r)? T(0) : *(s_ptr_ijk + r)) + ((i - r < i_l)? T(0) : *(s_ptr_ijk - r))) * d2_coeffs_x[r];
                                    T res_y = (((j + r > j_r)? T(0) : *(s_ptr_ijk + s_stride_y_r)) + ((j - r < j_l)? T(0) : *(s_ptr_ijk - s_stride_y_r))) * d2_coeffs_y[r];
                                    T res_z = (((k + r > k_r)? T(0) : *(s_ptr_ijk + s_stride_z_r)) + ((k - r < k_l)? T(0) : *(s_ptr_ijk - s_stride_z_r))) * d2_coeffs_z[r];
                                    res += res_x + res_y + res_z;
                                }
                                d_ptr_ijk[0] = res;
                            }
                        }
                    }
                }
            }
        }
    }
    return;
}

template<typename T>
void laplacian_3d_c2_o0_kernel_boundary_safe(
                            T const* const __restrict__ s_ptr, const int64_t s_stride_y, const int64_t s_stride_z, const int64_t s_stride_b,
                            T* const __restrict__ d_ptr, const int64_t d_stride_y, const int64_t d_stride_z, const int64_t d_stride_b,
                            const int64_t ni, const int64_t nj, const int64_t nk, const int64_t nb,
                            const int64_t i_l, const int64_t j_l, const int64_t k_l,
                            const int64_t i_r, const int64_t j_r, const int64_t k_r,
                            const int radius,
                            T const* const __restrict__ d2_coeffs_x,
                            T const* const __restrict__ d2_coeffs_y,
                            T const* const __restrict__ d2_coeffs_z,
                            T const coef_0,
                            T const* const __restrict__ A_ptr) {
    using traits = default_traits<T>;
    constexpr int64_t tile_size_j = traits::tile_size_j;
    constexpr int64_t tile_size_k = traits::tile_size_k;
    constexpr int64_t tile_size_i = traits::tile_size_i;
    // const T coef_0 = d2_coeffs_x[0] + d2_coeffs_y[0] + d2_coeffs_z[0];
    #ifdef USE_OPENMP
    #pragma omp for schedule(static) nowait
    #endif //USE_OPENMP
    for (int64_t b = 0; b < nb; b++) {
        for (int64_t kk = 0; kk < nk; kk += tile_size_k) {
            for (int64_t jj = 0; jj < nj; jj += tile_size_j) {
                for (int64_t ii = 0; ii < ni; ii += tile_size_i) {
                    for (int64_t k = kk; k < std::min(kk + tile_size_k, nk); k++) {
                        for (int64_t j = jj; j < std::min(jj + tile_size_j, nj); j++) {
                            for (int64_t i = ii; i < std::min(ii + tile_size_i, ni); i++) {
                                T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z + b * s_stride_b;
                                T* const __restrict__ d_ptr_ijk = d_ptr + i + j * d_stride_y + k * d_stride_z + b * d_stride_b;
                                T res = (coef_0 + A_ptr[i + j * d_stride_y + k * d_stride_z]) * s_ptr_ijk[0];
                                for (int r = 1; r <= radius; r++) {
                                    const int64_t s_stride_y_r = r * s_stride_y;
                                    const int64_t s_stride_z_r = r * s_stride_z;
                                    T res_x = (((i + r > i_r)? T(0) : *(s_ptr_ijk + r)) + ((i - r < i_l)? T(0) : *(s_ptr_ijk - r))) * d2_coeffs_x[r];
                                    T res_y = (((j + r > j_r)? T(0) : *(s_ptr_ijk + s_stride_y_r)) + ((j - r < j_l)? T(0) : *(s_ptr_ijk - s_stride_y_r))) * d2_coeffs_y[r];
                                    T res_z = (((k + r > k_r)? T(0) : *(s_ptr_ijk + s_stride_z_r)) + ((k - r < k_l)? T(0) : *(s_ptr_ijk - s_stride_z_r))) * d2_coeffs_z[r];
                                    res += res_x + res_y + res_z;
                                }
                                d_ptr_ijk[0] = res;
                            }
                        }
                    }
                }
            }
        }
    }
    return;
}

template<typename T>
void laplacian_3d_c2_o0_boundary_safe(T const* const __restrict__ s_ptr, const int64_t s_stride_y, const int64_t s_stride_z, const int64_t s_stride_b,
                                        T* const __restrict__ d_ptr, const int64_t d_stride_y, const int64_t d_stride_z, const int64_t d_stride_b,
                                        const int64_t ni, const int64_t nj, const int64_t nk, const int64_t nb,
                                        const int64_t i_l, const int64_t j_l, const int64_t k_l,
                                        const int64_t i_r, const int64_t j_r, const int64_t k_r,
                                        const int radius,
                                        T const* const __restrict__ d2_coeffs_x,
                                        T const* const __restrict__ d2_coeffs_y,
                                        T const* const __restrict__ d2_coeffs_z,
                                        T const coef_0,
                                        T const* const __restrict__ A_ptr)
{
    switch (radius) {
        case 7:
            laplacian_3d_c2_o0_kernel_boundary_safe<7, T>(s_ptr, s_stride_y, s_stride_z, s_stride_b,
                                      d_ptr, d_stride_y, d_stride_z, d_stride_b,
                                      ni, nj, nk, nb,
                                      i_l, j_l, k_l,
                                      i_r, j_r, k_r,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        case 6:
            Stencil3D<6, T>::lap_3d_c2_o0_boundary_safe(s_ptr, s_stride_y, s_stride_z, s_stride_b,
            // laplacian_3d_c2_o0_kernel_boundary_safe<6, T>(s_ptr, s_stride_y, s_stride_z, s_stride_b,
                                      d_ptr, d_stride_y, d_stride_z, d_stride_b,
                                      ni, nj, nk, nb,
                                      i_l, j_l, k_l,
                                      i_r, j_r, k_r,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        case 5:
            laplacian_3d_c2_o0_kernel_boundary_safe<5, T>(s_ptr, s_stride_y, s_stride_z, s_stride_b,
                                      d_ptr, d_stride_y, d_stride_z, d_stride_b,
                                      ni, nj, nk, nb,
                                      i_l, j_l, k_l,
                                      i_r, j_r, k_r,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        case 4:
            laplacian_3d_c2_o0_kernel_boundary_safe<4, T>(s_ptr, s_stride_y, s_stride_z, s_stride_b,
                                      d_ptr, d_stride_y, d_stride_z, d_stride_b,
                                      ni, nj, nk, nb,
                                      i_l, j_l, k_l,
                                      i_r, j_r, k_r,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        case 3:
            laplacian_3d_c2_o0_kernel_boundary_safe<3, T>(s_ptr, s_stride_y, s_stride_z, s_stride_b,
                                      d_ptr, d_stride_y, d_stride_z, d_stride_b,
                                      ni, nj, nk, nb,
                                      i_l, j_l, k_l,
                                      i_r, j_r, k_r,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        case 2:
            laplacian_3d_c2_o0_kernel_boundary_safe<7, T>(s_ptr, s_stride_y, s_stride_z, s_stride_b,
                                      d_ptr, d_stride_y, d_stride_z, d_stride_b,
                                      ni, nj, nk, nb,
                                      i_l, j_l, k_l,
                                      i_r, j_r, k_r,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        case 1:
            laplacian_3d_c2_o0_kernel_boundary_safe<1, T>(s_ptr, s_stride_y, s_stride_z, s_stride_b,
                                      d_ptr, d_stride_y, d_stride_z, d_stride_b,
                                      ni, nj, nk, nb,
                                      i_l, j_l, k_l,
                                      i_r, j_r, k_r,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
        default:
            laplacian_3d_c2_o0_kernel_boundary_safe<T>(s_ptr, s_stride_y, s_stride_z, s_stride_b,
                                      d_ptr, d_stride_y, d_stride_z, d_stride_b,
                                      ni, nj, nk, nb,
                                      i_l, j_l, k_l,
                                      i_r, j_r, k_r,
                                      radius,
                                      d2_coeffs_x, d2_coeffs_y, d2_coeffs_z,
                                      coef_0,
                                      A_ptr);
            break;
    }
}
template void laplacian_3d_c2_o0_boundary_safe<float>(float const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z, const int64_t s_stride_b,
                                                    float* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z, const int64_t d_stride_b,
                                                    const int64_t ni, const int64_t nj, const int64_t nk, const int64_t nb,
                                                    const int64_t i_l, const int64_t j_l, const int64_t k_l,
                                                    const int64_t i_r, const int64_t j_r, const int64_t k_r,
                                                    const int radius,
                                                    float const* const d2_coeffs_x,
                                                    float const* const d2_coeffs_y,
                                                    float const* const d2_coeffs_z,
                                                    float const coef_0,
                                                    float const* const A_ptr);
template void laplacian_3d_c2_o0_boundary_safe<double>(double const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z, const int64_t s_stride_b,
                                                    double* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z, const int64_t d_stride_b,
                                                    const int64_t ni, const int64_t nj, const int64_t nk, const int64_t nb,
                                                    const int64_t i_l, const int64_t j_l, const int64_t k_l,
                                                    const int64_t i_r, const int64_t j_r, const int64_t k_r,
                                                    const int radius,
                                                    double const* const d2_coeffs_x,
                                                    double const* const d2_coeffs_y,
                                                    double const* const d2_coeffs_z,
                                                    double const coef_0,
                                                    double const* const A_ptr);

#endif //__ARM_FEATURE_SVE
