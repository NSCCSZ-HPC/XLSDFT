#ifndef _STENCIL_KERNEL_H_
#define _STENCIL_KERNEL_H_

#include <cstdint>
#include <algorithm>
#ifdef USE_OPENMP
#include <omp.h>
#endif //USE_OPENMP

template<int Radius, typename T>
struct Stencil3D;

template<typename T>
struct Stencil3D<6, T> {
    static constexpr int radius = 6;
    static void lap_3d_c2_o0(T const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                            T* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                            const int64_t ni, const int64_t nj, const int64_t nk,
                            T const* const d2_coeffs_x,
                            T const* const d2_coeffs_y,
                            T const* const d2_coeffs_z,
                            T const coef_0,
                            T const* const A_ptr);
    static void lap_3d_c2_o0_boundary_safe(T const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                                        T* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                                        const int64_t ni, const int64_t nj, const int64_t nk,
                                        const int64_t i_l, const int64_t j_l, const int64_t k_l,
                                        const int64_t i_r, const int64_t j_r, const int64_t k_r,
                                        T const* const d2_coeffs_x,
                                        T const* const d2_coeffs_y,
                                        T const* const d2_coeffs_z,
                                        T const coef_0,
                                        T const* const A_ptr);
    static void lap_3d_c2_o0_boundary_safe(T const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,  const int64_t s_stride_b,
                                        T* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z, const int64_t d_stride_b,
                                        const int64_t ni, const int64_t nj, const int64_t nk, const int64_t nb,
                                        const int64_t i_l, const int64_t j_l, const int64_t k_l,
                                        const int64_t i_r, const int64_t j_r, const int64_t k_r,
                                        T const* const d2_coeffs_x,
                                        T const* const d2_coeffs_y,
                                        T const* const d2_coeffs_z,
                                        T const coef_0,
                                        T const* const A_ptr);
};

template<int Radius, typename T>
void laplacian_3d_c2_o0_kernel(T const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                                T* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                                const int64_t ni, const int64_t nj, const int64_t nk,
                                T const* const d2_coeffs_x,
                                T const* const d2_coeffs_y,
                                T const* const d2_coeffs_z,
                                T const coef_0,
                                T const* const A_ptr);

template<typename T>
void laplacian_3d_c2_o0_kernel(T const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                                T* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                                const int64_t ni, const int64_t nj, const int64_t nk,
                                const int64_t radius,
                                T const* const d2_coeffs_x,
                                T const* const d2_coeffs_y,
                                T const* const d2_coeffs_z,
                                T const coef_0,
                                T const* const A_ptr);

template<typename T>
void laplacian_3d_c2_o0(T const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                        T* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                        const int64_t ni, const int64_t nj, const int64_t nk,
                        const int radius,
                        T const* const d2_coeffs_x,
                        T const* const d2_coeffs_y,
                        T const* const d2_coeffs_z,
                        T const coef_0,
                        T const* const A_ptr);

template<int Radius, typename T>
void laplacian_3d_c2_o0_kernel_boundary_safe(
                                T const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                                T* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                                const int64_t ni, const int64_t nj, const int64_t nk,
                                const int64_t i_l, const int64_t j_l, const int64_t k_l,
                                const int64_t i_r, const int64_t j_r, const int64_t k_r,
                                T const* const d2_coeffs_x,
                                T const* const d2_coeffs_y,
                                T const* const d2_coeffs_z,
                                T const coef_0,
                                T const* const A_ptr);

template<typename T>
void laplacian_3d_c2_o0_kernel_boundary_safe(
                                T const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                                T* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                                const int64_t ni, const int64_t nj, const int64_t nk,
                                const int64_t i_l, const int64_t j_l, const int64_t k_l,
                                const int64_t i_r, const int64_t j_r, const int64_t k_r,
                                const int radius,
                                T const* const d2_coeffs_x,
                                T const* const d2_coeffs_y,
                                T const* const d2_coeffs_z,
                                T const coef_0,
                                T const* const A_ptr);

template<typename T>
void laplacian_3d_c2_o0_boundary_safe(T const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                        T* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                        const int64_t ni, const int64_t nj, const int64_t nk,
                        const int64_t i_l, const int64_t j_l, const int64_t k_l,
                        const int64_t i_r, const int64_t j_r, const int64_t k_r,
                        const int radius,
                        T const* const d2_coeffs_x,
                        T const* const d2_coeffs_y,
                        T const* const d2_coeffs_z,
                        T const coef_0,
                        T const* const A_ptr);

template<int Radius, typename T>
void laplacian_3d_c2_o0_kernel_boundary_safe(
                                T const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z, const int64_t s_stride_b,
                                T* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z, const int64_t d_stride_b,
                                const int64_t ni, const int64_t nj, const int64_t nk, const int64_t nb,
                                const int64_t i_l, const int64_t j_l, const int64_t k_l,
                                const int64_t i_r, const int64_t j_r, const int64_t k_r,
                                T const* const d2_coeffs_x,
                                T const* const d2_coeffs_y,
                                T const* const d2_coeffs_z,
                                T const coef_0,
                                T const* const A_ptr);

template<typename T>
void laplacian_3d_c2_o0_kernel_boundary_safe(
                                T const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z, const int64_t s_stride_b,
                                T* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z, const int64_t d_stride_b,
                                const int64_t ni, const int64_t nj, const int64_t nk, const int64_t nb,
                                const int64_t i_l, const int64_t j_l, const int64_t k_l,
                                const int64_t i_r, const int64_t j_r, const int64_t k_r,
                                const int radius,
                                T const* const d2_coeffs_x,
                                T const* const d2_coeffs_y,
                                T const* const d2_coeffs_z,
                                T const coef_0,
                                T const* const A_ptr);

template<typename T>
void laplacian_3d_c2_o0_boundary_safe(T const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z, const int64_t s_stride_b,
                        T* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z, const int64_t d_stride_b,
                        const int64_t ni, const int64_t nj, const int64_t nk, const int64_t nb,
                        const int64_t i_l, const int64_t j_l, const int64_t k_l,
                        const int64_t i_r, const int64_t j_r, const int64_t k_r,
                        const int radius,
                        T const* const d2_coeffs_x,
                        T const* const d2_coeffs_y,
                        T const* const d2_coeffs_z,
                        T const coef_0,
                        T const* const A_ptr);

#endif //_STENCIL_KERNEL_H_
