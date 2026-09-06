#include "stencil_kernel.h"

#if !defined(__ARM_FEATURE_SME) && defined(__ARM_FEATURE_SVE)
#pragma message("Building stencil_sve.cpp.")
#include <arm_sve.h>

template<class T>
struct sve_traits {
    static_assert(sizeof(T) == 0, "sve_traits is not implemented for this type");
};

template<>
struct sve_traits<float> {
    using sv_float__t = svfloat32_t;
    static inline int64_t svcnt_() {static const int64_t size = svcntw(); return size;}
    static inline svbool_t svwhilelt_b_(const int64_t i, const int64_t n) {return svwhilelt_b32(i, n);}
    static inline svbool_t svwhilegt_b_(const int64_t i, const int64_t n) {return svwhilegt_b32(i, n);}
    static inline sv_float__t svdup_f_(const float x) {return svdup_f32(x);}
    static inline svbool_t svptrue_b_() {return svptrue_b32();}
    static inline float svaddv_f_(svbool_t pg, sv_float__t v) {return svaddv_f32(pg, v);}
    static constexpr int64_t tile_size_k = 8;
    static constexpr int64_t tile_size_j = 8;
    static constexpr int64_t tile_size_i = 128;
};

template<>
struct sve_traits<double> {
    using sv_float__t = svfloat64_t;
    static inline int64_t svcnt_() {static const int64_t size = svcntd(); return size;}
    static inline svbool_t svwhilelt_b_(const int64_t i, const int64_t n) {return svwhilelt_b64(i, n);}
    static inline svbool_t svwhilegt_b_(const int64_t i, const int64_t n) {return svwhilegt_b64(i, n);}
    static inline sv_float__t svdup_f_(const double x) {return svdup_f64(x);}
    static inline svbool_t svptrue_b_() {return svptrue_b64();}
    static inline double svaddv_f_(svbool_t pg, sv_float__t v) {return svaddv_f64(pg, v);}
    static constexpr int64_t tile_size_k = 4;
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
    // constexpr int R = 6;
    using traits = sve_traits<T>;
    using sv_float__t = typename traits::sv_float__t;
    constexpr int64_t tile_size_k = traits::tile_size_k;
    constexpr int64_t tile_size_j = traits::tile_size_j;
    constexpr int64_t tile_size_i = traits::tile_size_i;
    sv_float__t sve_coef0 = traits::svdup_f_(coef_0);
    sv_float__t sve_coef_x1 = traits::svdup_f_(d2_coeffs_x[1]);
    sv_float__t sve_coef_x2 = traits::svdup_f_(d2_coeffs_x[2]);
    sv_float__t sve_coef_x3 = traits::svdup_f_(d2_coeffs_x[3]);
    sv_float__t sve_coef_x4 = traits::svdup_f_(d2_coeffs_x[4]);
    sv_float__t sve_coef_x5 = traits::svdup_f_(d2_coeffs_x[5]);
    sv_float__t sve_coef_x6 = traits::svdup_f_(d2_coeffs_x[6]);
    sv_float__t sve_coef_y1 = traits::svdup_f_(d2_coeffs_y[1]);
    sv_float__t sve_coef_y2 = traits::svdup_f_(d2_coeffs_y[2]);
    sv_float__t sve_coef_y3 = traits::svdup_f_(d2_coeffs_y[3]);
    sv_float__t sve_coef_y4 = traits::svdup_f_(d2_coeffs_y[4]);
    sv_float__t sve_coef_y5 = traits::svdup_f_(d2_coeffs_y[5]);
    sv_float__t sve_coef_y6 = traits::svdup_f_(d2_coeffs_y[6]);
    sv_float__t sve_coef_z1 = traits::svdup_f_(d2_coeffs_z[1]);
    sv_float__t sve_coef_z2 = traits::svdup_f_(d2_coeffs_z[2]);
    sv_float__t sve_coef_z3 = traits::svdup_f_(d2_coeffs_z[3]);
    sv_float__t sve_coef_z4 = traits::svdup_f_(d2_coeffs_z[4]);
    sv_float__t sve_coef_z5 = traits::svdup_f_(d2_coeffs_z[5]);
    sv_float__t sve_coef_z6 = traits::svdup_f_(d2_coeffs_z[6]);
    const int64_t svcnt = traits::svcnt_();
    #ifdef USE_OPENMP
    #pragma omp for schedule(static) collapse(2) nowait
    #endif //USE_OPENMP
    for (int64_t kk = 0; kk < nk; kk += tile_size_k) {
        for (int64_t jj = 0; jj < nj; jj += tile_size_j) {
            for (int64_t ii = 0; ii < ni; ii += tile_size_i) {
                for (int64_t k = kk; k < std::min(kk + tile_size_k, nk); k++) {
                    for (int64_t j = jj; j < std::min(jj + tile_size_j, nj); j++) {
                        for (int64_t i = ii; i < std::min(ii + tile_size_i, ni); i += svcnt) {
                            svbool_t pg = traits::svwhilelt_b_(i, ni);
                            T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z;
                            sv_float__t res = svmul_m(pg, svld1(pg, s_ptr_ijk), 
                                              svadd_m(pg, svld1(pg, A_ptr + i + j * d_stride_y + k * d_stride_z), sve_coef0));

                            res = svmla_m(pg, res, svadd_m(pg, svld1(pg, s_ptr_ijk + 1), svld1(pg, s_ptr_ijk - 1)), sve_coef_x1);
                            res = svmla_m(pg, res, svadd_m(pg, svld1(pg, s_ptr_ijk + 1 * s_stride_y), svld1(pg, s_ptr_ijk - 1 * s_stride_y)), sve_coef_y1);
                            res = svmla_m(pg, res, svadd_m(pg, svld1(pg, s_ptr_ijk + 1 * s_stride_z), svld1(pg, s_ptr_ijk - 1 * s_stride_z)), sve_coef_z1);

                            res = svmla_m(pg, res, svadd_m(pg, svld1(pg, s_ptr_ijk + 2), svld1(pg, s_ptr_ijk - 2)), sve_coef_x2);
                            res = svmla_m(pg, res, svadd_m(pg, svld1(pg, s_ptr_ijk + 2 * s_stride_y), svld1(pg, s_ptr_ijk - 2 * s_stride_y)), sve_coef_y2);
                            res = svmla_m(pg, res, svadd_m(pg, svld1(pg, s_ptr_ijk + 2 * s_stride_z), svld1(pg, s_ptr_ijk - 2 * s_stride_z)), sve_coef_z2);
    
                            res = svmla_m(pg, res, svadd_m(pg, svld1(pg, s_ptr_ijk + 3), svld1(pg, s_ptr_ijk - 3)), sve_coef_x3);
                            res = svmla_m(pg, res, svadd_m(pg, svld1(pg, s_ptr_ijk + 3 * s_stride_y), svld1(pg, s_ptr_ijk - 3 * s_stride_y)), sve_coef_y3);
                            res = svmla_m(pg, res, svadd_m(pg, svld1(pg, s_ptr_ijk + 3 * s_stride_z), svld1(pg, s_ptr_ijk - 3 * s_stride_z)), sve_coef_z3);

                            res = svmla_m(pg, res, svadd_m(pg, svld1(pg, s_ptr_ijk + 4), svld1(pg, s_ptr_ijk - 4)), sve_coef_x4);
                            res = svmla_m(pg, res, svadd_m(pg, svld1(pg, s_ptr_ijk + 4 * s_stride_y), svld1(pg, s_ptr_ijk - 4 * s_stride_y)), sve_coef_y4);
                            res = svmla_m(pg, res, svadd_m(pg, svld1(pg, s_ptr_ijk + 4 * s_stride_z), svld1(pg, s_ptr_ijk - 4 * s_stride_z)), sve_coef_z4);

                            res = svmla_m(pg, res, svadd_m(pg, svld1(pg, s_ptr_ijk + 5), svld1(pg, s_ptr_ijk - 5)), sve_coef_x5);
                            res = svmla_m(pg, res, svadd_m(pg, svld1(pg, s_ptr_ijk + 5 * s_stride_y), svld1(pg, s_ptr_ijk - 5 * s_stride_y)), sve_coef_y5);
                            res = svmla_m(pg, res, svadd_m(pg, svld1(pg, s_ptr_ijk + 5 * s_stride_z), svld1(pg, s_ptr_ijk - 5 * s_stride_z)), sve_coef_z5);

                            res = svmla_m(pg, res, svadd_m(pg, svld1(pg, s_ptr_ijk + 6), svld1(pg, s_ptr_ijk - 6)), sve_coef_x6);
                            res = svmla_m(pg, res, svadd_m(pg, svld1(pg, s_ptr_ijk + 6 * s_stride_y), svld1(pg, s_ptr_ijk - 6 * s_stride_y)), sve_coef_y6);
                            res = svmla_m(pg, res, svadd_m(pg, svld1(pg, s_ptr_ijk + 6 * s_stride_z), svld1(pg, s_ptr_ijk - 6 * s_stride_z)), sve_coef_z6);

                            T* const __restrict__ d_ptr_ijk = d_ptr + i + j * d_stride_y + k * d_stride_z;
                            svst1(pg, d_ptr_ijk, res);
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
    using traits = sve_traits<T>;
    using sv_float__t = typename traits::sv_float__t;
    constexpr int64_t tile_size_j = traits::tile_size_j;
    constexpr int64_t tile_size_k = traits::tile_size_k;
    constexpr int64_t tile_size_i = traits::tile_size_i;
    // const T coef_0 = d2_coeffs_x[0] + d2_coeffs_y[0] + d2_coeffs_z[0];
    const int64_t svcnt = traits::svcnt_();
    #ifdef USE_OPENMP
    #pragma omp for schedule(static) collapse(2) nowait
    #endif //USE_OPENMP
    for (int64_t kk = 0; kk < nk; kk += tile_size_k) {
        for (int64_t jj = 0; jj < nj; jj += tile_size_j) {
            for (int64_t ii = 0; ii < ni; ii += tile_size_i) {
                for (int64_t k = kk; k < std::min(kk + tile_size_k, nk); k++) {
                    for (int64_t j = jj; j < std::min(jj + tile_size_j, nj); j++) {
                        for (int64_t i = ii; i < std::min(ii + tile_size_i, ni); i += svcnt) {
                            svbool_t pg = traits::svwhilelt_b_(i, ni);
                            T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z;
                            T* const __restrict__ d_ptr_ijk = d_ptr + i + j * d_stride_y + k * d_stride_z;
                            sv_float__t sve_A = svadd_m(pg, svld1(pg, A_ptr + i + j * d_stride_y + k * d_stride_z), coef_0);
                            sv_float__t res = svmul_m(pg, svld1(pg, s_ptr_ijk), sve_A);

                            for (int r = 1; r <= Radius; r++) {
                                const int64_t s_stride_y_r = r * s_stride_y;
                                const int64_t s_stride_z_r = r * s_stride_z;

                                sv_float__t sum_x = svadd_m(pg, svld1(pg, s_ptr_ijk + r), svld1(pg, s_ptr_ijk - r));
                                sv_float__t sum_y = svadd_m(pg, svld1(pg, s_ptr_ijk + s_stride_y_r), svld1(pg, s_ptr_ijk - s_stride_y_r));
                                sv_float__t sum_z = svadd_m(pg, svld1(pg, s_ptr_ijk + s_stride_z_r), svld1(pg, s_ptr_ijk - s_stride_z_r));

                                res = svmla_m(pg, res, sum_x, d2_coeffs_x[r]);
                                res = svmla_m(pg, res, sum_y, d2_coeffs_y[r]);
                                res = svmla_m(pg, res, sum_z, d2_coeffs_z[r]);

                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk + r), d2_coeffs_x[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk - r), d2_coeffs_x[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk + s_stride_y_r), d2_coeffs_y[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk - s_stride_y_r), d2_coeffs_y[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk + s_stride_z_r), d2_coeffs_z[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk - s_stride_z_r), d2_coeffs_z[r]);
                            }
                            svst1(pg, d_ptr_ijk, res);
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
    using traits = sve_traits<T>;
    using sv_float__t = typename traits::sv_float__t;
    constexpr int64_t tile_size_j = traits::tile_size_j;
    constexpr int64_t tile_size_k = traits::tile_size_k;
    constexpr int64_t tile_size_i = traits::tile_size_i;
    // const T coef_0 = d2_coeffs_x[0] + d2_coeffs_y[0] + d2_coeffs_z[0];
    const int64_t svcnt = traits::svcnt_();
    #ifdef USE_OPENMP
    #pragma omp for schedule(static) collapse(2) nowait
    #endif //USE_OPENMP
    for (int64_t kk = 0; kk < nk; kk += tile_size_k) {
        for (int64_t jj = 0; jj < nj; jj += tile_size_j) {
            for (int64_t ii = 0; ii < ni; ii += tile_size_i) {
                for (int64_t k = kk; k < std::min(kk + tile_size_k, nk); k++) {
                    for (int64_t j = jj; j < std::min(jj + tile_size_j, nj); j++) {
                        for (int64_t i = ii; i < std::min(ii + tile_size_i, ni); i += svcnt) {
                            svbool_t pg = traits::svwhilelt_b_(i, ni);
                            T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z;
                            T* const __restrict__ d_ptr_ijk = d_ptr + i + j * d_stride_y + k * d_stride_z;
                            sv_float__t sve_A = svadd_m(pg, svld1(pg, A_ptr + i + j * d_stride_y + k * d_stride_z), coef_0);
                            sv_float__t res = svmul_m(pg, svld1(pg, s_ptr_ijk), sve_A);

                            for (int r = 1; r <= radius; r++) {
                                const int64_t s_stride_y_r = r * s_stride_y;
                                const int64_t s_stride_z_r = r * s_stride_z;

                                sv_float__t sum_x = svadd_m(pg, svld1(pg, s_ptr_ijk + r), svld1(pg, s_ptr_ijk - r));
                                sv_float__t sum_y = svadd_m(pg, svld1(pg, s_ptr_ijk + s_stride_y_r), svld1(pg, s_ptr_ijk - s_stride_y_r));
                                sv_float__t sum_z = svadd_m(pg, svld1(pg, s_ptr_ijk + s_stride_z_r), svld1(pg, s_ptr_ijk - s_stride_z_r));

                                res = svmla_m(pg, res, sum_x, d2_coeffs_x[r]);
                                res = svmla_m(pg, res, sum_y, d2_coeffs_y[r]);
                                res = svmla_m(pg, res, sum_z, d2_coeffs_z[r]);

                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk + r), d2_coeffs_x[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk - r), d2_coeffs_x[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk + s_stride_y_r), d2_coeffs_y[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk - s_stride_y_r), d2_coeffs_y[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk + s_stride_z_r), d2_coeffs_z[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk - s_stride_z_r), d2_coeffs_z[r]);
                            }
                            svst1(pg, d_ptr_ijk, res);
                        }
                    }
                }
            }
        }
    }
    return;
}

template<typename T>
void laplacian_3d_c2_o0(T const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                    T* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                    const int64_t ni, const int64_t nj, const int64_t nk,
                    const int radius,
                    T const* const d2_coeffs_x,
                    T const* const d2_coeffs_y,
                    T const* const d2_coeffs_z,
                    T const coef_0,
                    T const* const A_ptr)
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
    using traits = sve_traits<T>;
    using sv_float__t = typename traits::sv_float__t;
    const int64_t svcnt = traits::svcnt_();
    const svbool_t all_true = traits::svptrue_b_();
    const svbool_t all_false = svpfalse();
    constexpr int64_t tile_size_k = traits::tile_size_k;
    constexpr int64_t tile_size_j = traits::tile_size_j;
    constexpr int64_t tile_size_i = traits::tile_size_i;
    sv_float__t sve_coef0 = traits::svdup_f_(coef_0);
    sv_float__t sve_coef_x1 = traits::svdup_f_(d2_coeffs_x[1]);
    sv_float__t sve_coef_x2 = traits::svdup_f_(d2_coeffs_x[2]);
    sv_float__t sve_coef_x3 = traits::svdup_f_(d2_coeffs_x[3]);
    sv_float__t sve_coef_x4 = traits::svdup_f_(d2_coeffs_x[4]);
    sv_float__t sve_coef_x5 = traits::svdup_f_(d2_coeffs_x[5]);
    sv_float__t sve_coef_x6 = traits::svdup_f_(d2_coeffs_x[6]);
    sv_float__t sve_coef_y1 = traits::svdup_f_(d2_coeffs_y[1]);
    sv_float__t sve_coef_y2 = traits::svdup_f_(d2_coeffs_y[2]);
    sv_float__t sve_coef_y3 = traits::svdup_f_(d2_coeffs_y[3]);
    sv_float__t sve_coef_y4 = traits::svdup_f_(d2_coeffs_y[4]);
    sv_float__t sve_coef_y5 = traits::svdup_f_(d2_coeffs_y[5]);
    sv_float__t sve_coef_y6 = traits::svdup_f_(d2_coeffs_y[6]);
    sv_float__t sve_coef_z1 = traits::svdup_f_(d2_coeffs_z[1]);
    sv_float__t sve_coef_z2 = traits::svdup_f_(d2_coeffs_z[2]);
    sv_float__t sve_coef_z3 = traits::svdup_f_(d2_coeffs_z[3]);
    sv_float__t sve_coef_z4 = traits::svdup_f_(d2_coeffs_z[4]);
    sv_float__t sve_coef_z5 = traits::svdup_f_(d2_coeffs_z[5]);
    sv_float__t sve_coef_z6 = traits::svdup_f_(d2_coeffs_z[6]);
    #ifdef USE_OPENMP
    #pragma omp for schedule(static) collapse(2) nowait
    #endif //USE_OPENMP
    for (int64_t kk = 0; kk < nk; kk += tile_size_k) {
        for (int64_t jj = 0; jj < nj; jj += tile_size_j) {
            for (int64_t ii = 0; ii < ni; ii += tile_size_i) {
                const int64_t kb = std::min(kk + tile_size_k, nk);
                for (int64_t k = kk; k < kb; k++) {
                    const int64_t jb = std::min(jj + tile_size_j, nj);
                    for (int64_t j = jj; j < jb; j++) {
                        const int64_t ib = std::min(ii + tile_size_i, ni);
                        for (int64_t i = ii; i < ib; i+=svcnt) {
                            svbool_t pg = traits::svwhilelt_b_(i, ni);
                            T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z;
                            sv_float__t res = svmul_m(pg, svld1(pg, s_ptr_ijk),
                                            svadd_m(pg, svld1(pg, A_ptr + i + j * d_stride_y + k * d_stride_z), sve_coef0));
                            sv_float__t x_left;
                            // int64_t r = 1;
                            {
                                // const int64_t s_stride_y_r = r * s_stride_y;
                                // const int64_t s_stride_z_r = r * s_stride_z;
                                svbool_t pg_x_left_boundary = traits::svwhilegt_b_(i - 1 + svcnt, i_l);
                                svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + 1, i_r + 1);
                                const svbool_t& pg_y_left_boundary = j - 1 >= j_l ? all_true : all_false;
                                const svbool_t& pg_y_right_boundary = j + 1 <= j_r ? all_true : all_false;
                                const svbool_t& pg_z_left_boundary = k - 1 >= k_l ? all_true : all_false;
                                const svbool_t& pg_z_right_boundary = k + 1 <= k_r ? all_true : all_false;
                                x_left = svld1(pg_x_left_boundary, s_ptr_ijk - 1);
                                sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + 1),
                                                                x_left);
                                sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y),
                                                                svld1(pg_y_left_boundary, s_ptr_ijk - s_stride_y));
                                sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z),
                                                                svld1(pg_z_left_boundary, s_ptr_ijk - s_stride_z));
                                res = svmla_m(pg, res, sum_x, sve_coef_x1);
                                res = svmla_m(pg, res, sum_y, sve_coef_y1);
                                res = svmla_m(pg, res, sum_z, sve_coef_z1);
                            }

                            // r = 2;
                            {
                                const int64_t s_stride_y_r = 2 * s_stride_y;
                                const int64_t s_stride_z_r = 2 * s_stride_z;
                                // svbool_t pg_x_left_boundary = traits::svwhilegt_b_(i - 2 + svcnt, i_l);
                                svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + 2, i_r + 1);
                                const svbool_t& pg_y_left_boundary = j - 2 >= j_l ? all_true : all_false;
                                const svbool_t& pg_y_right_boundary = j + 2 <= j_r ? all_true : all_false;
                                const svbool_t& pg_z_left_boundary = k - 2 >= k_l ? all_true : all_false;
                                const svbool_t& pg_z_right_boundary = k + 2 <= k_r ? all_true : all_false;
                                x_left = i - 2 >= i_l ? svinsr(x_left, *(s_ptr_ijk - 2)) : svinsr(x_left, T(0));;
                                sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + 2),
                                                                x_left);
                                sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r),
                                                                svld1(pg_y_left_boundary, s_ptr_ijk - s_stride_y_r));
                                sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r),
                                                                svld1(pg_z_left_boundary, s_ptr_ijk - s_stride_z_r));
                                res = svmla_m(pg, res, sum_x, sve_coef_x2);
                                res = svmla_m(pg, res, sum_y, sve_coef_y2);
                                res = svmla_m(pg, res, sum_z, sve_coef_z2);
                            }

                            // r = 3;
                            {
                                const int64_t s_stride_y_r = 3 * s_stride_y;
                                const int64_t s_stride_z_r = 3 * s_stride_z;
                                // svbool_t pg_x_left_boundary = traits::svwhilegt_b_(i - 3 + svcnt, i_l);
                                svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + 3, i_r + 1);
                                const svbool_t& pg_y_left_boundary = j - 3 >= j_l ? all_true : all_false;
                                const svbool_t& pg_y_right_boundary = j + 3 <= j_r ? all_true : all_false;
                                const svbool_t& pg_z_left_boundary = k - 3 >= k_l ? all_true : all_false;
                                const svbool_t& pg_z_right_boundary = k + 3 <= k_r ? all_true : all_false;
                                x_left = i - 3 >= i_l ? svinsr(x_left, *(s_ptr_ijk - 3)) : svinsr(x_left, T(0));
                                sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + 3),
                                                                x_left);
                                sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r),
                                                                svld1(pg_y_left_boundary, s_ptr_ijk - s_stride_y_r));
                                sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r),
                                                                svld1(pg_z_left_boundary, s_ptr_ijk - s_stride_z_r));
                                res = svmla_m(pg, res, sum_x, sve_coef_x3);
                                res = svmla_m(pg, res, sum_y, sve_coef_y3);
                                res = svmla_m(pg, res, sum_z, sve_coef_z3);
                            }

                            // r = 4;
                            {
                                const int64_t s_stride_y_r = 4 * s_stride_y;
                                const int64_t s_stride_z_r = 4 * s_stride_z;
                                // svbool_t pg_x_left_boundary = traits::svwhilegt_b_(i - 4 + svcnt, i_l);
                                svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + 4, i_r + 1);
                                const svbool_t& pg_y_left_boundary = j - 4 >= j_l ? all_true : all_false;
                                const svbool_t& pg_y_right_boundary = j + 4 <= j_r ? all_true : all_false;
                                const svbool_t& pg_z_left_boundary = k - 4 >= k_l ? all_true : all_false;
                                const svbool_t& pg_z_right_boundary = k + 4 <= k_r ? all_true : all_false;
                                x_left = i - 4 >= i_l ? svinsr(x_left, *(s_ptr_ijk - 4)) : svinsr(x_left, T(0));
                                sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + 4),
                                                                x_left);
                                sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r),
                                                                svld1(pg_y_left_boundary, s_ptr_ijk - s_stride_y_r));
                                sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r),
                                                                svld1(pg_z_left_boundary, s_ptr_ijk - s_stride_z_r));
                                res = svmla_m(pg, res, sum_x, sve_coef_x4);
                                res = svmla_m(pg, res, sum_y, sve_coef_y4);
                                res = svmla_m(pg, res, sum_z, sve_coef_z4);
                            }

                            {
                                const int64_t s_stride_y_r = 5 * s_stride_y;
                                const int64_t s_stride_z_r = 5 * s_stride_z;
                                // svbool_t pg_x_left_boundary = traits::svwhilegt_b_(i - 5 + svcnt, i_l);
                                svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + 5, i_r + 1);
                                const svbool_t& pg_y_left_boundary = j - 5 >= j_l ? all_true : all_false;
                                const svbool_t& pg_y_right_boundary = j + 5 <= j_r ? all_true : all_false;
                                const svbool_t& pg_z_left_boundary = k - 5 >= k_l ? all_true : all_false;
                                const svbool_t& pg_z_right_boundary = k + 5 <= k_r ? all_true : all_false;
                                x_left = i - 5 >= i_l ? svinsr(x_left, *(s_ptr_ijk - 5)) : svinsr(x_left, T(0));
                                sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + 5),
                                                                    x_left);
                                sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r),
                                                                svld1(pg_y_left_boundary, s_ptr_ijk - s_stride_y_r));
                                sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r),
                                                                svld1(pg_z_left_boundary, s_ptr_ijk - s_stride_z_r));
                                res = svmla_m(pg, res, sum_x, sve_coef_x5);
                                res = svmla_m(pg, res, sum_y, sve_coef_y5);
                                res = svmla_m(pg, res, sum_z, sve_coef_z5);
                            }

                            // r = 6;
                            {
                                const int64_t s_stride_y_r = 6 * s_stride_y;
                                const int64_t s_stride_z_r = 6 * s_stride_z;
                                // svbool_t pg_x_left_boundary = traits::svwhilegt_b_(i - 6 + svcnt, i_l);
                                svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + 6, i_r + 1);
                                const svbool_t& pg_y_left_boundary = j - 6 >= j_l ? all_true : all_false;
                                const svbool_t& pg_y_right_boundary = j + 6 <= j_r ? all_true : all_false;
                                const svbool_t& pg_z_left_boundary = k - 6 >= k_l ? all_true : all_false;
                                const svbool_t& pg_z_right_boundary = k + 6 <= k_r ? all_true : all_false;
                                x_left = i - 6 >= i_l ? svinsr(x_left, *(s_ptr_ijk - 6)) : svinsr(x_left, T(0));
                                sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + 6),
                                                                x_left);
                                sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r),
                                                                svld1(pg_y_left_boundary, s_ptr_ijk - s_stride_y_r));
                                sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r),
                                                                svld1(pg_z_left_boundary, s_ptr_ijk - s_stride_z_r));
                                res = svmla_m(pg, res, sum_x, sve_coef_x6);
                                res = svmla_m(pg, res, sum_y, sve_coef_y6);
                                res = svmla_m(pg, res, sum_z, sve_coef_z6);
                            }

                            svst1(pg, d_ptr + i + j * d_stride_y + k * d_stride_z, res);
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
    using traits = sve_traits<T>;
    using sv_float__t = typename traits::sv_float__t;
    const int64_t svcnt = traits::svcnt_();
    const svbool_t all_true = traits::svptrue_b_();
    const svbool_t all_false = svpfalse();
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
                        for (int64_t i = ii; i < std::min(ii + tile_size_i, ni); i += svcnt) {
                            svbool_t pg = traits::svwhilelt_b_(i, ni);
                            T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z;
                            T* const __restrict__ d_ptr_ijk = d_ptr + i + j * d_stride_y + k * d_stride_z;
                            sv_float__t sve_A = svadd_m(pg, svld1(pg, A_ptr + i + j * d_stride_y + k * d_stride_z), coef_0);
                            sv_float__t res = svmul_m(pg, svld1(pg, s_ptr_ijk), sve_A);

                            for (int r = 1; r <= Radius; r++) {
                                const int64_t s_stride_y_r = r * s_stride_y;
                                const int64_t s_stride_z_r = r * s_stride_z;

                                svbool_t pg_x_left_boundary = traits::svwhilegt_b_(i - r + svcnt, i_l);
                                svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + r, i_r + 1);
                                const svbool_t& pg_y_left_boundary = j - r >= j_l ? all_true : all_false;
                                const svbool_t& pg_y_right_boundary = j + r <= j_r ? all_true : all_false;
                                const svbool_t& pg_z_left_boundary = k - r >= k_l ? all_true : all_false;
                                const svbool_t& pg_z_right_boundary = k + r <= k_r ? all_true : all_false;

                                sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + r),
                                                                svld1(pg_x_left_boundary, s_ptr_ijk - r));
                                sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r),
                                                                svld1(pg_y_left_boundary, s_ptr_ijk - s_stride_y_r));
                                sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r),
                                                                svld1(pg_z_left_boundary, s_ptr_ijk - s_stride_z_r));

                                res = svmla_m(pg, res, sum_x, d2_coeffs_x[r]);
                                res = svmla_m(pg, res, sum_y, d2_coeffs_y[r]);
                                res = svmla_m(pg, res, sum_z, d2_coeffs_z[r]);

                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk + r), d2_coeffs_x[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk - r), d2_coeffs_x[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk + s_stride_y_r), d2_coeffs_y[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk - s_stride_y_r), d2_coeffs_y[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk + s_stride_z_r), d2_coeffs_z[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk - s_stride_z_r), d2_coeffs_z[r]);
                            }
                            svst1(pg, d_ptr_ijk, res);
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
    using traits = sve_traits<T>;
    using sv_float__t = typename traits::sv_float__t;
    const int64_t svcnt = traits::svcnt_();
    const svbool_t all_true = traits::svptrue_b_();
    const svbool_t all_false = svpfalse();
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
                        for (int64_t i = ii; i < std::min(ii + tile_size_i, ni); i += svcnt) {
                            svbool_t pg = traits::svwhilelt_b_(i, ni);
                            T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z;
                            T* const __restrict__ d_ptr_ijk = d_ptr + i + j * d_stride_y + k * d_stride_z;
                            sv_float__t sve_A = svadd_m(pg, svld1(pg, A_ptr + i + j * d_stride_y + k * d_stride_z), coef_0);
                            sv_float__t res = svmul_m(pg, svld1(pg, s_ptr_ijk), sve_A);

                            for (int r = 1; r <= radius; r++) {
                                const int64_t s_stride_y_r = r * s_stride_y;
                                const int64_t s_stride_z_r = r * s_stride_z;

                                svbool_t pg_x_left_boundary = traits::svwhilegt_b_(i - r + svcnt, i_l);
                                svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + r, i_r + 1);
                                const svbool_t& pg_y_left_boundary = j - r >= j_l ? all_true : all_false;
                                const svbool_t& pg_y_right_boundary = j + r <= j_r ? all_true : all_false;
                                const svbool_t& pg_z_left_boundary = k - r >= k_l ? all_true : all_false;
                                const svbool_t& pg_z_right_boundary = k + r <= k_r ? all_true : all_false;

                                sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + r),
                                                                svld1(pg_x_left_boundary, s_ptr_ijk - r));
                                sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r),
                                                                svld1(pg_y_left_boundary, s_ptr_ijk - s_stride_y_r));
                                sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r),
                                                                svld1(pg_z_left_boundary, s_ptr_ijk - s_stride_z_r));

                                res = svmla_m(pg, res, sum_x, d2_coeffs_x[r]);
                                res = svmla_m(pg, res, sum_y, d2_coeffs_y[r]);
                                res = svmla_m(pg, res, sum_z, d2_coeffs_z[r]);

                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk + r), d2_coeffs_x[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk - r), d2_coeffs_x[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk + s_stride_y_r), d2_coeffs_y[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk - s_stride_y_r), d2_coeffs_y[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk + s_stride_z_r), d2_coeffs_z[r]);
                                // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk - s_stride_z_r), d2_coeffs_z[r]);
                            }
                            svst1(pg, d_ptr_ijk, res);
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
            laplacian_3d_c2_o0_kernel_boundary_safe<6, T>(s_ptr, s_stride_y, s_stride_z,
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
    using traits = sve_traits<T>;
    using sv_float__t = typename traits::sv_float__t;
    const int64_t svcnt = traits::svcnt_();
    const svbool_t all_true = traits::svptrue_b_();
    const svbool_t all_false = svpfalse();
    constexpr int64_t tile_size_k = traits::tile_size_k;
    constexpr int64_t tile_size_j = traits::tile_size_j;
    constexpr int64_t tile_size_i = traits::tile_size_i;
    sv_float__t sve_coef0 = traits::svdup_f_(coef_0);
    sv_float__t sve_coef_x1 = traits::svdup_f_(d2_coeffs_x[1]);
    sv_float__t sve_coef_x2 = traits::svdup_f_(d2_coeffs_x[2]);
    sv_float__t sve_coef_x3 = traits::svdup_f_(d2_coeffs_x[3]);
    sv_float__t sve_coef_x4 = traits::svdup_f_(d2_coeffs_x[4]);
    sv_float__t sve_coef_x5 = traits::svdup_f_(d2_coeffs_x[5]);
    sv_float__t sve_coef_x6 = traits::svdup_f_(d2_coeffs_x[6]);
    sv_float__t sve_coef_y1 = traits::svdup_f_(d2_coeffs_y[1]);
    sv_float__t sve_coef_y2 = traits::svdup_f_(d2_coeffs_y[2]);
    sv_float__t sve_coef_y3 = traits::svdup_f_(d2_coeffs_y[3]);
    sv_float__t sve_coef_y4 = traits::svdup_f_(d2_coeffs_y[4]);
    sv_float__t sve_coef_y5 = traits::svdup_f_(d2_coeffs_y[5]);
    sv_float__t sve_coef_y6 = traits::svdup_f_(d2_coeffs_y[6]);
    sv_float__t sve_coef_z1 = traits::svdup_f_(d2_coeffs_z[1]);
    sv_float__t sve_coef_z2 = traits::svdup_f_(d2_coeffs_z[2]);
    sv_float__t sve_coef_z3 = traits::svdup_f_(d2_coeffs_z[3]);
    sv_float__t sve_coef_z4 = traits::svdup_f_(d2_coeffs_z[4]);
    sv_float__t sve_coef_z5 = traits::svdup_f_(d2_coeffs_z[5]);
    sv_float__t sve_coef_z6 = traits::svdup_f_(d2_coeffs_z[6]);
    #ifdef USE_OPENMP
    #pragma omp for schedule(static) nowait
    #endif //USE_OPENMP
    for (int64_t b = 0; b < nb; b++) {
        for (int64_t kk = 0; kk < nk; kk += tile_size_k) {
            for (int64_t jj = 0; jj < nj; jj += tile_size_j) {
                for (int64_t ii = 0; ii < ni; ii += tile_size_i) {
                    const int64_t kb = std::min(kk + tile_size_k, nk);
                    for (int64_t k = kk; k < kb; k++) {
                        const int64_t jb = std::min(jj + tile_size_j, nj);
                        for (int64_t j = jj; j < jb; j++) {
                            const int64_t ib = std::min(ii + tile_size_i, ni);
                            for (int64_t i = ii; i < ib; i+=svcnt) {
                                svbool_t pg = traits::svwhilelt_b_(i, ni);
                                T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z + b * s_stride_b;
                                sv_float__t res = svmul_m(pg, svld1(pg, s_ptr_ijk),
                                                svadd_m(pg, svld1(pg, A_ptr + i + j * d_stride_y + k * d_stride_z), sve_coef0));
                                sv_float__t x_left;
                                // int64_t r = 1;
                                {
                                    // const int64_t s_stride_y_r = r * s_stride_y;
                                    // const int64_t s_stride_z_r = r * s_stride_z;
                                    svbool_t pg_x_left_boundary = traits::svwhilegt_b_(i - 1 + svcnt, i_l);
                                    svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + 1, i_r + 1);
                                    const svbool_t& pg_y_left_boundary = j - 1 >= j_l ? all_true : all_false;
                                    const svbool_t& pg_y_right_boundary = j + 1 <= j_r ? all_true : all_false;
                                    const svbool_t& pg_z_left_boundary = k - 1 >= k_l ? all_true : all_false;
                                    const svbool_t& pg_z_right_boundary = k + 1 <= k_r ? all_true : all_false;
                                    x_left = svld1(pg_x_left_boundary, s_ptr_ijk - 1);
                                    sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + 1),
                                                                    x_left);
                                    sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y),
                                                                    svld1(pg_y_left_boundary, s_ptr_ijk - s_stride_y));
                                    sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z),
                                                                    svld1(pg_z_left_boundary, s_ptr_ijk - s_stride_z));
                                    res = svmla_m(pg, res, sum_x, sve_coef_x1);
                                    res = svmla_m(pg, res, sum_y, sve_coef_y1);
                                    res = svmla_m(pg, res, sum_z, sve_coef_z1);
                                }

                                // r = 2;
                                {
                                    const int64_t s_stride_y_r = 2 * s_stride_y;
                                    const int64_t s_stride_z_r = 2 * s_stride_z;
                                    // svbool_t pg_x_left_boundary = traits::svwhilegt_b_(i - 2 + svcnt, i_l);
                                    svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + 2, i_r + 1);
                                    const svbool_t& pg_y_left_boundary = j - 2 >= j_l ? all_true : all_false;
                                    const svbool_t& pg_y_right_boundary = j + 2 <= j_r ? all_true : all_false;
                                    const svbool_t& pg_z_left_boundary = k - 2 >= k_l ? all_true : all_false;
                                    const svbool_t& pg_z_right_boundary = k + 2 <= k_r ? all_true : all_false;
                                    x_left = i - 2 >= i_l ? svinsr(x_left, *(s_ptr_ijk - 2)) : svinsr(x_left, T(0));;
                                    sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + 2),
                                                                    x_left);
                                    sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r),
                                                                    svld1(pg_y_left_boundary, s_ptr_ijk - s_stride_y_r));
                                    sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r),
                                                                    svld1(pg_z_left_boundary, s_ptr_ijk - s_stride_z_r));
                                    res = svmla_m(pg, res, sum_x, sve_coef_x2);
                                    res = svmla_m(pg, res, sum_y, sve_coef_y2);
                                    res = svmla_m(pg, res, sum_z, sve_coef_z2);
                                }

                                // r = 3;
                                {
                                    const int64_t s_stride_y_r = 3 * s_stride_y;
                                    const int64_t s_stride_z_r = 3 * s_stride_z;
                                    // svbool_t pg_x_left_boundary = traits::svwhilegt_b_(i - 3 + svcnt, i_l);
                                    svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + 3, i_r + 1);
                                    const svbool_t& pg_y_left_boundary = j - 3 >= j_l ? all_true : all_false;
                                    const svbool_t& pg_y_right_boundary = j + 3 <= j_r ? all_true : all_false;
                                    const svbool_t& pg_z_left_boundary = k - 3 >= k_l ? all_true : all_false;
                                    const svbool_t& pg_z_right_boundary = k + 3 <= k_r ? all_true : all_false;
                                    x_left = i - 3 >= i_l ? svinsr(x_left, *(s_ptr_ijk - 3)) : svinsr(x_left, T(0));
                                    sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + 3),
                                                                    x_left);
                                    sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r),
                                                                    svld1(pg_y_left_boundary, s_ptr_ijk - s_stride_y_r));
                                    sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r),
                                                                    svld1(pg_z_left_boundary, s_ptr_ijk - s_stride_z_r));
                                    res = svmla_m(pg, res, sum_x, sve_coef_x3);
                                    res = svmla_m(pg, res, sum_y, sve_coef_y3);
                                    res = svmla_m(pg, res, sum_z, sve_coef_z3);
                                }

                                // r = 4;
                                {
                                    const int64_t s_stride_y_r = 4 * s_stride_y;
                                    const int64_t s_stride_z_r = 4 * s_stride_z;
                                    // svbool_t pg_x_left_boundary = traits::svwhilegt_b_(i - 4 + svcnt, i_l);
                                    svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + 4, i_r + 1);
                                    const svbool_t& pg_y_left_boundary = j - 4 >= j_l ? all_true : all_false;
                                    const svbool_t& pg_y_right_boundary = j + 4 <= j_r ? all_true : all_false;
                                    const svbool_t& pg_z_left_boundary = k - 4 >= k_l ? all_true : all_false;
                                    const svbool_t& pg_z_right_boundary = k + 4 <= k_r ? all_true : all_false;
                                    x_left = i - 4 >= i_l ? svinsr(x_left, *(s_ptr_ijk - 4)) : svinsr(x_left, T(0));
                                    sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + 4),
                                                                    x_left);
                                    sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r),
                                                                    svld1(pg_y_left_boundary, s_ptr_ijk - s_stride_y_r));
                                    sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r),
                                                                    svld1(pg_z_left_boundary, s_ptr_ijk - s_stride_z_r));
                                    res = svmla_m(pg, res, sum_x, sve_coef_x4);
                                    res = svmla_m(pg, res, sum_y, sve_coef_y4);
                                    res = svmla_m(pg, res, sum_z, sve_coef_z4);
                                }

                                {
                                    const int64_t s_stride_y_r = 5 * s_stride_y;
                                    const int64_t s_stride_z_r = 5 * s_stride_z;
                                    // svbool_t pg_x_left_boundary = traits::svwhilegt_b_(i - 5 + svcnt, i_l);
                                    svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + 5, i_r + 1);
                                    const svbool_t& pg_y_left_boundary = j - 5 >= j_l ? all_true : all_false;
                                    const svbool_t& pg_y_right_boundary = j + 5 <= j_r ? all_true : all_false;
                                    const svbool_t& pg_z_left_boundary = k - 5 >= k_l ? all_true : all_false;
                                    const svbool_t& pg_z_right_boundary = k + 5 <= k_r ? all_true : all_false;
                                    x_left = i - 5 >= i_l ? svinsr(x_left, *(s_ptr_ijk - 5)) : svinsr(x_left, T(0));
                                    sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + 5),
                                                                     x_left);
                                    sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r),
                                                                    svld1(pg_y_left_boundary, s_ptr_ijk - s_stride_y_r));
                                    sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r),
                                                                    svld1(pg_z_left_boundary, s_ptr_ijk - s_stride_z_r));
                                    res = svmla_m(pg, res, sum_x, sve_coef_x5);
                                    res = svmla_m(pg, res, sum_y, sve_coef_y5);
                                    res = svmla_m(pg, res, sum_z, sve_coef_z5);
                                }

                                // r = 6;
                                {
                                    const int64_t s_stride_y_r = 6 * s_stride_y;
                                    const int64_t s_stride_z_r = 6 * s_stride_z;
                                    // svbool_t pg_x_left_boundary = traits::svwhilegt_b_(i - 6 + svcnt, i_l);
                                    svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + 6, i_r + 1);
                                    const svbool_t& pg_y_left_boundary = j - 6 >= j_l ? all_true : all_false;
                                    const svbool_t& pg_y_right_boundary = j + 6 <= j_r ? all_true : all_false;
                                    const svbool_t& pg_z_left_boundary = k - 6 >= k_l ? all_true : all_false;
                                    const svbool_t& pg_z_right_boundary = k + 6 <= k_r ? all_true : all_false;
                                    x_left = i - 6 >= i_l ? svinsr(x_left, *(s_ptr_ijk - 6)) : svinsr(x_left, T(0));
                                    sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + 6),
                                                                    x_left);
                                    sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r),
                                                                    svld1(pg_y_left_boundary, s_ptr_ijk - s_stride_y_r));
                                    sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r),
                                                                    svld1(pg_z_left_boundary, s_ptr_ijk - s_stride_z_r));
                                    res = svmla_m(pg, res, sum_x, sve_coef_x6);
                                    res = svmla_m(pg, res, sum_y, sve_coef_y6);
                                    res = svmla_m(pg, res, sum_z, sve_coef_z6);
                                }

                                svst1(pg, d_ptr + i + j * d_stride_y + k * d_stride_z + b * d_stride_b, res);
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
    using traits = sve_traits<T>;
    using sv_float__t = typename traits::sv_float__t;
    const int64_t svcnt = traits::svcnt_();
    const svbool_t all_true = traits::svptrue_b_();
    const svbool_t all_false = svpfalse();
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
                            for (int64_t i = ii; i < std::min(ii + tile_size_i, ni); i += svcnt) {
                                svbool_t pg = traits::svwhilelt_b_(i, ni);
                                T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z + b * s_stride_b;
                                T* const __restrict__ d_ptr_ijk = d_ptr + i + j * d_stride_y + k * d_stride_z + b * d_stride_b;
                                sv_float__t sve_A = svadd_m(pg, svld1(pg, A_ptr + i + j * d_stride_y + k * d_stride_z), coef_0);
                                sv_float__t res = svmul_m(pg, svld1(pg, s_ptr_ijk), sve_A);

                                for (int r = 1; r <= Radius; r++) {
                                    const int64_t s_stride_y_r = r * s_stride_y;
                                    const int64_t s_stride_z_r = r * s_stride_z;

                                    svbool_t pg_x_left_boundary = traits::svwhilegt_b_(i - r + svcnt, i_l);
                                    svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + r, i_r + 1);
                                    const svbool_t& pg_y_left_boundary = j - r >= j_l ? all_true : all_false;
                                    const svbool_t& pg_y_right_boundary = j + r <= j_r ? all_true : all_false;
                                    const svbool_t& pg_z_left_boundary = k - r >= k_l ? all_true : all_false;
                                    const svbool_t& pg_z_right_boundary = k + r <= k_r ? all_true : all_false;

                                    sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + r),
                                                                    svld1(pg_x_left_boundary, s_ptr_ijk - r));
                                    sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r),
                                                                    svld1(pg_y_left_boundary, s_ptr_ijk - s_stride_y_r));
                                    sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r),
                                                                    svld1(pg_z_left_boundary, s_ptr_ijk - s_stride_z_r));

                                    res = svmla_m(pg, res, sum_x, d2_coeffs_x[r]);
                                    res = svmla_m(pg, res, sum_y, d2_coeffs_y[r]);
                                    res = svmla_m(pg, res, sum_z, d2_coeffs_z[r]);

                                    // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk + r), d2_coeffs_x[r]);
                                    // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk - r), d2_coeffs_x[r]);
                                    // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk + s_stride_y_r), d2_coeffs_y[r]);
                                    // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk - s_stride_y_r), d2_coeffs_y[r]);
                                    // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk + s_stride_z_r), d2_coeffs_z[r]);
                                    // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk - s_stride_z_r), d2_coeffs_z[r]);
                                }
                                svst1(pg, d_ptr_ijk, res);
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
    using traits = sve_traits<T>;
    using sv_float__t = typename traits::sv_float__t;
    const int64_t svcnt = traits::svcnt_();
    const svbool_t all_true = traits::svptrue_b_();
    const svbool_t all_false = svpfalse();
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
                            for (int64_t i = ii; i < std::min(ii + tile_size_i, ni); i += svcnt) {
                                svbool_t pg = traits::svwhilelt_b_(i, ni);
                                T const* const __restrict__ s_ptr_ijk = s_ptr + i + j * s_stride_y + k * s_stride_z + b * s_stride_b;
                                T* const __restrict__ d_ptr_ijk = d_ptr + i + j * d_stride_y + k * d_stride_z + b * d_stride_b;
                                sv_float__t sve_A = svadd_m(pg, svld1(pg, A_ptr + i + j * d_stride_y + k * d_stride_z), coef_0);
                                sv_float__t res = svmul_m(pg, svld1(pg, s_ptr_ijk), sve_A);

                                for (int r = 1; r <= radius; r++) {
                                    const int64_t s_stride_y_r = r * s_stride_y;
                                    const int64_t s_stride_z_r = r * s_stride_z;

                                    svbool_t pg_x_left_boundary = traits::svwhilegt_b_(i - r + svcnt, i_l);
                                    svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + r, i_r + 1);
                                    const svbool_t& pg_y_left_boundary = j - r >= j_l ? all_true : all_false;
                                    const svbool_t& pg_y_right_boundary = j + r <= j_r ? all_true : all_false;
                                    const svbool_t& pg_z_left_boundary = k - r >= k_l ? all_true : all_false;
                                    const svbool_t& pg_z_right_boundary = k + r <= k_r ? all_true : all_false;

                                    sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + r),
                                                                    svld1(pg_x_left_boundary, s_ptr_ijk - r));
                                    sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r),
                                                                    svld1(pg_y_left_boundary, s_ptr_ijk - s_stride_y_r));
                                    sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r),
                                                                    svld1(pg_z_left_boundary, s_ptr_ijk - s_stride_z_r));

                                    res = svmla_m(pg, res, sum_x, d2_coeffs_x[r]);
                                    res = svmla_m(pg, res, sum_y, d2_coeffs_y[r]);
                                    res = svmla_m(pg, res, sum_z, d2_coeffs_z[r]);

                                    // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk + r), d2_coeffs_x[r]);
                                    // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk - r), d2_coeffs_x[r]);
                                    // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk + s_stride_y_r), d2_coeffs_y[r]);
                                    // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk - s_stride_y_r), d2_coeffs_y[r]);
                                    // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk + s_stride_z_r), d2_coeffs_z[r]);
                                    // res = svmla_m(pg, res, svld1(pg, s_ptr_ijk - s_stride_z_r), d2_coeffs_z[r]);
                                }
                                svst1(pg, d_ptr_ijk, res);
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

#endif //!defined(__ARM_FEATURE_SME) && defined(__ARM_FEATURE_SVE)
