#include "stencil_kernel.h"

#ifdef __ARM_FEATURE_SME
#pragma message("Building stencil_sme.cpp.")
#include <arm_sve.h>
#include <arm_sme.h>

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

// template<>
// struct Stencil3D<6, double> {
//     static constexpr int radius = 6;
//     static void lap_3d_c2_o0(double const* const s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
//                             double* const d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
//                             const int64_t ni, const int64_t nj, const int64_t nk,
//                             double const* const d2_coeffs_x,
//                             double const* const d2_coeffs_y,
//                             double const* const d2_coeffs_z,
//                             double const* const A_ptr) __arm_streaming;
// };

// __arm_new("za")
// void Stencil3D<6, double>::lap_3d_c2_o0(double const* const __restrict__ s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
//                                     double* const __restrict__ d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
//                                     const int64_t ni, const int64_t nj, const int64_t nk,
//                                     double const* const __restrict__ d2_coeffs_x,
//                                     double const* const __restrict__ d2_coeffs_y,
//                                     double const* const __restrict__ d2_coeffs_z,
//                                     double const* const __restrict__ A_ptr)
// __arm_streaming 
// {
//     constexpr int R = radius;
//     using traits = sve_traits<double>;
//     using sv_float__t = typename traits::sv_float__t;
//     const int64_t svcnt = traits::svcnt_();
//     constexpr int64_t tile_size_k = 8;
//     constexpr int64_t tile_size_j = 8;
//     constexpr int64_t tile_size_i = 8;
//     double s_st[tile_size_k+R*2][tile_size_j+R*2][tile_size_i+R*2], (* const s)[tile_size_j+R*2][tile_size_i+R*2] = (double(*)[tile_size_j+R*2][tile_size_i+R*2])&s_st[R][R][R];
//     double d[tile_size_k][tile_size_j][tile_size_i];
//     const double coef_0 = d2_coeffs_x[0] + d2_coeffs_y[0] + d2_coeffs_z[0];
//     #ifdef USE_OPENMP
//     #pragma omp for schedule(static) collapse(2) nowait
//     #endif //USE_OPENMP
//     for (int64_t kk = 0; kk < nk; kk += tile_size_k) {
//         for (int64_t jj = 0; jj < nj; jj += tile_size_j) {
//             for (int64_t ii = 0; ii < ni; ii += tile_size_i) {
//                 // print64_tf("load %d %d\n", kk, jj);
//                 const int64_t kt = std::min(kk + tile_size_k, nk);
//                 const int64_t jt = std::min(jj + tile_size_j, nj);
//                 const int64_t it = std::min(ii + tile_size_i, ni);
//                 for (int64_t k = kk; k < kt; k++) {
//                     for (int64_t j = jj; j < jt; j++) {
//                         #pragma omp simd
//                         for (int64_t i = ii-R; i < it+R; i ++) {
//                             s[k-kk][j-jj][i-ii] = s_ptr[i + j * s_stride_y + k * s_stride_z];
//                         }
//                     }
//                 }
//                 // print64_tf("load km\n", kk, jj);
//                 for (int64_t k = kk-R; k < kk; k++) {
//                     for (int64_t j = jj; j < jt; j++) {
//                         #pragma omp simd
//                         for (int64_t i = ii; i < it; i ++) {
//                             s[k-kk][j-jj][i-ii] = s_ptr[i + j * s_stride_y + k * s_stride_z];
//                         }
//                     }
//                 }
//                 // print64_tf("load kp\n", kk, jj);
//                 for (int64_t k = kt; k < kt+R; k++) {
//                     for (int64_t j = jj; j < jt; j++) {
//                         #pragma omp simd
//                         for (int64_t i = ii; i < it; i ++) {
//                             s[k-kk][j-jj][i-ii] = s_ptr[i + j * s_stride_y + k * s_stride_z];
//                         }
//                     }
//                 }
//                 // print64_tf("load jm\n", kk, jj);
//                 for (int64_t k = kk; k < kt; k++) {
//                     for (int64_t j = jj-R; j < jj; j++) {
//                         #pragma omp simd
//                         for (int64_t i = ii; i < it; i ++) {
//                             s[k-kk][j-jj][i-ii] = s_ptr[i + j * s_stride_y + k * s_stride_z];
//                         }
//                     }
//                 }
//                 for (int64_t k = kk; k < kt; k++) {
//                     for (int64_t j = jt; j < jt+R; j++) {
//                         #pragma omp simd
//                         for (int64_t i = ii; i < it; i ++) {
//                             s[k-kk][j-jj][i-ii] = s_ptr[i + j * s_stride_y + k * s_stride_z];
//                         }
//                     }
//                 }
//                 for (int64_t k = kk; k < kt; k++) {
//                     for (int64_t j = jj; j < jt; j++) {
//                         #pragma omp simd
//                         for (int64_t i = ii; i < it; i ++) {
//                             d[k-kk][j-jj][i-ii] = A_ptr[i + j * d_stride_y + k * d_stride_z];
//                         }
//                     }
//                 }
//                 // print64_tf("calc %d %d\n", kk, jj);
//                 for (int64_t k = 0; k < tile_size_k; k++) {
//                     // for (int64_t j = 0; j < tile_size_j; j++) {
//                     for (int64_t i = 0; i < tile_size_i; i += svcnt) {
//                         svbool_t pg = traits::svwhilelt_b_(i+ii, ni);
//                         constexpr int64_t j = 0;
//                         sv_float__t sve_A0 = svadd_m(pg, svld1(pg, &d[k][j+0][i]), coef_0);
//                         sv_float__t sve_A1 = svadd_m(pg, svld1(pg, &d[k][j+1][i]), coef_0);
//                         sv_float__t sve_A2 = svadd_m(pg, svld1(pg, &d[k][j+2][i]), coef_0);
//                         sv_float__t sve_A3 = svadd_m(pg, svld1(pg, &d[k][j+3][i]), coef_0);
//                         sv_float__t sve_A4 = svadd_m(pg, svld1(pg, &d[k][j+4][i]), coef_0);
//                         sv_float__t sve_A5 = svadd_m(pg, svld1(pg, &d[k][j+5][i]), coef_0);
//                         sv_float__t sve_A6 = svadd_m(pg, svld1(pg, &d[k][j+6][i]), coef_0);
//                         sv_float__t sve_A7 = svadd_m(pg, svld1(pg, &d[k][j+7][i]), coef_0);
//                         sv_float__t res0 = svmul_m(pg, svld1(pg, &s[k][j+0][i]), sve_A0);
//                         sv_float__t res1 = svmul_m(pg, svld1(pg, &s[k][j+1][i]), sve_A1);
//                         sv_float__t res2 = svmul_m(pg, svld1(pg, &s[k][j+2][i]), sve_A2);
//                         sv_float__t res3 = svmul_m(pg, svld1(pg, &s[k][j+3][i]), sve_A3);
//                         sv_float__t res4 = svmul_m(pg, svld1(pg, &s[k][j+4][i]), sve_A4);
//                         sv_float__t res5 = svmul_m(pg, svld1(pg, &s[k][j+5][i]), sve_A5);
//                         sv_float__t res6 = svmul_m(pg, svld1(pg, &s[k][j+6][i]), sve_A6);
//                         sv_float__t res7 = svmul_m(pg, svld1(pg, &s[k][j+7][i]), sve_A7);

//                         for (int r = 1; r <= R; r++) {
//                             // sv_float__t sum_x0 = svadd_m(pg, svld1(pg, &s[k][j+0][i+r]), svld1(pg, &s[k][j+0][i-r]));
//                             // sv_float__t sum_x1 = svadd_m(pg, svld1(pg, &s[k][j+1][i+r]), svld1(pg, &s[k][j+1][i-r]));
//                             // sv_float__t sum_x2 = svadd_m(pg, svld1(pg, &s[k][j+2][i+r]), svld1(pg, &s[k][j+2][i-r]));
//                             // sv_float__t sum_x3 = svadd_m(pg, svld1(pg, &s[k][j+3][i+r]), svld1(pg, &s[k][j+3][i-r]));
//                             // sv_float__t sum_x4 = svadd_m(pg, svld1(pg, &s[k][j+4][i+r]), svld1(pg, &s[k][j+4][i-r]));
//                             // sv_float__t sum_x5 = svadd_m(pg, svld1(pg, &s[k][j+5][i+r]), svld1(pg, &s[k][j+5][i-r]));
//                             // sv_float__t sum_x6 = svadd_m(pg, svld1(pg, &s[k][j+6][i+r]), svld1(pg, &s[k][j+6][i-r]));
//                             // sv_float__t sum_x7 = svadd_m(pg, svld1(pg, &s[k][j+7][i+r]), svld1(pg, &s[k][j+7][i-r]));
//                             res0 = svmla_m(pg, res0, svld1(pg, &s[k][j+0][i+r]), d2_coeffs_x[r]);
//                             res1 = svmla_m(pg, res1, svld1(pg, &s[k][j+1][i+r]), d2_coeffs_x[r]);
//                             res2 = svmla_m(pg, res2, svld1(pg, &s[k][j+2][i+r]), d2_coeffs_x[r]);
//                             res3 = svmla_m(pg, res3, svld1(pg, &s[k][j+3][i+r]), d2_coeffs_x[r]);
//                             res4 = svmla_m(pg, res4, svld1(pg, &s[k][j+4][i+r]), d2_coeffs_x[r]);
//                             res5 = svmla_m(pg, res5, svld1(pg, &s[k][j+5][i+r]), d2_coeffs_x[r]);
//                             res6 = svmla_m(pg, res6, svld1(pg, &s[k][j+6][i+r]), d2_coeffs_x[r]);
//                             res7 = svmla_m(pg, res7, svld1(pg, &s[k][j+7][i+r]), d2_coeffs_x[r]);
//                             res0 = svmla_m(pg, res0, svld1(pg, &s[k][j+0][i-r]), d2_coeffs_x[r]);
//                             res1 = svmla_m(pg, res1, svld1(pg, &s[k][j+1][i-r]), d2_coeffs_x[r]);
//                             res2 = svmla_m(pg, res2, svld1(pg, &s[k][j+2][i-r]), d2_coeffs_x[r]);
//                             res3 = svmla_m(pg, res3, svld1(pg, &s[k][j+3][i-r]), d2_coeffs_x[r]);
//                             res4 = svmla_m(pg, res4, svld1(pg, &s[k][j+4][i-r]), d2_coeffs_x[r]);
//                             res5 = svmla_m(pg, res5, svld1(pg, &s[k][j+5][i-r]), d2_coeffs_x[r]);
//                             res6 = svmla_m(pg, res6, svld1(pg, &s[k][j+6][i-r]), d2_coeffs_x[r]);
//                             res7 = svmla_m(pg, res7, svld1(pg, &s[k][j+7][i-r]), d2_coeffs_x[r]);
//                         }
//                         svst1(pg, &d[k][j+0][i], res0);
//                         svst1(pg, &d[k][j+1][i], res1);
//                         svst1(pg, &d[k][j+2][i], res2);
//                         svst1(pg, &d[k][j+3][i], res3);
//                         svst1(pg, &d[k][j+4][i], res4);
//                         svst1(pg, &d[k][j+5][i], res5);
//                         svst1(pg, &d[k][j+6][i], res6);
//                         svst1(pg, &d[k][j+7][i], res7);
//                     }
//                 }

//                 svzero_za(); 
//                 // y direction
//                 constexpr int64_t i = 0;
//                 {
//                     svbool_t pg = traits::svwhilelt_b_(i+ii, ni);
//                     //load d into ZA [k][j][i]
//                     for (int64_t j = 0; j < std::min(nj - jj, svcnt); j++) {
//                         svld1_hor_za64(0, j, pg, &d[0][j][i]);
//                         svld1_hor_za64(1, j, pg, &d[1][j][i]);
//                         svld1_hor_za64(2, j, pg, &d[2][j][i]);
//                         svld1_hor_za64(3, j, pg, &d[3][j][i]);
//                         svld1_hor_za64(4, j, pg, &d[4][j][i]);
//                         svld1_hor_za64(5, j, pg, &d[5][j][i]);
//                         svld1_hor_za64(6, j, pg, &d[6][j][i]);
//                         svld1_hor_za64(7, j, pg, &d[7][j][i]);
//                     }

//                     const svbool_t svptrue = traits::svptrue_b_();
//                     sv_float__t coef_y = traits::svdup_f_(0.0);
//                     sv_float__t sve_y0;
//                     sv_float__t sve_y1;
//                     sv_float__t sve_y2;
//                     sv_float__t sve_y3;
//                     sv_float__t sve_y4;
//                     sv_float__t sve_y5;
//                     sv_float__t sve_y6;
//                     sv_float__t sve_y7;

//                     coef_y = svinsr(coef_y, d2_coeffs_y[6]);
//                     sve_y0 = svld1(pg, &s[0][-6][i]); svmopa_za64_m(0, svptrue, svptrue, coef_y, sve_y0);
//                     sve_y1 = svld1(pg, &s[1][-6][i]); svmopa_za64_m(1, svptrue, svptrue, coef_y, sve_y1);
//                     sve_y2 = svld1(pg, &s[2][-6][i]); svmopa_za64_m(2, svptrue, svptrue, coef_y, sve_y2);
//                     sve_y3 = svld1(pg, &s[3][-6][i]); svmopa_za64_m(3, svptrue, svptrue, coef_y, sve_y3);
//                     sve_y4 = svld1(pg, &s[4][-6][i]); svmopa_za64_m(4, svptrue, svptrue, coef_y, sve_y4);
//                     sve_y5 = svld1(pg, &s[5][-6][i]); svmopa_za64_m(5, svptrue, svptrue, coef_y, sve_y5);
//                     sve_y6 = svld1(pg, &s[6][-6][i]); svmopa_za64_m(6, svptrue, svptrue, coef_y, sve_y6);
//                     sve_y7 = svld1(pg, &s[7][-6][i]); svmopa_za64_m(7, svptrue, svptrue, coef_y, sve_y7);

//                     coef_y = svinsr(coef_y, d2_coeffs_y[5]);
//                     sve_y0 = svld1(pg, &s[0][-5][i]); svmopa_za64_m(0, svptrue, svptrue, coef_y, sve_y0);
//                     sve_y1 = svld1(pg, &s[1][-5][i]); svmopa_za64_m(1, svptrue, svptrue, coef_y, sve_y1);
//                     sve_y2 = svld1(pg, &s[2][-5][i]); svmopa_za64_m(2, svptrue, svptrue, coef_y, sve_y2);
//                     sve_y3 = svld1(pg, &s[3][-5][i]); svmopa_za64_m(3, svptrue, svptrue, coef_y, sve_y3);
//                     sve_y4 = svld1(pg, &s[4][-5][i]); svmopa_za64_m(4, svptrue, svptrue, coef_y, sve_y4);
//                     sve_y5 = svld1(pg, &s[5][-5][i]); svmopa_za64_m(5, svptrue, svptrue, coef_y, sve_y5);
//                     sve_y6 = svld1(pg, &s[6][-5][i]); svmopa_za64_m(6, svptrue, svptrue, coef_y, sve_y6);
//                     sve_y7 = svld1(pg, &s[7][-5][i]); svmopa_za64_m(7, svptrue, svptrue, coef_y, sve_y7);

//                     coef_y = svinsr(coef_y, d2_coeffs_y[4]);
//                     sve_y0 = svld1(pg, &s[0][-4][i]); svmopa_za64_m(0, svptrue, svptrue, coef_y, sve_y0);
//                     sve_y1 = svld1(pg, &s[1][-4][i]); svmopa_za64_m(1, svptrue, svptrue, coef_y, sve_y1);
//                     sve_y2 = svld1(pg, &s[2][-4][i]); svmopa_za64_m(2, svptrue, svptrue, coef_y, sve_y2);
//                     sve_y3 = svld1(pg, &s[3][-4][i]); svmopa_za64_m(3, svptrue, svptrue, coef_y, sve_y3);
//                     sve_y4 = svld1(pg, &s[4][-4][i]); svmopa_za64_m(4, svptrue, svptrue, coef_y, sve_y4);
//                     sve_y5 = svld1(pg, &s[5][-4][i]); svmopa_za64_m(5, svptrue, svptrue, coef_y, sve_y5);
//                     sve_y6 = svld1(pg, &s[6][-4][i]); svmopa_za64_m(6, svptrue, svptrue, coef_y, sve_y6);
//                     sve_y7 = svld1(pg, &s[7][-4][i]); svmopa_za64_m(7, svptrue, svptrue, coef_y, sve_y7);
                    
//                     coef_y = svinsr(coef_y, d2_coeffs_y[3]);
//                     sve_y0 = svld1(pg, &s[0][-3][i]); svmopa_za64_m(0, svptrue, svptrue, coef_y, sve_y0);
//                     sve_y1 = svld1(pg, &s[1][-3][i]); svmopa_za64_m(1, svptrue, svptrue, coef_y, sve_y1);
//                     sve_y2 = svld1(pg, &s[2][-3][i]); svmopa_za64_m(2, svptrue, svptrue, coef_y, sve_y2);
//                     sve_y3 = svld1(pg, &s[3][-3][i]); svmopa_za64_m(3, svptrue, svptrue, coef_y, sve_y3);
//                     sve_y4 = svld1(pg, &s[4][-3][i]); svmopa_za64_m(4, svptrue, svptrue, coef_y, sve_y4);
//                     sve_y5 = svld1(pg, &s[5][-3][i]); svmopa_za64_m(5, svptrue, svptrue, coef_y, sve_y5);
//                     sve_y6 = svld1(pg, &s[6][-3][i]); svmopa_za64_m(6, svptrue, svptrue, coef_y, sve_y6);
//                     sve_y7 = svld1(pg, &s[7][-3][i]); svmopa_za64_m(7, svptrue, svptrue, coef_y, sve_y7);

//                     coef_y = svinsr(coef_y, d2_coeffs_y[2]);
//                     sve_y0 = svld1(pg, &s[0][-2][i]); svmopa_za64_m(0, svptrue, svptrue, coef_y, sve_y0);
//                     sve_y1 = svld1(pg, &s[1][-2][i]); svmopa_za64_m(1, svptrue, svptrue, coef_y, sve_y1);
//                     sve_y2 = svld1(pg, &s[2][-2][i]); svmopa_za64_m(2, svptrue, svptrue, coef_y, sve_y2);
//                     sve_y3 = svld1(pg, &s[3][-2][i]); svmopa_za64_m(3, svptrue, svptrue, coef_y, sve_y3);
//                     sve_y4 = svld1(pg, &s[4][-2][i]); svmopa_za64_m(4, svptrue, svptrue, coef_y, sve_y4);
//                     sve_y5 = svld1(pg, &s[5][-2][i]); svmopa_za64_m(5, svptrue, svptrue, coef_y, sve_y5);
//                     sve_y6 = svld1(pg, &s[6][-2][i]); svmopa_za64_m(6, svptrue, svptrue, coef_y, sve_y6);
//                     sve_y7 = svld1(pg, &s[7][-2][i]); svmopa_za64_m(7, svptrue, svptrue, coef_y, sve_y7);

//                     coef_y = svinsr(coef_y, d2_coeffs_y[1]);
//                     sve_y0 = svld1(pg, &s[0][-1][i]); svmopa_za64_m(0, svptrue, svptrue, coef_y, sve_y0);
//                     sve_y1 = svld1(pg, &s[1][-1][i]); svmopa_za64_m(1, svptrue, svptrue, coef_y, sve_y1);
//                     sve_y2 = svld1(pg, &s[2][-1][i]); svmopa_za64_m(2, svptrue, svptrue, coef_y, sve_y2);
//                     sve_y3 = svld1(pg, &s[3][-1][i]); svmopa_za64_m(3, svptrue, svptrue, coef_y, sve_y3);
//                     sve_y4 = svld1(pg, &s[4][-1][i]); svmopa_za64_m(4, svptrue, svptrue, coef_y, sve_y4);
//                     sve_y5 = svld1(pg, &s[5][-1][i]); svmopa_za64_m(5, svptrue, svptrue, coef_y, sve_y5);
//                     sve_y6 = svld1(pg, &s[6][-1][i]); svmopa_za64_m(6, svptrue, svptrue, coef_y, sve_y6);
//                     sve_y7 = svld1(pg, &s[7][-1][i]); svmopa_za64_m(7, svptrue, svptrue, coef_y, sve_y7);

//                     coef_y = svinsr(coef_y, 0.0);
//                     sve_y0 = svld1(pg, &s[0][0][i]); svmopa_za64_m(0, svptrue, svptrue, coef_y, sve_y0);
//                     sve_y1 = svld1(pg, &s[1][0][i]); svmopa_za64_m(1, svptrue, svptrue, coef_y, sve_y1);
//                     sve_y2 = svld1(pg, &s[2][0][i]); svmopa_za64_m(2, svptrue, svptrue, coef_y, sve_y2);
//                     sve_y3 = svld1(pg, &s[3][0][i]); svmopa_za64_m(3, svptrue, svptrue, coef_y, sve_y3);
//                     sve_y4 = svld1(pg, &s[4][0][i]); svmopa_za64_m(4, svptrue, svptrue, coef_y, sve_y4);
//                     sve_y5 = svld1(pg, &s[5][0][i]); svmopa_za64_m(5, svptrue, svptrue, coef_y, sve_y5);
//                     sve_y6 = svld1(pg, &s[6][0][i]); svmopa_za64_m(6, svptrue, svptrue, coef_y, sve_y6);
//                     sve_y7 = svld1(pg, &s[7][0][i]); svmopa_za64_m(7, svptrue, svptrue, coef_y, sve_y7);

//                     coef_y = svinsr(coef_y, d2_coeffs_y[1]);
//                     sve_y0 = svld1(pg, &s[0][1][i]); svmopa_za64_m(0, svptrue, svptrue, coef_y, sve_y0);
//                     sve_y1 = svld1(pg, &s[1][1][i]); svmopa_za64_m(1, svptrue, svptrue, coef_y, sve_y1);
//                     sve_y2 = svld1(pg, &s[2][1][i]); svmopa_za64_m(2, svptrue, svptrue, coef_y, sve_y2);
//                     sve_y3 = svld1(pg, &s[3][1][i]); svmopa_za64_m(3, svptrue, svptrue, coef_y, sve_y3);
//                     sve_y4 = svld1(pg, &s[4][1][i]); svmopa_za64_m(4, svptrue, svptrue, coef_y, sve_y4);
//                     sve_y5 = svld1(pg, &s[5][1][i]); svmopa_za64_m(5, svptrue, svptrue, coef_y, sve_y5);
//                     sve_y6 = svld1(pg, &s[6][1][i]); svmopa_za64_m(6, svptrue, svptrue, coef_y, sve_y6);
//                     sve_y7 = svld1(pg, &s[7][1][i]); svmopa_za64_m(7, svptrue, svptrue, coef_y, sve_y7);

//                     coef_y = svinsr(coef_y, d2_coeffs_y[2]);
//                     sve_y0 = svld1(pg, &s[0][2][i]); svmopa_za64_m(0, svptrue, svptrue, coef_y, sve_y0);
//                     sve_y1 = svld1(pg, &s[1][2][i]); svmopa_za64_m(1, svptrue, svptrue, coef_y, sve_y1);
//                     sve_y2 = svld1(pg, &s[2][2][i]); svmopa_za64_m(2, svptrue, svptrue, coef_y, sve_y2);
//                     sve_y3 = svld1(pg, &s[3][2][i]); svmopa_za64_m(3, svptrue, svptrue, coef_y, sve_y3);
//                     sve_y4 = svld1(pg, &s[4][2][i]); svmopa_za64_m(4, svptrue, svptrue, coef_y, sve_y4);
//                     sve_y5 = svld1(pg, &s[5][2][i]); svmopa_za64_m(5, svptrue, svptrue, coef_y, sve_y5);
//                     sve_y6 = svld1(pg, &s[6][2][i]); svmopa_za64_m(6, svptrue, svptrue, coef_y, sve_y6);
//                     sve_y7 = svld1(pg, &s[7][2][i]); svmopa_za64_m(7, svptrue, svptrue, coef_y, sve_y7);

//                     coef_y = svinsr(coef_y, d2_coeffs_y[3]);
//                     sve_y0 = svld1(pg, &s[0][3][i]); svmopa_za64_m(0, svptrue, svptrue, coef_y, sve_y0);
//                     sve_y1 = svld1(pg, &s[1][3][i]); svmopa_za64_m(1, svptrue, svptrue, coef_y, sve_y1);
//                     sve_y2 = svld1(pg, &s[2][3][i]); svmopa_za64_m(2, svptrue, svptrue, coef_y, sve_y2);
//                     sve_y3 = svld1(pg, &s[3][3][i]); svmopa_za64_m(3, svptrue, svptrue, coef_y, sve_y3);
//                     sve_y4 = svld1(pg, &s[4][3][i]); svmopa_za64_m(4, svptrue, svptrue, coef_y, sve_y4);
//                     sve_y5 = svld1(pg, &s[5][3][i]); svmopa_za64_m(5, svptrue, svptrue, coef_y, sve_y5);
//                     sve_y6 = svld1(pg, &s[6][3][i]); svmopa_za64_m(6, svptrue, svptrue, coef_y, sve_y6);
//                     sve_y7 = svld1(pg, &s[7][3][i]); svmopa_za64_m(7, svptrue, svptrue, coef_y, sve_y7);

//                     coef_y = svinsr(coef_y, d2_coeffs_y[4]);
//                     sve_y0 = svld1(pg, &s[0][4][i]); svmopa_za64_m(0, svptrue, svptrue, coef_y, sve_y0);
//                     sve_y1 = svld1(pg, &s[1][4][i]); svmopa_za64_m(1, svptrue, svptrue, coef_y, sve_y1);
//                     sve_y2 = svld1(pg, &s[2][4][i]); svmopa_za64_m(2, svptrue, svptrue, coef_y, sve_y2);
//                     sve_y3 = svld1(pg, &s[3][4][i]); svmopa_za64_m(3, svptrue, svptrue, coef_y, sve_y3);
//                     sve_y4 = svld1(pg, &s[4][4][i]); svmopa_za64_m(4, svptrue, svptrue, coef_y, sve_y4);
//                     sve_y5 = svld1(pg, &s[5][4][i]); svmopa_za64_m(5, svptrue, svptrue, coef_y, sve_y5);
//                     sve_y6 = svld1(pg, &s[6][4][i]); svmopa_za64_m(6, svptrue, svptrue, coef_y, sve_y6);
//                     sve_y7 = svld1(pg, &s[7][4][i]); svmopa_za64_m(7, svptrue, svptrue, coef_y, sve_y7);

//                     coef_y = svinsr(coef_y, d2_coeffs_y[5]);
//                     sve_y0 = svld1(pg, &s[0][5][i]); svmopa_za64_m(0, svptrue, svptrue, coef_y, sve_y0);
//                     sve_y1 = svld1(pg, &s[1][5][i]); svmopa_za64_m(1, svptrue, svptrue, coef_y, sve_y1);
//                     sve_y2 = svld1(pg, &s[2][5][i]); svmopa_za64_m(2, svptrue, svptrue, coef_y, sve_y2);
//                     sve_y3 = svld1(pg, &s[3][5][i]); svmopa_za64_m(3, svptrue, svptrue, coef_y, sve_y3);
//                     sve_y4 = svld1(pg, &s[4][5][i]); svmopa_za64_m(4, svptrue, svptrue, coef_y, sve_y4);
//                     sve_y5 = svld1(pg, &s[5][5][i]); svmopa_za64_m(5, svptrue, svptrue, coef_y, sve_y5);
//                     sve_y6 = svld1(pg, &s[6][5][i]); svmopa_za64_m(6, svptrue, svptrue, coef_y, sve_y6);
//                     sve_y7 = svld1(pg, &s[7][5][i]); svmopa_za64_m(7, svptrue, svptrue, coef_y, sve_y7);

//                     coef_y = svinsr(coef_y, d2_coeffs_y[6]);
//                     sve_y0 = svld1(pg, &s[0][6][i]); svmopa_za64_m(0, svptrue, svptrue, coef_y, sve_y0);
//                     sve_y1 = svld1(pg, &s[1][6][i]); svmopa_za64_m(1, svptrue, svptrue, coef_y, sve_y1);
//                     sve_y2 = svld1(pg, &s[2][6][i]); svmopa_za64_m(2, svptrue, svptrue, coef_y, sve_y2);
//                     sve_y3 = svld1(pg, &s[3][6][i]); svmopa_za64_m(3, svptrue, svptrue, coef_y, sve_y3);
//                     sve_y4 = svld1(pg, &s[4][6][i]); svmopa_za64_m(4, svptrue, svptrue, coef_y, sve_y4);
//                     sve_y5 = svld1(pg, &s[5][6][i]); svmopa_za64_m(5, svptrue, svptrue, coef_y, sve_y5);
//                     sve_y6 = svld1(pg, &s[6][6][i]); svmopa_za64_m(6, svptrue, svptrue, coef_y, sve_y6);
//                     sve_y7 = svld1(pg, &s[7][6][i]); svmopa_za64_m(7, svptrue, svptrue, coef_y, sve_y7);

//                     for (int64_t j = 1; j < std::min(nj - jj, svcnt); j++) {
//                         coef_y = svinsr(coef_y, 0.0);
//                         sve_y0 = svld1(pg, &s[0][6+j][i]); svmopa_za64_m(0, svptrue, svptrue, coef_y, sve_y0);
//                         sve_y1 = svld1(pg, &s[1][6+j][i]); svmopa_za64_m(1, svptrue, svptrue, coef_y, sve_y1);
//                         sve_y2 = svld1(pg, &s[2][6+j][i]); svmopa_za64_m(2, svptrue, svptrue, coef_y, sve_y2);
//                         sve_y3 = svld1(pg, &s[3][6+j][i]); svmopa_za64_m(3, svptrue, svptrue, coef_y, sve_y3);
//                         sve_y4 = svld1(pg, &s[4][6+j][i]); svmopa_za64_m(4, svptrue, svptrue, coef_y, sve_y4);
//                         sve_y5 = svld1(pg, &s[5][6+j][i]); svmopa_za64_m(5, svptrue, svptrue, coef_y, sve_y5);
//                         sve_y6 = svld1(pg, &s[6][6+j][i]); svmopa_za64_m(6, svptrue, svptrue, coef_y, sve_y6);
//                         sve_y7 = svld1(pg, &s[7][6+j][i]); svmopa_za64_m(7, svptrue, svptrue, coef_y, sve_y7);
//                     }
//                 }

//                 // trans ZA from y to z direction 
//                 {
//                     const svbool_t svptrue = traits::svptrue_b_();
//                     sv_float__t sve_k_y;
//                     sv_float__t sve_y_k;

//                     // 0 1
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 0, 1);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 1, 0);
//                     svwrite_hor_za64_f64_m(0, 1, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(1, 0, svptrue, sve_k_y);

//                     // 0 2
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 0, 2);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 2, 0);
//                     svwrite_hor_za64_f64_m(0, 2, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(2, 0, svptrue, sve_k_y);

//                     // 0 3
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 0, 3);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 3, 0);
//                     svwrite_hor_za64_f64_m(0, 3, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(3, 0, svptrue, sve_k_y);

//                     // 0 4
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 0, 4);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 4, 0);
//                     svwrite_hor_za64_f64_m(0, 4, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(4, 0, svptrue, sve_k_y);

//                     // 0 5
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 0, 5);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 5, 0);
//                     svwrite_hor_za64_f64_m(0, 5, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(5, 0, svptrue, sve_k_y);

//                     // 0 6
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 0, 6);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 6, 0);
//                     svwrite_hor_za64_f64_m(0, 6, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(6, 0, svptrue, sve_k_y);

//                     // 0 7
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 0, 7);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 7, 0);
//                     svwrite_hor_za64_f64_m(0, 7, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(7, 0, svptrue, sve_k_y);

//                     // 1 2
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 1, 2);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 2, 1);
//                     svwrite_hor_za64_f64_m(1, 2, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(2, 1, svptrue, sve_k_y);

//                     // 1 3
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 1, 3);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 3, 1);
//                     svwrite_hor_za64_f64_m(1, 3, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(3, 1, svptrue, sve_k_y);

//                     // 1 4
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 1, 4);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 4, 1);
//                     svwrite_hor_za64_f64_m(1, 4, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(4, 1, svptrue, sve_k_y);

//                     // 1 5
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 1, 5);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 5, 1);
//                     svwrite_hor_za64_f64_m(1, 5, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(5, 1, svptrue, sve_k_y);

//                     // 1 6
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 1, 6);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 6, 1);
//                     svwrite_hor_za64_f64_m(1, 6, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(6, 1, svptrue, sve_k_y);

//                     // 1 7
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 1, 7);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 7, 1);
//                     svwrite_hor_za64_f64_m(1, 7, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(7, 1, svptrue, sve_k_y);

//                     // 2 3
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 2, 3);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 3, 2);
//                     svwrite_hor_za64_f64_m(2, 3, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(3, 2, svptrue, sve_k_y);

//                     // 2 4
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 2, 4);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 4, 2);
//                     svwrite_hor_za64_f64_m(2, 4, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(4, 2, svptrue, sve_k_y);

//                     // 2 5
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 2, 5);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 5, 2);
//                     svwrite_hor_za64_f64_m(2, 5, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(5, 2, svptrue, sve_k_y);

//                     // 2 6
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 2, 6);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 6, 2);
//                     svwrite_hor_za64_f64_m(2, 6, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(6, 2, svptrue, sve_k_y);

//                     // 2 7
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 2, 7);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 7, 2);
//                     svwrite_hor_za64_f64_m(2, 7, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(7, 2, svptrue, sve_k_y);

//                     // 3 4
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 3, 4);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 4, 3);
//                     svwrite_hor_za64_f64_m(3, 4, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(4, 3, svptrue, sve_k_y);

//                     // 3 5
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 3, 5);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 5, 3);
//                     svwrite_hor_za64_f64_m(3, 5, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(5, 3, svptrue, sve_k_y);

//                     // 3 6
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 3, 6);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 6, 3);
//                     svwrite_hor_za64_f64_m(3, 6, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(6, 3, svptrue, sve_k_y);

//                     // 3 7
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 3, 7);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 7, 3);
//                     svwrite_hor_za64_f64_m(3, 7, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(7, 3, svptrue, sve_k_y);

//                     // 4 5
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 4, 5);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 5, 4);
//                     svwrite_hor_za64_f64_m(4, 5, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(5, 4, svptrue, sve_k_y);

//                     // 4 6
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 4, 6);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 6, 4);
//                     svwrite_hor_za64_f64_m(4, 6, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(6, 4, svptrue, sve_k_y);

//                     // 4 7
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 4, 7);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 7, 4);
//                     svwrite_hor_za64_f64_m(4, 7, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(7, 4, svptrue, sve_k_y);

//                     // 5 6
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 5, 6);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 6, 5);
//                     svwrite_hor_za64_f64_m(5, 6, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(6, 5, svptrue, sve_k_y);

//                     // 5 7
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 5, 7);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 7, 5);
//                     svwrite_hor_za64_f64_m(5, 7, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(7, 5, svptrue, sve_k_y);

//                     // 6 7
//                     sve_k_y = svread_hor_za64_f64_m(sve_k_y, svptrue, 6, 7);
//                     sve_y_k = svread_hor_za64_f64_m(sve_y_k, svptrue, 7, 6);
//                     svwrite_hor_za64_f64_m(6, 7, svptrue, sve_y_k);
//                     svwrite_hor_za64_f64_m(7, 6, svptrue, sve_k_y);
//                 }

//                 {
//                     svbool_t pg = traits::svwhilelt_b_(i+ii, ni);
//                     const svbool_t svptrue = traits::svptrue_b_();
//                     sv_float__t coef_z = traits::svdup_f_(0.0);
//                     sv_float__t sve_z0;
//                     sv_float__t sve_z1;
//                     sv_float__t sve_z2;
//                     sv_float__t sve_z3;
//                     sv_float__t sve_z4;
//                     sv_float__t sve_z5;
//                     sv_float__t sve_z6;
//                     sv_float__t sve_z7;

//                     coef_z = svinsr(coef_z, d2_coeffs_z[6]);
//                     sve_z0 = svld1(pg, &s[-6][0][i]); svmopa_za64_m(0, svptrue, svptrue, coef_z, sve_z0);
//                     sve_z1 = svld1(pg, &s[-6][1][i]); svmopa_za64_m(1, svptrue, svptrue, coef_z, sve_z1);
//                     sve_z2 = svld1(pg, &s[-6][2][i]); svmopa_za64_m(2, svptrue, svptrue, coef_z, sve_z2);
//                     sve_z3 = svld1(pg, &s[-6][3][i]); svmopa_za64_m(3, svptrue, svptrue, coef_z, sve_z3);
//                     sve_z4 = svld1(pg, &s[-6][4][i]); svmopa_za64_m(4, svptrue, svptrue, coef_z, sve_z4);
//                     sve_z5 = svld1(pg, &s[-6][5][i]); svmopa_za64_m(5, svptrue, svptrue, coef_z, sve_z5);
//                     sve_z6 = svld1(pg, &s[-6][6][i]); svmopa_za64_m(6, svptrue, svptrue, coef_z, sve_z6);
//                     sve_z7 = svld1(pg, &s[-6][7][i]); svmopa_za64_m(7, svptrue, svptrue, coef_z, sve_z7);

//                     coef_z = svinsr(coef_z, d2_coeffs_z[5]);
//                     sve_z0 = svld1(pg, &s[-5][0][i]); svmopa_za64_m(0, svptrue, svptrue, coef_z, sve_z0);
//                     sve_z1 = svld1(pg, &s[-5][1][i]); svmopa_za64_m(1, svptrue, svptrue, coef_z, sve_z1);
//                     sve_z2 = svld1(pg, &s[-5][2][i]); svmopa_za64_m(2, svptrue, svptrue, coef_z, sve_z2);
//                     sve_z3 = svld1(pg, &s[-5][3][i]); svmopa_za64_m(3, svptrue, svptrue, coef_z, sve_z3);
//                     sve_z4 = svld1(pg, &s[-5][4][i]); svmopa_za64_m(4, svptrue, svptrue, coef_z, sve_z4);
//                     sve_z5 = svld1(pg, &s[-5][5][i]); svmopa_za64_m(5, svptrue, svptrue, coef_z, sve_z5);
//                     sve_z6 = svld1(pg, &s[-5][6][i]); svmopa_za64_m(6, svptrue, svptrue, coef_z, sve_z6);
//                     sve_z7 = svld1(pg, &s[-5][7][i]); svmopa_za64_m(7, svptrue, svptrue, coef_z, sve_z7);

//                     coef_z = svinsr(coef_z, d2_coeffs_z[4]);
//                     sve_z0 = svld1(pg, &s[-4][0][i]); svmopa_za64_m(0, svptrue, svptrue, coef_z, sve_z0);
//                     sve_z1 = svld1(pg, &s[-4][1][i]); svmopa_za64_m(1, svptrue, svptrue, coef_z, sve_z1);
//                     sve_z2 = svld1(pg, &s[-4][2][i]); svmopa_za64_m(2, svptrue, svptrue, coef_z, sve_z2);
//                     sve_z3 = svld1(pg, &s[-4][3][i]); svmopa_za64_m(3, svptrue, svptrue, coef_z, sve_z3);
//                     sve_z4 = svld1(pg, &s[-4][4][i]); svmopa_za64_m(4, svptrue, svptrue, coef_z, sve_z4);
//                     sve_z5 = svld1(pg, &s[-4][5][i]); svmopa_za64_m(5, svptrue, svptrue, coef_z, sve_z5);
//                     sve_z6 = svld1(pg, &s[-4][6][i]); svmopa_za64_m(6, svptrue, svptrue, coef_z, sve_z6);
//                     sve_z7 = svld1(pg, &s[-4][7][i]); svmopa_za64_m(7, svptrue, svptrue, coef_z, sve_z7);
                    
//                     coef_z = svinsr(coef_z, d2_coeffs_z[3]);
//                     sve_z0 = svld1(pg, &s[-3][0][i]); svmopa_za64_m(0, svptrue, svptrue, coef_z, sve_z0);
//                     sve_z1 = svld1(pg, &s[-3][1][i]); svmopa_za64_m(1, svptrue, svptrue, coef_z, sve_z1);
//                     sve_z2 = svld1(pg, &s[-3][2][i]); svmopa_za64_m(2, svptrue, svptrue, coef_z, sve_z2);
//                     sve_z3 = svld1(pg, &s[-3][3][i]); svmopa_za64_m(3, svptrue, svptrue, coef_z, sve_z3);
//                     sve_z4 = svld1(pg, &s[-3][4][i]); svmopa_za64_m(4, svptrue, svptrue, coef_z, sve_z4);
//                     sve_z5 = svld1(pg, &s[-3][5][i]); svmopa_za64_m(5, svptrue, svptrue, coef_z, sve_z5);
//                     sve_z6 = svld1(pg, &s[-3][6][i]); svmopa_za64_m(6, svptrue, svptrue, coef_z, sve_z6);
//                     sve_z7 = svld1(pg, &s[-3][7][i]); svmopa_za64_m(7, svptrue, svptrue, coef_z, sve_z7);

//                     coef_z = svinsr(coef_z, d2_coeffs_z[2]);
//                     sve_z0 = svld1(pg, &s[-2][0][i]); svmopa_za64_m(0, svptrue, svptrue, coef_z, sve_z0);
//                     sve_z1 = svld1(pg, &s[-2][1][i]); svmopa_za64_m(1, svptrue, svptrue, coef_z, sve_z1);
//                     sve_z2 = svld1(pg, &s[-2][2][i]); svmopa_za64_m(2, svptrue, svptrue, coef_z, sve_z2);
//                     sve_z3 = svld1(pg, &s[-2][3][i]); svmopa_za64_m(3, svptrue, svptrue, coef_z, sve_z3);
//                     sve_z4 = svld1(pg, &s[-2][4][i]); svmopa_za64_m(4, svptrue, svptrue, coef_z, sve_z4);
//                     sve_z5 = svld1(pg, &s[-2][5][i]); svmopa_za64_m(5, svptrue, svptrue, coef_z, sve_z5);
//                     sve_z6 = svld1(pg, &s[-2][6][i]); svmopa_za64_m(6, svptrue, svptrue, coef_z, sve_z6);
//                     sve_z7 = svld1(pg, &s[-2][7][i]); svmopa_za64_m(7, svptrue, svptrue, coef_z, sve_z7);

//                     coef_z = svinsr(coef_z, d2_coeffs_z[1]);
//                     sve_z0 = svld1(pg, &s[-1][0][i]); svmopa_za64_m(0, svptrue, svptrue, coef_z, sve_z0);
//                     sve_z1 = svld1(pg, &s[-1][1][i]); svmopa_za64_m(1, svptrue, svptrue, coef_z, sve_z1);
//                     sve_z2 = svld1(pg, &s[-1][2][i]); svmopa_za64_m(2, svptrue, svptrue, coef_z, sve_z2);
//                     sve_z3 = svld1(pg, &s[-1][3][i]); svmopa_za64_m(3, svptrue, svptrue, coef_z, sve_z3);
//                     sve_z4 = svld1(pg, &s[-1][4][i]); svmopa_za64_m(4, svptrue, svptrue, coef_z, sve_z4);
//                     sve_z5 = svld1(pg, &s[-1][5][i]); svmopa_za64_m(5, svptrue, svptrue, coef_z, sve_z5);
//                     sve_z6 = svld1(pg, &s[-1][6][i]); svmopa_za64_m(6, svptrue, svptrue, coef_z, sve_z6);
//                     sve_z7 = svld1(pg, &s[-1][7][i]); svmopa_za64_m(7, svptrue, svptrue, coef_z, sve_z7);

//                     coef_z = svinsr(coef_z, 0.0);
//                     sve_z0 = svld1(pg, &s[0][0][i]); svmopa_za64_m(0, svptrue, svptrue, coef_z, sve_z0);
//                     sve_z1 = svld1(pg, &s[0][1][i]); svmopa_za64_m(1, svptrue, svptrue, coef_z, sve_z1);
//                     sve_z2 = svld1(pg, &s[0][2][i]); svmopa_za64_m(2, svptrue, svptrue, coef_z, sve_z2);
//                     sve_z3 = svld1(pg, &s[0][3][i]); svmopa_za64_m(3, svptrue, svptrue, coef_z, sve_z3);
//                     sve_z4 = svld1(pg, &s[0][4][i]); svmopa_za64_m(4, svptrue, svptrue, coef_z, sve_z4);
//                     sve_z5 = svld1(pg, &s[0][5][i]); svmopa_za64_m(5, svptrue, svptrue, coef_z, sve_z5);
//                     sve_z6 = svld1(pg, &s[0][6][i]); svmopa_za64_m(6, svptrue, svptrue, coef_z, sve_z6);
//                     sve_z7 = svld1(pg, &s[0][7][i]); svmopa_za64_m(7, svptrue, svptrue, coef_z, sve_z7);

//                     coef_z = svinsr(coef_z, d2_coeffs_z[1]);
//                     sve_z0 = svld1(pg, &s[1][0][i]); svmopa_za64_m(0, svptrue, svptrue, coef_z, sve_z0);
//                     sve_z1 = svld1(pg, &s[1][1][i]); svmopa_za64_m(1, svptrue, svptrue, coef_z, sve_z1);
//                     sve_z2 = svld1(pg, &s[1][2][i]); svmopa_za64_m(2, svptrue, svptrue, coef_z, sve_z2);
//                     sve_z3 = svld1(pg, &s[1][3][i]); svmopa_za64_m(3, svptrue, svptrue, coef_z, sve_z3);
//                     sve_z4 = svld1(pg, &s[1][4][i]); svmopa_za64_m(4, svptrue, svptrue, coef_z, sve_z4);
//                     sve_z5 = svld1(pg, &s[1][5][i]); svmopa_za64_m(5, svptrue, svptrue, coef_z, sve_z5);
//                     sve_z6 = svld1(pg, &s[1][6][i]); svmopa_za64_m(6, svptrue, svptrue, coef_z, sve_z6);
//                     sve_z7 = svld1(pg, &s[1][7][i]); svmopa_za64_m(7, svptrue, svptrue, coef_z, sve_z7);

//                     coef_z = svinsr(coef_z, d2_coeffs_z[2]);
//                     sve_z0 = svld1(pg, &s[2][0][i]); svmopa_za64_m(0, svptrue, svptrue, coef_z, sve_z0);
//                     sve_z1 = svld1(pg, &s[2][1][i]); svmopa_za64_m(1, svptrue, svptrue, coef_z, sve_z1);
//                     sve_z2 = svld1(pg, &s[2][2][i]); svmopa_za64_m(2, svptrue, svptrue, coef_z, sve_z2);
//                     sve_z3 = svld1(pg, &s[2][3][i]); svmopa_za64_m(3, svptrue, svptrue, coef_z, sve_z3);
//                     sve_z4 = svld1(pg, &s[2][4][i]); svmopa_za64_m(4, svptrue, svptrue, coef_z, sve_z4);
//                     sve_z5 = svld1(pg, &s[2][5][i]); svmopa_za64_m(5, svptrue, svptrue, coef_z, sve_z5);
//                     sve_z6 = svld1(pg, &s[2][6][i]); svmopa_za64_m(6, svptrue, svptrue, coef_z, sve_z6);
//                     sve_z7 = svld1(pg, &s[2][7][i]); svmopa_za64_m(7, svptrue, svptrue, coef_z, sve_z7);

//                     coef_z = svinsr(coef_z, d2_coeffs_z[3]);
//                     sve_z0 = svld1(pg, &s[3][0][i]); svmopa_za64_m(0, svptrue, svptrue, coef_z, sve_z0);
//                     sve_z1 = svld1(pg, &s[3][1][i]); svmopa_za64_m(1, svptrue, svptrue, coef_z, sve_z1);
//                     sve_z2 = svld1(pg, &s[3][2][i]); svmopa_za64_m(2, svptrue, svptrue, coef_z, sve_z2);
//                     sve_z3 = svld1(pg, &s[3][3][i]); svmopa_za64_m(3, svptrue, svptrue, coef_z, sve_z3);
//                     sve_z4 = svld1(pg, &s[3][4][i]); svmopa_za64_m(4, svptrue, svptrue, coef_z, sve_z4);
//                     sve_z5 = svld1(pg, &s[3][5][i]); svmopa_za64_m(5, svptrue, svptrue, coef_z, sve_z5);
//                     sve_z6 = svld1(pg, &s[3][6][i]); svmopa_za64_m(6, svptrue, svptrue, coef_z, sve_z6);
//                     sve_z7 = svld1(pg, &s[3][7][i]); svmopa_za64_m(7, svptrue, svptrue, coef_z, sve_z7);

//                     coef_z = svinsr(coef_z, d2_coeffs_z[4]);
//                     sve_z0 = svld1(pg, &s[4][0][i]); svmopa_za64_m(0, svptrue, svptrue, coef_z, sve_z0);
//                     sve_z1 = svld1(pg, &s[4][1][i]); svmopa_za64_m(1, svptrue, svptrue, coef_z, sve_z1);
//                     sve_z2 = svld1(pg, &s[4][2][i]); svmopa_za64_m(2, svptrue, svptrue, coef_z, sve_z2);
//                     sve_z3 = svld1(pg, &s[4][3][i]); svmopa_za64_m(3, svptrue, svptrue, coef_z, sve_z3);
//                     sve_z4 = svld1(pg, &s[4][4][i]); svmopa_za64_m(4, svptrue, svptrue, coef_z, sve_z4);
//                     sve_z5 = svld1(pg, &s[4][5][i]); svmopa_za64_m(5, svptrue, svptrue, coef_z, sve_z5);
//                     sve_z6 = svld1(pg, &s[4][6][i]); svmopa_za64_m(6, svptrue, svptrue, coef_z, sve_z6);
//                     sve_z7 = svld1(pg, &s[4][7][i]); svmopa_za64_m(7, svptrue, svptrue, coef_z, sve_z7);

//                     coef_z = svinsr(coef_z, d2_coeffs_z[5]);
//                     sve_z0 = svld1(pg, &s[5][0][i]); svmopa_za64_m(0, svptrue, svptrue, coef_z, sve_z0);
//                     sve_z1 = svld1(pg, &s[5][1][i]); svmopa_za64_m(1, svptrue, svptrue, coef_z, sve_z1);
//                     sve_z2 = svld1(pg, &s[5][2][i]); svmopa_za64_m(2, svptrue, svptrue, coef_z, sve_z2);
//                     sve_z3 = svld1(pg, &s[5][3][i]); svmopa_za64_m(3, svptrue, svptrue, coef_z, sve_z3);
//                     sve_z4 = svld1(pg, &s[5][4][i]); svmopa_za64_m(4, svptrue, svptrue, coef_z, sve_z4);
//                     sve_z5 = svld1(pg, &s[5][5][i]); svmopa_za64_m(5, svptrue, svptrue, coef_z, sve_z5);
//                     sve_z6 = svld1(pg, &s[5][6][i]); svmopa_za64_m(6, svptrue, svptrue, coef_z, sve_z6);
//                     sve_z7 = svld1(pg, &s[5][7][i]); svmopa_za64_m(7, svptrue, svptrue, coef_z, sve_z7);

//                     coef_z = svinsr(coef_z, d2_coeffs_z[6]);
//                     sve_z0 = svld1(pg, &s[6][0][i]); svmopa_za64_m(0, svptrue, svptrue, coef_z, sve_z0);
//                     sve_z1 = svld1(pg, &s[6][1][i]); svmopa_za64_m(1, svptrue, svptrue, coef_z, sve_z1);
//                     sve_z2 = svld1(pg, &s[6][2][i]); svmopa_za64_m(2, svptrue, svptrue, coef_z, sve_z2);
//                     sve_z3 = svld1(pg, &s[6][3][i]); svmopa_za64_m(3, svptrue, svptrue, coef_z, sve_z3);
//                     sve_z4 = svld1(pg, &s[6][4][i]); svmopa_za64_m(4, svptrue, svptrue, coef_z, sve_z4);
//                     sve_z5 = svld1(pg, &s[6][5][i]); svmopa_za64_m(5, svptrue, svptrue, coef_z, sve_z5);
//                     sve_z6 = svld1(pg, &s[6][6][i]); svmopa_za64_m(6, svptrue, svptrue, coef_z, sve_z6);
//                     sve_z7 = svld1(pg, &s[6][7][i]); svmopa_za64_m(7, svptrue, svptrue, coef_z, sve_z7);

//                     for (int64_t k = 1; k < std::min(nk - kk, svcnt); k++) {
//                         coef_z = svinsr(coef_z, 0.0);
//                         sve_z0 = svld1(pg, &s[6+k][0][i]); svmopa_za64_m(0, svptrue, svptrue, coef_z, sve_z0);
//                         sve_z1 = svld1(pg, &s[6+k][1][i]); svmopa_za64_m(1, svptrue, svptrue, coef_z, sve_z1);
//                         sve_z2 = svld1(pg, &s[6+k][2][i]); svmopa_za64_m(2, svptrue, svptrue, coef_z, sve_z2);
//                         sve_z3 = svld1(pg, &s[6+k][3][i]); svmopa_za64_m(3, svptrue, svptrue, coef_z, sve_z3);
//                         sve_z4 = svld1(pg, &s[6+k][4][i]); svmopa_za64_m(4, svptrue, svptrue, coef_z, sve_z4);
//                         sve_z5 = svld1(pg, &s[6+k][5][i]); svmopa_za64_m(5, svptrue, svptrue, coef_z, sve_z5);
//                         sve_z6 = svld1(pg, &s[6+k][6][i]); svmopa_za64_m(6, svptrue, svptrue, coef_z, sve_z6);
//                         sve_z7 = svld1(pg, &s[6+k][7][i]); svmopa_za64_m(7, svptrue, svptrue, coef_z, sve_z7);
//                     }

//                     for (int64_t k = 0; k < std::min(nk - kk, svcnt); k++) {
//                         svst1_hor_za64(0, k, pg, &d[k][0][i]);
//                         svst1_hor_za64(1, k, pg, &d[k][1][i]);
//                         svst1_hor_za64(2, k, pg, &d[k][2][i]);
//                         svst1_hor_za64(3, k, pg, &d[k][3][i]);
//                         svst1_hor_za64(4, k, pg, &d[k][4][i]);
//                         svst1_hor_za64(5, k, pg, &d[k][5][i]);
//                         svst1_hor_za64(6, k, pg, &d[k][6][i]);
//                         svst1_hor_za64(7, k, pg, &d[k][7][i]);
//                     }
//                 }
                
//                 for (int64_t k = kk; k < kt; k++) {
//                     for (int64_t j = jj; j < jt; j++) {
//                         #pragma omp simd
//                         for (int64_t i = ii; i < it; i ++) {
//                             d_ptr[i + j * d_stride_y + k * d_stride_z] = d[k-kk][j-jj][i-ii];
//                         }
//                     }
//                 }

//             }
//         }
//     }
//     return;
// }

template<typename T>
void Stencil3D<6, T>::lap_3d_c2_o0(T const* const __restrict__ s_ptr, const int64_t s_stride_y, const int64_t s_stride_z,
                                    T* const __restrict__ d_ptr, const int64_t d_stride_y, const int64_t d_stride_z,
                                    const int64_t ni, const int64_t nj, const int64_t nk,
                                    T const* const __restrict__ d2_coeffs_x,
                                    T const* const __restrict__ d2_coeffs_y,
                                    T const* const __restrict__ d2_coeffs_z,
                                    T const coef_0,
                                    T const* const __restrict__ A_ptr) {
    constexpr int R = radius;
    //laplacian_3d_c2_o0_kernel<T>(s_ptr, s_stride_y, s_stride_z, d_ptr, d_stride_y, d_stride_z,
    //                             ni, nj, nk, R, d2_coeffs_x, d2_coeffs_y, d2_coeffs_z, A_ptr);
    // return;
    // Credit: Xiaohui Duan
    using traits = sve_traits<T>;
    using sv_float__t = typename traits::sv_float__t;
    constexpr int64_t tile_size_k = 8;
    constexpr int64_t tile_size_j = 8;
    constexpr int64_t tile_size_i = traits::tile_size_i;
    // const T coef_0 = d2_coeffs_x[0] + d2_coeffs_y[0] + d2_coeffs_z[0];
    const int64_t svcnt = traits::svcnt_();
    T s_st[tile_size_k+R*2][tile_size_j+R*2][tile_size_i+R*2], (* const s)[tile_size_j+R*2][tile_size_i+R*2] = (T(*)[tile_size_j+R*2][tile_size_i+R*2])&s_st[R][R][R];
    T d[tile_size_k][tile_size_j][tile_size_i];
    #ifdef USE_OPENMP
    #pragma omp for schedule(static) collapse(2) nowait
    #endif //USE_OPENMP
    for (int64_t kk = 0; kk < nk; kk += tile_size_k) {
        for (int64_t jj = 0; jj < nj; jj += tile_size_j) {
            for (int64_t ii = 0; ii < ni; ii += tile_size_i) {
                // print64_tf("load %d %d\n", kk, jj);
                const int64_t kt = std::min(kk + tile_size_k, nk);
                const int64_t jt = std::min(jj + tile_size_j, nj);
                const int64_t it = std::min(ii + tile_size_i, ni);
                for (int64_t k = kk; k < kt; k++) {
                    for (int64_t j = jj; j < jt; j++) {
                        #pragma omp simd
                        for (int64_t i = ii-R; i < it+R; i ++) {
                            s[k-kk][j-jj][i-ii] = s_ptr[i + j * s_stride_y + k * s_stride_z];
                        }
                    }
                }
                // print64_tf("load km\n", kk, jj);
                for (int64_t k = kk-R; k < kk; k++) {
                    for (int64_t j = jj; j < jt; j++) {
                        #pragma omp simd
                        for (int64_t i = ii; i < it; i ++) {
                            s[k-kk][j-jj][i-ii] = s_ptr[i + j * s_stride_y + k * s_stride_z];
                        }
                    }
                }
                // print64_tf("load kp\n", kk, jj);
                for (int64_t k = kt; k < kt+R; k++) {
                    for (int64_t j = jj; j < jt; j++) {
                        #pragma omp simd
                        for (int64_t i = ii; i < it; i ++) {
                            s[k-kk][j-jj][i-ii] = s_ptr[i + j * s_stride_y + k * s_stride_z];
                        }
                    }
                }
                // print64_tf("load jm\n", kk, jj);
                for (int64_t k = kk; k < kt; k++) {
                    for (int64_t j = jj-R; j < jj; j++) {
                        #pragma omp simd
                        for (int64_t i = ii; i < it; i ++) {
                            s[k-kk][j-jj][i-ii] = s_ptr[i + j * s_stride_y + k * s_stride_z];
                        }
                    }
                }
                for (int64_t k = kk; k < kt; k++) {
                    for (int64_t j = jt; j < jt+R; j++) {
                        #pragma omp simd
                        for (int64_t i = ii; i < it; i ++) {
                            s[k-kk][j-jj][i-ii] = s_ptr[i + j * s_stride_y + k * s_stride_z];
                        }
                    }
                }
                for (int64_t k = kk; k < kt; k++) {
                    for (int64_t j = jj; j < jt; j++) {
                        #pragma omp simd
                        for (int64_t i = ii; i < it; i ++) {
                            d[k-kk][j-jj][i-ii] = A_ptr[i + j * d_stride_y + k * d_stride_z];
                        }
                    }
                }
                // print64_tf("calc %d %d\n", kk, jj);
                for (int64_t k = 0; k < tile_size_k; k++) {
                    // for (int64_t j = 0; j < tile_size_j; j++) {
                    for (int64_t i = 0; i < tile_size_i; i += svcnt) {
                        svbool_t pg = traits::svwhilelt_b_(i+ii, ni);
                        constexpr int64_t j = 0;
                        sv_float__t sve_A0 = svadd_m(pg, svld1(pg, &d[k][j+0][i]), coef_0);
                        sv_float__t sve_A1 = svadd_m(pg, svld1(pg, &d[k][j+1][i]), coef_0);
                        sv_float__t sve_A2 = svadd_m(pg, svld1(pg, &d[k][j+2][i]), coef_0);
                        sv_float__t sve_A3 = svadd_m(pg, svld1(pg, &d[k][j+3][i]), coef_0);
                        sv_float__t sve_A4 = svadd_m(pg, svld1(pg, &d[k][j+4][i]), coef_0);
                        sv_float__t sve_A5 = svadd_m(pg, svld1(pg, &d[k][j+5][i]), coef_0);
                        sv_float__t sve_A6 = svadd_m(pg, svld1(pg, &d[k][j+6][i]), coef_0);
                        sv_float__t sve_A7 = svadd_m(pg, svld1(pg, &d[k][j+7][i]), coef_0);
                        sv_float__t res0 = svmul_m(pg, svld1(pg, &s[k][j+0][i]), sve_A0);
                        sv_float__t res1 = svmul_m(pg, svld1(pg, &s[k][j+1][i]), sve_A1);
                        sv_float__t res2 = svmul_m(pg, svld1(pg, &s[k][j+2][i]), sve_A2);
                        sv_float__t res3 = svmul_m(pg, svld1(pg, &s[k][j+3][i]), sve_A3);
                        sv_float__t res4 = svmul_m(pg, svld1(pg, &s[k][j+4][i]), sve_A4);
                        sv_float__t res5 = svmul_m(pg, svld1(pg, &s[k][j+5][i]), sve_A5);
                        sv_float__t res6 = svmul_m(pg, svld1(pg, &s[k][j+6][i]), sve_A6);
                        sv_float__t res7 = svmul_m(pg, svld1(pg, &s[k][j+7][i]), sve_A7);

                        for (int r = 1; r <= R; r++) {
                            // sv_float__t sum_x0 = svadd_m(pg, svld1(pg, &s[k][j+0][i+r]), svld1(pg, &s[k][j+0][i-r]));
                            // sv_float__t sum_x1 = svadd_m(pg, svld1(pg, &s[k][j+1][i+r]), svld1(pg, &s[k][j+1][i-r]));
                            // sv_float__t sum_x2 = svadd_m(pg, svld1(pg, &s[k][j+2][i+r]), svld1(pg, &s[k][j+2][i-r]));
                            // sv_float__t sum_x3 = svadd_m(pg, svld1(pg, &s[k][j+3][i+r]), svld1(pg, &s[k][j+3][i-r]));
                            // sv_float__t sum_x4 = svadd_m(pg, svld1(pg, &s[k][j+4][i+r]), svld1(pg, &s[k][j+4][i-r]));
                            // sv_float__t sum_x5 = svadd_m(pg, svld1(pg, &s[k][j+5][i+r]), svld1(pg, &s[k][j+5][i-r]));
                            // sv_float__t sum_x6 = svadd_m(pg, svld1(pg, &s[k][j+6][i+r]), svld1(pg, &s[k][j+6][i-r]));
                            // sv_float__t sum_x7 = svadd_m(pg, svld1(pg, &s[k][j+7][i+r]), svld1(pg, &s[k][j+7][i-r]));
                            res0 = svmla_m(pg, res0, svld1(pg, &s[k][j+0][i+r]), d2_coeffs_x[r]);
                            res1 = svmla_m(pg, res1, svld1(pg, &s[k][j+1][i+r]), d2_coeffs_x[r]);
                            res2 = svmla_m(pg, res2, svld1(pg, &s[k][j+2][i+r]), d2_coeffs_x[r]);
                            res3 = svmla_m(pg, res3, svld1(pg, &s[k][j+3][i+r]), d2_coeffs_x[r]);
                            res4 = svmla_m(pg, res4, svld1(pg, &s[k][j+4][i+r]), d2_coeffs_x[r]);
                            res5 = svmla_m(pg, res5, svld1(pg, &s[k][j+5][i+r]), d2_coeffs_x[r]);
                            res6 = svmla_m(pg, res6, svld1(pg, &s[k][j+6][i+r]), d2_coeffs_x[r]);
                            res7 = svmla_m(pg, res7, svld1(pg, &s[k][j+7][i+r]), d2_coeffs_x[r]);
                            res0 = svmla_m(pg, res0, svld1(pg, &s[k][j+0][i-r]), d2_coeffs_x[r]);
                            res1 = svmla_m(pg, res1, svld1(pg, &s[k][j+1][i-r]), d2_coeffs_x[r]);
                            res2 = svmla_m(pg, res2, svld1(pg, &s[k][j+2][i-r]), d2_coeffs_x[r]);
                            res3 = svmla_m(pg, res3, svld1(pg, &s[k][j+3][i-r]), d2_coeffs_x[r]);
                            res4 = svmla_m(pg, res4, svld1(pg, &s[k][j+4][i-r]), d2_coeffs_x[r]);
                            res5 = svmla_m(pg, res5, svld1(pg, &s[k][j+5][i-r]), d2_coeffs_x[r]);
                            res6 = svmla_m(pg, res6, svld1(pg, &s[k][j+6][i-r]), d2_coeffs_x[r]);
                            res7 = svmla_m(pg, res7, svld1(pg, &s[k][j+7][i-r]), d2_coeffs_x[r]);
                        }

                        sv_float__t sjn6 = svld1(pg, &s[k][j-6][i]);
                        res0 = svmla_m(pg, res0, sjn6, d2_coeffs_y[6]);
                        sv_float__t sjn5 = svld1(pg, &s[k][j-5][i]);
                        res0 = svmla_m(pg, res0, sjn5, d2_coeffs_y[5]);
                        res1 = svmla_m(pg, res1, sjn5, d2_coeffs_y[6]);
                        sv_float__t sjn4 = svld1(pg, &s[k][j-4][i]);
                        res0 = svmla_m(pg, res0, sjn4, d2_coeffs_y[4]);
                        res1 = svmla_m(pg, res1, sjn4, d2_coeffs_y[5]);
                        res2 = svmla_m(pg, res2, sjn4, d2_coeffs_y[6]);
                        sv_float__t sjn3 = svld1(pg, &s[k][j-3][i]);
                        res0 = svmla_m(pg, res0, sjn3, d2_coeffs_y[3]);
                        res1 = svmla_m(pg, res1, sjn3, d2_coeffs_y[4]);
                        res2 = svmla_m(pg, res2, sjn3, d2_coeffs_y[5]);
                        res3 = svmla_m(pg, res3, sjn3, d2_coeffs_y[6]);
                        sv_float__t sjn2 = svld1(pg, &s[k][j-2][i]);
                        res0 = svmla_m(pg, res0, sjn2, d2_coeffs_y[2]);
                        res1 = svmla_m(pg, res1, sjn2, d2_coeffs_y[3]);
                        res2 = svmla_m(pg, res2, sjn2, d2_coeffs_y[4]);
                        res3 = svmla_m(pg, res3, sjn2, d2_coeffs_y[5]);
                        res4 = svmla_m(pg, res4, sjn2, d2_coeffs_y[6]);
                        sv_float__t sjn1 = svld1(pg, &s[k][j-1][i]);
                        res0 = svmla_m(pg, res0, sjn1, d2_coeffs_y[1]);
                        res1 = svmla_m(pg, res1, sjn1, d2_coeffs_y[2]);
                        res2 = svmla_m(pg, res2, sjn1, d2_coeffs_y[3]);
                        res3 = svmla_m(pg, res3, sjn1, d2_coeffs_y[4]);
                        res4 = svmla_m(pg, res4, sjn1, d2_coeffs_y[5]);
                        res5 = svmla_m(pg, res5, sjn1, d2_coeffs_y[6]);
                        sv_float__t sjp0 = svld1(pg, &s[k][j+0][i]);
                        res1 = svmla_m(pg, res1, sjp0, d2_coeffs_y[1]);
                        res2 = svmla_m(pg, res2, sjp0, d2_coeffs_y[2]);
                        res3 = svmla_m(pg, res3, sjp0, d2_coeffs_y[3]);
                        res4 = svmla_m(pg, res4, sjp0, d2_coeffs_y[4]);
                        res5 = svmla_m(pg, res5, sjp0, d2_coeffs_y[5]);
                        res6 = svmla_m(pg, res6, sjp0, d2_coeffs_y[6]);
                        sv_float__t sjp1 = svld1(pg, &s[k][j+1][i]);
                        res0 = svmla_m(pg, res0, sjp1, d2_coeffs_y[1]);
                        res2 = svmla_m(pg, res2, sjp1, d2_coeffs_y[1]);
                        res3 = svmla_m(pg, res3, sjp1, d2_coeffs_y[2]);
                        res4 = svmla_m(pg, res4, sjp1, d2_coeffs_y[3]);
                        res5 = svmla_m(pg, res5, sjp1, d2_coeffs_y[4]);
                        res6 = svmla_m(pg, res6, sjp1, d2_coeffs_y[5]);
                        res7 = svmla_m(pg, res7, sjp1, d2_coeffs_y[6]);
                        sv_float__t sjp2 = svld1(pg, &s[k][j+2][i]);
                        res0 = svmla_m(pg, res0, sjp2, d2_coeffs_y[2]);
                        res1 = svmla_m(pg, res1, sjp2, d2_coeffs_y[1]);
                        res3 = svmla_m(pg, res3, sjp2, d2_coeffs_y[1]);
                        res4 = svmla_m(pg, res4, sjp2, d2_coeffs_y[2]);
                        res5 = svmla_m(pg, res5, sjp2, d2_coeffs_y[3]);
                        res6 = svmla_m(pg, res6, sjp2, d2_coeffs_y[4]);
                        res7 = svmla_m(pg, res7, sjp2, d2_coeffs_y[5]);
                        sv_float__t sjp3 = svld1(pg, &s[k][j+3][i]);
                        res0 = svmla_m(pg, res0, sjp3, d2_coeffs_y[3]);
                        res1 = svmla_m(pg, res1, sjp3, d2_coeffs_y[2]);
                        res2 = svmla_m(pg, res2, sjp3, d2_coeffs_y[1]);
                        res4 = svmla_m(pg, res4, sjp3, d2_coeffs_y[1]);
                        res5 = svmla_m(pg, res5, sjp3, d2_coeffs_y[2]);
                        res6 = svmla_m(pg, res6, sjp3, d2_coeffs_y[3]);
                        res7 = svmla_m(pg, res7, sjp3, d2_coeffs_y[4]);
                        sv_float__t sjp4 = svld1(pg, &s[k][j+4][i]);
                        res0 = svmla_m(pg, res0, sjp4, d2_coeffs_y[4]);
                        res1 = svmla_m(pg, res1, sjp4, d2_coeffs_y[3]);
                        res2 = svmla_m(pg, res2, sjp4, d2_coeffs_y[2]);
                        res3 = svmla_m(pg, res3, sjp4, d2_coeffs_y[1]);
                        res5 = svmla_m(pg, res5, sjp4, d2_coeffs_y[1]);
                        res6 = svmla_m(pg, res6, sjp4, d2_coeffs_y[2]);
                        res7 = svmla_m(pg, res7, sjp4, d2_coeffs_y[3]);
                        sv_float__t sjp5 = svld1(pg, &s[k][j+5][i]);
                        res0 = svmla_m(pg, res0, sjp5, d2_coeffs_y[5]);
                        res1 = svmla_m(pg, res1, sjp5, d2_coeffs_y[4]);
                        res2 = svmla_m(pg, res2, sjp5, d2_coeffs_y[3]);
                        res3 = svmla_m(pg, res3, sjp5, d2_coeffs_y[2]);
                        res4 = svmla_m(pg, res4, sjp5, d2_coeffs_y[1]);
                        res6 = svmla_m(pg, res6, sjp5, d2_coeffs_y[1]);
                        res7 = svmla_m(pg, res7, sjp5, d2_coeffs_y[2]);
                        sv_float__t sjp6 = svld1(pg, &s[k][j+6][i]);
                        res0 = svmla_m(pg, res0, sjp6, d2_coeffs_y[6]);
                        res1 = svmla_m(pg, res1, sjp6, d2_coeffs_y[5]);
                        res2 = svmla_m(pg, res2, sjp6, d2_coeffs_y[4]);
                        res3 = svmla_m(pg, res3, sjp6, d2_coeffs_y[3]);
                        res4 = svmla_m(pg, res4, sjp6, d2_coeffs_y[2]);
                        res5 = svmla_m(pg, res5, sjp6, d2_coeffs_y[1]);
                        res7 = svmla_m(pg, res7, sjp6, d2_coeffs_y[1]);
                        sv_float__t sjp7 = svld1(pg, &s[k][j+7][i]);
                        res1 = svmla_m(pg, res1, sjp7, d2_coeffs_y[6]);
                        res2 = svmla_m(pg, res2, sjp7, d2_coeffs_y[5]);
                        res3 = svmla_m(pg, res3, sjp7, d2_coeffs_y[4]);
                        res4 = svmla_m(pg, res4, sjp7, d2_coeffs_y[3]);
                        res5 = svmla_m(pg, res5, sjp7, d2_coeffs_y[2]);
                        res6 = svmla_m(pg, res6, sjp7, d2_coeffs_y[1]);

                        sv_float__t sjp8 = svld1(pg, &s[k][j+8][i]);
                        res2 = svmla_m(pg, res2, sjp8, d2_coeffs_y[6]);
                        res3 = svmla_m(pg, res3, sjp8, d2_coeffs_y[5]);
                        res4 = svmla_m(pg, res4, sjp8, d2_coeffs_y[4]);
                        res5 = svmla_m(pg, res5, sjp8, d2_coeffs_y[3]);
                        res6 = svmla_m(pg, res6, sjp8, d2_coeffs_y[2]);
                        res7 = svmla_m(pg, res7, sjp8, d2_coeffs_y[1]);

                        sv_float__t sjp9 = svld1(pg, &s[k][j+9][i]);
                        res3 = svmla_m(pg, res3, sjp9, d2_coeffs_y[6]);
                        res4 = svmla_m(pg, res4, sjp9, d2_coeffs_y[5]);
                        res5 = svmla_m(pg, res5, sjp9, d2_coeffs_y[4]);
                        res6 = svmla_m(pg, res6, sjp9, d2_coeffs_y[3]);
                        res7 = svmla_m(pg, res7, sjp9, d2_coeffs_y[2]);

                        sv_float__t sjpA = svld1(pg, &s[k][j+0xA][i]);
                        res4 = svmla_m(pg, res4, sjpA, d2_coeffs_y[6]);
                        res5 = svmla_m(pg, res5, sjpA, d2_coeffs_y[5]);
                        res6 = svmla_m(pg, res6, sjpA, d2_coeffs_y[4]);
                        res7 = svmla_m(pg, res7, sjpA, d2_coeffs_y[3]);

                        sv_float__t sjpB = svld1(pg, &s[k][j+0xB][i]);
                        res5 = svmla_m(pg, res5, sjpB, d2_coeffs_y[6]);
                        res6 = svmla_m(pg, res6, sjpB, d2_coeffs_y[5]);
                        res7 = svmla_m(pg, res7, sjpB, d2_coeffs_y[4]);

                        sv_float__t sjpC = svld1(pg, &s[k][j+0xC][i]);
                        res6 = svmla_m(pg, res6, sjpC, d2_coeffs_y[6]);
                        res7 = svmla_m(pg, res7, sjpC, d2_coeffs_y[5]);

                        sv_float__t sjpD = svld1(pg, &s[k][j+0xD][i]);
                        res7 = svmla_m(pg, res7, sjpD, d2_coeffs_y[6]);

                        svst1(pg, &d[k][j+0][i], res0);
                        svst1(pg, &d[k][j+1][i], res1);
                        svst1(pg, &d[k][j+2][i], res2);
                        svst1(pg, &d[k][j+3][i], res3);
                        svst1(pg, &d[k][j+4][i], res4);
                        svst1(pg, &d[k][j+5][i], res5);
                        svst1(pg, &d[k][j+6][i], res6);
                        svst1(pg, &d[k][j+7][i], res7);
                    }
                }

                for (int64_t j = 0; j < tile_size_j; j++) {
                    // for (int64_t j = 0; j < tile_size_j; j++) {
                    for (int64_t i = 0; i < tile_size_i; i += svcnt) {
                        svbool_t pg = traits::svwhilelt_b_(i+ii, ni);
                        constexpr int64_t k = 0;
                        sv_float__t res0 = svld1(pg, &d[k+0][j][i]);  //svmul_m(pg, svld1(pg, &s[k][j+0][i]), sve_A0);
                        sv_float__t res1 = svld1(pg, &d[k+1][j][i]);  //svmul_m(pg, svld1(pg, &s[k][j+1][i]), sve_A1);
                        sv_float__t res2 = svld1(pg, &d[k+2][j][i]);  //svmul_m(pg, svld1(pg, &s[k][j+2][i]), sve_A2);
                        sv_float__t res3 = svld1(pg, &d[k+3][j][i]);  //svmul_m(pg, svld1(pg, &s[k][j+3][i]), sve_A3);
                        sv_float__t res4 = svld1(pg, &d[k+4][j][i]);  //svmul_m(pg, svld1(pg, &s[k][j+4][i]), sve_A4);
                        sv_float__t res5 = svld1(pg, &d[k+5][j][i]);  //svmul_m(pg, svld1(pg, &s[k][j+5][i]), sve_A5);
                        sv_float__t res6 = svld1(pg, &d[k+6][j][i]);  //svmul_m(pg, svld1(pg, &s[k][j+6][i]), sve_A6);
                        sv_float__t res7 = svld1(pg, &d[k+7][j][i]);  //svmul_m(pg, svld1(pg, &s[k][j+7][i]), sve_A7);

                        sv_float__t skn6 = svld1(pg, &s[k-6][j][i]);
                        res0 = svmla_m(pg, res0, skn6, d2_coeffs_z[6]);
                        sv_float__t skn5 = svld1(pg, &s[k-5][j][i]);
                        res0 = svmla_m(pg, res0, skn5, d2_coeffs_z[5]);
                        res1 = svmla_m(pg, res1, skn5, d2_coeffs_z[6]);
                        sv_float__t skn4 = svld1(pg, &s[k-4][j][i]);
                        res0 = svmla_m(pg, res0, skn4, d2_coeffs_z[4]);
                        res1 = svmla_m(pg, res1, skn4, d2_coeffs_z[5]);
                        res2 = svmla_m(pg, res2, skn4, d2_coeffs_z[6]);
                        sv_float__t skn3 = svld1(pg, &s[k-3][j][i]);
                        res0 = svmla_m(pg, res0, skn3, d2_coeffs_z[3]);
                        res1 = svmla_m(pg, res1, skn3, d2_coeffs_z[4]);
                        res2 = svmla_m(pg, res2, skn3, d2_coeffs_z[5]);
                        res3 = svmla_m(pg, res3, skn3, d2_coeffs_z[6]);
                        sv_float__t skn2 = svld1(pg, &s[k-2][j][i]);
                        res0 = svmla_m(pg, res0, skn2, d2_coeffs_z[2]);
                        res1 = svmla_m(pg, res1, skn2, d2_coeffs_z[3]);
                        res2 = svmla_m(pg, res2, skn2, d2_coeffs_z[4]);
                        res3 = svmla_m(pg, res3, skn2, d2_coeffs_z[5]);
                        res4 = svmla_m(pg, res4, skn2, d2_coeffs_z[6]);
                        sv_float__t skn1 = svld1(pg, &s[k-1][j][i]);
                        res0 = svmla_m(pg, res0, skn1, d2_coeffs_z[1]);
                        res1 = svmla_m(pg, res1, skn1, d2_coeffs_z[2]);
                        res2 = svmla_m(pg, res2, skn1, d2_coeffs_z[3]);
                        res3 = svmla_m(pg, res3, skn1, d2_coeffs_z[4]);
                        res4 = svmla_m(pg, res4, skn1, d2_coeffs_z[5]);
                        res5 = svmla_m(pg, res5, skn1, d2_coeffs_z[6]);
                        sv_float__t skp0 = svld1(pg, &s[k+0][j][i]);
                        res1 = svmla_m(pg, res1, skp0, d2_coeffs_z[1]);
                        res2 = svmla_m(pg, res2, skp0, d2_coeffs_z[2]);
                        res3 = svmla_m(pg, res3, skp0, d2_coeffs_z[3]);
                        res4 = svmla_m(pg, res4, skp0, d2_coeffs_z[4]);
                        res5 = svmla_m(pg, res5, skp0, d2_coeffs_z[5]);
                        res6 = svmla_m(pg, res6, skp0, d2_coeffs_z[6]);
                        sv_float__t skp1 = svld1(pg, &s[k+1][j][i]);
                        res0 = svmla_m(pg, res0, skp1, d2_coeffs_z[1]);
                        res2 = svmla_m(pg, res2, skp1, d2_coeffs_z[1]);
                        res3 = svmla_m(pg, res3, skp1, d2_coeffs_z[2]);
                        res4 = svmla_m(pg, res4, skp1, d2_coeffs_z[3]);
                        res5 = svmla_m(pg, res5, skp1, d2_coeffs_z[4]);
                        res6 = svmla_m(pg, res6, skp1, d2_coeffs_z[5]);
                        res7 = svmla_m(pg, res7, skp1, d2_coeffs_z[6]);
                        sv_float__t skp2 = svld1(pg, &s[k+2][j][i]);
                        res0 = svmla_m(pg, res0, skp2, d2_coeffs_z[2]);
                        res1 = svmla_m(pg, res1, skp2, d2_coeffs_z[1]);
                        res3 = svmla_m(pg, res3, skp2, d2_coeffs_z[1]);
                        res4 = svmla_m(pg, res4, skp2, d2_coeffs_z[2]);
                        res5 = svmla_m(pg, res5, skp2, d2_coeffs_z[3]);
                        res6 = svmla_m(pg, res6, skp2, d2_coeffs_z[4]);
                        res7 = svmla_m(pg, res7, skp2, d2_coeffs_z[5]);
                        sv_float__t skp3 = svld1(pg, &s[k+3][j][i]);
                        res0 = svmla_m(pg, res0, skp3, d2_coeffs_z[3]);
                        res1 = svmla_m(pg, res1, skp3, d2_coeffs_z[2]);
                        res2 = svmla_m(pg, res2, skp3, d2_coeffs_z[1]);
                        res4 = svmla_m(pg, res4, skp3, d2_coeffs_z[1]);
                        res5 = svmla_m(pg, res5, skp3, d2_coeffs_z[2]);
                        res6 = svmla_m(pg, res6, skp3, d2_coeffs_z[3]);
                        res7 = svmla_m(pg, res7, skp3, d2_coeffs_z[4]);
                        sv_float__t skp4 = svld1(pg, &s[k+4][j][i]);
                        res0 = svmla_m(pg, res0, skp4, d2_coeffs_z[4]);
                        res1 = svmla_m(pg, res1, skp4, d2_coeffs_z[3]);
                        res2 = svmla_m(pg, res2, skp4, d2_coeffs_z[2]);
                        res3 = svmla_m(pg, res3, skp4, d2_coeffs_z[1]);
                        res5 = svmla_m(pg, res5, skp4, d2_coeffs_z[1]);
                        res6 = svmla_m(pg, res6, skp4, d2_coeffs_z[2]);
                        res7 = svmla_m(pg, res7, skp4, d2_coeffs_z[3]);
                        sv_float__t skp5 = svld1(pg, &s[k+5][j][i]);
                        res0 = svmla_m(pg, res0, skp5, d2_coeffs_z[5]);
                        res1 = svmla_m(pg, res1, skp5, d2_coeffs_z[4]);
                        res2 = svmla_m(pg, res2, skp5, d2_coeffs_z[3]);
                        res3 = svmla_m(pg, res3, skp5, d2_coeffs_z[2]);
                        res4 = svmla_m(pg, res4, skp5, d2_coeffs_z[1]);
                        res6 = svmla_m(pg, res6, skp5, d2_coeffs_z[1]);
                        res7 = svmla_m(pg, res7, skp5, d2_coeffs_z[2]);
                        sv_float__t skp6 = svld1(pg, &s[k+6][j][i]);
                        res0 = svmla_m(pg, res0, skp6, d2_coeffs_z[6]);
                        res1 = svmla_m(pg, res1, skp6, d2_coeffs_z[5]);
                        res2 = svmla_m(pg, res2, skp6, d2_coeffs_z[4]);
                        res3 = svmla_m(pg, res3, skp6, d2_coeffs_z[3]);
                        res4 = svmla_m(pg, res4, skp6, d2_coeffs_z[2]);
                        res5 = svmla_m(pg, res5, skp6, d2_coeffs_z[1]);
                        res7 = svmla_m(pg, res7, skp6, d2_coeffs_z[1]);
                        sv_float__t skp7 = svld1(pg, &s[k+7][j][i]);
                        res1 = svmla_m(pg, res1, skp7, d2_coeffs_z[6]);
                        res2 = svmla_m(pg, res2, skp7, d2_coeffs_z[5]);
                        res3 = svmla_m(pg, res3, skp7, d2_coeffs_z[4]);
                        res4 = svmla_m(pg, res4, skp7, d2_coeffs_z[3]);
                        res5 = svmla_m(pg, res5, skp7, d2_coeffs_z[2]);
                        res6 = svmla_m(pg, res6, skp7, d2_coeffs_z[1]);

                        sv_float__t skp8 = svld1(pg, &s[k+8][j][i]);
                        res2 = svmla_m(pg, res2, skp8, d2_coeffs_z[6]);
                        res3 = svmla_m(pg, res3, skp8, d2_coeffs_z[5]);
                        res4 = svmla_m(pg, res4, skp8, d2_coeffs_z[4]);
                        res5 = svmla_m(pg, res5, skp8, d2_coeffs_z[3]);
                        res6 = svmla_m(pg, res6, skp8, d2_coeffs_z[2]);
                        res7 = svmla_m(pg, res7, skp8, d2_coeffs_z[1]);

                        sv_float__t skp9 = svld1(pg, &s[k+9][j][i]);
                        res3 = svmla_m(pg, res3, skp9, d2_coeffs_z[6]);
                        res4 = svmla_m(pg, res4, skp9, d2_coeffs_z[5]);
                        res5 = svmla_m(pg, res5, skp9, d2_coeffs_z[4]);
                        res6 = svmla_m(pg, res6, skp9, d2_coeffs_z[3]);
                        res7 = svmla_m(pg, res7, skp9, d2_coeffs_z[2]);

                        sv_float__t skpA = svld1(pg, &s[k+0xA][j][i]);
                        res4 = svmla_m(pg, res4, skpA, d2_coeffs_z[6]);
                        res5 = svmla_m(pg, res5, skpA, d2_coeffs_z[5]);
                        res6 = svmla_m(pg, res6, skpA, d2_coeffs_z[4]);
                        res7 = svmla_m(pg, res7, skpA, d2_coeffs_z[3]);

                        sv_float__t skpB = svld1(pg, &s[k+0xB][j][i]);
                        res5 = svmla_m(pg, res5, skpB, d2_coeffs_z[6]);
                        res6 = svmla_m(pg, res6, skpB, d2_coeffs_z[5]);
                        res7 = svmla_m(pg, res7, skpB, d2_coeffs_z[4]);

                        sv_float__t skpC = svld1(pg, &s[k+0xC][j][i]);
                        res6 = svmla_m(pg, res6, skpC, d2_coeffs_z[6]);
                        res7 = svmla_m(pg, res7, skpC, d2_coeffs_z[5]);

                        sv_float__t skpD = svld1(pg, &s[k+0xD][j][i]);
                        res7 = svmla_m(pg, res7, skpD, d2_coeffs_z[6]);

                        svst1(pg, &d[k+0][j][i], res0);
                        svst1(pg, &d[k+1][j][i], res1);
                        svst1(pg, &d[k+2][j][i], res2);
                        svst1(pg, &d[k+3][j][i], res3);
                        svst1(pg, &d[k+4][j][i], res4);
                        svst1(pg, &d[k+5][j][i], res5);
                        svst1(pg, &d[k+6][j][i], res6);
                        svst1(pg, &d[k+7][j][i], res7);
                    }
                }
                for (int64_t k = kk; k < kt; k++) {
                    for (int64_t j = jj; j < jt; j++) {
                        #pragma omp simd
                        for (int64_t i = ii; i < it; i ++) {
                            d_ptr[i + j * d_stride_y + k * d_stride_z] = d[k-kk][j-jj][i-ii];
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
                            // int r = 1;
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
    const sv_float__t sve_coef0 = traits::svdup_f_(coef_0);
    const sv_float__t sve_coef_x1 = traits::svdup_f_(d2_coeffs_x[1]);
    const sv_float__t sve_coef_x2 = traits::svdup_f_(d2_coeffs_x[2]);
    const sv_float__t sve_coef_x3 = traits::svdup_f_(d2_coeffs_x[3]);
    const sv_float__t sve_coef_x4 = traits::svdup_f_(d2_coeffs_x[4]);
    const sv_float__t sve_coef_x5 = traits::svdup_f_(d2_coeffs_x[5]);
    const sv_float__t sve_coef_x6 = traits::svdup_f_(d2_coeffs_x[6]);
    const sv_float__t sve_coef_y1 = traits::svdup_f_(d2_coeffs_y[1]);
    const sv_float__t sve_coef_y2 = traits::svdup_f_(d2_coeffs_y[2]);
    const sv_float__t sve_coef_y3 = traits::svdup_f_(d2_coeffs_y[3]);
    const sv_float__t sve_coef_y4 = traits::svdup_f_(d2_coeffs_y[4]);
    const sv_float__t sve_coef_y5 = traits::svdup_f_(d2_coeffs_y[5]);
    const sv_float__t sve_coef_y6 = traits::svdup_f_(d2_coeffs_y[6]);
    const sv_float__t sve_coef_z1 = traits::svdup_f_(d2_coeffs_z[1]);
    const sv_float__t sve_coef_z2 = traits::svdup_f_(d2_coeffs_z[2]);
    const sv_float__t sve_coef_z3 = traits::svdup_f_(d2_coeffs_z[3]);
    const sv_float__t sve_coef_z4 = traits::svdup_f_(d2_coeffs_z[4]);
    const sv_float__t sve_coef_z5 = traits::svdup_f_(d2_coeffs_z[5]);
    const sv_float__t sve_coef_z6 = traits::svdup_f_(d2_coeffs_z[6]);

    const int64_t s_stride_y_r1_byte = sizeof(T) * 1 * s_stride_y;
    const int64_t s_stride_y_r2_byte = sizeof(T) * 2 * s_stride_y;
    const int64_t s_stride_y_r3_byte = sizeof(T) * 3 * s_stride_y;
    const int64_t s_stride_y_r4_byte = sizeof(T) * 4 * s_stride_y;
    const int64_t s_stride_y_r5_byte = sizeof(T) * 5 * s_stride_y;
    const int64_t s_stride_y_r6_byte = sizeof(T) * 6 * s_stride_y;
    const int64_t s_stride_z_r1_byte = sizeof(T) * 1 * s_stride_z;
    const int64_t s_stride_z_r2_byte = sizeof(T) * 2 * s_stride_z;
    const int64_t s_stride_z_r3_byte = sizeof(T) * 3 * s_stride_z;
    const int64_t s_stride_z_r4_byte = sizeof(T) * 4 * s_stride_z;
    const int64_t s_stride_z_r5_byte = sizeof(T) * 5 * s_stride_z;
    const int64_t s_stride_z_r6_byte = sizeof(T) * 6 * s_stride_z;

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

                                T* addr_plus = nullptr;
                                T* addr_minus = nullptr;

                                {
                                    // const int64_t s_stride_y_r = 1 * s_stride_y;
                                    // const int64_t s_stride_z_r = 1 * s_stride_z;

                                    svbool_t pg_x_left_boundary  = traits::svwhilegt_b_(i - 1 + svcnt, i_l);
                                    svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + 1, i_r + 1);

                                    x_left = svld1(pg_x_left_boundary, s_ptr_ijk - 1);
                                    sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + 1),
                                                                    x_left);
                                    res = svmla_m(pg, res, sum_x, sve_coef_x1);

                                    svbool_t pg_y_left_boundary  = j - 1 >= j_l ? all_true : all_false;
                                    svbool_t pg_y_right_boundary = j + 1 <= j_r ? all_true : all_false;

                                    asm volatile (
                                        "mov x19, %[base]        \n\t"
                                        "mov x20, %[stride]      \n\t"
                                        "add %[out_plus], x19, x20 \n\t"
                                        "sub %[out_minus], x19, x20 \n\t"
                                        : [out_plus] "=r" (addr_plus),
                                          [out_minus] "=r" (addr_minus)
                                        : [base] "ir" (s_ptr_ijk),
                                          [stride] "ir" (s_stride_y_r1_byte)
                                        : "x19", "x20", "cc"
                                    );

                                    sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, addr_plus ),
                                                                    svld1(pg_y_left_boundary , addr_minus));
                                    // sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r1),
                                    //                                 svld1(pg_y_left_boundary , s_ptr_ijk - s_stride_y_r1));
                                    res = svmla_m(pg, res, sum_y, sve_coef_y1);

                                    svbool_t pg_z_left_boundary  = k - 1 >= k_l ? all_true : all_false;
                                    svbool_t pg_z_right_boundary = k + 1 <= k_r ? all_true : all_false;

                                    asm volatile (
                                        "mov x19, %[base]        \n\t"
                                        "mov x20, %[stride]      \n\t"
                                        "add %[out_plus], x19, x20 \n\t"
                                        "sub %[out_minus], x19, x20 \n\t"
                                        : [out_plus] "=r" (addr_plus),
                                          [out_minus] "=r" (addr_minus)
                                        : [base] "ir" (s_ptr_ijk),
                                          [stride] "ir" (s_stride_z_r1_byte)
                                        : "x19", "x20", "cc"
                                    );
                                    // // printf("debug %p %d %p %p\n", (void*)s_ptr_ijk, s_stride_z_r1_byte, (void*)(s_ptr_ijk+s_stride_z_r1_byte/8), (void*)addr_plus);

                                    sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, addr_plus ),
                                                                    svld1(pg_z_left_boundary , addr_minus));
                                    // sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r1),
                                    //                                 svld1(pg_z_left_boundary , s_ptr_ijk - s_stride_z_r1));
                                    res = svmla_m(pg, res, sum_z, sve_coef_z1);
                                }
                                {
                                    // const int64_t s_stride_y_r = 2 * s_stride_y;
                                    // const int64_t s_stride_z_r = 2 * s_stride_z;

                                    // svbool_t pg_x_left_boundary  = traits::svwhilegt_b_(i - 2 + svcnt, i_l);
                                    svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + 2, i_r + 1);

                                    x_left = i - 2 >= i_l ? svinsr(x_left, *(s_ptr_ijk - 2)) : svinsr(x_left, 0.0);
                                    sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + 2),
                                                                    x_left);
                                    res = svmla_m(pg, res, sum_x, sve_coef_x2);

                                    svbool_t pg_y_left_boundary  = j - 2 >= j_l ? all_true : all_false;
                                    svbool_t pg_y_right_boundary = j + 2 <= j_r ? all_true : all_false;

                                    asm volatile (
                                        "mov x19, %[base]        \n\t"
                                        "mov x20, %[stride]      \n\t"
                                        "add %[out_plus], x19, x20 \n\t"
                                        "sub %[out_minus], x19, x20 \n\t"
                                        : [out_plus] "=r" (addr_plus),
                                          [out_minus] "=r" (addr_minus)
                                        : [base] "ir" (s_ptr_ijk),
                                          [stride] "ir" (s_stride_y_r2_byte)
                                        : "x19", "x20", "cc"
                                    );

                                    sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, addr_plus ),
                                                                    svld1(pg_y_left_boundary , addr_minus));
                                    // sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r2),
                                    //                                 svld1(pg_y_left_boundary , s_ptr_ijk - s_stride_y_r2));
                                    res = svmla_m(pg, res, sum_y, sve_coef_y2);

                                    svbool_t pg_z_left_boundary  = k - 2 >= k_l ? all_true : all_false;
                                    svbool_t pg_z_right_boundary = k + 2 <= k_r ? all_true : all_false;

                                    asm volatile (
                                        "mov x19, %[base]        \n\t"
                                        "mov x20, %[stride]      \n\t"
                                        "add %[out_plus], x19, x20 \n\t"
                                        "sub %[out_minus], x19, x20 \n\t"
                                        : [out_plus] "=r" (addr_plus),
                                          [out_minus] "=r" (addr_minus)
                                        : [base] "ir" (s_ptr_ijk),
                                          [stride] "ir" (s_stride_z_r2_byte)
                                        : "x19", "x20", "cc"
                                    );

                                    sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, addr_plus ),
                                                                    svld1(pg_z_left_boundary , addr_minus));
                                    // sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r2),
                                    //                                 svld1(pg_z_left_boundary , s_ptr_ijk - s_stride_z_r2));
                                    res = svmla_m(pg, res, sum_z, sve_coef_z2);
                                }
                                {
                                    // const int64_t s_stride_y_r = 3 * s_stride_y;
                                    // const int64_t s_stride_z_r = 3 * s_stride_z;

                                    // svbool_t pg_x_left_boundary  = traits::svwhilegt_b_(i - 3 + svcnt, i_l);
                                    svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + 3, i_r + 1);

                                    x_left = i - 2 >= i_l ? svinsr(x_left, *(s_ptr_ijk - 3)) : svinsr(x_left, 0.0);
                                    sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + 3),
                                                                    x_left);
                                    res = svmla_m(pg, res, sum_x, sve_coef_x3);

                                    svbool_t pg_y_left_boundary  = j - 3 >= j_l ? all_true : all_false;
                                    svbool_t pg_y_right_boundary = j + 3 <= j_r ? all_true : all_false;

                                    asm volatile (
                                        "mov x19, %[base]        \n\t"
                                        "mov x20, %[stride]      \n\t"
                                        "add %[out_plus], x19, x20 \n\t"
                                        "sub %[out_minus], x19, x20 \n\t"
                                        : [out_plus] "=r" (addr_plus),
                                          [out_minus] "=r" (addr_minus)
                                        : [base] "ir" (s_ptr_ijk),
                                          [stride] "ir" (s_stride_y_r3_byte)
                                        : "x19", "x20", "cc"
                                    );

                                    sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, addr_plus ),
                                                                    svld1(pg_y_left_boundary , addr_minus));
                                    // sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r3),
                                    //                                 svld1(pg_y_left_boundary , s_ptr_ijk - s_stride_y_r3));
                                    res = svmla_m(pg, res, sum_y, sve_coef_y3);

                                    svbool_t pg_z_left_boundary  = k - 3 >= k_l ? all_true : all_false;
                                    svbool_t pg_z_right_boundary = k + 3 <= k_r ? all_true : all_false;

                                    asm volatile (
                                        "mov x19, %[base]        \n\t"
                                        "mov x20, %[stride]      \n\t"
                                        "add %[out_plus], x19, x20 \n\t"
                                        "sub %[out_minus], x19, x20 \n\t"
                                        : [out_plus] "=r" (addr_plus),
                                          [out_minus] "=r" (addr_minus)
                                        : [base] "ir" (s_ptr_ijk),
                                          [stride] "ir" (s_stride_z_r3_byte)
                                        : "x19", "x20", "cc"
                                    );

                                    sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, addr_plus ),
                                                                    svld1(pg_z_left_boundary , addr_minus));
                                    // sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r3),
                                    //                                 svld1(pg_z_left_boundary , s_ptr_ijk - s_stride_z_r3));
                                    res = svmla_m(pg, res, sum_z, sve_coef_z3);
                                }
                                {
                                    // const int64_t s_stride_y_r = 4 * s_stride_y;
                                    // const int64_t s_stride_z_r = 4 * s_stride_z;

                                    // svbool_t pg_x_left_boundary  = traits::svwhilegt_b_(i - 4 + svcnt, i_l);
                                    svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + 4, i_r + 1);

                                    x_left = i - 2 >= i_l ? svinsr(x_left, *(s_ptr_ijk - 4)) : svinsr(x_left, 0.0);
                                    sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + 4),
                                                                    x_left);
                                    res = svmla_m(pg, res, sum_x, sve_coef_x4);

                                    svbool_t pg_y_left_boundary  = j - 4 >= j_l ? all_true : all_false;
                                    svbool_t pg_y_right_boundary = j + 4 <= j_r ? all_true : all_false;

                                    asm volatile (
                                        "mov x19, %[base]        \n\t"
                                        "mov x20, %[stride]      \n\t"
                                        "add %[out_plus], x19, x20 \n\t"
                                        "sub %[out_minus], x19, x20 \n\t"
                                        : [out_plus] "=r" (addr_plus),
                                          [out_minus] "=r" (addr_minus)
                                        : [base] "ir" (s_ptr_ijk),
                                          [stride] "ir" (s_stride_y_r4_byte)
                                        : "x19", "x20", "cc"
                                    );

                                    sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, addr_plus ),
                                                                    svld1(pg_y_left_boundary , addr_minus));
                                    // sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r4),
                                    //                                 svld1(pg_y_left_boundary , s_ptr_ijk - s_stride_y_r4));
                                    res = svmla_m(pg, res, sum_y, sve_coef_y4);

                                    svbool_t pg_z_left_boundary  = k - 4 >= k_l ? all_true : all_false;
                                    svbool_t pg_z_right_boundary = k + 4 <= k_r ? all_true : all_false;

                                    asm volatile (
                                        "mov x19, %[base]        \n\t"
                                        "mov x20, %[stride]      \n\t"
                                        "add %[out_plus], x19, x20 \n\t"
                                        "sub %[out_minus], x19, x20 \n\t"
                                        : [out_plus] "=r" (addr_plus),
                                          [out_minus] "=r" (addr_minus)
                                        : [base] "ir" (s_ptr_ijk),
                                          [stride] "ir" (s_stride_z_r4_byte)
                                        : "x19", "x20", "cc"
                                    );

                                    sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, addr_plus ),
                                                                    svld1(pg_z_left_boundary , addr_minus));
                                    // sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r4),
                                    //                                 svld1(pg_z_left_boundary , s_ptr_ijk - s_stride_z_r4));
                                    res = svmla_m(pg, res, sum_z, sve_coef_z4);
                                }
                                {
                                    // const int64_t s_stride_y_r = 5 * s_stride_y;
                                    // const int64_t s_stride_z_r = 5 * s_stride_z;

                                    // svbool_t pg_x_left_boundary  = traits::svwhilegt_b_(i - 5 + svcnt, i_l);
                                    svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + 5, i_r + 1);

                                    x_left = i - 2 >= i_l ? svinsr(x_left, *(s_ptr_ijk - 5)) : svinsr(x_left, 0.0);
                                    sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + 5),
                                                                    x_left);
                                    res = svmla_m(pg, res, sum_x, sve_coef_x5);

                                    svbool_t pg_y_left_boundary  = j - 5 >= j_l ? all_true : all_false;
                                    svbool_t pg_y_right_boundary = j + 5 <= j_r ? all_true : all_false;

                                    asm volatile (
                                        "mov x19, %[base]        \n\t"
                                        "mov x20, %[stride]      \n\t"
                                        "add %[out_plus], x19, x20 \n\t"
                                        "sub %[out_minus], x19, x20 \n\t"
                                        : [out_plus] "=r" (addr_plus),
                                          [out_minus] "=r" (addr_minus)
                                        : [base] "ir" (s_ptr_ijk),
                                          [stride] "ir" (s_stride_y_r5_byte)
                                        : "x19", "x20", "cc"
                                    );

                                    sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, addr_plus ),
                                                                    svld1(pg_y_left_boundary , addr_minus));
                                    // sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r5),
                                    //                                 svld1(pg_y_left_boundary , s_ptr_ijk - s_stride_y_r5));
                                    res = svmla_m(pg, res, sum_y, sve_coef_y5);

                                    svbool_t pg_z_left_boundary  = k - 5 >= k_l ? all_true : all_false;
                                    svbool_t pg_z_right_boundary = k + 5 <= k_r ? all_true : all_false;

                                    asm volatile (
                                        "mov x19, %[base]        \n\t"
                                        "mov x20, %[stride]      \n\t"
                                        "add %[out_plus], x19, x20 \n\t"
                                        "sub %[out_minus], x19, x20 \n\t"
                                        : [out_plus] "=r" (addr_plus),
                                          [out_minus] "=r" (addr_minus)
                                        : [base] "ir" (s_ptr_ijk),
                                          [stride] "ir" (s_stride_z_r5_byte)
                                        : "x19", "x20", "cc"
                                    );

                                    sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, addr_plus ),
                                                                    svld1(pg_z_left_boundary , addr_minus));
                                    // sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r5),
                                    //                                 svld1(pg_z_left_boundary , s_ptr_ijk - s_stride_z_r5));
                                    res = svmla_m(pg, res, sum_z, sve_coef_z5);
                                }
                                {
                                    // const int64_t s_stride_y_r = 6 * s_stride_y;
                                    // const int64_t s_stride_z_r = 6 * s_stride_z;

                                    // svbool_t pg_x_left_boundary  = traits::svwhilegt_b_(i - 6 + svcnt, i_l);
                                    svbool_t pg_x_right_boundary = traits::svwhilelt_b_(i + 6, i_r + 1);

                                    x_left = i - 2 >= i_l ? svinsr(x_left, *(s_ptr_ijk - 6)) : svinsr(x_left, 0.0);
                                    sv_float__t sum_x = svadd_m(pg, svld1(pg_x_right_boundary, s_ptr_ijk + 6),
                                                                    x_left);
                                    res = svmla_m(pg, res, sum_x, sve_coef_x6);

                                    svbool_t pg_y_left_boundary  = j - 6 >= j_l ? all_true : all_false;
                                    svbool_t pg_y_right_boundary = j + 6 <= j_r ? all_true : all_false;

                                    asm volatile (
                                        "mov x19, %[base]        \n\t"
                                        "mov x20, %[stride]      \n\t"
                                        "add %[out_plus], x19, x20 \n\t"
                                        "sub %[out_minus], x19, x20 \n\t"
                                        : [out_plus] "=r" (addr_plus),
                                          [out_minus] "=r" (addr_minus)
                                        : [base] "ir" (s_ptr_ijk),
                                          [stride] "ir" (s_stride_y_r6_byte)
                                        : "x19", "x20", "cc"
                                    );

                                    sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, addr_plus ),
                                                                    svld1(pg_y_left_boundary , addr_minus));
                                    // sv_float__t sum_y = svadd_m(pg, svld1(pg_y_right_boundary, s_ptr_ijk + s_stride_y_r6),
                                    //                                 svld1(pg_y_left_boundary , s_ptr_ijk - s_stride_y_r6));
                                    res = svmla_m(pg, res, sum_y, sve_coef_y6);

                                    svbool_t pg_z_left_boundary  = k - 6 >= k_l ? all_true : all_false;
                                    svbool_t pg_z_right_boundary = k + 6 <= k_r ? all_true : all_false;

                                    asm volatile (
                                        "mov x19, %[base]        \n\t"
                                        "mov x20, %[stride]      \n\t"
                                        "add %[out_plus], x19, x20 \n\t"
                                        "sub %[out_minus], x19, x20 \n\t"
                                        : [out_plus] "=r" (addr_plus),
                                          [out_minus] "=r" (addr_minus)
                                        : [base] "ir" (s_ptr_ijk),
                                          [stride] "ir" (s_stride_z_r6_byte)
                                        : "x19", "x20", "cc"
                                    );

                                    sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, addr_plus ),
                                                                    svld1(pg_z_left_boundary , addr_minus));
                                    // sv_float__t sum_z = svadd_m(pg, svld1(pg_z_right_boundary, s_ptr_ijk + s_stride_z_r6),
                                    //                                 svld1(pg_z_left_boundary , s_ptr_ijk - s_stride_z_r6));
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
            // Stencil3D<4, T>::lap_3d_c2_o0_boundary_safe(s_ptr, s_stride_y, s_stride_z, s_stride_b,
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
            // Stencil3D<3, T>::lap_3d_c2_o0_boundary_safe(s_ptr, s_stride_y, s_stride_z, s_stride_b,
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

#endif //__ARM_FEATURE_SME
