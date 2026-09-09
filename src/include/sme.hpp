#pragma once

#include <arm_sme.h>
#include <arm_sve.h>

inline void svzero() __arm_streaming __arm_out("za") { svzero_za(); }

inline void svld1_f64_2x4(const double * __restrict__ y,
                          int N) __arm_streaming __arm_out("za") {
  const double* __restrict__ y0 = y, * __restrict__ y1 = y + 8 * N;
#pragma clang loop unroll(disable)
  for (int i = 0; i < 8; ++i) {
    svld1_hor_za64(0, i, svptrue_b64(), y0);
    svld1_hor_za64(1, i, svptrue_b64(), y0 + 8);
    svld1_hor_za64(2, i, svptrue_b64(), y0 + 16);
    svld1_hor_za64(3, i, svptrue_b64(), y0 + 24);
    y0 += N;
    svld1_hor_za64(4, i, svptrue_b64(), y1);
    svld1_hor_za64(5, i, svptrue_b64(), y1 + 8);
    svld1_hor_za64(6, i, svptrue_b64(), y1 + 16);
    svld1_hor_za64(7, i, svptrue_b64(), y1 + 24);
    y1 += N;
  }
}

inline void svld1_f64_2x4_ver(const double * __restrict__ y,
                              int N) __arm_streaming __arm_out("za") {
  const double* __restrict__ y0 = y, * __restrict__ y1 = y + 8 * N;
#pragma clang loop unroll(disable)
  for (int i = 0; i < 8; ++i) {
    svld1_ver_za64(0, i, svptrue_b64(), y0);
    svld1_ver_za64(2, i, svptrue_b64(), y0 + 8);
    svld1_ver_za64(4, i, svptrue_b64(), y0 + 16);
    svld1_ver_za64(6, i, svptrue_b64(), y0 + 24);
    y0 += N;
    svld1_ver_za64(1, i, svptrue_b64(), y1);
    svld1_ver_za64(3, i, svptrue_b64(), y1 + 8);
    svld1_ver_za64(5, i, svptrue_b64(), y1 + 16);
    svld1_ver_za64(7, i, svptrue_b64(), y1 + 24);
    y1 += N;
  }
}

inline void svld1_f64_4x2(const double * __restrict__ y,
                          int N) __arm_streaming __arm_out("za") {
  const double* __restrict__ y0 = y, * __restrict__ y1 = y + 8 * N, * __restrict__ y2 = y + 16 * N, 
                            * __restrict__ y3 = y + 24 * N;
#pragma clang loop unroll(disable)
  for (int i = 0; i < 8; ++i) {
    svld1_hor_za64(0, i, svptrue_b64(), y0);
    svld1_hor_za64(1, i, svptrue_b64(), y0 + 8);
    y0 += N;
    svld1_hor_za64(2, i, svptrue_b64(), y1);
    svld1_hor_za64(3, i, svptrue_b64(), y1 + 8);
    y1 += N;
    svld1_hor_za64(4, i, svptrue_b64(), y2);
    svld1_hor_za64(5, i, svptrue_b64(), y2 + 8);
    y2 += N;
    svld1_hor_za64(6, i, svptrue_b64(), y3);
    svld1_hor_za64(7, i, svptrue_b64(), y3 + 8);
    y3 += N;
  }
}

inline void svld1_f64_4x2_ver(const double * __restrict__ y,
                              int N) __arm_streaming __arm_out("za") {
  const double* __restrict__ y0 = y, * __restrict__ y1 = y + 8 * N, * __restrict__ y2 = y + 16 * N, 
                            * __restrict__ y3 = y + 24 * N;
#pragma clang loop unroll(disable)
  for (int i = 0; i < 8; ++i) {
    svld1_ver_za64(0, i, svptrue_b64(), y0);
    svld1_ver_za64(4, i, svptrue_b64(), y0 + 8);
    y0 += N;
    svld1_ver_za64(1, i, svptrue_b64(), y1);
    svld1_ver_za64(5, i, svptrue_b64(), y1 + 8);
    y1 += N;
    svld1_ver_za64(2, i, svptrue_b64(), y2);
    svld1_ver_za64(6, i, svptrue_b64(), y2 + 8);
    y2 += N;
    svld1_ver_za64(3, i, svptrue_b64(), y3);
    svld1_ver_za64(7, i, svptrue_b64(), y3 + 8);
    y3 += N;
  }
}

inline void svld1_f32_2x2(const float * __restrict__ y,
                          int N) __arm_streaming __arm_out("za") {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float* __restrict__ y0 = y + i * N, * __restrict__ y1 = y + (i + 16) * N;
    svld1_hor_za32(0, i, svptrue_b32(), y0);
    svld1_hor_za32(1, i, svptrue_b32(), y0 + 16);
    svld1_hor_za32(2, i, svptrue_b32(), y1);
    svld1_hor_za32(3, i, svptrue_b32(), y1 + 16);
  }
}

inline void svld1_f32_2x2_ver(const float * __restrict__ y,
                              int N) __arm_streaming __arm_out("za") {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float* __restrict__ y0 = y + i * N, * __restrict__ y1 = y + (i + 16) * N;
    svld1_ver_za32(0, i, svptrue_b32(), y0);
    svld1_ver_za32(2, i, svptrue_b32(), y0 + 16);
    svld1_ver_za32(1, i, svptrue_b32(), y1);
    svld1_ver_za32(3, i, svptrue_b32(), y1 + 16);
  }
}

inline void svst1_f64_2x4(double * __restrict__ y,
                          int N) __arm_streaming __arm_preserves("za") {
  double* __restrict__ y0 = y, * __restrict__ y1 = y + 8 * N;
#pragma clang loop unroll(disable)
  for (int i = 0; i < 8; ++i) {
    svst1_hor_za64(0, i, svptrue_b64(), y0);
    svst1_hor_za64(1, i, svptrue_b64(), y0 + 8);
    svst1_hor_za64(2, i, svptrue_b64(), y0 + 16);
    svst1_hor_za64(3, i, svptrue_b64(), y0 + 24);
    y0 += N;
    svst1_hor_za64(4, i, svptrue_b64(), y1);
    svst1_hor_za64(5, i, svptrue_b64(), y1 + 8);
    svst1_hor_za64(6, i, svptrue_b64(), y1 + 16);
    svst1_hor_za64(7, i, svptrue_b64(), y1 + 24);
    y1 += N;
  }
}

inline void svst1_f64_2x4_ver(double * __restrict__ y,
                              int N) __arm_streaming __arm_preserves("za") {
  double* __restrict__ y0 = y, * __restrict__ y1 = y + 8 * N;
#pragma clang loop unroll(disable)
  for (int i = 0; i < 8; ++i) {
    svst1_ver_za64(0, i, svptrue_b64(), y0);
    svst1_ver_za64(2, i, svptrue_b64(), y0 + 8);
    svst1_ver_za64(4, i, svptrue_b64(), y0 + 16);
    svst1_ver_za64(6, i, svptrue_b64(), y0 + 24);
    y0 += N;
    svst1_ver_za64(1, i, svptrue_b64(), y1);
    svst1_ver_za64(3, i, svptrue_b64(), y1 + 8);
    svst1_ver_za64(5, i, svptrue_b64(), y1 + 16);
    svst1_ver_za64(7, i, svptrue_b64(), y1 + 24);
    y1 += N;
  }
}

inline void svst1_f64_4x2(double * __restrict__ y,
                          int N) __arm_streaming __arm_preserves("za") {
  double* __restrict__ y0 = y, * __restrict__ y1 = y + 8 * N, * __restrict__ y2 = y + 16 * N, 
                            * __restrict__ y3 = y + 24 * N;
#pragma clang loop unroll(disable)
  for (int i = 0; i < 8; ++i) {
    svst1_hor_za64(0, i, svptrue_b64(), y0);
    svst1_hor_za64(1, i, svptrue_b64(), y0 + 8);
    y0 += N;
    svst1_hor_za64(2, i, svptrue_b64(), y1);
    svst1_hor_za64(3, i, svptrue_b64(), y1 + 8);
    y1 += N;
    svst1_hor_za64(4, i, svptrue_b64(), y2);
    svst1_hor_za64(5, i, svptrue_b64(), y2 + 8);
    y2 += N;
    svst1_hor_za64(6, i, svptrue_b64(), y3);
    svst1_hor_za64(7, i, svptrue_b64(), y3 + 8);
    y3 += N;
  }
}

inline void svst1_f64_4x2_ver(double * __restrict__ y,
                              int N) __arm_streaming __arm_preserves("za") {
  double* __restrict__ y0 = y, * __restrict__ y1 = y + 8 * N, * __restrict__ y2 = y + 16 * N, 
                            * __restrict__ y3 = y + 24 * N;
#pragma clang loop unroll(disable)
  for (int i = 0; i < 8; ++i) {
    svst1_ver_za64(0, i, svptrue_b64(), y0);
    svst1_ver_za64(4, i, svptrue_b64(), y0 + 8);
    y0 += N;
    svst1_ver_za64(1, i, svptrue_b64(), y1);
    svst1_ver_za64(5, i, svptrue_b64(), y1 + 8);
    y1 += N;
    svst1_ver_za64(2, i, svptrue_b64(), y2);
    svst1_ver_za64(6, i, svptrue_b64(), y2 + 8);
    y2 += N;
    svst1_ver_za64(3, i, svptrue_b64(), y3);
    svst1_ver_za64(7, i, svptrue_b64(), y3 + 8);
    y3 += N;
  }
}

inline void svst1_f32_2x2(float * __restrict__ y,
                          int N) __arm_streaming __arm_preserves("za") {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    float* __restrict__ y0 = y + i * N, * __restrict__ y1 = y + (i + 16) * N;
    svst1_hor_za32(0, i, svptrue_b32(), y0);
    svst1_hor_za32(1, i, svptrue_b32(), y0 + 16);
    svst1_hor_za32(2, i, svptrue_b32(), y1);
    svst1_hor_za32(3, i, svptrue_b32(), y1 + 16);
  }
}

inline void svst1_f32_2x2_ver(float * __restrict__ y,
                              int N) __arm_streaming __arm_preserves("za") {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    float* __restrict__ y0 = y + i * N, * __restrict__ y1 = y + (i + 16) * N;
    svst1_ver_za32(0, i, svptrue_b32(), y0);
    svst1_ver_za32(2, i, svptrue_b32(), y0 + 16);
    svst1_ver_za32(1, i, svptrue_b32(), y1);
    svst1_ver_za32(3, i, svptrue_b32(), y1 + 16);
  }
}
inline void fmopa_f64_2x4(const double * __restrict__ a,
                          const double * __restrict__ b) __arm_streaming __arm_inout("za") {
  svfloat64_t a0 = svld1(svptrue_b64(), a);
  svfloat64_t a1 = svld1_vnum(svptrue_b64(), a, 1);
  svfloat64_t b0 = svld1(svptrue_b64(), b);
  svmopa_za64_f64_m(0, svptrue_b64(), svptrue_b64(), a0, b0);
  svmopa_za64_f64_m(4, svptrue_b64(), svptrue_b64(), a1, b0);
  svfloat64_t b1 = svld1_vnum(svptrue_b64(), b, 1);
  svmopa_za64_f64_m(1, svptrue_b64(), svptrue_b64(), a0, b1);
  svmopa_za64_f64_m(5, svptrue_b64(), svptrue_b64(), a1, b1);
  svfloat64_t b2 = svld1_vnum(svptrue_b64(), b, 2);
  svmopa_za64_f64_m(2, svptrue_b64(), svptrue_b64(), a0, b2);
  svmopa_za64_f64_m(6, svptrue_b64(), svptrue_b64(), a1, b2);
  svfloat64_t b3 = svld1_vnum(svptrue_b64(), b, 3);
  svmopa_za64_f64_m(3, svptrue_b64(), svptrue_b64(), a0, b3);
  svmopa_za64_f64_m(7, svptrue_b64(), svptrue_b64(), a1, b3);
}

inline void fmopa_f64_4x2(const double * __restrict__ a,
                          const double * __restrict__ b) __arm_streaming __arm_inout("za") {
  svfloat64_t a0 = svld1(svptrue_b64(), a);
  svfloat64_t b0 = svld1(svptrue_b64(), b),
              b1 = svld1_vnum(svptrue_b64(), b, 1);
  svmopa_za64_f64_m(0, svptrue_b64(), svptrue_b64(), a0, b0);
  svmopa_za64_f64_m(1, svptrue_b64(), svptrue_b64(), a0, b1);
  svfloat64_t a1 = svld1_vnum(svptrue_b64(), a, 1);
  svmopa_za64_f64_m(2, svptrue_b64(), svptrue_b64(), a1, b0);
  svmopa_za64_f64_m(3, svptrue_b64(), svptrue_b64(), a1, b1);
  svfloat64_t a2 = svld1_vnum(svptrue_b64(), a, 2);
  svmopa_za64_f64_m(4, svptrue_b64(), svptrue_b64(), a2, b0);
  svmopa_za64_f64_m(5, svptrue_b64(), svptrue_b64(), a2, b1);
  svfloat64_t a3 = svld1_vnum(svptrue_b64(), a, 3);
  svmopa_za64_f64_m(6, svptrue_b64(), svptrue_b64(), a3, b0);
  svmopa_za64_f64_m(7, svptrue_b64(), svptrue_b64(), a3, b1);
}

inline void fmopa_f32_2x2(const float * __restrict__ a,
                          const float * __restrict__ b) __arm_streaming __arm_inout("za") {
  svfloat32_t a0 = svld1(svptrue_b32(), a);
  svfloat32_t b0 = svld1(svptrue_b32(), b),
              b1 = svld1_vnum(svptrue_b32(), b, 1);
  svmopa_za32_f32_m(0, svptrue_b32(), svptrue_b32(), a0, b0);
  svmopa_za32_f32_m(1, svptrue_b32(), svptrue_b32(), a0, b1);
  svfloat32_t a1 = svld1_vnum(svptrue_b32(), a, 1);
  svmopa_za32_f32_m(2, svptrue_b32(), svptrue_b32(), a1, b0);
  svmopa_za32_f32_m(3, svptrue_b32(), svptrue_b32(), a1, b1);
}
