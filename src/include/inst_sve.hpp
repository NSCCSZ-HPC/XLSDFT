#pragma once

#include <arm_sve.h>

inline void vmov_f64(double * __restrict__ y, const double * __restrict__ x) __arm_streaming_compatible {
  svfloat64_t y_sve = svld1(svptrue_b64(), x);
  svst1(svptrue_b64(), y, y_sve);
}

inline void vmov_f32(float * __restrict__ y, const float * __restrict__ x) __arm_streaming_compatible {
  svfloat32_t y_sve = svld1(svptrue_b32(), x);
  svst1(svptrue_b32(), y, y_sve);
}

inline void vadd_f64(double * __restrict__ y, const double * __restrict__ a,
                     const double * __restrict__ b) __arm_streaming_compatible {
  svfloat64_t a_sve = svld1(svptrue_b64(), a);
  svfloat64_t b_sve = svld1(svptrue_b64(), b);
  svfloat64_t y_sve = svadd_z(svptrue_b64(), a_sve, b_sve);
  svst1(svptrue_b64(), y, y_sve);
}

inline void vadd_f32(float * __restrict__ y, const float * __restrict__ a,
                     const float * __restrict__ b) __arm_streaming_compatible {
  svfloat32_t a_sve = svld1(svptrue_b32(), a);
  svfloat32_t b_sve = svld1(svptrue_b32(), b);
  svfloat32_t y_sve = svadd_z(svptrue_b32(), a_sve, b_sve);
  svst1(svptrue_b32(), y, y_sve);
}

inline void vsub_f64(double * __restrict__ y, const double * __restrict__ a,
                     const double * __restrict__ b) __arm_streaming_compatible {
  svfloat64_t a_sve = svld1(svptrue_b64(), a);
  svfloat64_t b_sve = svld1(svptrue_b64(), b);
  svfloat64_t y_sve = svsub_z(svptrue_b64(), a_sve, b_sve);
  svst1(svptrue_b64(), y, y_sve);
}

inline void vsub_f32(float * __restrict__ y, const float * __restrict__ a,
                     const float * __restrict__ b) __arm_streaming_compatible {
  svfloat32_t a_sve = svld1(svptrue_b32(), a);
  svfloat32_t b_sve = svld1(svptrue_b32(), b);
  svfloat32_t y_sve = svsub_z(svptrue_b32(), a_sve, b_sve);
  svst1(svptrue_b32(), y, y_sve);
}

inline void vmul_f64(double * __restrict__ y, const double * __restrict__ a,
                     const double * __restrict__ b) __arm_streaming_compatible {
  svfloat64_t a_sve = svld1(svptrue_b64(), a);
  svfloat64_t b_sve = svld1(svptrue_b64(), b);
  svfloat64_t y_sve = svmul_z(svptrue_b64(), a_sve, b_sve);
  svst1(svptrue_b64(), y, y_sve);
}

inline void vmul_f32(float * __restrict__ y, const float * __restrict__ a,
                     const float * __restrict__ b) __arm_streaming_compatible {
  svfloat32_t a_sve = svld1(svptrue_b32(), a);
  svfloat32_t b_sve = svld1(svptrue_b32(), b);
  svfloat32_t y_sve = svmul_z(svptrue_b32(), a_sve, b_sve);
  svst1(svptrue_b32(), y, y_sve);
}

inline void vmad_f64(double * __restrict__ y, const double * __restrict__ a,
                     const double * __restrict__ b) __arm_streaming_compatible {
  svfloat64_t a_sve = svld1(svptrue_b64(), a);
  svfloat64_t b_sve = svld1(svptrue_b64(), b);
  svfloat64_t y_sve = svmad_z(svptrue_b64(), y_sve, a_sve, b_sve);
  svst1(svptrue_b64(), y, y_sve);
}

inline void vmad_f32(float * __restrict__ y, const float * __restrict__ a,
                     const float * __restrict__  b) __arm_streaming_compatible {
  svfloat32_t a_sve = svld1(svptrue_b32(), a);
  svfloat32_t b_sve = svld1(svptrue_b32(), b);
  svfloat32_t y_sve = svmad_z(svptrue_b32(), y_sve, a_sve, b_sve);
  svst1(svptrue_b32(), y, y_sve);
}

inline void vfma_f64(double * __restrict__ y, const double * __restrict__ a,
                     const double * __restrict__ b) __arm_streaming_compatible {
  svfloat64_t a_sve = svld1(svptrue_b64(), a);
  svfloat64_t b_sve = svld1(svptrue_b64(), b);
  svfloat64_t y_sve = svmla_z(svptrue_b64(), a_sve, b_sve, y_sve);
  svst1(svptrue_b64(), y, y_sve);
}

inline void vfma_f32(float * __restrict__ y, const float * __restrict__ a,
                     const float * __restrict__ b) __arm_streaming_compatible {
  svfloat32_t a_sve = svld1(svptrue_b32(), a);
  svfloat32_t b_sve = svld1(svptrue_b32(), b);
  svfloat32_t y_sve = svmla_z(svptrue_b32(), a_sve, b_sve, y_sve);
  svst1(svptrue_b32(), y, y_sve);
}

inline void vmin_f64(double * __restrict__ y, const double * __restrict__ a,
                     const double * __restrict__ b) __arm_streaming_compatible {
  svfloat64_t a_sve = svld1(svptrue_b64(), a);
  svfloat64_t b_sve = svld1(svptrue_b64(), b);
  svfloat64_t y_sve = svmin_z(svptrue_b64(), a_sve, b_sve);
  svst1(svptrue_b64(), y, y_sve);
}

inline void vmin_f32(float * __restrict__ y, const float * __restrict__ a,
                     const float * __restrict__ b) __arm_streaming_compatible {
  svfloat32_t a_sve = svld1(svptrue_b32(), a);
  svfloat32_t b_sve = svld1(svptrue_b32(), b);
  svfloat32_t y_sve = svmin_z(svptrue_b32(), a_sve, b_sve);
  svst1(svptrue_b32(), y, y_sve);
}

inline void vmax_f64(double * __restrict__ y, const double * __restrict__ a,
                     const double * __restrict__ b) __arm_streaming_compatible {
  svfloat64_t a_sve = svld1(svptrue_b64(), a);
  svfloat64_t b_sve = svld1(svptrue_b64(), b);
  svfloat64_t y_sve = svmax_z(svptrue_b64(), a_sve, b_sve);
  svst1(svptrue_b64(), y, y_sve);
}

inline void vmax_f32(float * __restrict__ y, const float * __restrict__ a,
                     const float * __restrict__ b) __arm_streaming_compatible {
  svfloat32_t a_sve = svld1(svptrue_b32(), a);
  svfloat32_t b_sve = svld1(svptrue_b32(), b);
  svfloat32_t y_sve = svmax_z(svptrue_b32(), a_sve, b_sve);
  svst1(svptrue_b32(), y, y_sve);
}

inline void vabs_f64(double * __restrict__ y, const double * __restrict__ x) __arm_streaming_compatible {
  svfloat64_t x_sve = svld1(svptrue_b64(), x);
  svfloat64_t y_sve = svabs_z(svptrue_b64(), x_sve);
  svst1(svptrue_b64(), y, y_sve);
}

inline void vabs_f32(float * __restrict__ y, const float * __restrict__ x) __arm_streaming_compatible {
  svfloat32_t x_sve = svld1(svptrue_b32(), x);
  svfloat32_t y_sve = svabs_z(svptrue_b32(), x_sve);
  svst1(svptrue_b32(), y, y_sve);
}

inline void vneg_f64(double * __restrict__ y, const double * __restrict__ x) __arm_streaming_compatible {
  svfloat64_t x_sve = svld1(svptrue_b64(), x);
  svfloat64_t y_sve = svneg_z(svptrue_b64(), x_sve);
  svst1(svptrue_b64(), y, y_sve);
}

inline void vneg_f32(float * __restrict__ y, const float * __restrict__ x) __arm_streaming_compatible {
  svfloat32_t x_sve = svld1(svptrue_b32(), x);
  svfloat32_t y_sve = svneg_z(svptrue_b32(), x_sve);
  svst1(svptrue_b32(), y, y_sve);
}
