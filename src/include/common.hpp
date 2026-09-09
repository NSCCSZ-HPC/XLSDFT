#pragma once

template <typename T>
inline void mov_scalar(T* __restrict__ y, const T* __restrict__ x) __arm_streaming_compatible {
  *y = *x;
}

template <typename T>
inline void add_scalar(T* __restrict__ y, const T* __restrict__ a,
                       const T* __restrict__ b) __arm_streaming_compatible {
  *y = *a + *b;
}

template <typename T>
inline void sub_scalar(T* __restrict__ y, const T* __restrict__ a,
                       const T* __restrict__ b) __arm_streaming_compatible {
  *y = *a - *b;
}

template <typename T>
inline void mul_scalar(T* __restrict__ y, const T* __restrict__ a,
                       const T* __restrict__ b) __arm_streaming_compatible {
  *y = *a * *b;
}

template <typename T>
inline void div_scalar(T* __restrict__ y, const T* __restrict__ a,
                       const T* __restrict__ b) __arm_streaming_compatible {
  *y = *a / *b;
}

template <typename T>
inline void mad_scalar(T* __restrict__ y, const T* __restrict__ a, const T* __restrict__ b,
                       const T* __restrict__ c) __arm_streaming_compatible {
  *y = *a * *b + *c;
}

template <typename T>
inline void min_scalar(T* __restrict__ y, const T* __restrict__ a,
                       const T* __restrict__ b) __arm_streaming_compatible {
  *y = *a < *b ? *a : *b;
}

template <typename T>
inline void max_scalar(T* __restrict__ y, const T* __restrict__ a,
                       const T* __restrict__ b) __arm_streaming_compatible {
  *y = *a > *b ? *a : *b;
}

template <typename T>
inline void abs_scalar(T* __restrict__ y, const T* __restrict__ x) __arm_streaming_compatible {
  *y = *x < 0 ? -*x : *x;
}

template <typename T>
inline void neg_scalar(T* __restrict__ y, const T* __restrict__ x) __arm_streaming_compatible {
  *y = -*x;
}

template <typename T>
inline void relu_scalar(T* __restrict__ y, const T* __restrict__ x) __arm_streaming_compatible {
  *y = *x > 0 ? *x : 0;
}
