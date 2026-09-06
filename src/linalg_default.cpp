#include "linalg.h"

#ifndef __ARM_FEATURE_SVE
#pragma message("Building linalg_default.cpp.")

#define STR(x) #x
#define EXPAND(x) x
// #define STR_EXPAND(x) STR(x)

#if (defined(USE_OPENMP_SIMD) && defined(USE_OPENMP))
    // #pragma omp for simd schedule(static, (length - 1)/omp_get_num_threads() + 1) nowait
    #define FOR_WHILE(init, condition, update, chunk) \
    _Pragma(STR(omp for simd schedule(static, EXPAND(chunk)) nowait)) \
    for (init; condition; update)
#elif defined(USE_OPENMP)
    // #pragma omp for schedule(static, (length - 1)/omp_get_num_threads() + 1) nowait
    #define FOR_WHILE(init, condition, update, chunk) \
    _Pragma(STR(omp for schedule(static, EXPAND(chunk)) nowait)) \
    for (init; condition; update)
#elif defined(USE_OPENMP_SIMD)
    // #pragma omp simd
    #define FOR_WHILE(init, condition, update, chunk) \
    _Pragma(STR(omp simd)) \
    for (init; condition; update)
#else
    #define FOR_WHILE(init, condition, update, chunk) \
    for (init; condition; update)
#endif

#ifdef USE_OPENMP
#pragma omp declare reduction (+ : std::complex<float> : omp_out += omp_in) initializer(omp_priv = std::complex<float>(0.0f, 0.0f))
#pragma omp declare reduction (+ : std::complex<double> : omp_out += omp_in) initializer(omp_priv = std::complex<double>(0.0, 0.0))
#endif

namespace Linalg {

//Vertor functions
template<typename T>
T vector_sum(T const* const ptr, uint64_t const length, const MPI_Comm comm) {
    #ifdef USE_OPENMP
        static T total_Sum;
        #pragma omp single nowait
        total_Sum = static_cast<T>(0);
        T local_Sum = static_cast<T>(0);
        FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
            local_Sum += ptr[i];
        }
        #pragma omp barrier
        #pragma omp atomic
        total_Sum += local_Sum;
        #pragma omp barrier
    #else
        T total_Sum = static_cast<T>(0);
        for (uint64_t i = 0; i < length; i++) {
            total_Sum += ptr[i];
        }
    #endif //USE_OPENMP
    if (comm == MPI_COMM_NULL) return total_Sum;
    int size;
    MPI_Comm_size(comm, &size);
    if (size > 1) {
        #ifdef USE_OPENMP
            #pragma omp master
        #endif //USE_OPENMP
        MPI_Allreduce(MPI_IN_PLACE, &total_Sum, 1, get_mpi_datatype<T>(), MPI_SUM, comm);
        #ifdef USE_OPENMP
            #pragma omp barrier
        #endif //USE_OPENMP
    }
    return total_Sum;
}
template float vector_sum<float>(float const* const ptr, uint64_t const length, const MPI_Comm comm);
template double vector_sum<double>(double const* const ptr, uint64_t const length, const MPI_Comm comm);

template<typename T>
T vector_norm_square_sum(T const* const ptr, uint64_t const length, const MPI_Comm comm) {
    #ifdef USE_OPENMP
        static T total_Sum;
        #pragma omp single nowait
        total_Sum = static_cast<T>(0);
        T local_Sum = static_cast<T>(0);
        FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
            local_Sum += ptr[i] * ptr[i];
        }
        #pragma omp barrier
        #pragma omp atomic
        total_Sum += local_Sum;
        #pragma omp barrier
    #else
        T total_Sum = static_cast<T>(0);
        for (uint64_t i = 0; i < length; i++) {
            total_Sum += ptr[i] * ptr[i];
        }
    #endif //USE_OPENMP
    if (comm == MPI_COMM_NULL) return total_Sum;
    int size;
    MPI_Comm_size(comm, &size);
    if (size > 1) {
        #ifdef USE_OPENMP
            #pragma omp master
        #endif //USE_OPENMP
        MPI_Allreduce(MPI_IN_PLACE, &total_Sum, 1, get_mpi_datatype<T>(), MPI_SUM, comm);
        #ifdef USE_OPENMP
            #pragma omp barrier
        #endif //USE_OPENMP
    }
    return total_Sum;
}
template float vector_norm_square_sum<float>(float const* const ptr, uint64_t const length, const MPI_Comm comm);
template double vector_norm_square_sum<double>(double const* const ptr, uint64_t const length, const MPI_Comm comm);


template<typename T>
T vector_dot_product(T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length, const MPI_Comm comm) {
    #ifdef USE_OPENMP
        static T total_Sum;
        #pragma omp single nowait
        total_Sum = static_cast<T>(0);
        T local_Sum = static_cast<T>(0);
        FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
            local_Sum += ptr_A[i] * ptr_B[i];
        }
        #pragma omp barrier
        #pragma omp atomic
        total_Sum += local_Sum;
        #pragma omp barrier
    #else
        T total_Sum = static_cast<T>(0);
        for (uint64_t i = 0; i < length; i++) {
            total_Sum += ptr_A[i] * ptr_B[i];
        }
    #endif //USE_OPENMP
    if (comm == MPI_COMM_NULL) return total_Sum;
    int size;
    MPI_Comm_size(comm, &size);
    if (size > 1) {
        #ifdef USE_OPENMP
            #pragma omp master
        #endif //USE_OPENMP
        MPI_Allreduce(MPI_IN_PLACE, &total_Sum, 1, get_mpi_datatype<T>(), MPI_SUM, comm);
        #ifdef USE_OPENMP
            #pragma omp barrier
        #endif //USE_OPENMP
    }
    return total_Sum;
}
template float vector_dot_product<float>(float const* const ptr_A, float const* const ptr_B, const uint64_t length, const MPI_Comm comm);
template double vector_dot_product<double>(double const* const ptr_A, double const* const ptr_B, const uint64_t length, const MPI_Comm comm);

template<typename T>
void vector_norm_square(T* const out_ptr, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1){
        out_ptr[i] *= out_ptr[i];
    }
}
template void vector_norm_square<float>(float* const out_ptr, const uint64_t length);
template void vector_norm_square<double>(double* const out_ptr, const uint64_t length);

template<typename T>
void vector_norm_square(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] * ptr_A[i];
    }
}
template void vector_norm_square<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length);
template void vector_norm_square<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length);

template<typename T>
void vector_norm_square(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A,
                    const uint64_t length, const T alpha) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] * ptr_A[i] * alpha;
    }
}
template void vector_norm_square<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length, const float alpha);
template void vector_norm_square<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length, const double alpha);

template<typename T>
void accumulate_vector_norm_square(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length,
                           const T alpha) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1){
        out_ptr[i] += ptr_A[i] * ptr_A[i] * alpha;
    }
}
template void accumulate_vector_norm_square<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length, const float alpha);
template void accumulate_vector_norm_square<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length, const double alpha);

template<typename T>
void set_value_general(T* const out_ptr, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = num;
    }
}
template void set_value_general<float>(float* const out_ptr, const float num, const uint64_t length);
template void set_value_general<double>(double* const out_ptr, const double num, const uint64_t length);

template<typename T>
void set_value_general(T* const out_ptr, T const* const ptr_A, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i];
    }
    return;
}
template void set_value_general<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length);
template void set_value_general<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length);

template<typename T>
void hadamard_plus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i];
    }
}
template void hadamard_plus_general<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length);
template void hadamard_plus_general<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length);

template<typename T>
void hadamard_minus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] -= ptr_A[i];
    }
}
template void hadamard_minus_general<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length);
template void hadamard_minus_general<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length);

template<typename T>
void hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] *= ptr_A[i];
    }
}
template void hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length);
template void hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length);

template<typename T>
void hadamard_divide_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] /= ptr_A[i];
    }
}
template void hadamard_divide_general<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length);
template void hadamard_divide_general<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length);

template<typename T>
void hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length, T const alpha) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] *= alpha * ptr_A[i];
    }
}
template void hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length, float const alpha);
template void hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length, double const alpha);

template<typename T>
void hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length, T const alpha, T const beta) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = alpha * ptr_A[i] * out_ptr[i] + beta * out_ptr[i];
    }
}
template void hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length, float const alpha, float const beta);
template void hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length, double const alpha, double const beta);

template<typename T>
void hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length, T const alpha, T const beta, T const gamma) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = alpha * ptr_A[i] * out_ptr[i] + beta * out_ptr[i] + gamma;
    }
}
template void hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length, float const alpha, float const beta, float const gamma);
template void hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length, double const alpha, double const beta, double const gamma);

template<typename T>
void hadamard_plus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] + ptr_B[i];
    }
}
template void hadamard_plus_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length);
template void hadamard_plus_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length);

template<typename T>
void hadamard_minus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] - ptr_B[i];
    }
}
template void hadamard_minus_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length);
template void hadamard_minus_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length);

template<typename T>
void hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] * ptr_B[i];
    }
}
template void hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length);
template void hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length);

template<typename T>
void hadamard_divide_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] / ptr_B[i];
    }
}
template void hadamard_divide_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length);
template void hadamard_divide_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length);

template<typename T>
void hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length, T const alpha) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = alpha * ptr_A[i] * ptr_B[i];
    }
}
template void hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length, float const alpha);
template void hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length, double const alpha);

template<typename T>
void hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length, T const alpha, T const beta) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = alpha * ptr_A[i] * ptr_B[i] + beta * out_ptr[i];
    }
}
template void hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length, float const alpha, float const beta);
template void hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length, double const alpha, double const beta);

template<typename T>
void hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length, T const alpha, T const beta, T const gamma) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = alpha * ptr_A[i] * ptr_B[i] + beta * out_ptr[i] + gamma;
    }
}
template void hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length, float const alpha, float const beta, float const gamma);
template void hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length, double const alpha, double const beta, double const gamma);

template<typename T>
void accumulate_hadamard_plus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] + ptr_B[i];
    }
}
template void accumulate_hadamard_plus_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length);
template void accumulate_hadamard_plus_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length);

template<typename T>
void accumulate_hadamard_minus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] - ptr_B[i];
    }
}
template void accumulate_hadamard_minus_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length);
template void accumulate_hadamard_minus_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length);

template<typename T>
void accumulate_hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] * ptr_B[i];
    }
}
template void accumulate_hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length);
template void accumulate_hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length);

template<typename T>
void accumulate_hadamard_divide_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] / ptr_B[i];
    }
}
template void accumulate_hadamard_divide_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length);
template void accumulate_hadamard_divide_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length);

template<typename T>
void accumulate_hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length, T const alpha) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += alpha * ptr_A[i] * ptr_B[i];
    }
}
template void accumulate_hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length, float const alpha);
template void accumulate_hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length, double const alpha);

template<typename T>
void accumulate_hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length, T const alpha, T const beta) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += alpha * ptr_A[i] * ptr_B[i] + beta * out_ptr[i];
    }
}
template void accumulate_hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length, float const alpha, float const beta);
template void accumulate_hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length, double const alpha, double const beta);

template<typename T>
void accumulate_hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length, T const alpha, T const beta, T const gamma) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += alpha * ptr_A[i] * ptr_B[i] + beta * out_ptr[i] + gamma;
    }
}
template void accumulate_hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length, float const alpha, float const beta, float const gamma);
template void accumulate_hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length, double const alpha, double const beta, double const gamma);

template<typename T>
void scalar_plus_general(T* const __restrict__ out_ptr, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += num;
    }
}
template void scalar_plus_general<float>(float* const out_ptr, const float num, const uint64_t length);
template void scalar_plus_general<double>(double* const out_ptr, const double num, const uint64_t length);

template<typename T>
void scalar_minus_general(T* const __restrict__ out_ptr, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] -= num;
    }
}
template void scalar_minus_general<float>(float* const out_ptr, const float num, const uint64_t length);
template void scalar_minus_general<double>(double* const out_ptr, const double num, const uint64_t length);

template<typename T1, typename T2>
void scalar_product_general(T1* const __restrict__ out_ptr, const T2 num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] *= num;
    }
}
template void scalar_product_general<float>(float* const out_ptr, const float num, const uint64_t length);
template void scalar_product_general<double>(double* const out_ptr, const double num, const uint64_t length);

template<typename T>
void scalar_divide_general(T* const __restrict__ out_ptr, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] /= num;
    }
}
template void scalar_divide_general<float>(float* const out_ptr, const float num, const uint64_t length);
template void scalar_divide_general<double>(double* const out_ptr, const double num, const uint64_t length);

template<typename T>
void scalar_product_general(T* const __restrict__ out_ptr, const T num, const uint64_t length,
                                    T const* const __restrict__ ptr_B) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = num * out_ptr[i] + ptr_B[i];
    }
}
template void scalar_product_general<float>(float* const out_ptr, const float num, const uint64_t length,
                                                    float const* const ptr_B);
template void scalar_product_general<double>(double* const out_ptr, const double num, const uint64_t length,
                                                     double const* const ptr_B);

template<typename T1, typename T2>
void scalar_product_general(T1* const __restrict__ out_ptr, const T2 num, const uint64_t length,
                                    T1 const* const __restrict__ ptr_B, T2 const beta) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = num * out_ptr[i] + beta * ptr_B[i];
    }
}
template void scalar_product_general<float>(float* const out_ptr, const float num, const uint64_t length,
                                                    float const* const ptr_B, float const beta);
template void scalar_product_general<double>(double* const out_ptr, const double num, const uint64_t length,
                                                     double const* const ptr_B, double const beta);

template<typename T>
void scalar_product_general(T* const __restrict__ out_ptr, const T num, const uint64_t length,
                                    T const* const __restrict__ ptr_B, T const beta, T const gamma) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = num * out_ptr[i] + beta * ptr_B[i] + gamma;
    }
}
template void scalar_product_general<float>(float* const out_ptr, const float num, const uint64_t length,
                                            float const* const ptr_B, float const beta, float const gamma);
template void scalar_product_general<double>(double* const out_ptr, const double num, const uint64_t length,
                                                     double const* const ptr_B, double const beta, double const gamma);

template<typename T>
void scalar_plus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] + num;
    }
}
template void scalar_plus_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length);
template void scalar_plus_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length);

template<typename T>
void scalar_minus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] - num;
    }
}
template void scalar_minus_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length);
template void scalar_minus_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length);

template<typename T1, typename T2>
void scalar_product_general(T1* const __restrict__ out_ptr, T1 const* const __restrict__ ptr_A, const T2 num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] * num;
    }
}
template void scalar_product_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length);
template void scalar_product_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length);

template<typename T>
void scalar_divide_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] / num;
    }
}
template void scalar_divide_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length);
template void scalar_divide_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length);

template<typename T>
void scalar_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length,
                                    T const* const __restrict__ ptr_B) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] * num + ptr_B[i];
    }
}
template void scalar_product_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length,
                                                    float const* const ptr_B);
template void scalar_product_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length,
                                                     double const* const ptr_B);

template<typename T>
void scalar_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length,
                                    T const* const __restrict__ ptr_B, T const beta) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] * num + ptr_B[i] * beta;
    }
}
template void scalar_product_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length,
                                                    float const* const ptr_B, float const beta);
template void scalar_product_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length,
                                                     double const* const ptr_B, double const beta);

template<typename T>
void scalar_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length,
                                    T const* const __restrict__ ptr_B, T const beta, T const gamma) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] * num + ptr_B[i] * beta + gamma;
    }
}
template void scalar_product_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length,
                                                    float const* const ptr_B, float const beta, float const gamma);
template void scalar_product_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length,
                                                     double const* const ptr_B, double const beta, double const gamma);

template<typename T>
void accumulate_scalar_plus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] + num;
    }
}
template void accumulate_scalar_plus_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length);
template void accumulate_scalar_plus_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length);

template<typename T>
void accumulate_scalar_minus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] - num;
    }
}
template void accumulate_scalar_minus_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length);
template void accumulate_scalar_minus_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length);

template<typename T1, typename T2>
void accumulate_scalar_product_general(T1* const __restrict__ out_ptr, T1 const* const __restrict__ ptr_A, const T2 num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] * num;
    }
}
template void accumulate_scalar_product_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length);
template void accumulate_scalar_product_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length);

template<typename T>
void accumulate_scalar_divide_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] / num;
    }
}
template void accumulate_scalar_divide_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length);
template void accumulate_scalar_divide_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length);

template<typename T>
void accumulate_scalar_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length,
                                    T const* const __restrict__ ptr_B) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] * num + ptr_B[i];
    }
}
template void accumulate_scalar_product_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length,
                                                    float const* const ptr_B);
template void accumulate_scalar_product_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length,
                                                     double const* const ptr_B);

template<typename T>
void accumulate_scalar_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length,
                                    T const* const __restrict__ ptr_B, T const beta) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] * num + ptr_B[i] * beta;
    }
}
template void accumulate_scalar_product_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length,
                                                    float const* const ptr_B, float const beta);
template void accumulate_scalar_product_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length,
                                                     double const* const ptr_B, double const beta);

template<typename T>
void accumulate_scalar_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length,
                                    T const* const __restrict__ ptr_B, T const beta, T const gamma) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] * num + ptr_B[i] * beta + gamma;
    }
}
template void accumulate_scalar_product_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length,
                                                    float const* const ptr_B, float const beta, float const gamma);
template void accumulate_scalar_product_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length,
                                                     double const* const ptr_B, double const beta, double const gamma);

} // namespace Linalg

#undef FOR_WHILE

#endif //__ARM_FEATURE_SVE
