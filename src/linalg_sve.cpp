
#include "linalg.h"

#ifdef __ARM_FEATURE_SVE
#pragma message("Building linalg_sve.cpp.")
#include <arm_sve.h>

#define STR(x) #x
#define EXPAND(x) x

#if defined(USE_OPENMP)
    #define FOR_WHILE(init, condition, update, chunk) \
    _Pragma(STR(omp for schedule(static, EXPAND(chunk)) nowait)) \
    for (init; condition; update)
#else
    #define FOR_WHILE(init, condition, update, chunk) \
    for (init; condition; update)
#endif

#define SVE_TARGET __attribute__((target("sve")))

namespace Linalg {

//Vertor functions
template<>
SVE_TARGET
float vector_sum<float>(float const* const ptr, uint64_t const length, const MPI_Comm comm)
{
    #ifdef USE_OPENMP
        #pragma omp barrier
        static float total_Sum;
        #pragma omp single nowait
        total_Sum = 0.0f;
        svfloat32_t vsum = svdup_f32(0.0f);
        const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
            svbool_t pg = svwhilelt_b32(i, length);
            svfloat32_t v = svld1(pg, ptr + i);
            vsum = svadd_m(pg, vsum, v);
        }
        float local_Sum = svaddv_f32(svptrue_b32(), vsum);
        #pragma omp barrier
        #pragma omp atomic
        total_Sum += local_Sum;
        #pragma omp barrier
    #else
        svfloat32_t vsum = svdup_f32(0.0f);
        for (uint64_t i = 0; i < length; i += svcntw()) {
            svbool_t pg = svwhilelt_b32(i, length);
            svfloat32_t v = svld1(pg, ptr + i);
            vsum = svadd_m(pg, vsum, v);
        }
        float total_Sum = svaddv_f32(svptrue_b32(), vsum);
    #endif

    if (comm == MPI_COMM_NULL) return total_Sum;

    int size;
    MPI_Comm_size(comm, &size);
    if (size > 1) {
    #ifdef USE_OPENMP
            #pragma omp master
            MPI_Allreduce(MPI_IN_PLACE, &total_Sum, 1, MPI_FLOAT, MPI_SUM, comm);
            #pragma omp barrier
    #else
            MPI_Allreduce(MPI_IN_PLACE, &total_Sum, 1, MPI_FLOAT, MPI_SUM, comm);
    #endif
    }
    return total_Sum;
}

template<>
SVE_TARGET
double vector_sum<double>(double const* const ptr, uint64_t const length, const MPI_Comm comm)
{
    #ifdef USE_OPENMP
        #pragma omp barrier
        static double total_Sum;
        #pragma omp single nowait
        total_Sum = 0.0;
        svfloat64_t vsum = svdup_f64(0.0);
        const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
            svbool_t pg = svwhilelt_b64(i, length);
            svfloat64_t v = svld1(pg, ptr + i);
            vsum = svadd_m(pg, vsum, v);
        }
        double local_Sum = svaddv_f64(svptrue_b64(), vsum);
        #pragma omp barrier
        #pragma omp atomic
        total_Sum += local_Sum;
        #pragma omp barrier
    #else
        svfloat64_t vsum = svdup_f64(0.0);
        for (uint64_t i = 0; i < length; i += svcntd()) {
            svbool_t pg = svwhilelt_b64(i, length);
            svfloat64_t v = svld1(pg, ptr + i);
            vsum = svadd_m(pg, vsum, v);
        }
        double total_Sum = svaddv_f64(svptrue_b64(), vsum);
    #endif

    if (comm == MPI_COMM_NULL) return total_Sum;

    int size;
    MPI_Comm_size(comm, &size);
    if (size > 1) {
        #ifdef USE_OPENMP
                #pragma omp master
                MPI_Allreduce(MPI_IN_PLACE, &total_Sum, 1, MPI_DOUBLE, MPI_SUM, comm);
                #pragma omp barrier
        #else
                MPI_Allreduce(MPI_IN_PLACE, &total_Sum, 1, MPI_DOUBLE, MPI_SUM, comm);
        #endif
    }
    return total_Sum;
}

template<>
SVE_TARGET
float vector_norm_square_sum<float>(float const* const ptr, uint64_t const length, const MPI_Comm comm) {
    #ifdef USE_OPENMP
        #pragma omp barrier
        static float total_Sum;
        #pragma omp single nowait
        total_Sum = 0.0f;
        svfloat32_t vSum = svdup_f32(0.0f);
        const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
            svbool_t pg = svwhilelt_b32(i, length);
            svfloat32_t v = svld1(pg, ptr + i);
            vSum = svmla_m(pg, vSum, v, v);
        }
        float local_Sum = svaddv_f32(svptrue_b32(), vSum);
        #pragma omp barrier
        #pragma omp atomic
        total_Sum += local_Sum;
        #pragma omp barrier
    #else
        svfloat32_t vSum = svdup_f32(0.0f);
        for (uint64_t i = 0; i < length; i += svcntw()) {
            svbool_t pg = svwhilelt_b32(i, length);
            svfloat32_t v = svld1(pg, ptr + i);
            vSum = svmla_m(pg, vSum, v, v);
        }
        float total_Sum = svaddv_f32(svptrue_b32(), vSum);
    #endif // USE_OPENMP

    if (comm == MPI_COMM_NULL) return total_Sum;

    int size;
    MPI_Comm_size(comm, &size);
    if (size > 1) {
        #ifdef USE_OPENMP
            #pragma omp master
            MPI_Allreduce(MPI_IN_PLACE, &total_Sum, 1, MPI_FLOAT, MPI_SUM, comm);
            #pragma omp barrier
        #else
            MPI_Allreduce(MPI_IN_PLACE, &total_Sum, 1, MPI_FLOAT, MPI_SUM, comm);
        #endif
    }
    return total_Sum;
}

template<>
SVE_TARGET
double vector_norm_square_sum<double>(double const* const ptr, uint64_t const length, const MPI_Comm comm) {
    #ifdef USE_OPENMP
        #pragma omp barrier
        static double total_Sum;
        #pragma omp single nowait
        total_Sum = 0.0;
        svfloat64_t vSum = svdup_f64(0.0);
        const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
            svbool_t pg = svwhilelt_b64(i, length);
            svfloat64_t v = svld1(pg, ptr + i);
            vSum = svmla_m(pg, vSum, v, v);
        }
        double local_Sum = svaddv_f64(svptrue_b64(), vSum);
        #pragma omp barrier
        #pragma omp atomic
        total_Sum += local_Sum;
        #pragma omp barrier
    #else
        svfloat64_t vSum = svdup_f64(0.0);
        for (uint64_t i = 0; i < length; i += svcntd()) {
            svbool_t pg = svwhilelt_b64(i, length);
            svfloat64_t v = svld1(pg, ptr + i);
            vSum = svmla_m(pg, vSum, v, v);
        }
        double total_Sum = svaddv_f64(svptrue_b64(), vSum);
    #endif // USE_OPENMP

    if (comm == MPI_COMM_NULL) return total_Sum;

    int size;
    MPI_Comm_size(comm, &size);
    if (size > 1) {
        #ifdef USE_OPENMP
            #pragma omp master
            MPI_Allreduce(MPI_IN_PLACE, &total_Sum, 1, MPI_DOUBLE, MPI_SUM, comm);
            #pragma omp barrier
        #else
            MPI_Allreduce(MPI_IN_PLACE, &total_Sum, 1, MPI_DOUBLE, MPI_SUM, comm);
        #endif
    }
    return total_Sum;
}

template<>
SVE_TARGET
float vector_dot_product<float>(float const* const ptr_A, float const* const ptr_B, uint64_t const length, const MPI_Comm comm) {
    #ifdef USE_OPENMP
        #pragma omp barrier
        static float total_Sum;
        #pragma omp single nowait
        total_Sum = 0.0f;
        svfloat32_t vSum = svdup_f32(0.0f);
        const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
            svbool_t pg = svwhilelt_b32(i, length);
            svfloat32_t vA = svld1(pg, ptr_A + i);
            svfloat32_t vB = svld1(pg, ptr_B + i);
            vSum = svmla_m(pg, vSum, vA, vB);
        }
        float local_Sum = svaddv_f32(svptrue_b32(), vSum);
        #pragma omp barrier
        #pragma omp atomic
        total_Sum += local_Sum;
        #pragma omp barrier
    #else
        svfloat32_t vSum = svdup_f32(0.0f);
        for (uint64_t i = 0; i < length; i += svcntw()) {
            svbool_t pg = svwhilelt_b32(i, length);
            svfloat32_t vA = svld1(pg, ptr_A + i);
            svfloat32_t vB = svld1(pg, ptr_B + i);
            vSum = svmla_m(pg, vSum, vA, vB);
        }
        float total_Sum = svaddv_f32(svptrue_b32(), vSum);
    #endif // USE_OPENMP

    if (comm == MPI_COMM_NULL) return total_Sum;

    int size;
    MPI_Comm_size(comm, &size);
    if (size > 1) {
        #ifdef USE_OPENMP
            #pragma omp master
            MPI_Allreduce(MPI_IN_PLACE, &total_Sum, 1, MPI_FLOAT, MPI_SUM, comm);
            #pragma omp barrier
        #else
            MPI_Allreduce(MPI_IN_PLACE, &total_Sum, 1, MPI_FLOAT, MPI_SUM, comm);
        #endif
    }
    return total_Sum;
}

template<>
SVE_TARGET
double vector_dot_product<double>(double const* const ptr_A, double const* const ptr_B, uint64_t const length, const MPI_Comm comm) {
    #ifdef USE_OPENMP
        #pragma omp barrier
        static double total_Sum;
        #pragma omp single nowait
        total_Sum = 0.0;
        svfloat64_t vSum = svdup_f64(0.0);
        const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
            svbool_t pg = svwhilelt_b64(i, length);
            svfloat64_t vA = svld1(pg, ptr_A + i);
            svfloat64_t vB = svld1(pg, ptr_B + i);
            vSum = svmla_m(pg, vSum, vA, vB);
        }
        double local_Sum = svaddv_f64(svptrue_b64(), vSum);
        #pragma omp barrier
        #pragma omp atomic
        total_Sum += local_Sum;
        #pragma omp barrier
    #else
        svfloat64_t vSum = svdup_f64(0.0);
        for (uint64_t i = 0; i < length; i += svcntd()) {
            svbool_t pg = svwhilelt_b64(i, length);
            svfloat64_t vA = svld1(pg, ptr_A + i);
            svfloat64_t vB = svld1(pg, ptr_B + i);
            vSum = svmla_m(pg, vSum, vA, vB);
        }
        double total_Sum = svaddv_f64(svptrue_b64(), vSum);
    #endif // USE_OPENMP

    if (comm == MPI_COMM_NULL) return total_Sum;

    int size;
    MPI_Comm_size(comm, &size);
    if (size > 1) {
        #ifdef USE_OPENMP
            #pragma omp master
            MPI_Allreduce(MPI_IN_PLACE, &total_Sum, 1, MPI_DOUBLE, MPI_SUM, comm);
            #pragma omp barrier
        #else
            MPI_Allreduce(MPI_IN_PLACE, &total_Sum, 1, MPI_DOUBLE, MPI_SUM, comm);
        #endif
    }
    return total_Sum;
}

template<>
SVE_TARGET
void vector_norm_square<float>(float* const out_ptr, const uint64_t length) {
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        vOut = svmul_m(pg, vOut, vOut);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void vector_norm_square<double>(double* const out_ptr, const uint64_t length) {
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        vOut = svmul_m(pg, vOut, vOut);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void vector_norm_square<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A, const uint64_t length) {
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vRes = svmul_m(pg, vA, vA);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void vector_norm_square<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A, const uint64_t length) {
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vRes = svmul_m(pg, vA, vA);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void vector_norm_square<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A, const uint64_t length, const float alpha) {
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vSq = svmul_m(pg, vA, vA);
        svfloat32_t vRes = svmul_m(pg, vSq, alpha);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void vector_norm_square<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A, const uint64_t length, const double alpha) {
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vSq = svmul_m(pg, vA, vA);
        svfloat64_t vRes = svmul_m(pg, vSq, alpha);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_vector_norm_square<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                            const uint64_t length, const float alpha) {
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        svfloat32_t vA   = svld1(pg, ptr_A + i);
        svfloat32_t vSq  = svmul_m(pg, vA, vA);
        svfloat32_t vRes = svmla_m(pg, vOut, vSq, alpha);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_vector_norm_square<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                            const uint64_t length, const double alpha) {
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        svfloat64_t vA   = svld1(pg, ptr_A + i);
        svfloat64_t vSq  = svmul_m(pg, vA, vA);
        svfloat64_t vRes = svmla_m(pg, vOut, vSq, alpha);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void set_value_general<float>(float* const __restrict__ out_ptr, const float num, const uint64_t length) {
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vnum = svdup_f32(num);
        svst1(pg, out_ptr + i, vnum);
    }
    return;
}

template<>
SVE_TARGET
void set_value_general<double>(double* const __restrict__ out_ptr, const double num, const uint64_t length) {
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vnum = svdup_f64(num);
        svst1(pg, out_ptr + i, vnum);
    }
    return;
}

template<>
SVE_TARGET
void set_value_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A, const uint64_t length) {
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svst1(pg, out_ptr + i, vA);
    }
    return;
}

template<>
SVE_TARGET
void set_value_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A, const uint64_t length) {
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svst1(pg, out_ptr + i, vA);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_plus_general<float>(float* const __restrict__ out_ptr,
                                   float const* const __restrict__ ptr_A,
                                   const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vRes = svadd_m(pg, vOut, vA);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_plus_general<double>(double* const __restrict__ out_ptr,
                                   double const* const __restrict__ ptr_A,
                                   const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vRes = svadd_m(pg, vOut, vA);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_minus_general<float>(float* const __restrict__ out_ptr,
                                   float const* const __restrict__ ptr_A,
                                   const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vRes = svsub_m(pg, vOut, vA);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_minus_general<double>(double* const __restrict__ out_ptr,
                                   double const* const __restrict__ ptr_A,
                                   const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vRes = svsub_m(pg, vOut, vA);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_product_general<float>(float* const __restrict__ out_ptr,
                                   float const* const __restrict__ ptr_A,
                                   const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vRes = svmul_m(pg, vOut, vA);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_product_general<double>(double* const __restrict__ out_ptr,
                                   double const* const __restrict__ ptr_A,
                                   const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vRes = svmul_m(pg, vOut, vA);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_divide_general<float>(float* const __restrict__ out_ptr,
                                   float const* const __restrict__ ptr_A,
                                   const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vRes = svdiv_m(pg, vOut, vA);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_divide_general<double>(double* const __restrict__ out_ptr,
                                   double const* const __restrict__ ptr_A,
                                   const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vRes = svdiv_m(pg, vOut, vA);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_product_general<float>(float* const __restrict__ out_ptr,
                                   float const* const __restrict__ ptr_A,
                                   const uint64_t length, float const alpha)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vRes = svmul_m(pg, vOut, vA);
        vRes = svmul_m(pg, vRes, alpha);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_product_general<double>(double* const __restrict__ out_ptr,
                                   double const* const __restrict__ ptr_A,
                                   const uint64_t length, double const alpha)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vRes = svmul_m(pg, vOut, vA);
        vRes = svmul_m(pg, vRes, alpha);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_product_general<float>(float* const __restrict__ out_ptr,
                                   float const* const __restrict__ ptr_A,
                                   const uint64_t length, float const alpha, float const beta)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vRes1 = svmul_m(pg, vOut, vA);
        svfloat32_t vRes = svmla_m(pg, svmul_m(pg, vOut, beta), vRes1, alpha);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_product_general<double>(double* const __restrict__ out_ptr,
                                   double const* const __restrict__ ptr_A,
                                   const uint64_t length, double const alpha, double const beta)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vRes1 = svmul_m(pg, vOut, vA);
        svfloat64_t vRes = svmla_m(pg, svmul_m(pg, vOut, beta), vRes1, alpha);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_product_general<float>(float* const __restrict__ out_ptr,
                                   float const* const __restrict__ ptr_A,
                                   const uint64_t length, float const alpha, float const beta, float const gamma)
{
    svfloat32_t vGamma = svdup_f32(gamma);
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vRes1 = svmul_m(pg, vOut, vA);
        svfloat32_t vRes = svmla_m(pg, svmla_m(pg, vGamma, vOut, beta), vRes1, alpha);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_product_general<double>(double* const __restrict__ out_ptr,
                                    double const* const __restrict__ ptr_A,
                                    const uint64_t length, double const alpha, double const beta, double const gamma)
{
    svfloat64_t vGamma = svdup_f64(gamma);
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vRes1 = svmul_m(pg, vOut, vA);
        svfloat64_t vRes = svmla_m(pg, svmla_m(pg, vGamma, vOut, beta), vRes1, alpha);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_plus_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A, 
                                    float const* const __restrict__ ptr_B, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vRes = svadd_m(pg, vA, vB);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_plus_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A, 
                                    double const* const __restrict__ ptr_B, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vRes = svadd_m(pg, vA, vB);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_minus_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                    float const* const __restrict__ ptr_B, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vRes = svsub_m(pg, vA, vB);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_minus_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                    double const* const __restrict__ ptr_B, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vRes = svsub_m(pg, vA, vB);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_product_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                        float const* const __restrict__ ptr_B, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vRes = svmul_m(pg, vA, vB);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_product_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                        double const* const __restrict__ ptr_B, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vRes = svmul_m(pg, vA, vB);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_divide_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                    float const* const __restrict__ ptr_B, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vRes = svdiv_m(pg, vA, vB);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_divide_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                    double const* const __restrict__ ptr_B, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vRes = svdiv_m(pg, vA, vB);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_product_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                    float const* const __restrict__ ptr_B, const uint64_t length,
                                    float const alpha)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vRes = svmul_m(pg, vA, vB);
        vRes = svmul_m(pg, vRes, alpha);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_product_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                    double const* const __restrict__ ptr_B, const uint64_t length,
                                    double const alpha)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vRes = svmul_m(pg, vA, vB);
        vRes = svmul_m(pg, vRes, alpha);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_product_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                    float const* const __restrict__ ptr_B, const uint64_t length,
                                    float const alpha, float const beta)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        svfloat32_t vAB = svmul_m(pg, vA, vB);
        svfloat32_t vRes = svmla_m(pg, svmul_m(pg, vAB, alpha), vOut, beta);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_product_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                    double const* const __restrict__ ptr_B, const uint64_t length,
                                    double const alpha, double const beta)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        svfloat64_t vAB = svmul_m(pg, vA, vB);
        svfloat64_t vRes = svmla_m(pg, svmul_m(pg, vAB, alpha), vOut, beta);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_product_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                    float const* const __restrict__ ptr_B, const uint64_t length,
                                    float const alpha, float const beta, float const gamma)
{
    svfloat32_t vGamma = svdup_f32(gamma);
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        svfloat32_t vAB = svmul_m(pg, vA, vB);
        svfloat32_t vRes = svmla_m(pg, svmla_m(pg, vGamma, vAB, alpha), vOut, beta);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void hadamard_product_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                    double const* const __restrict__ ptr_B, const uint64_t length,
                                    double const alpha, double const beta, double const gamma)
{
    svfloat64_t vGamma = svdup_f64(gamma);
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        svfloat64_t vAB = svmul_m(pg, vA, vB);
        svfloat64_t vRes = svmla_m(pg, svmla_m(pg, vGamma, vAB, alpha), vOut, beta);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_hadamard_plus_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                            float const* const __restrict__ ptr_B, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vRes = svadd_m(pg, vA, vB);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        vRes = svadd_m(pg, vRes, vOut);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_hadamard_plus_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                            double const* const __restrict__ ptr_B, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vRes = svadd_m(pg, vA, vB);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        vRes = svadd_m(pg, vRes, vOut);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_hadamard_minus_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                            float const* const __restrict__ ptr_B, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vRes = svsub_m(pg, vA, vB);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        vRes = svadd_m(pg, vRes, vOut);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_hadamard_minus_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                            double const* const __restrict__ ptr_B, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vRes = svsub_m(pg, vA, vB);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        vRes = svadd_m(pg, vRes, vOut);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_hadamard_product_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                                float const* const __restrict__ ptr_B, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        svfloat32_t vRes = svmla_m(pg, vOut, vA, vB);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_hadamard_product_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                                double const* const __restrict__ ptr_B, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        svfloat64_t vRes = svmla_m(pg, vOut, vA, vB);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_hadamard_divide_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                                float const* const __restrict__ ptr_B, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vRes = svdiv_m(pg, vA, vB);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        vRes = svadd_m(pg, vRes, vOut);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_hadamard_divide_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                                double const* const __restrict__ ptr_B, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vRes = svdiv_m(pg, vA, vB);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        vRes = svadd_m(pg, vRes, vOut);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_hadamard_product_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                                float const* const __restrict__ ptr_B, const uint64_t length,
                                                float const alpha)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        svfloat32_t vAB = svmul_m(pg, vA, vB);
        svfloat32_t vRes = svmla_m(pg, vOut, vAB, alpha);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_hadamard_product_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                                double const* const __restrict__ ptr_B, const uint64_t length,
                                                double const alpha)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        svfloat64_t vAB = svmul_m(pg, vA, vB);
        svfloat64_t vRes = svmla_m(pg, vOut, vAB, alpha);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_hadamard_product_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                                float const* const __restrict__ ptr_B, const uint64_t length,
                                                float const alpha, float const beta)
{
    float const beta_p1 = beta + 1.0f;
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        svfloat32_t vAB = svmul_m(pg, vA, vB);
        svfloat32_t vRes = svmla_m(pg, svmul_m(pg, vOut, beta_p1), vAB, alpha);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_hadamard_product_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                                double const* const __restrict__ ptr_B, const uint64_t length,
                                                double const alpha, double const beta)
{
    const double beta_p1 = beta + 1.0;
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        svfloat64_t vAB = svmul_m(pg, vA, vB);
        svfloat64_t vRes = svmla_m(pg, svmul_m(pg, vOut, beta_p1), vAB, alpha);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_hadamard_product_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                                float const* const __restrict__ ptr_B, const uint64_t length,
                                                float const alpha, float const beta, float const gamma)
{
    float const beta_p1 = beta + 1.0f;
    svfloat32_t vGamma = svdup_f32(gamma);
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        svfloat32_t vAB = svmul_m(pg, vA, vB);
        svfloat32_t vRes = svmla_m(pg, svmla_m(pg, vGamma, vOut, beta_p1), vAB, alpha);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_hadamard_product_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                                double const* const __restrict__ ptr_B, const uint64_t length,
                                                double const alpha, double const beta, double const gamma)
{
    double const beta_p1 = beta + 1.0;
    svfloat64_t vGamma = svdup_f64(gamma);
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        svfloat64_t vAB = svmul_m(pg, vA, vB);
        svfloat64_t vRes = svmla_m(pg, svmla_m(pg, vGamma, vOut, beta_p1), vAB, alpha);
        svst1(pg, out_ptr + i, vRes);
    }
    return;
}

template<>
SVE_TARGET
void scalar_plus_general<float>(float* const __restrict__ out_ptr, const float num, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        vOut = svadd_m(pg, vOut, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_plus_general<double>(double* const __restrict__ out_ptr, const double num, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        vOut = svadd_m(pg, vOut, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_minus_general<float>(float* const __restrict__ out_ptr, const float num, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        vOut = svsub_m(pg, vOut, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_minus_general<double>(double* const __restrict__ out_ptr, const double num, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        vOut = svsub_m(pg, vOut, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_product_general<float>(float* const __restrict__ out_ptr, const float num, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        vOut = svmul_m(pg, vOut, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_product_general<double>(double* const __restrict__ out_ptr, const double num, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        vOut = svmul_m(pg, vOut, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_divide_general<float>(float* const __restrict__ out_ptr, const float num, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        vOut = svdiv_m(pg, vOut, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_divide_general<double>(double* const __restrict__ out_ptr, const double num, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        vOut = svdiv_m(pg, vOut, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_product_general<float>(float* const __restrict__ out_ptr, const float num, const uint64_t length,
                                    float const* const __restrict__ ptr_B)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        vOut = svmla_m(pg, vB, vOut, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_product_general<double>(double* const __restrict__ out_ptr, const double num, const uint64_t length,
                                    double const* const __restrict__ ptr_B)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        vOut = svmla_m(pg, vB, vOut, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_product_general<float>(float* const __restrict__ out_ptr, const float num, const uint64_t length,
                                    float const* const __restrict__ ptr_B, float const beta)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        vOut = svmla_m(pg, svmul_m(pg, vB, beta), vOut, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_product_general<double>(double* const __restrict__ out_ptr, const double num, const uint64_t length,
                                    double const* const __restrict__ ptr_B, double const beta)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        vOut = svmla_m(pg, svmul_m(pg, vB, beta), vOut, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_product_general<float>(float* const __restrict__ out_ptr, const float num, const uint64_t length,
                                    float const* const __restrict__ ptr_B, float const beta, float const gamma)
{
    svfloat32_t vGamma = svdup_f32(gamma);
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        vOut = svmla_m(pg, svmla_m(pg, vGamma, vB, beta), vOut, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_product_general<double>(double* const __restrict__ out_ptr, const double num, const uint64_t length,
                                    double const* const __restrict__ ptr_B, double const beta, double const gamma)
{
    svfloat64_t vGamma = svdup_f64(gamma);
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        vOut = svmla_m(pg, svmla_m(pg, vGamma, vB, beta), vOut, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_plus_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                const float num, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vOut = svadd_m(pg, vA, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_plus_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                const double num, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vOut = svadd_m(pg, vA, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_minus_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                const float num, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vOut = svsub_m(pg, vA, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_minus_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                const double num, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vOut = svsub_m(pg, vA, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_product_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                const float num, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vOut = svmul_m(pg, vA, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_product_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                const double num, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vOut = svmul_m(pg, vA, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_divide_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                const float num, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vOut = svdiv_m(pg, vA, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_divide_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                const double num, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vOut = svdiv_m(pg, vA, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_product_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                    const float num, const uint64_t length,
                                    float const* const __restrict__ ptr_B)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vOut = svmla_m(pg, vB, vA, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_product_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                    const double num, const uint64_t length,
                                    double const* const __restrict__ ptr_B)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vOut = svmla_m(pg, vB, vA, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_product_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                    const float num, const uint64_t length,
                                    float const* const __restrict__ ptr_B, float const beta)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vOut = svmla_m(pg, svmul_m(pg, vB, beta), vA, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_product_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                    const double num, const uint64_t length,
                                    double const* const __restrict__ ptr_B, double const beta)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vOut = svmla_m(pg, svmul_m(pg, vB, beta), vA, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_product_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                    const float num, const uint64_t length,
                                    float const* const __restrict__ ptr_B, float const beta,
                                    float const gamma)
{
    svfloat32_t vGamma = svdup_f32(gamma);
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vOut = svmla_m(pg, svmla_m(pg, vGamma, vB, beta), vA, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void scalar_product_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                    const double num, const uint64_t length,
                                    double const* const __restrict__ ptr_B, double const beta,
                                    double const gamma)
{
    svfloat64_t vGamma = svdup_f64(gamma);
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vOut = svmla_m(pg, svmla_m(pg, vGamma, vB, beta), vA, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_scalar_plus_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                            const float num, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        vA = svadd_m(pg, vA, num);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        vOut = svadd_m(pg, vOut, vA);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_scalar_plus_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                            const double num, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        vA = svadd_m(pg, vA, num);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        vOut = svadd_m(pg, vOut, vA);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_scalar_minus_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                            const float num, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        vA = svsub_m(pg, vA, num);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        vOut = svadd_m(pg, vOut, vA);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_scalar_minus_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                            const double num, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        vA = svsub_m(pg, vA, num);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        vOut = svadd_m(pg, vOut, vA);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_scalar_product_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                            const float num, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        vOut = svmla_m(pg, vOut, vA, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_scalar_product_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                            const double num, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        vOut = svmla_m(pg, vOut, vA, num);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_scalar_divide_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                            const float num, const uint64_t length)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        vA = svdiv_m(pg, vA, num);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        vOut = svadd_m(pg, vOut, vA);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_scalar_divide_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                            const double num, const uint64_t length)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        vA = svdiv_m(pg, vA, num);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        vOut = svadd_m(pg, vOut, vA);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_scalar_product_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                            const float num, const uint64_t length, float const* const __restrict__ ptr_B)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        vOut = svmla_m(pg, vOut, vA, num);
        vOut = svadd_m(pg, vOut, vB);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_scalar_product_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                                const double num, const uint64_t length, double const* const __restrict__ ptr_B)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        vOut = svmla_m(pg, vOut, vA, num);
        vOut = svadd_m(pg, vOut, vB);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_scalar_product_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                                const float num, const uint64_t length,
                                                float const* const __restrict__ ptr_B, float const beta)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        vOut = svmla_m(pg, vOut, vA, num);
        vOut = svmla_m(pg, vOut, vB, beta);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_scalar_product_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                                const double num, const uint64_t length,
                                                double const* const __restrict__ ptr_B, double const beta)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        vOut = svmla_m(pg, vOut, vA, num);
        vOut = svmla_m(pg, vOut, vB, beta);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_scalar_product_general<float>(float* const __restrict__ out_ptr, float const* const __restrict__ ptr_A,
                                                const float num, const uint64_t length,
                                                float const* const __restrict__ ptr_B, float const beta,
                                                float const gamma)
{
    const uint64_t svcnt = svcntw();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b32(i, length);
        svfloat32_t vA = svld1(pg, ptr_A + i);
        svfloat32_t vB = svld1(pg, ptr_B + i);
        svfloat32_t vOut = svld1(pg, out_ptr + i);
        vOut = svmla_m(pg, vOut, vA, num);
        vOut = svmla_m(pg, vOut, vB, beta);
        vOut = svadd_m(pg, vOut, gamma);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}

template<>
SVE_TARGET
void accumulate_scalar_product_general<double>(double* const __restrict__ out_ptr, double const* const __restrict__ ptr_A,
                                                const double num, const uint64_t length,
                                                double const* const __restrict__ ptr_B, double const beta,
                                                double const gamma)
{
    const uint64_t svcnt = svcntd();
    FOR_WHILE (uint64_t i = 0, i < length, i += svcnt, (length - 1)/(omp_get_num_threads() * svcnt) + 1) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vA = svld1(pg, ptr_A + i);
        svfloat64_t vB = svld1(pg, ptr_B + i);
        svfloat64_t vOut = svld1(pg, out_ptr + i);
        vOut = svmla_m(pg, vOut, vA, num);
        vOut = svmla_m(pg, vOut, vB, beta);
        vOut = svadd_m(pg, vOut, gamma);
        svst1(pg, out_ptr + i, vOut);
    }
    return;
}



} //Linalg

#undef FOR_WHILE
#undef SVE_TARGET
#undef STR
#undef EXPAND

#endif //__ARM_FEATURE_SVE
