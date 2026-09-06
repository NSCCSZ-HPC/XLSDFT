#include <stdio.h>
#include "linalg.h"

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

#ifdef USE_OPENMP

    void init_omp_env() {
        int nthreads = 0;
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        (void) nthreads;
        #ifdef USE_KML
            // printf("USE_KML:: \n");
            #ifdef USE_MY_OPENMP_BLAS
                BlasSetNumThreads(1);
                BlasSetNumThreadsLocal(1);
            #elif defined(USE_BLAS_OPEMNP_NESTED)
                BlasSetNumThreads(1);
                BlasSetNumThreadsLocal(nthreads);
            #else
                BlasSetNumThreads(nthreads);
                BlasSetNumThreadsLocal(nthreads);
            #endif //USE_MY_OPENMP_BLAS

            #ifdef USE_OPENMP_NESTED
                // omp_set_nested(0);
                omp_set_max_active_levels(1);
            #else
                // omp_set_nested(0);
                omp_set_max_active_levels(1);
            #endif //USE_OPENMP_NESTED
        #elif defined(USE_MKL)
            // printf("USE_MKL:: \n");
            mkl_set_num_threads(nthreads);
            mkl_set_num_threads_local(nthreads);
            #ifdef USE_OPENMP_NESTED
            // omp_set_nested(0);
            omp_set_max_active_levels(1);
            #else
            // omp_set_nested(0);
            omp_set_max_active_levels(1);
            #endif //USE_OPENMP_NESTED
        #elif defined(USE_OPENBLAS)
            // printf("USE_OPENBLAS:: \n");
            const char* config = openblas_get_config();
            printf("OpenBLAS config: %s\n", config);
            std::cout << RED << "Warning: OpenBLAS may have issues when called inside an OpenMP parallel region." << RESET << std::endl;
            std::cout << RED << "Warning: pthread version OpenBLAS will print a lot of err." << RESET << std::endl;
            std::cout << RED << "Warning: OpenMP version OpenBLAS will run with only single thread." << RESET << std::endl;
            std::cout << RED << "Warning: I recommend pthread version OpenBLAS with warning related code remove in the source file." << RESET << std::endl;
            openblas_set_num_threads(nthreads);
            #ifdef USE_OPENMP_NESTED
            // omp_set_nested(0);
            omp_set_max_active_levels(1);
            #else
            // omp_set_nested(0);
            omp_set_max_active_levels(1);
            #endif //USE_OPENMP_NESTED
        #endif
        return;
    }
#endif

void set_kblas_nthread() {
    #ifdef USE_KML
        int nthreads = 1;
        #pragma omp parallel
        nthreads = omp_get_num_threads();
        BlasSetNumThreads(nthreads);
        BlasSetNumThreadsLocal(nthreads);
    #endif
}

void set_kblas_1() {
    #ifdef USE_KML
        BlasSetNumThreads(1);
        BlasSetNumThreadsLocal(1);
    #endif
}

uint64_t get_chunksize(const uint64_t nthreads, const uint64_t njobs) {
    assert(nthreads != 0);
    return (njobs - 1 + nthreads)/nthreads;
}

template<>
uint64_t get_type_id<int>() {
    return 0;
}

template<>
uint64_t get_type_id<float>() {
    return 1;
}

template<>
uint64_t get_type_id<double>() {
    return 2;
}

#ifdef FP16_FLAG
template<>
uint64_t get_type_id<__fp16>() {
    return 3;
}
#endif //FP16_FLAG

template<>
uint64_t get_type_id<std::complex<float>>() {
    return 4;
}

template<>
uint64_t get_type_id<std::complex<double>>() {
    return 5;
}

#ifdef FP16_FLAG
template<>
uint64_t get_type_id<std::complex<__fp16>>() {
    return 6;
}
#endif //FP16_FLAG

template<typename T1, typename T2>
void convert_type(T1* const out_ptr, T2 const* const in_ptr, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = (T1)in_ptr[i];
    }
    return;
}
template void convert_type<double, double>(double* const out_ptr, double const* const in_ptr, const uint64_t length);
template void convert_type<float, float>(float* const out_ptr, float const* const in_ptr, const uint64_t length);
template void convert_type<int, int>(int* const out_ptr, int const* const in_ptr, const uint64_t length);
template void convert_type<double, float>(double* const out_ptr, float const* const in_ptr, const uint64_t length);
template void convert_type<float, double>(float* const out_ptr, double const* const in_ptr, const uint64_t length);
template void convert_type<int, float>(int* const out_ptr, float const* const in_ptr, const uint64_t length);
template void convert_type<float, int>(float* const out_ptr, int const* const in_ptr, const uint64_t length);
template void convert_type<double, int>(double* const out_ptr, int const* const in_ptr, const uint64_t length);
template void convert_type<int, double>(int* const out_ptr, double const* const in_ptr, const uint64_t length);
template void convert_type<std::complex<float>, std::complex<double>>(std::complex<float>* const out_ptr, std::complex<double> const* const in_ptr, const uint64_t length);
template void convert_type<std::complex<double>, std::complex<float>>(std::complex<double>* const out_ptr, std::complex<float> const* const in_ptr, const uint64_t length);
template void convert_type<std::complex<double>, std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const in_ptr, const uint64_t length);
template void convert_type<std::complex<float>, std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const in_ptr, const uint64_t length);
#ifdef FP16_FLAG
template void convert_type<__fp16, __fp16>(__fp16* const out_ptr, __fp16 const* const in_ptr, const uint64_t length);
template void convert_type<double, __fp16>(double* const out_ptr, __fp16 const* const in_ptr, const uint64_t length);
template void convert_type<__fp16, double>(__fp16* const out_ptr, double const* const in_ptr, const uint64_t length);
template void convert_type<float, __fp16>(float* const out_ptr, __fp16 const* const in_ptr, const uint64_t length);
template void convert_type<__fp16, float>(__fp16* const out_ptr, float const* const in_ptr, const uint64_t length);
#endif //FP16_FLAG

template<typename T1, typename T2>
void convert_type(std::complex<T1>* const out_ptr, T2 const* const in_ptr, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = std::complex<T1>(in_ptr[i]);
    }
    return;
}
template void convert_type<std::complex<float>, float>(std::complex<float>* const out_ptr, float const* const in_ptr, const uint64_t length);
template void convert_type<std::complex<double>, double>(std::complex<double>* const out_ptr, double const* const in_ptr, const uint64_t length);
template void convert_type<std::complex<double>, float>(std::complex<double>* const out_ptr, float const* const in_ptr, const uint64_t length);
template void convert_type<std::complex<float>, double>(std::complex<float>* const out_ptr, double const* const in_ptr, const uint64_t length);

template<> MPI_Datatype get_mpi_datatype<int>() {
    return MPI_INT;
}

template<> MPI_Datatype get_mpi_datatype<float>() {
    return MPI_FLOAT;
}

template<> MPI_Datatype get_mpi_datatype<double>() {
    return MPI_DOUBLE;
}

#ifdef FP16_FLAG
template<> MPI_Datatype get_mpi_datatype<__fp16>() {
    assert(!"no __fp16 mpi_datatype");
    return MPI_DOUBLE;
}
#endif //FP16_FLAG

template<> MPI_Datatype get_mpi_datatype<std::complex<float>>() {
    #ifdef MPI_CXX_FLOAT_COMPLEX
    return MPI_CXX_FLOAT_COMPLEX;
    #else
    return MPI_C_FLOAT_COMPLEX;
    #endif
}

template<> MPI_Datatype get_mpi_datatype<std::complex<double>>() {
    #ifdef MPI_CXX_DOUBLE_COMPLEX
    return MPI_CXX_DOUBLE_COMPLEX;
    #else
    return MPI_C_DOUBLE_COMPLEX;
    #endif
}

#ifdef FP16_FLAG
template<> MPI_Datatype get_mpi_datatype<std::complex<__fp16>>() {
    assert(!"no __fp16 mpi_datatype");
    return MPI_CXX_DOUBLE_COMPLEX;
}
#endif //FP16_FLAG

template<typename T>
void print_vector(T const* const ptr, uint64_t const length) {
    if (get_type_id<T>() == 0) std::cout << "this is a int vector" << std::endl;
    if (get_type_id<T>() == 1) std::cout << "this is a float vector" << std::endl;
    if (get_type_id<T>() == 2) std::cout << "this is a double vector" << std::endl;
    if (get_type_id<T>() == 4) std::cout << "this is a std::complex<float> vector" << std::endl;
    if (get_type_id<T>() == 5) std::cout << "this is a std::complex<double> vector" << std::endl;
    for (uint64_t i = 0; i < length; i++)
    {
        std::cout << std::setw(14) << std::setprecision(6)
        // outfile << std::setw(22) << std::setprecision(13)
            << std::scientific << std::uppercase
            << ptr[i];
        if (i % 6 == 5)
            std::cout << std::endl;
    }
    std::cout << std::endl;
    return;
}
template void print_vector<int>(int const* const ptr, uint64_t const length);
template void print_vector<float>(float const* const ptr, uint64_t const length);
template void print_vector<double>(double const* const ptr, uint64_t const length);
template void print_vector<std::complex<float>>(std::complex<float> const* const ptr, uint64_t const length);
template void print_vector<std::complex<double>>(std::complex<double> const* const ptr, uint64_t const length);
#ifdef FP16_FLAG
template void print_vector<__fp16>(__fp16 const* const ptr, uint64_t const length);
template void print_vector<std::complex<__fp16>>(std::complex<__fp16> const* const ptr, uint64_t const length);
#endif //FP16_FLAG

template<typename T> 
void print_vector(T const* const ptr, uint64_t const length, const std::string& fname) {
    std::ofstream outfile(fname);
    if (get_type_id<T>() == 0) outfile << "this is a int vector" << std::endl;
    if (get_type_id<T>() == 1) outfile << "this is a float vector" << std::endl;
    if (get_type_id<T>() == 2) outfile << "this is a double vector" << std::endl;
    if (get_type_id<T>() == 4) outfile << "this is a std::complex<float> vector" << std::endl;
    if (get_type_id<T>() == 5) outfile << "this is a std::complex<double> vector" << std::endl;
    for (uint64_t i = 0; i < length; i++)
    {
        outfile << std::setw(14) << std::setprecision(6)
        // outfile << std::setw(22) << std::setprecision(13)
            << std::scientific << std::uppercase
            << ptr[i];
        if (i % 6 == 5)
            outfile << std::endl;
    }
    outfile.close();
    return;
}
template void print_vector<int>(int const* const ptr, uint64_t const length, const std::string& fname);
template void print_vector<float>(float const* const ptr, uint64_t const length, const std::string& fname);
template void print_vector<double>(double const* const ptr, uint64_t const length, const std::string& fname);
template void print_vector<std::complex<float>>(std::complex<float> const* const ptr, uint64_t const length, const std::string& fname);
template void print_vector<std::complex<double>>(std::complex<double> const* const ptr, uint64_t const length, const std::string& fname);
#ifdef FP16_FLAG
template void print_vector<__fp16>(__fp16 const* const ptr, uint64_t const length, const std::string& fname);
template void print_vector<std::complex<__fp16>>(std::complex<__fp16> const* const ptr, uint64_t const length, const std::string& fname);
#endif //FP16_FLAG

template<typename T> void print_vectorX(const T* ptr, const uint& length, const std::string& fname) {
    std::ofstream outfile(fname);
    if (get_type_id<T>() == 0) outfile << "this is a int vector" << std::endl;
    if (get_type_id<T>() == 1) outfile << "this is a float vector" << std::endl;
    if (get_type_id<T>() == 2) outfile << "this is a double vector" << std::endl;
    if (get_type_id<T>() == 4) outfile << "this is a std::complex<float> vector" << std::endl;
    if (get_type_id<T>() == 5) outfile << "this is a std::complex<double> vector" << std::endl;
    if (sizeof(T) == 8) {
        long long int* dataX = (long long int*)ptr;
        for (uint64_t index = 0; index < length; index++) {
            outfile << std::hex << std::setw(18) << dataX[index];
            if (index % 6 == 5)
                outfile << std::endl;
        }
        outfile << std::endl;
    } else if (sizeof(T) == 4) {
        int* dataX = (int*)ptr;
        for (uint64_t index = 0; index < length; index++) {
            outfile << std::hex << std::setw(18) << dataX[index];
            if (index % 6 == 5)
                outfile << std::endl;
        }
        outfile << std::endl;
    }
    outfile.close();
    return;
}
template void print_vectorX<int>(const int* ptr, const uint& length, const std::string& fname);
template void print_vectorX<float>(const float* ptr, const uint& length, const std::string& fname);
template void print_vectorX<double>(const double* ptr, const uint& length, const std::string& fname);
#ifdef FP16_FLAG
template void print_vectorX<__fp16>(const __fp16* ptr, const uint& length, const std::string& fname);
#endif //FP16_FLAG

//return a rand uint64_t in [0, UINT64_MAX]
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

template<typename T>
void seededrand(T* const ptr, uint64_t const length,
                T const rand_min, T const rand_max,
                uint64_t const seed) {
    #ifdef USE_SRAND
    std::srand(seed);
    for (uint64_t i = 0; i < length; i++) {
        ptr[i] = rand_min + (rand_max - rand_min) * ((T) std::rand() / (T)RAND_MAX);
    }
    #else
    for (uint64_t i = 0; i < length; i++) {
        ptr[i] = rand_min + (rand_max - rand_min) * ((T) splitmix64(seed + i) / (T)UINT64_MAX);
    }
    #endif
    return;
}
template void seededrand<int>(int* const ptr, uint64_t const length, int const rand_min, int const rand_max, uint64_t const seed);
template void seededrand<float>(float* const ptr, uint64_t const length, float const rand_min, float const rand_max, uint64_t const seed);
template void seededrand<double>(double* const ptr, uint64_t const length, double const rand_min, double const rand_max, uint64_t const seed);
#ifdef FP16_FLAG
template void seededrand<__fp16>(__fp16* const ptr, uint64_t const length, __fp16 const rand_min, __fp16 const rand_max, uint64_t const seed);
#endif //FP16_FLAG

template<typename T>
void seededrand(std::complex<T>* const ptr, uint64_t const length,
                T const rand_min, T const rand_max,
                uint64_t const seed) {
    #ifdef USE_SRAND
    std::srand(seed);
    for (uint64_t i = 0; i < length; i++) {
        // ptr[i] = std::complex<T>(rand_min + (rand_max - rand_min) * ((double) rand() / (double)RAND_MAX),
        //                     rand_min + (rand_max - rand_min) * ((double) rand() / (double)RAND_MAX));
        ptr[i] = std::complex<T>(rand_min + (rand_max - rand_min) * ((T) std::rand() / (T)RAND_MAX));
    }
    #else
    for (uint64_t i = 0; i < length; i++) {
        ptr[i] = std::complex<T>(rand_min + (rand_max - rand_min) * ((T) splitmix64(seed + i) / (T)UINT64_MAX));
    }
    #endif
    return;
}
template void seededrand<float>(std::complex<float>* const ptr, uint64_t const length, float const rand_min, float const rand_max, uint64_t const seed);
template void seededrand<double>(std::complex<double>* const ptr, uint64_t const length, double const rand_min, double const rand_max, uint64_t const seed);
#ifdef FP16_FLAG
template void seededrand<__fp16>(std::complex<__fp16>* const ptr, uint64_t const length, __fp16 const rand_min, __fp16 const rand_max, uint64_t const seed);
#endif //FP16_FLAG

template<typename T1, typename T2>
void seededrand_sequential(T1* const ptr, uint64_t const length,
                            T2 const rand_min, T2 const rand_max,
                            uint64_t const seed) {
    #ifdef USE_SRAND
    int seed_temp = seed;
    for (uint64_t i = 0; i < length; i++) {
        std::srand(seed_temp++);
        ptr[i] = rand_min + (rand_max - rand_min) * ((T1) std::rand() / (T1)RAND_MAX);
    }
    #else
    uint64_t seed_temp = seed;
    for (uint64_t i = 0; i < length; i++) {
        ptr[i] = rand_min + (rand_max - rand_min) * ((T1) splitmix64(seed_temp++) / (T1)UINT64_MAX);
    }
    #endif
    return;
}
template void seededrand_sequential<int>(int* const ptr, uint64_t const length, int const rand_min, int const rand_max, uint64_t const seed);
template void seededrand_sequential<float>(float* const ptr, uint64_t const length, float const rand_min, float const rand_max, uint64_t const seed);
template void seededrand_sequential<double>(double* const ptr, uint64_t const length, double const rand_min, double const rand_max, uint64_t const seed);
template void seededrand_sequential<int, float>(int* const ptr, uint64_t const length, float const rand_min, float const rand_max, uint64_t const seed);
template void seededrand_sequential<int, double>(int* const ptr, uint64_t const length, double const rand_min, double const rand_max, uint64_t const seed);
#ifdef FP16_FLAG
template void seededrand_sequential<__fp16>(__fp16* const ptr, uint64_t const length, __fp16 const rand_min, __fp16 const rand_max, uint64_t const seed);
#endif //FP16_FLAG

template<typename T1, typename T2>
void seededrand_sequential(std::complex<T1>* const ptr, uint64_t const length,
                            T2 const rand_min, T2 const rand_max,
                            uint64_t const seed) {
    #ifdef USE_SRAND
    int seed_temp = seed;
    for (uint64_t i = 0; i < length; i++) {
        std::srand(seed_temp++);
        // ptr[i] = std::complex<T1>(rand_min + (rand_max - rand_min) * ((double) rand() / (double)RAND_MAX),
        //                          rand_min + (rand_max - rand_min) * ((double) rand() / (double)RAND_MAX));
        ptr[i] = std::complex<T1>(rand_min + (rand_max - rand_min) * ((T1) std::rand() / (T1)RAND_MAX));
    }
    #else
    uint64_t seed_temp = seed;
    for (uint64_t i = 0; i < length; i++) {
        ptr[i] = std::complex<T1>(rand_min + (rand_max - rand_min) * ((T1) splitmix64(seed_temp++) / (T1)UINT64_MAX));
    }
    #endif
    return;
}
template void seededrand_sequential<float>(std::complex<float>* const ptr, uint64_t const length, float const rand_min, float const rand_max, uint64_t const seed);
template void seededrand_sequential<double>(std::complex<double>* const ptr, uint64_t const length, double const rand_min, double const rand_max, uint64_t const seed);
#ifdef FP16_FLAG
template void seededrand_sequential<__fp16>(std::complex<__fp16>* const ptr, uint64_t const length, __fp16 const rand_min, __fp16 const rand_max, uint64_t const seed);
#endif //FP16_FLAG

//Vertor functions
template<typename T>
T vector_sum(T const* const ptr, uint64_t const length, const MPI_Comm comm) {
    #ifdef USE_OPENMP
        #pragma omp barrier
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
template int vector_sum<int>(int const* const ptr, uint64_t const length, const MPI_Comm comm);
// template float vector_sum<float>(float const* const ptr, uint64_t const length, const MPI_Comm comm);
// template double vector_sum<double>(double const* const ptr, uint64_t const length, const MPI_Comm comm);
#ifdef FP16_FLAG
template __fp16 vector_sum<__fp16>(__fp16 const* const ptr, uint64_t const length, const MPI_Comm comm);
#endif //FP16_FLAG

template<typename T>
std::complex<T> vector_sum(std::complex<T> const* const ptr, uint64_t const length, const MPI_Comm comm) {
    #ifdef USE_OPENMP
        #pragma omp barrier
        static std::complex<T> total_Sum;
        #pragma omp single nowait
        total_Sum = static_cast<std::complex<T>>(0);
        std::complex<T> local_Sum = static_cast<std::complex<T>>(0);
        FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
            local_Sum += ptr[i];
        }
        #pragma omp barrier
        #pragma omp critical
        total_Sum += local_Sum;
        #pragma omp barrier
    #else
        std::complex<T> total_Sum = static_cast<T>(0);
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
        MPI_Allreduce(MPI_IN_PLACE, &total_Sum, 1, get_mpi_datatype<std::complex<T>>(), MPI_SUM, comm);
        #ifdef USE_OPENMP
            #pragma omp barrier
        #endif //USE_OPENMP
    }
    return total_Sum;
}
template std::complex<float> vector_sum<float>(std::complex<float> const* const ptr, uint64_t const length, const MPI_Comm comm);
template std::complex<double> vector_sum<double>(std::complex<double> const* const ptr, uint64_t const length, const MPI_Comm comm);

template<typename T>
T vector_norm_square_sum(T const* const ptr, uint64_t const length, const MPI_Comm comm) {
    #ifdef USE_OPENMP
        #pragma omp barrier
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
template int vector_norm_square_sum<int>(int const* const ptr, uint64_t const length, const MPI_Comm comm);
// template float vector_norm_square_sum<float>(float const* const ptr, uint64_t const length, const MPI_Comm comm);
// template double vector_norm_square_sum<double>(double const* const ptr, uint64_t const length, const MPI_Comm comm);
#ifdef FP16_FLAG
template __fp16 vector_norm_square_sum<__fp16>(__fp16 const* const ptr, uint64_t const length, const MPI_Comm comm);
#endif //FP16_FLAG

template<typename T>
T vector_norm_square_sum(std::complex<T> const* const ptr, uint64_t const length, const MPI_Comm comm) {
    #ifdef USE_OPENMP
        #pragma omp barrier
        static T total_Sum;
        #pragma omp single nowait
        total_Sum = static_cast<T>(0);
        T local_Sum = static_cast<T>(0);
        FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
            local_Sum += std::norm(ptr[i]);
        }
        #pragma omp barrier
        #pragma omp critical
        total_Sum += local_Sum;
        #pragma omp barrier
    #else
        T total_Sum = static_cast<T>(0);
        for (uint64_t i = 0; i < length; i++) {
            total_Sum += std::norm(ptr[i]);
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
template float vector_norm_square_sum<float>(std::complex<float> const* const ptr, uint64_t const length, const MPI_Comm comm);
template double vector_norm_square_sum<double>(std::complex<double> const* const ptr, uint64_t const length, const MPI_Comm comm);
#ifdef FP16_FLAG
template __fp16 vector_norm_square_sum<__fp16>(std::complex<__fp16> const* const ptr, uint64_t const length, const MPI_Comm comm);
#endif //FP16_FLAG

template<typename T>
T vector_dot_product(T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length, const MPI_Comm comm) {
    #ifdef USE_OPENMP
        #pragma omp barrier
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
template int vector_dot_product<int>(int const* const ptr_A, int const* const ptr_B, const uint64_t length, const MPI_Comm comm);
// template float vector_dot_product<float>(float const* const ptr_A, float const* const ptr_B, const uint64_t length, const MPI_Comm comm);
// template double vector_dot_product<double>(double const* const ptr_A, double const* const ptr_B, const uint64_t length, const MPI_Comm comm);
#ifdef FP16_FLAG
template __fp16 vector_dot_product<__fp16>(__fp16 const* const ptr_A, __fp16 const* const ptr_B, const uint64_t length, const MPI_Comm comm);
#endif //FP16_FLAG

template<typename T>
std::complex<T> vector_dot_product(std::complex<T> const* const __restrict__ ptr_A, std::complex<T> const* const __restrict__ ptr_B, const uint64_t length, const MPI_Comm comm) {
    #ifdef USE_OPENMP
        #pragma omp barrier
        static std::complex<T> total_Sum;
        #pragma omp single nowait
        total_Sum = static_cast<std::complex<T>>(0);
        std::complex<T> local_Sum = static_cast<std::complex<T>>(0);
        FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
            local_Sum += ptr_A[i] * ptr_B[i];
        }
        #pragma omp barrier
        #pragma omp critical
        total_Sum += local_Sum;
        #pragma omp barrier
    #else
        std::complex<T> total_Sum = static_cast<std::complex<T>>(0);
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
        MPI_Allreduce(MPI_IN_PLACE, &total_Sum, 1, get_mpi_datatype<std::complex<T>>(), MPI_SUM, comm);
        #ifdef USE_OPENMP
            #pragma omp barrier
        #endif //USE_OPENMP
    }
    return total_Sum;
}
template std::complex<float> vector_dot_product<float>(std::complex<float> const* const ptr_A, std::complex<float> const* const ptr_B, const uint64_t length, const MPI_Comm comm);
template std::complex<double> vector_dot_product<double>(std::complex<double> const* const ptr_A, std::complex<double> const* const ptr_B, const uint64_t length, const MPI_Comm comm);
#ifdef FP16_FLAG
template std::complex<__fp16> vector_dot_product<__fp16>(std::complex<__fp16> const* const ptr_A, std::complex<__fp16> const* const ptr_B, const uint64_t length, const MPI_Comm comm);
#endif //FP16_FLAG

template<typename T>
T vector_2norm(T const* const ptr, const uint64_t length, const MPI_Comm comm) {
    return std::sqrt(vector_norm_square_sum(ptr, length, comm));
}
template int vector_2norm<int>(int const* const ptr, const uint64_t length, const MPI_Comm comm);
template float vector_2norm<float>(float const* const ptr, const uint64_t length, const MPI_Comm comm);
template double vector_2norm<double>(double const* const ptr, const uint64_t length, const MPI_Comm comm);
#ifdef FP16_FLAG
template __fp16 vector_2norm<__fp16>(__fp16 const* const ptr, const uint64_t length, const MPI_Comm comm);
#endif //FP16_FLAG

template<typename T>
T vector_2norm(std::complex<T> const* const ptr, const uint64_t length, const MPI_Comm comm) {
    return std::sqrt(vector_norm_square_sum(ptr, length, comm));
}
template float vector_2norm<float>(std::complex<float> const* const ptr, const uint64_t length, const MPI_Comm comm);
template double vector_2norm<double>(std::complex<double> const* const ptr, const uint64_t length, const MPI_Comm comm);
#ifdef FP16_FLAG
template __fp16 vector_2norm<__fp16>(std::complex<__fp16> const* const ptr, const uint64_t length, const MPI_Comm comm);
#endif //FP16_FLAG

template<typename T>
void vector_normalize(T* const ptr, const uint64_t length, const MPI_Comm comm) {
    T ratio = (T)1.0 / vector_2norm(ptr, length, comm);
    Linalg::scalar_product_general(ptr, ratio, length);
    return;
}
template void vector_normalize<int>(int* const ptr, const uint64_t length, const MPI_Comm comm);
template void vector_normalize<float>(float* const ptr, const uint64_t length, const MPI_Comm comm);
template void vector_normalize<double>(double* const ptr, const uint64_t length, const MPI_Comm comm);
template void vector_normalize<std::complex<float>>(std::complex<float>* const ptr, const uint64_t length, const MPI_Comm comm);
template void vector_normalize<std::complex<double>>(std::complex<double>* const ptr, const uint64_t length, const MPI_Comm comm);
#ifdef FP16_FLAG
template void vector_normalize<__fp16>(__fp16* const ptr, const uint64_t length, const MPI_Comm comm);
template void vector_normalize<std::complex<__fp16>>(std::complex<__fp16>* const ptr, const uint64_t length, const MPI_Comm comm);
#endif //FP16_FLAG

template<typename T>
void vector_orthogonalize(T* const inout_ptr, T const* const ptr_A, const uint64_t length, const MPI_Comm comm) {
    T dot_product = vector_dot_product(inout_ptr, ptr_A, length, comm);
    accumulate_scalar_product_general<T>(inout_ptr, ptr_A, - dot_product, length);
    return;
}
template void vector_orthogonalize<int>(int* const inout_ptr, int const* const ptr_A, const uint64_t length, const MPI_Comm comm);
template void vector_orthogonalize<float>(float* const inout_ptr, float const* const ptr_A, const uint64_t length, const MPI_Comm comm);
template void vector_orthogonalize<double>(double* const inout_ptr, double const* const ptr_A, const uint64_t length, const MPI_Comm comm);
#ifdef FP16_FLAG
template void vector_orthogonalize<__fp16>(__fp16* const inout_ptr, __fp16 const* const ptr_A, const uint64_t length, const MPI_Comm comm);
#endif //FP16_FLAG

template<typename T>
void vector_norm_square(T* const out_ptr, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1){
        out_ptr[i] *= out_ptr[i];
    }
}
template void vector_norm_square<int>(int* const out_ptr, const uint64_t length);
// template void vector_norm_square<float>(float* const out_ptr, const uint64_t length);
// template void vector_norm_square<double>(double* const out_ptr, const uint64_t length);
#ifdef FP16_FLAG
template void vector_norm_square<__fp16>(__fp16* const out_ptr, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void vector_norm_square(std::complex<T>* const out_ptr, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = std::complex<T>(std::norm(out_ptr[i]));
    }
}
template void vector_norm_square<std::complex<float>>(std::complex<float>* const out_ptr, const uint64_t length);
template void vector_norm_square<std::complex<double>>(std::complex<double>* const out_ptr, const uint64_t length);
#ifdef FP16_FLAG
template void vector_norm_square<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void vector_norm_square(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] * ptr_A[i];
    }
}
template void vector_norm_square<int>(int* const out_ptr, int const* const ptr_A, const uint64_t length);
// template void vector_norm_square<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length);
// template void vector_norm_square<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length);
#ifdef FP16_FLAG
template void vector_norm_square<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void vector_norm_square(T* const __restrict__ out_ptr, std::complex<T> const* const __restrict__ ptr_A, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = std::norm(ptr_A[i]);
    }
}
template void vector_norm_square<float>(float* const out_ptr, std::complex<float> const* const ptr_A, const uint64_t length);
template void vector_norm_square<double>(double* const out_ptr, std::complex<double> const* const ptr_A, const uint64_t length);
#ifdef FP16_FLAG
template void vector_norm_square<__fp16>(__fp16* const out_ptr, std::complex<__fp16> const* const ptr_A, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void vector_norm_square(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A,
                    const uint64_t length, const T alpha) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] * ptr_A[i] * alpha;
    }
}
template void vector_norm_square<int>(int* const out_ptr, int const* const ptr_A, const uint64_t length, const int alpha);
// template void vector_norm_square<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length, const float alpha);
// template void vector_norm_square<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length, const double alpha);
#ifdef FP16_FLAG
template void vector_norm_square<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const uint64_t length, const __fp16 alpha);
#endif //FP16_FLAG

template<typename T>
void vector_norm_square(T* const __restrict__ out_ptr, std::complex<T> const* const __restrict__ ptr_A,
                    const uint64_t length, const T alpha) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = std::norm(ptr_A[i]) * alpha;
    }
}
template void vector_norm_square<float>(float* const out_ptr, std::complex<float> const* const ptr_A, const uint64_t length, const float alpha);
template void vector_norm_square<double>(double* const out_ptr, std::complex<double> const* const ptr_A, const uint64_t length, const double alpha);
#ifdef FP16_FLAG
template void vector_norm_square<__fp16>(__fp16* const out_ptr, std::complex<__fp16> const* const ptr_A, const uint64_t length, const __fp16 alpha);
#endif //FP16_FLAG

template<typename T>
void accumulate_vector_norm_square(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length,
                           const T alpha) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1){
        out_ptr[i] += ptr_A[i] * ptr_A[i] * alpha;
    }
}
template void accumulate_vector_norm_square<int>(int* const out_ptr, int const* const ptr_A, const uint64_t length, const int alpha);
// template void accumulate_vector_norm_square<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length, const float alpha);
// template void accumulate_vector_norm_square<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length, const double alpha);
#ifdef FP16_FLAG
template void accumulate_vector_norm_square<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const uint64_t length, const __fp16 alpha);
#endif //FP16_FLAG

template<typename T>
void accumulate_vector_norm_square(T* const __restrict__ out_ptr, std::complex<T> const* const __restrict__ ptr_A,
                    const uint64_t length, const T alpha) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += std::norm(ptr_A[i]) * alpha;
    }
}
template void accumulate_vector_norm_square<float>(float* const out_ptr, std::complex<float> const* const ptr_A, const uint64_t length, const float alpha);
template void accumulate_vector_norm_square<double>(double* const out_ptr, std::complex<double> const* const ptr_A, const uint64_t length, const double alpha);
#ifdef FP16_FLAG
template void accumulate_vector_norm_square<__fp16>(__fp16* const out_ptr, std::complex<__fp16> const* const ptr_A, const uint64_t length, const __fp16 alpha);
#endif //FP16_FLAG

template<typename T>
void set_value_general(T* const out_ptr, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = num;
    }
}
template void set_value_general<int>(int* const out_ptr, const int num, const uint64_t length);
// template void set_value_general<float>(float* const out_ptr, const float num, const uint64_t length);
// template void set_value_general<double>(double* const out_ptr, const double num, const uint64_t length);
template void set_value_general<std::complex<float>>(std::complex<float>* const out_ptr, const std::complex<float> num, const uint64_t length);
template void set_value_general<std::complex<double>>(std::complex<double>* const out_ptr, const std::complex<double> num, const uint64_t length);
#ifdef FP16_FLAG
template void set_value_general<__fp16>(__fp16* const out_ptr, const __fp16 num, const uint64_t length);
template void set_value_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, const std::complex<__fp16> num, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void set_value_general(T* const out_ptr, T const* const ptr_A, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i];
    }
    return;
}
template void set_value_general<int>(int* const out_ptr, int const* const ptr_A, const uint64_t length);
// template void set_value_general<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length);
// template void set_value_general<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length);
template void set_value_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const uint64_t length);
template void set_value_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const uint64_t length);
#ifdef FP16_FLAG
template void set_value_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const uint64_t length);
template void set_value_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void vector_cbrt(T* const out_ptr, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1){
        out_ptr[i] = std::cbrt(out_ptr[i]);
    }
}
template void vector_cbrt<int>(int* const out_ptr, const uint64_t length);
template void vector_cbrt<float>(float* const out_ptr, const uint64_t length);
template void vector_cbrt<double>(double* const out_ptr, const uint64_t length);
#ifdef FP16_FLAG
template void vector_cbrt<__fp16>(__fp16* const out_ptr, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void vector_cbrt(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1){
        out_ptr[i] = std::cbrt(ptr_A[i]);
    }
}
template void vector_cbrt<int>(int* const out_ptr, int const* const ptr_A, const uint64_t length);
template void vector_cbrt<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length);
template void vector_cbrt<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length);
#ifdef FP16_FLAG
template void vector_cbrt<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void conj_general(std::complex<T>* const out_ptr, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = std::conj(out_ptr[i]);
    }
    return;
}
template void conj_general<float>(std::complex<float>* const out_ptr, const uint64_t length);
template void conj_general<double>(std::complex<double>* const out_ptr, const uint64_t length);
#ifdef FP16_FLAG
template void conj_general<__fp16>(std::complex<__fp16>* const out_ptr, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void conj_general(std::complex<T>* const out_ptr, std::complex<T> const* const ptr_A, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = std::conj(ptr_A[i]);
    }
    return;
}
template void conj_general<float>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const uint64_t length);
template void conj_general<double>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const uint64_t length);
#ifdef FP16_FLAG
template void conj_general<__fp16>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void conj_product(std::complex<T>* const out_ptr, std::complex<T> const* const ptr_A, std::complex<T> const* const ptr_B, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = std::conj(ptr_A[i]) * ptr_B[i];
    }
    return;
}
template void conj_product<float>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, std::complex<float> const* const ptr_B, const uint64_t length);
template void conj_product<double>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, std::complex<double> const* const ptr_B, const uint64_t length);
#ifdef FP16_FLAG
template void conj_product<__fp16>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, std::complex<__fp16> const* const ptr_B, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void hadamard_plus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i];
    }
}
template void hadamard_plus_general<int>(int* const out_ptr, int const* const ptr_A, const uint64_t length);
// template void hadamard_plus_general<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length);
// template void hadamard_plus_general<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length);
template void hadamard_plus_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const uint64_t length);
template void hadamard_plus_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const uint64_t length);
#ifdef FP16_FLAG
template void hadamard_plus_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const uint64_t length);
template void hadamard_plus_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void hadamard_minus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] -= ptr_A[i];
    }
}
template void hadamard_minus_general<int>(int* const out_ptr, int const* const ptr_A, const uint64_t length);
// template void hadamard_minus_general<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length);
// template void hadamard_minus_general<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length);
template void hadamard_minus_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const uint64_t length);
template void hadamard_minus_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const uint64_t length);
#ifdef FP16_FLAG
template void hadamard_minus_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const uint64_t length);
template void hadamard_minus_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] *= ptr_A[i];
    }
}
template void hadamard_product_general<int>(int* const out_ptr, int const* const ptr_A, const uint64_t length);
// template void hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length);
// template void hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length);
template void hadamard_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const uint64_t length);
template void hadamard_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const uint64_t length);
#ifdef FP16_FLAG
template void hadamard_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const uint64_t length);
template void hadamard_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void hadamard_divide_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] /= ptr_A[i];
    }
}
template void hadamard_divide_general<int>(int* const out_ptr, int const* const ptr_A, const uint64_t length);
// template void hadamard_divide_general<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length);
// template void hadamard_divide_general<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length);
template void hadamard_divide_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const uint64_t length);
template void hadamard_divide_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const uint64_t length);
#ifdef FP16_FLAG
template void hadamard_divide_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const uint64_t length);
template void hadamard_divide_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length, T const alpha) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] *= alpha * ptr_A[i];
    }
}
template void hadamard_product_general<int>(int* const out_ptr, int const* const ptr_A, const uint64_t length, int const alpha);
// template void hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length, float const alpha);
// template void hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length, double const alpha);
template void hadamard_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const uint64_t length, std::complex<float> const alpha);
template void hadamard_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const uint64_t length, std::complex<double> const alpha);
#ifdef FP16_FLAG
template void hadamard_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const uint64_t length, __fp16 const alpha);
template void hadamard_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const uint64_t length, std::complex<__fp16> const alpha);
#endif //FP16_FLAG

template<typename T>
void hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length, T const alpha, T const beta) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = alpha * ptr_A[i] * out_ptr[i] + beta * out_ptr[i];
    }
}
template void hadamard_product_general<int>(int* const out_ptr, int const* const ptr_A, const uint64_t length, int const alpha, int const beta);
// template void hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length, float const alpha, float const beta);
// template void hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length, double const alpha, double const beta);
template void hadamard_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const uint64_t length, std::complex<float> const alpha, std::complex<float> const beta);
template void hadamard_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const uint64_t length, std::complex<double> const alpha, std::complex<double> const beta);
#ifdef FP16_FLAG
template void hadamard_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const uint64_t length, __fp16 const alpha, __fp16 const beta);
template void hadamard_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const uint64_t length, std::complex<__fp16> const alpha, std::complex<__fp16> const beta);
#endif //FP16_FLAG

template<typename T>
void hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const uint64_t length, T const alpha, T const beta, T const gamma) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = alpha * ptr_A[i] * out_ptr[i] + beta * out_ptr[i] + gamma;
    }
}
template void hadamard_product_general<int>(int* const out_ptr, int const* const ptr_A, const uint64_t length, int const alpha, int const beta, int const gamma);
// template void hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, const uint64_t length, float const alpha, float const beta, float const gamma);
// template void hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, const uint64_t length, double const alpha, double const beta, double const gamma);
template void hadamard_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const uint64_t length, std::complex<float> const alpha, std::complex<float> const beta, std::complex<float> const gamma);
template void hadamard_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const uint64_t length, std::complex<double> const alpha, std::complex<double> const beta, std::complex<double> const gamma);
#ifdef FP16_FLAG
template void hadamard_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const uint64_t length, __fp16 const alpha, __fp16 const beta, __fp16 const gamma);
template void hadamard_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const uint64_t length, std::complex<__fp16> const alpha, std::complex<__fp16> const beta, std::complex<__fp16> const gamma);
#endif //FP16_FLAG

template<typename T>
void hadamard_plus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] + ptr_B[i];
    }
}
template void hadamard_plus_general<int>(int* const out_ptr, int const* const ptr_A, int const* const ptr_B, const uint64_t length);
// template void hadamard_plus_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length);
// template void hadamard_plus_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length);
template void hadamard_plus_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, std::complex<float> const* const ptr_B, const uint64_t length);
template void hadamard_plus_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, std::complex<double> const* const ptr_B, const uint64_t length);
#ifdef FP16_FLAG
template void hadamard_plus_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, __fp16 const* const ptr_B, const uint64_t length);
template void hadamard_plus_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, std::complex<__fp16> const* const ptr_B, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void hadamard_minus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] - ptr_B[i];
    }
}
template void hadamard_minus_general<int>(int* const out_ptr, int const* const ptr_A, int const* const ptr_B, const uint64_t length);
// template void hadamard_minus_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length);
// template void hadamard_minus_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length);
template void hadamard_minus_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, std::complex<float> const* const ptr_B, const uint64_t length);
template void hadamard_minus_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, std::complex<double> const* const ptr_B, const uint64_t length);
#ifdef FP16_FLAG
template void hadamard_minus_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, __fp16 const* const ptr_B, const uint64_t length);
template void hadamard_minus_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, std::complex<__fp16> const* const ptr_B, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] * ptr_B[i];
    }
}
template void hadamard_product_general<int>(int* const out_ptr, int const* const ptr_A, int const* const ptr_B, const uint64_t length);
// template void hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length);
// template void hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length);
template void hadamard_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, std::complex<float> const* const ptr_B, const uint64_t length);
template void hadamard_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, std::complex<double> const* const ptr_B, const uint64_t length);
#ifdef FP16_FLAG
template void hadamard_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, __fp16 const* const ptr_B, const uint64_t length);
template void hadamard_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, std::complex<__fp16> const* const ptr_B, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void hadamard_divide_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] / ptr_B[i];
    }
}
template void hadamard_divide_general<int>(int* const out_ptr, int const* const ptr_A, int const* const ptr_B, const uint64_t length);
// template void hadamard_divide_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length);
// template void hadamard_divide_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length);
template void hadamard_divide_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, std::complex<float> const* const ptr_B, const uint64_t length);
template void hadamard_divide_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, std::complex<double> const* const ptr_B, const uint64_t length);
#ifdef FP16_FLAG
template void hadamard_divide_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, __fp16 const* const ptr_B, const uint64_t length);
template void hadamard_divide_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, std::complex<__fp16> const* const ptr_B, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length, T const alpha) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = alpha * ptr_A[i] * ptr_B[i];
    }
}
template void hadamard_product_general<int>(int* const out_ptr, int const* const ptr_A, int const* const ptr_B, const uint64_t length, int const alpha);
// template void hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length, float const alpha);
// template void hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length, double const alpha);
template void hadamard_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, std::complex<float> const* const ptr_B, const uint64_t length, std::complex<float> const alpha);
template void hadamard_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, std::complex<double> const* const ptr_B, const uint64_t length, std::complex<double> const alpha);
#ifdef FP16_FLAG
template void hadamard_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, __fp16 const* const ptr_B, const uint64_t length, __fp16 const alpha);
template void hadamard_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, std::complex<__fp16> const* const ptr_B, const uint64_t length, std::complex<__fp16> const alpha);
#endif //FP16_FLAG

template<typename T>
void hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length, T const alpha, T const beta) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = alpha * ptr_A[i] * ptr_B[i] + beta * out_ptr[i];
    }
}
template void hadamard_product_general<int>(int* const out_ptr, int const* const ptr_A, int const* const ptr_B, const uint64_t length, int const alpha, int const beta);
// template void hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length, float const alpha, float const beta);
// template void hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length, double const alpha, double const beta);
template void hadamard_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, std::complex<float> const* const ptr_B, const uint64_t length, std::complex<float> const alpha, std::complex<float> const beta);
template void hadamard_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, std::complex<double> const* const ptr_B, const uint64_t length, std::complex<double> const alpha, std::complex<double> const beta);
#ifdef FP16_FLAG
template void hadamard_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, __fp16 const* const ptr_B, const uint64_t length, __fp16 const alpha, __fp16 const beta);
template void hadamard_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, std::complex<__fp16> const* const ptr_B, const uint64_t length, std::complex<__fp16> const alpha, std::complex<__fp16> const beta);
#endif //FP16_FLAG

template<typename T>
void hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length, T const alpha, T const beta, T const gamma) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = alpha * ptr_A[i] * ptr_B[i] + beta * out_ptr[i] + gamma;
    }
}
template void hadamard_product_general<int>(int* const out_ptr, int const* const ptr_A, int const* const ptr_B, const uint64_t length, int const alpha, int const beta, int const gamma);
// template void hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length, float const alpha, float const beta, float const gamma);
// template void hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length, double const alpha, double const beta, double const gamma);
template void hadamard_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, std::complex<float> const* const ptr_B, const uint64_t length, std::complex<float> const alpha, std::complex<float> const beta, std::complex<float> const gamma);
template void hadamard_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, std::complex<double> const* const ptr_B, const uint64_t length, std::complex<double> const alpha, std::complex<double> const beta, std::complex<double> const gamma);
#ifdef FP16_FLAG
template void hadamard_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, __fp16 const* const ptr_B, const uint64_t length, __fp16 const alpha, __fp16 const beta, __fp16 const gamma);
template void hadamard_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, std::complex<__fp16> const* const ptr_B, const uint64_t length, std::complex<__fp16> const alpha, std::complex<__fp16> const beta, std::complex<__fp16> const gamma);
#endif //FP16_FLAG

template<typename T>
void accumulate_hadamard_plus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] + ptr_B[i];
    }
}
template void accumulate_hadamard_plus_general<int>(int* const out_ptr, int const* const ptr_A, int const* const ptr_B, const uint64_t length);
// template void accumulate_hadamard_plus_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length);
// template void accumulate_hadamard_plus_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length);
template void accumulate_hadamard_plus_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, std::complex<float> const* const ptr_B, const uint64_t length);
template void accumulate_hadamard_plus_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, std::complex<double> const* const ptr_B, const uint64_t length);
#ifdef FP16_FLAG
template void accumulate_hadamard_plus_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, __fp16 const* const ptr_B, const uint64_t length);
template void accumulate_hadamard_plus_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, std::complex<__fp16> const* const ptr_B, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void accumulate_hadamard_minus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] - ptr_B[i];
    }
}
template void accumulate_hadamard_minus_general<int>(int* const out_ptr, int const* const ptr_A, int const* const ptr_B, const uint64_t length);
// template void accumulate_hadamard_minus_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length);
// template void accumulate_hadamard_minus_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length);
template void accumulate_hadamard_minus_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, std::complex<float> const* const ptr_B, const uint64_t length);
template void accumulate_hadamard_minus_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, std::complex<double> const* const ptr_B, const uint64_t length);
#ifdef FP16_FLAG
template void accumulate_hadamard_minus_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, __fp16 const* const ptr_B, const uint64_t length);
template void accumulate_hadamard_minus_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, std::complex<__fp16> const* const ptr_B, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void accumulate_hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] * ptr_B[i];
    }
}
template void accumulate_hadamard_product_general<int>(int* const out_ptr, int const* const ptr_A, int const* const ptr_B, const uint64_t length);
// template void accumulate_hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length);
// template void accumulate_hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length);
template void accumulate_hadamard_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, std::complex<float> const* const ptr_B, const uint64_t length);
template void accumulate_hadamard_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, std::complex<double> const* const ptr_B, const uint64_t length);
#ifdef FP16_FLAG
template void accumulate_hadamard_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, __fp16 const* const ptr_B, const uint64_t length);
template void accumulate_hadamard_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, std::complex<__fp16> const* const ptr_B, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void accumulate_hadamard_divide_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] / ptr_B[i];
    }
}
template void accumulate_hadamard_divide_general<int>(int* const out_ptr, int const* const ptr_A, int const* const ptr_B, const uint64_t length);
// template void accumulate_hadamard_divide_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length);
// template void accumulate_hadamard_divide_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length);
template void accumulate_hadamard_divide_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, std::complex<float> const* const ptr_B, const uint64_t length);
template void accumulate_hadamard_divide_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, std::complex<double> const* const ptr_B, const uint64_t length);
#ifdef FP16_FLAG
template void accumulate_hadamard_divide_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, __fp16 const* const ptr_B, const uint64_t length);
template void accumulate_hadamard_divide_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, std::complex<__fp16> const* const ptr_B, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void accumulate_hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length, T const alpha) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += alpha * ptr_A[i] * ptr_B[i];
    }
}
template void accumulate_hadamard_product_general<int>(int* const out_ptr, int const* const ptr_A, int const* const ptr_B, const uint64_t length, int const alpha);
// template void accumulate_hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length, float const alpha);
// template void accumulate_hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length, double const alpha);
template void accumulate_hadamard_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, std::complex<float> const* const ptr_B, const uint64_t length, std::complex<float> const alpha);
template void accumulate_hadamard_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, std::complex<double> const* const ptr_B, const uint64_t length, std::complex<double> const alpha);
#ifdef FP16_FLAG
template void accumulate_hadamard_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, __fp16 const* const ptr_B, const uint64_t length, __fp16 const alpha);
template void accumulate_hadamard_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, std::complex<__fp16> const* const ptr_B, const uint64_t length, std::complex<__fp16> const alpha);
#endif //FP16_FLAG

template<typename T>
void accumulate_hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length, T const alpha, T const beta) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += alpha * ptr_A[i] * ptr_B[i] + beta * out_ptr[i];
    }
}
template void accumulate_hadamard_product_general<int>(int* const out_ptr, int const* const ptr_A, int const* const ptr_B, const uint64_t length, int const alpha, int const beta);
// template void accumulate_hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length, float const alpha, float const beta);
// template void accumulate_hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length, double const alpha, double const beta);
template void accumulate_hadamard_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, std::complex<float> const* const ptr_B, const uint64_t length, std::complex<float> const alpha, std::complex<float> const beta);
template void accumulate_hadamard_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, std::complex<double> const* const ptr_B, const uint64_t length, std::complex<double> const alpha, std::complex<double> const beta);
#ifdef FP16_FLAG
template void accumulate_hadamard_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, __fp16 const* const ptr_B, const uint64_t length, __fp16 const alpha, __fp16 const beta);
template void accumulate_hadamard_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, std::complex<__fp16> const* const ptr_B, const uint64_t length, std::complex<__fp16> const alpha, std::complex<__fp16> const beta);
#endif //FP16_FLAG

template<typename T>
void accumulate_hadamard_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, T const* const __restrict__ ptr_B, const uint64_t length, T const alpha, T const beta, T const gamma) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += alpha * ptr_A[i] * ptr_B[i] + beta * out_ptr[i] + gamma;
    }
}
template void accumulate_hadamard_product_general<int>(int* const out_ptr, int const* const ptr_A, int const* const ptr_B, const uint64_t length, int const alpha, int const beta, int const gamma);
// template void accumulate_hadamard_product_general<float>(float* const out_ptr, float const* const ptr_A, float const* const ptr_B, const uint64_t length, float const alpha, float const beta, float const gamma);
// template void accumulate_hadamard_product_general<double>(double* const out_ptr, double const* const ptr_A, double const* const ptr_B, const uint64_t length, double const alpha, double const beta, double const gamma);
template void accumulate_hadamard_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, std::complex<float> const* const ptr_B, const uint64_t length, std::complex<float> const alpha, std::complex<float> const beta, std::complex<float> const gamma);
template void accumulate_hadamard_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, std::complex<double> const* const ptr_B, const uint64_t length, std::complex<double> const alpha, std::complex<double> const beta, std::complex<double> const gamma);
#ifdef FP16_FLAG
template void accumulate_hadamard_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, __fp16 const* const ptr_B, const uint64_t length, __fp16 const alpha, __fp16 const beta, __fp16 const gamma);
template void accumulate_hadamard_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, std::complex<__fp16> const* const ptr_B, const uint64_t length, std::complex<__fp16> const alpha, std::complex<__fp16> const beta, std::complex<__fp16> const gamma);
#endif //FP16_FLAG

template<typename T>
void scalar_plus_general(T* const __restrict__ out_ptr, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += num;
    }
}
template void scalar_plus_general<int>(int* const out_ptr, const int num, const uint64_t length);
// template void scalar_plus_general<float>(float* const out_ptr, const float num, const uint64_t length);
// template void scalar_plus_general<double>(double* const out_ptr, const double num, const uint64_t length);
template void scalar_plus_general<std::complex<float>>(std::complex<float>* const out_ptr, const std::complex<float> num, const uint64_t length);
template void scalar_plus_general<std::complex<double>>(std::complex<double>* const out_ptr, const std::complex<double> num, const uint64_t length);
#ifdef FP16_FLAG
template void scalar_plus_general<__fp16>(__fp16* const out_ptr, const __fp16 num, const uint64_t length);
template void scalar_plus_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, const std::complex<__fp16> num, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void scalar_minus_general(T* const __restrict__ out_ptr, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] -= num;
    }
}
template void scalar_minus_general<int>(int* const out_ptr, const int num, const uint64_t length);
// template void scalar_minus_general<float>(float* const out_ptr, const float num, const uint64_t length);
// template void scalar_minus_general<double>(double* const out_ptr, const double num, const uint64_t length);
template void scalar_minus_general<std::complex<float>>(std::complex<float>* const out_ptr, const std::complex<float> num, const uint64_t length);
template void scalar_minus_general<std::complex<double>>(std::complex<double>* const out_ptr, const std::complex<double> num, const uint64_t length);
#ifdef FP16_FLAG
template void scalar_minus_general<__fp16>(__fp16* const out_ptr, const __fp16 num, const uint64_t length);
template void scalar_minus_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, const std::complex<__fp16> num, const uint64_t length);
#endif //FP16_FLAG

template<typename T1, typename T2>
void scalar_product_general(T1* const __restrict__ out_ptr, const T2 num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] *= num;
    }
}
template void scalar_product_general<int>(int* const out_ptr, const int num, const uint64_t length);
// template void scalar_product_general<float>(float* const out_ptr, const float num, const uint64_t length);
// template void scalar_product_general<double>(double* const out_ptr, const double num, const uint64_t length);
template void scalar_product_general<float, double>(float* const out_ptr, const double num, const uint64_t length);
template void scalar_product_general<double, float>(double* const out_ptr, const float num, const uint64_t length);
template void scalar_product_general<std::complex<float>>(std::complex<float>* const out_ptr, const std::complex<float> num, const uint64_t length);
template void scalar_product_general<std::complex<double>>(std::complex<double>* const out_ptr, const std::complex<double> num, const uint64_t length);
template void scalar_product_general<std::complex<float>, float>(std::complex<float>* const out_ptr, const float num, const uint64_t length);
template void scalar_product_general<std::complex<double>, double>(std::complex<double>* const out_ptr, const double num, const uint64_t length);
#ifdef FP16_FLAG
template void scalar_product_general<__fp16>(__fp16* const out_ptr, const __fp16 num, const uint64_t length);
template void scalar_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, const std::complex<__fp16> num, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void scalar_divide_general(T* const __restrict__ out_ptr, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] /= num;
    }
}
template void scalar_divide_general<int>(int* const out_ptr, const int num, const uint64_t length);
// template void scalar_divide_general<float>(float* const out_ptr, const float num, const uint64_t length);
// template void scalar_divide_general<double>(double* const out_ptr, const double num, const uint64_t length);
template void scalar_divide_general<std::complex<float>>(std::complex<float>* const out_ptr, const std::complex<float> num, const uint64_t length);
template void scalar_divide_general<std::complex<double>>(std::complex<double>* const out_ptr, const std::complex<double> num, const uint64_t length);
#ifdef FP16_FLAG
template void scalar_divide_general<__fp16>(__fp16* const out_ptr, const __fp16 num, const uint64_t length);
template void scalar_divide_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, const std::complex<__fp16> num, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void scalar_product_general(T* const __restrict__ out_ptr, const T num, const uint64_t length,
                                    T const* const __restrict__ ptr_B) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = num * out_ptr[i] + ptr_B[i];
    }
}
template void scalar_product_general<int>(int* const out_ptr, const int num, const uint64_t length,
                                                  int const* const ptr_B);
// template void scalar_product_general<float>(float* const out_ptr, const float num, const uint64_t length,
//                                                     float const* const ptr_B);
// template void scalar_product_general<double>(double* const out_ptr, const double num, const uint64_t length,
//                                                      double const* const ptr_B);
template void scalar_product_general<std::complex<float>>(std::complex<float>* const out_ptr, const std::complex<float> num, const uint64_t length,
                                                                    std::complex<float> const* const ptr_B);
template void scalar_product_general<std::complex<double>>(std::complex<double>* const out_ptr, const std::complex<double> num, const uint64_t length,
                                                                     std::complex<double> const* const ptr_B);
#ifdef FP16_FLAG
template void scalar_product_general<__fp16>(__fp16* const out_ptr, const __fp16 num, const uint64_t length,
                                                    __fp16 const* const ptr_B);
template void scalar_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, const std::complex<__fp16> num, const uint64_t length,
                                                                        std::complex<__fp16> const* const ptr_B);
#endif //FP16_FLAG

template<typename T1, typename T2>
void scalar_product_general(T1* const __restrict__ out_ptr, const T2 num, const uint64_t length,
                                    T1 const* const __restrict__ ptr_B, T2 const beta) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = num * out_ptr[i] + beta * ptr_B[i];
    }
}
template void scalar_product_general<int>(int* const out_ptr, const int num, const uint64_t length,
                                                  int const* const ptr_B, int const beta);
// template void scalar_product_general<float>(float* const out_ptr, const float num, const uint64_t length,
//                                                     float const* const ptr_B, float const beta);
// template void scalar_product_general<double>(double* const out_ptr, const double num, const uint64_t length,
//                                                      double const* const ptr_B, double const beta);
template void scalar_product_general<std::complex<float>>(std::complex<float>* const out_ptr, const std::complex<float> num, const uint64_t length,
                                                                    std::complex<float> const* const ptr_B, std::complex<float> const beta);
template void scalar_product_general<std::complex<double>>(std::complex<double>* const out_ptr, const std::complex<double> num, const uint64_t length,
                                                                     std::complex<double> const* const ptr_B, std::complex<double> const beta);
template void scalar_product_general<std::complex<float>, float>(std::complex<float>* const out_ptr, const float num, const uint64_t length,
                                                                    std::complex<float> const* const ptr_B, float const beta);
template void scalar_product_general<std::complex<double>, double>(std::complex<double>* const out_ptr, const double num, const uint64_t length,
                                                                     std::complex<double> const* const ptr_B, double const beta);
#ifdef FP16_FLAG
template void scalar_product_general<__fp16>(__fp16* const out_ptr, const __fp16 num, const uint64_t length,
                                                    __fp16 const* const ptr_B, __fp16 const beta);
template void scalar_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, const std::complex<__fp16> num, const uint64_t length,
                                                                        std::complex<__fp16> const* const ptr_B, std::complex<__fp16> const beta);
#endif //FP16_FLAG

template<typename T>
void scalar_product_general(T* const __restrict__ out_ptr, const T num, const uint64_t length,
                                    T const* const __restrict__ ptr_B, T const beta, T const gamma) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = num * out_ptr[i] + beta * ptr_B[i] + gamma;
    }
}
template void scalar_product_general<int>(int* const out_ptr, const int num, const uint64_t length,
                                            int const* const ptr_B, int const beta, int const gamma);
// template void scalar_product_general<float>(float* const out_ptr, const float num, const uint64_t length,
//                                             float const* const ptr_B, float const beta, float const gamma);
// template void scalar_product_general<double>(double* const out_ptr, const double num, const uint64_t length,
//                                                      double const* const ptr_B, double const beta, double const gamma);
template void scalar_product_general<std::complex<float>>(std::complex<float>* const out_ptr, const std::complex<float> num, const uint64_t length,
                                                                    std::complex<float> const* const ptr_B, std::complex<float> const beta, std::complex<float> const gamma);
template void scalar_product_general<std::complex<double>>(std::complex<double>* const out_ptr, const std::complex<double> num, const uint64_t length,
                                                                     std::complex<double> const* const ptr_B, std::complex<double> const beta, std::complex<double> const gamma);
#ifdef FP16_FLAG
template void scalar_product_general<__fp16>(__fp16* const out_ptr, const __fp16 num, const uint64_t length,
                                                    __fp16 const* const ptr_B, __fp16 const beta, __fp16 const gamma);
template void scalar_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, const std::complex<__fp16> num, const uint64_t length,
                                                                        std::complex<__fp16> const* const ptr_B, std::complex<__fp16> const beta, std::complex<__fp16> const gamma);
#endif //FP16_FLAG

template<typename T>
void scalar_plus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] + num;
    }
}
template void scalar_plus_general<int>(int* const out_ptr, int const* const ptr_A, const int num, const uint64_t length);
// template void scalar_plus_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length);
// template void scalar_plus_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length);
template void scalar_plus_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const std::complex<float> num, const uint64_t length);
template void scalar_plus_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const std::complex<double> num, const uint64_t length);
#ifdef FP16_FLAG
template void scalar_plus_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const __fp16 num, const uint64_t length);
template void scalar_plus_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const std::complex<__fp16> num, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void scalar_minus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] - num;
    }
}
template void scalar_minus_general<int>(int* const out_ptr, int const* const ptr_A, const int num, const uint64_t length);
// template void scalar_minus_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length);
// template void scalar_minus_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length);
template void scalar_minus_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const std::complex<float> num, const uint64_t length);
template void scalar_minus_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const std::complex<double> num, const uint64_t length);
#ifdef FP16_FLAG
template void scalar_minus_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const __fp16 num, const uint64_t length);
template void scalar_minus_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const std::complex<__fp16> num, const uint64_t length);
#endif //FP16_FLAG

template<typename T1, typename T2>
void scalar_product_general(T1* const __restrict__ out_ptr, T1 const* const __restrict__ ptr_A, const T2 num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] * num;
    }
}
template void scalar_product_general<int>(int* const out_ptr, int const* const ptr_A, const int num, const uint64_t length);
// template void scalar_product_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length);
// template void scalar_product_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length);
template void scalar_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const std::complex<float> num, const uint64_t length);
template void scalar_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const std::complex<double> num, const uint64_t length);
template void scalar_product_general<std::complex<float>, float>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const float num, const uint64_t length);
template void scalar_product_general<std::complex<double>, double>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const double num, const uint64_t length);
#ifdef FP16_FLAG
template void scalar_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const __fp16 num, const uint64_t length);
template void scalar_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const std::complex<__fp16> num, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void scalar_divide_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] / num;
    }
}
template void scalar_divide_general<int>(int* const out_ptr, int const* const ptr_A, const int num, const uint64_t length);
// template void scalar_divide_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length);
// template void scalar_divide_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length);
template void scalar_divide_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const std::complex<float> num, const uint64_t length);
template void scalar_divide_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const std::complex<double> num, const uint64_t length);
#ifdef FP16_FLAG
template void scalar_divide_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const __fp16 num, const uint64_t length);
template void scalar_divide_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const std::complex<__fp16> num, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void scalar_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length,
                                    T const* const __restrict__ ptr_B) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] * num + ptr_B[i];
    }
}
template void scalar_product_general<int>(int* const out_ptr, int const* const ptr_A, const int num, const uint64_t length,
                                                  int const* const ptr_B);
// template void scalar_product_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length,
//                                                     float const* const ptr_B);
// template void scalar_product_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length,
//                                                      double const* const ptr_B);
template void scalar_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const std::complex<float> num, const uint64_t length,
                                                                    std::complex<float> const* const ptr_B);
template void scalar_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const std::complex<double> num, const uint64_t length,
                                                                     std::complex<double> const* const ptr_B);
#ifdef FP16_FLAG
template void scalar_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const __fp16 num, const uint64_t length,
                                                     __fp16 const* const ptr_B);
template void scalar_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const std::complex<__fp16> num, const uint64_t length,
                                                                        std::complex<__fp16> const* const ptr_B);
#endif //FP16_FLAG

template<typename T>
void scalar_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length,
                                    T const* const __restrict__ ptr_B, T const beta) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] * num + ptr_B[i] * beta;
    }
}
template void scalar_product_general<int>(int* const out_ptr, int const* const ptr_A, const int num, const uint64_t length,
                                                  int const* const ptr_B, int const beta);
// template void scalar_product_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length,
//                                                     float const* const ptr_B, float const beta);
// template void scalar_product_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length,
//                                                      double const* const ptr_B, double const beta);
template void scalar_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const std::complex<float> num, const uint64_t length,
                                                                    std::complex<float> const* const ptr_B, std::complex<float> const beta);
template void scalar_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const std::complex<double> num, const uint64_t length,
                                                                     std::complex<double> const* const ptr_B, std::complex<double> const beta);
#ifdef FP16_FLAG
template void scalar_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const __fp16 num, const uint64_t length,
                                                     __fp16 const* const ptr_B, __fp16 const beta);
template void scalar_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const std::complex<__fp16> num, const uint64_t length,
                                                                        std::complex<__fp16> const* const ptr_B, std::complex<__fp16> const beta);
#endif //FP16_FLAG

template<typename T>
void scalar_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length,
                                    T const* const __restrict__ ptr_B, T const beta, T const gamma) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] = ptr_A[i] * num + ptr_B[i] * beta + gamma;
    }
}
template void scalar_product_general<int>(int* const out_ptr, int const* const ptr_A, const int num, const uint64_t length,
                                                  int const* const ptr_B, int const beta, int const gamma);
// template void scalar_product_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length,
//                                                     float const* const ptr_B, float const beta, float const gamma);
// template void scalar_product_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length,
//                                                      double const* const ptr_B, double const beta, double const gamma);
template void scalar_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const std::complex<float> num, const uint64_t length,
                                                                    std::complex<float> const* const ptr_B, std::complex<float> const beta, std::complex<float> const gamma);
template void scalar_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const std::complex<double> num, const uint64_t length,
                                                                     std::complex<double> const* const ptr_B, std::complex<double> const beta, std::complex<double> const gamma);
#ifdef FP16_FLAG
template void scalar_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const __fp16 num, const uint64_t length,
                                                     __fp16 const* const ptr_B, __fp16 const beta, __fp16 const gamma);
template void scalar_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const std::complex<__fp16> num, const uint64_t length,
                                                                        std::complex<__fp16> const* const ptr_B, std::complex<__fp16> const beta, std::complex<__fp16> const gamma);
#endif //FP16_FLAG

template<typename T>
void accumulate_scalar_plus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] + num;
    }
}
template void accumulate_scalar_plus_general<int>(int* const out_ptr, int const* const ptr_A, const int num, const uint64_t length);
// template void accumulate_scalar_plus_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length);
// template void accumulate_scalar_plus_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length);
template void accumulate_scalar_plus_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const std::complex<float> num, const uint64_t length);
template void accumulate_scalar_plus_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const std::complex<double> num, const uint64_t length);
#ifdef FP16_FLAG
template void accumulate_scalar_plus_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const __fp16 num, const uint64_t length);
template void accumulate_scalar_plus_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const std::complex<__fp16> num, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void accumulate_scalar_minus_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] - num;
    }
}
template void accumulate_scalar_minus_general<int>(int* const out_ptr, int const* const ptr_A, const int num, const uint64_t length);
// template void accumulate_scalar_minus_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length);
// template void accumulate_scalar_minus_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length);
template void accumulate_scalar_minus_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const std::complex<float> num, const uint64_t length);
template void accumulate_scalar_minus_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const std::complex<double> num, const uint64_t length);
#ifdef FP16_FLAG
template void accumulate_scalar_minus_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const __fp16 num, const uint64_t length);
template void accumulate_scalar_minus_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const std::complex<__fp16> num, const uint64_t length);
#endif //FP16_FLAG

template<typename T1, typename T2>
void accumulate_scalar_product_general(T1* const __restrict__ out_ptr, T1 const* const __restrict__ ptr_A, const T2 num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] * num;
    }
}
template void accumulate_scalar_product_general<int>(int* const out_ptr, int const* const ptr_A, const int num, const uint64_t length);
// template void accumulate_scalar_product_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length);
// template void accumulate_scalar_product_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length);
template void accumulate_scalar_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const std::complex<float> num, const uint64_t length);
template void accumulate_scalar_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const std::complex<double> num, const uint64_t length);
template void accumulate_scalar_product_general<std::complex<float>, float>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const float num, const uint64_t length);
template void accumulate_scalar_product_general<std::complex<double>, double>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const double num, const uint64_t length);
#ifdef FP16_FLAG
template void accumulate_scalar_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const __fp16 num, const uint64_t length);
template void accumulate_scalar_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const std::complex<__fp16> num, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void accumulate_scalar_divide_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] / num;
    }
}
template void accumulate_scalar_divide_general<int>(int* const out_ptr, int const* const ptr_A, const int num, const uint64_t length);
// template void accumulate_scalar_divide_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length);
// template void accumulate_scalar_divide_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length);
template void accumulate_scalar_divide_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const std::complex<float> num, const uint64_t length);
template void accumulate_scalar_divide_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const std::complex<double> num, const uint64_t length);
#ifdef FP16_FLAG
template void accumulate_scalar_divide_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const __fp16 num, const uint64_t length);
template void accumulate_scalar_divide_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const std::complex<__fp16> num, const uint64_t length);
#endif //FP16_FLAG

template<typename T>
void accumulate_scalar_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length,
                                    T const* const __restrict__ ptr_B) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] * num + ptr_B[i];
    }
}
template void accumulate_scalar_product_general<int>(int* const out_ptr, int const* const ptr_A, const int num, const uint64_t length,
                                                  int const* const ptr_B);
// template void accumulate_scalar_product_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length,
//                                                     float const* const ptr_B);
// template void accumulate_scalar_product_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length,
//                                                      double const* const ptr_B);
template void accumulate_scalar_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const std::complex<float> num, const uint64_t length,
                                                                    std::complex<float> const* const ptr_B);
template void accumulate_scalar_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const std::complex<double> num, const uint64_t length,
                                                                     std::complex<double> const* const ptr_B);
#ifdef FP16_FLAG
template void accumulate_scalar_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const __fp16 num, const uint64_t length,
                                                     __fp16 const* const ptr_B);
template void accumulate_scalar_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const std::complex<__fp16> num, const uint64_t length,
                                                                        std::complex<__fp16> const* const ptr_B);
#endif //FP16_FLAG

template<typename T>
void accumulate_scalar_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length,
                                    T const* const __restrict__ ptr_B, T const beta) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] * num + ptr_B[i] * beta;
    }
}
template void accumulate_scalar_product_general<int>(int* const out_ptr, int const* const ptr_A, const int num, const uint64_t length,
                                                  int const* const ptr_B, int const beta);
// template void accumulate_scalar_product_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length,
//                                                     float const* const ptr_B, float const beta);
// template void accumulate_scalar_product_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length,
//                                                      double const* const ptr_B, double const beta);
template void accumulate_scalar_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const std::complex<float> num, const uint64_t length,
                                                                    std::complex<float> const* const ptr_B, std::complex<float> const beta);
template void accumulate_scalar_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const std::complex<double> num, const uint64_t length,
                                                                     std::complex<double> const* const ptr_B, std::complex<double> const beta);
#ifdef FP16_FLAG
template void accumulate_scalar_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const __fp16 num, const uint64_t length,
                                                     __fp16 const* const ptr_B, __fp16 const beta);
template void accumulate_scalar_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const std::complex<__fp16> num, const uint64_t length,
                                                                        std::complex<__fp16> const* const ptr_B, std::complex<__fp16> const beta);
#endif //FP16_FLAG

template<typename T>
void accumulate_scalar_product_general(T* const __restrict__ out_ptr, T const* const __restrict__ ptr_A, const T num, const uint64_t length,
                                    T const* const __restrict__ ptr_B, T const beta, T const gamma) {
    FOR_WHILE(uint64_t i = 0, i < length, i++, (length - 1)/omp_get_num_threads() + 1) {
        out_ptr[i] += ptr_A[i] * num + ptr_B[i] * beta + gamma;
    }
}
template void accumulate_scalar_product_general<int>(int* const out_ptr, int const* const ptr_A, const int num, const uint64_t length,
                                                  int const* const ptr_B, int const beta, int const gamma);
// template void accumulate_scalar_product_general<float>(float* const out_ptr, float const* const ptr_A, const float num, const uint64_t length,
//                                                     float const* const ptr_B, float const beta, float const gamma);
// template void accumulate_scalar_product_general<double>(double* const out_ptr, double const* const ptr_A, const double num, const uint64_t length,
//                                                      double const* const ptr_B, double const beta, double const gamma);
template void accumulate_scalar_product_general<std::complex<float>>(std::complex<float>* const out_ptr, std::complex<float> const* const ptr_A, const std::complex<float> num, const uint64_t length,
                                                                    std::complex<float> const* const ptr_B, std::complex<float> const beta, std::complex<float> const gamma);
template void accumulate_scalar_product_general<std::complex<double>>(std::complex<double>* const out_ptr, std::complex<double> const* const ptr_A, const std::complex<double> num, const uint64_t length,
                                                                     std::complex<double> const* const ptr_B, std::complex<double> const beta, std::complex<double> const gamma);
#ifdef FP16_FLAG
template void accumulate_scalar_product_general<__fp16>(__fp16* const out_ptr, __fp16 const* const ptr_A, const __fp16 num, const uint64_t length,
                                                     __fp16 const* const ptr_B, __fp16 const beta, __fp16 const gamma);
template void accumulate_scalar_product_general<std::complex<__fp16>>(std::complex<__fp16>* const out_ptr, std::complex<__fp16> const* const ptr_A, const std::complex<__fp16> num, const uint64_t length,
                                                                        std::complex<__fp16> const* const ptr_B, std::complex<__fp16> const beta, std::complex<__fp16> const gamma);
#endif //FP16_FLAG

//Matrix functions
template<typename T>
void matrix_vector_product(const T* ptr_A, const uint is_Col_Maj_A,
                                   const T* ptr_B, T* ptr_C,
                                   const int64_t m, const int64_t n) {
    #ifdef USE_CBLAS
    matrix_vector_product_cblas(ptr_A, is_Col_Maj_A, ptr_B, ptr_C, m, n);
    #else
    matrix_product_general(ptr_A, is_Col_Maj_A, ptr_B, 1,
                                   ptr_C, 1, m, 1, n);
    #endif
    return;
}
template void matrix_vector_product<float>(const float* ptr_A, const uint is_Col_Maj_A,
                                                   const float* ptr_B, float* ptr_C,
                                                   const int64_t m, const int64_t n);
template void matrix_vector_product<double>(const double* ptr_A, const uint is_Col_Maj_A,
                                                    const double* ptr_B, double* ptr_C,
                                                    const int64_t m, const int64_t n);

template<typename T>
void matrix_vector_product(const T* ptr_A, const uint is_Col_Maj_A,
                                   const T* ptr_B, T* ptr_C,
                                   const int64_t m, const int64_t n,
                                   const T alpha) {
    #ifdef USE_CBLAS
    matrix_vector_product_cblas(ptr_A, is_Col_Maj_A, ptr_B, ptr_C, m, n, alpha);
    #else
    matrix_product_general(ptr_A, is_Col_Maj_A, ptr_B, 1,
                                   ptr_C, 1, m, 1, n, alpha);
    #endif
    return;
}
template void matrix_vector_product<float>(const float* ptr_A, const uint is_Col_Maj_A,
                                                   const float* ptr_B, float* ptr_C,
                                                   const int64_t m, const int64_t n,
                                                   const float alpha);
template void matrix_vector_product<double>(const double* ptr_A, const uint is_Col_Maj_A,
                                                    const double* ptr_B, double* ptr_C,
                                                    const int64_t m, const int64_t n,
                                                    const double alpha);

template<typename T>
void matrix_vector_product(const T* ptr_A, const uint is_Col_Maj_A,
                                   const T* ptr_B, T* ptr_C,
                                   const int64_t m, const int64_t n,
                                   const T alpha, const T beta) {
    #ifdef USE_CBLAS
    matrix_vector_product_cblas(ptr_A, is_Col_Maj_A, ptr_B, ptr_C, m, n, alpha, beta);
    #else
    matrix_product_general(ptr_A, is_Col_Maj_A, ptr_B, 1,
                                   ptr_C, 1, m, 1, n, alpha, beta);
    #endif
    return;
}
template void matrix_vector_product<float>(const float* ptr_A, const uint is_Col_Maj_A,
                                                   const float* ptr_B, float* ptr_C,
                                                   const int64_t m, const int64_t n,
                                                   const float alpha, const float beta);
template void matrix_vector_product<double>(const double* ptr_A, const uint is_Col_Maj_A,
                                                    const double* ptr_B, double* ptr_C,
                                                    const int64_t m, const int64_t n,
                                                    const double alpha, const double beta);

template<typename T>
void matrix_vector_product(const T* ptr_A, const uint is_Col_Maj_A,
                                   const T* ptr_B, T* ptr_C,
                                   const int64_t m, const int64_t n,
                                   const T alpha, const T beta, const T gamma) {
    #ifdef USE_CBLAS
    matrix_vector_product_cblas(ptr_A, is_Col_Maj_A, ptr_B, ptr_C, m, n, alpha, beta, gamma);
    #else
    matrix_product_general(ptr_A, is_Col_Maj_A, ptr_B, 1,
                                   ptr_C, 1, m, 1, n, alpha, beta, gamma);
    #endif
    return;
}
template void matrix_vector_product<float>(const float* ptr_A, const uint is_Col_Maj_A,
                                                   const float* ptr_B, float* ptr_C,
                                                   const int64_t m, const int64_t n,
                                                   const float alpha, const float beta, const float gamma);
template void matrix_vector_product<double>(const double* ptr_A, const uint is_Col_Maj_A,
                                                    const double* ptr_B, double* ptr_C,
                                                    const int64_t m, const int64_t n,
                                                    const double alpha, const double beta, const double gamma);

template<typename T>
void matrix_product(const T* ptr_A, const uint is_Col_Maj_A, const T* ptr_B, const uint is_Col_Maj_B,
                            T* ptr_C, const uint is_Col_Maj_C, const int64_t m, const int64_t n, const int64_t k) {
    #ifdef USE_CBLAS
    matrix_product_cblas(ptr_A, is_Col_Maj_A, ptr_B, is_Col_Maj_B,
                                 ptr_C, is_Col_Maj_C, m, n, k);
    #else
    matrix_product_general(ptr_A, is_Col_Maj_A, ptr_B, is_Col_Maj_B,
                                   ptr_C, is_Col_Maj_C, m, n, k);
    #endif
}
template void matrix_product<float>(const float* ptr_A, const uint is_Col_Maj_A,
                                            const float* ptr_B, const uint is_Col_Maj_B,
                                            float* ptr_C, const uint is_Col_Maj_C,
                                            const int64_t m, const int64_t n, const int64_t k);
template void matrix_product<double>(const double* ptr_A, const uint is_Col_Maj_A,
                                             const double* ptr_B, const uint is_Col_Maj_B,
                                             double* ptr_C, const uint is_Col_Maj_C,
                                             const int64_t m, const int64_t n, const int64_t k);

template<typename T>
void matrix_product(const T* ptr_A, const uint is_Col_Maj_A, const T* ptr_B, const uint is_Col_Maj_B,
                            T* ptr_C, const uint is_Col_Maj_C, const int64_t m, const int64_t n, const int64_t k,
                            const T alpha) {
    #ifdef USE_CBLAS
    matrix_product_cblas(ptr_A, is_Col_Maj_A, ptr_B, is_Col_Maj_B,
                                 ptr_C, is_Col_Maj_C, m, n, k, alpha);
    #else
    matrix_product_general(ptr_A, is_Col_Maj_A, ptr_B, is_Col_Maj_B,
                                   ptr_C, is_Col_Maj_C, m, n, k, alpha);
    #endif
}
template void matrix_product<float>(const float* ptr_A, const uint is_Col_Maj_A,
                                            const float* ptr_B, const uint is_Col_Maj_B,
                                            float* ptr_C, const uint is_Col_Maj_C,
                                            const int64_t m, const int64_t n, const int64_t k,
                                            const float alpha);
template void matrix_product<double>(const double* ptr_A, const uint is_Col_Maj_A,
                                             const double* ptr_B, const uint is_Col_Maj_B,
                                             double* ptr_C, const uint is_Col_Maj_C,
                                             const int64_t m, const int64_t n, const int64_t k,
                                             const double alpha);

template<typename T> 
void matrix_product(const T* ptr_A, const uint is_Col_Maj_A, const T* ptr_B, const uint is_Col_Maj_B,
                            T* ptr_C, const uint is_Col_Maj_C, const int64_t m, const int64_t n, const int64_t k,
                            const T alpha, const T beta) {
    #ifdef USE_CBLAS
    matrix_product_cblas(ptr_A, is_Col_Maj_A, ptr_B, is_Col_Maj_B,
                                 ptr_C, is_Col_Maj_C, m, n, k, alpha, beta);
    #else
    matrix_product_general(ptr_A, is_Col_Maj_A, ptr_B, is_Col_Maj_B,
                                   ptr_C, is_Col_Maj_C, m, n, k, alpha, beta);
    #endif
}
template void matrix_product<float>(const float* ptr_A, const uint is_Col_Maj_A,
                                            const float* ptr_B, const uint is_Col_Maj_B,
                                            float* ptr_C, const uint is_Col_Maj_C,
                                            const int64_t m, const int64_t n, const int64_t k,
                                            const float alpha, const float beta);
template void matrix_product<double>(const double* ptr_A, const uint is_Col_Maj_A,
                                             const double* ptr_B, const uint is_Col_Maj_B,
                                             double* ptr_C, const uint is_Col_Maj_C,
                                             const int64_t m, const int64_t n, const int64_t k,
                                             const double alpha, const double beta);

template<typename T>                                          
void matrix_product(const T* ptr_A, const uint is_Col_Maj_A, const T* ptr_B, const uint is_Col_Maj_B,
                            T* ptr_C, const uint is_Col_Maj_C, const int64_t m, const int64_t n, const int64_t k,
                            const T alpha, const T beta, const T gamma) {
    #ifdef USE_CBLAS
    matrix_product_cblas(ptr_A, is_Col_Maj_A, ptr_B, is_Col_Maj_B,
                                 ptr_C, is_Col_Maj_C, m, n, k, alpha, beta, gamma);
    #else
    matrix_product_general(ptr_A, is_Col_Maj_A, ptr_B, is_Col_Maj_B,
                                   ptr_C, is_Col_Maj_C, m, n, k, alpha, beta, gamma);
    #endif
}
template void matrix_product<float>(const float* ptr_A, const uint is_Col_Maj_A,
                                            const float* ptr_B, const uint is_Col_Maj_B,
                                            float* ptr_C, const uint is_Col_Maj_C,
                                            const int64_t m, const int64_t n, const int64_t k,
                                            const float alpha, const float beta, const float gamma);
template void matrix_product<double>(const double* ptr_A, const uint is_Col_Maj_A,
                                             const double* ptr_B, const uint is_Col_Maj_B,
                                             double* ptr_C, const uint is_Col_Maj_C,
                                             const int64_t m, const int64_t n, const int64_t k,
                                             const double alpha, const double beta, const double gamma);

template<typename T>
void matrix_product_general(const T* ptr_A, const uint is_Col_Maj_A,
                                    const T* ptr_B, const uint is_Col_Maj_B,
                                    T* ptr_C, const uint is_Col_Maj_C,
                                    const int64_t m, const int64_t n, const int64_t k) {
    uint64_t (*A_get_index)(const uint64_t i, const uint64_t j, const uint64_t ld) = nullptr;
    uint64_t (*B_get_index)(const uint64_t i, const uint64_t j, const uint64_t ld) = nullptr;
    uint64_t (*C_get_i)(const uint64_t index, const uint64_t ld) = nullptr;
    uint64_t (*C_get_j)(const uint64_t index, const uint64_t ld) = nullptr;
    int64_t A_ld, B_ld, C_ld;
    switch(4 * is_Col_Maj_A + 2 * is_Col_Maj_B + is_Col_Maj_C) {
        case 3 :  // A: Row B: Col C: Col
            A_get_index = Index::get_index_Row_Maj;
            A_ld = k;
            B_get_index = Index::get_index_Col_Maj;
            B_ld = k;
            C_get_i = Index::get_i_Col_Maj;
            C_get_j = Index::get_j_Col_Maj;
            C_ld = m;
            break;
        case 7 :  // A: Col B: Col C: Col
            A_get_index = Index::get_index_Col_Maj;
            A_ld = m;
            B_get_index = Index::get_index_Col_Maj;
            B_ld = k;
            C_get_i = Index::get_i_Col_Maj;
            C_get_j = Index::get_j_Col_Maj;
            C_ld = m;
            break;
        case 2 :  // A: Row B: Col C: Row
            A_get_index = Index::get_index_Row_Maj;
            A_ld = k;
            B_get_index = Index::get_index_Col_Maj;
            B_ld = k;
            C_get_i = Index::get_i_Row_Maj;
            C_get_j = Index::get_j_Row_Maj;
            C_ld = n;
            break;
        case 6 :  // A: Col B: Col C: Row
            A_get_index = Index::get_index_Col_Maj;
            A_ld = m;
            B_get_index = Index::get_index_Col_Maj;
            B_ld = k;
            C_get_i = Index::get_i_Row_Maj;
            C_get_j = Index::get_j_Row_Maj;
            C_ld = n;
            break;
        case 1 :  // A: Row B: Row C: Col
            A_get_index = Index::get_index_Row_Maj;
            A_ld = k;
            B_get_index = Index::get_index_Row_Maj;
            B_ld = n;
            C_get_i = Index::get_i_Col_Maj;
            C_get_j = Index::get_j_Col_Maj;
            C_ld = m;
            break;
        case 5 :  // A: Col B: Row C: Col
            A_get_index = Index::get_index_Col_Maj;
            A_ld = m;
            B_get_index = Index::get_index_Row_Maj;
            B_ld = n;
            C_get_i = Index::get_i_Col_Maj;
            C_get_j = Index::get_j_Col_Maj;
            C_ld = m;
            break;
        case 0 :  // A: Row B: Row C: Row
            A_get_index = Index::get_index_Row_Maj;
            A_ld = k;
            B_get_index = Index::get_index_Row_Maj;
            B_ld = n;
            C_get_i = Index::get_i_Row_Maj;
            C_get_j = Index::get_j_Row_Maj;
            C_ld = n;
            break;
        case 4 :  // A: Col B: Row C: Row
            A_get_index = Index::get_index_Col_Maj;
            A_ld = m;
            B_get_index = Index::get_index_Row_Maj;
            B_ld = n;
            C_get_i = Index::get_i_Row_Maj;
            C_get_j = Index::get_j_Row_Maj;
            C_ld = n;
            break;
        default :
            assert(0 && "The matrix should be Col(1) or Row(0) Major!");
    }
    int64_t length = m * n;
    #if (defined(USE_OPENMP_SIMD) && defined(USE_OPENMP))
    #pragma omp for simd schedule(static, (length - 1)/omp_get_num_threads() + 1) nowait
    #elif defined(USE_OPENMP)
    #pragma omp for schedule(static, (length - 1)/omp_get_num_threads() + 1) nowait
    #elif defined(USE_OPENMP_SIMD)
    #pragma omp simd
    #endif
    for (int64_t index_C = 0; index_C < length; index_C++) {
        ptr_C[index_C] = 0;
        int64_t i_C = C_get_i(index_C, C_ld);
        int64_t j_C = C_get_j(index_C, C_ld);
        for (int64_t j_A = 0, i_B = 0; j_A < k; j_A++, i_B++)
        {
            ptr_C[index_C] += ptr_A[A_get_index(i_C, j_A, A_ld)] * ptr_B[B_get_index(i_B, j_C, B_ld)];
        }
    }          
}
// template void matrix_product_general<int>(const int* ptr_A, const uint is_Col_Maj_A,
//                                                   const int* ptr_B, const uint is_Col_Maj_B,
//                                                   int* ptr_C, const uint is_Col_Maj_C,
//                                                   const uint m, const uint n, const uint k);
// template void matrix_product_general<float>(const float* ptr_A, const uint is_Col_Maj_A,
//                                                     const float* ptr_B, const uint is_Col_Maj_B,
//                                                     float* ptr_C, const uint is_Col_Maj_C,
//                                                     const uint m, const uint n, const uint k);
// template void matrix_product_general<double>(const double* ptr_A, const uint is_Col_Maj_A,
//                                                      const double* ptr_B, const uint is_Col_Maj_B,
//                                                      double* ptr_C, const uint is_Col_Maj_C,
//                                                      const uint m, const uint n, const uint k);

template<typename T>
void matrix_product_general(const T* ptr_A, const uint is_Col_Maj_A,
                                    const T* ptr_B, const uint is_Col_Maj_B,
                                    T* ptr_C, const uint is_Col_Maj_C,
                                    const int64_t m, const int64_t n, const int64_t k,
                                    const T alpha) {
    uint64_t (*A_get_index)(const uint64_t i, const uint64_t j, const uint64_t ld) = nullptr;
    uint64_t (*B_get_index)(const uint64_t i, const uint64_t j, const uint64_t ld) = nullptr;
    uint64_t (*C_get_i)(const uint64_t index, const uint64_t ld) = nullptr;
    uint64_t (*C_get_j)(const uint64_t index, const uint64_t ld) = nullptr;
    int64_t A_ld, B_ld, C_ld;
    switch(4 * is_Col_Maj_A + 2 * is_Col_Maj_B + is_Col_Maj_C) {
        case 3 :  // A: Row B: Col C: Col
            A_get_index = Index::get_index_Row_Maj;
            A_ld = k;
            B_get_index = Index::get_index_Col_Maj;
            B_ld = k;
            C_get_i = Index::get_i_Col_Maj;
            C_get_j = Index::get_j_Col_Maj;
            C_ld = m;
            break;
        case 7 :  // A: Col B: Col C: Col
            A_get_index = Index::get_index_Col_Maj;
            A_ld = m;
            B_get_index = Index::get_index_Col_Maj;
            B_ld = k;
            C_get_i = Index::get_i_Col_Maj;
            C_get_j = Index::get_j_Col_Maj;
            C_ld = m;
            break;
        case 2 :  // A: Row B: Col C: Row
            A_get_index = Index::get_index_Row_Maj;
            A_ld = k;
            B_get_index = Index::get_index_Col_Maj;
            B_ld = k;
            C_get_i = Index::get_i_Row_Maj;
            C_get_j = Index::get_j_Row_Maj;
            C_ld = n;
            break;
        case 6 :  // A: Col B: Col C: Row
            A_get_index = Index::get_index_Col_Maj;
            A_ld = m;
            B_get_index = Index::get_index_Col_Maj;
            B_ld = k;
            C_get_i = Index::get_i_Row_Maj;
            C_get_j = Index::get_j_Row_Maj;
            C_ld = n;
            break;
        case 1 :  // A: Row B: Row C: Col
            A_get_index = Index::get_index_Row_Maj;
            A_ld = k;
            B_get_index = Index::get_index_Row_Maj;
            B_ld = n;
            C_get_i = Index::get_i_Col_Maj;
            C_get_j = Index::get_j_Col_Maj;
            C_ld = m;
            break;
        case 5 :  // A: Col B: Row C: Col
            A_get_index = Index::get_index_Col_Maj;
            A_ld = m;
            B_get_index = Index::get_index_Row_Maj;
            B_ld = n;
            C_get_i = Index::get_i_Col_Maj;
            C_get_j = Index::get_j_Col_Maj;
            C_ld = m;
            break;
        case 0 :  // A: Row B: Row C: Row
            A_get_index = Index::get_index_Row_Maj;
            A_ld = k;
            B_get_index = Index::get_index_Row_Maj;
            B_ld = n;
            C_get_i = Index::get_i_Row_Maj;
            C_get_j = Index::get_j_Row_Maj;
            C_ld = n;
            break;
        case 4 :  // A: Col B: Row C: Row
            A_get_index = Index::get_index_Col_Maj;
            A_ld = m;
            B_get_index = Index::get_index_Row_Maj;
            B_ld = n;
            C_get_i = Index::get_i_Row_Maj;
            C_get_j = Index::get_j_Row_Maj;
            C_ld = n;
            break;
        default :
            assert(0 && "The matrix should be Col(1) or Row(0) Major!");
    }
    int64_t length = m * n;
    #if (defined(USE_OPENMP_SIMD) && defined(USE_OPENMP))
    #pragma omp for simd schedule(static, (length - 1)/omp_get_num_threads() + 1) nowait
    #elif defined(USE_OPENMP)
    #pragma omp for schedule(static, (length - 1)/omp_get_num_threads() + 1) nowait
    #elif defined(USE_OPENMP_SIMD)
    #pragma omp simd
    #endif
    for (int64_t index_C = 0; index_C < length; index_C++) {
        ptr_C[index_C] = 0;
        int64_t i_C = C_get_i(index_C, C_ld);
        int64_t j_C = C_get_j(index_C, C_ld);
        for (int64_t j_A = 0, i_B = 0; j_A < k; j_A++, i_B++)
        {
            ptr_C[index_C] += ptr_A[A_get_index(i_C, j_A, A_ld)] * ptr_B[B_get_index(i_B, j_C, B_ld)];
        }
        ptr_C[index_C] *= alpha;
    }          
}
// template void matrix_product_general<int>(const int* ptr_A, const uint is_Col_Maj_A,
//                                                   const int* ptr_B, const uint is_Col_Maj_B,
//                                                   int* ptr_C, const uint is_Col_Maj_C,
//                                                   const uint m, const uint n, const uint k,
//                                                   const int alpha);
// template void matrix_product_general<float>(const float* ptr_A, const uint is_Col_Maj_A,
//                                                     const float* ptr_B, const uint is_Col_Maj_B,
//                                                     float* ptr_C, const uint is_Col_Maj_C,
//                                                     const uint m, const uint n, const uint k,
//                                                     const float alpha);
// template void matrix_product_general<double>(const double* ptr_A, const uint is_Col_Maj_A,
//                                                      const double* ptr_B, const uint is_Col_Maj_B,
//                                                      double* ptr_C, const uint is_Col_Maj_C,
//                                                      const uint m, const uint n, const uint kk,
//                                                      const double alpha);

template<typename T>
void matrix_product_general(const T* ptr_A, const uint is_Col_Maj_A,
                                    const T* ptr_B, const uint is_Col_Maj_B,
                                    T* ptr_C, const uint is_Col_Maj_C,
                                    const int64_t m, const int64_t n, const int64_t k,
                                    const T alpha, const T beta) {
    uint64_t (*A_get_index)(const uint64_t i, const uint64_t j, const uint64_t ld) = nullptr;
    uint64_t (*B_get_index)(const uint64_t i, const uint64_t j, const uint64_t ld) = nullptr;
    uint64_t (*C_get_i)(const uint64_t index, const uint64_t ld) = nullptr;
    uint64_t (*C_get_j)(const uint64_t index, const uint64_t ld) = nullptr;
    int64_t A_ld, B_ld, C_ld;
    switch(4 * is_Col_Maj_A + 2 * is_Col_Maj_B + is_Col_Maj_C) {
        case 3 :  // A: Row B: Col C: Col
            A_get_index = Index::get_index_Row_Maj;
            A_ld = k;
            B_get_index = Index::get_index_Col_Maj;
            B_ld = k;
            C_get_i = Index::get_i_Col_Maj;
            C_get_j = Index::get_j_Col_Maj;
            C_ld = m;
            break;
        case 7 :  // A: Col B: Col C: Col
            A_get_index = Index::get_index_Col_Maj;
            A_ld = m;
            B_get_index = Index::get_index_Col_Maj;
            B_ld = k;
            C_get_i = Index::get_i_Col_Maj;
            C_get_j = Index::get_j_Col_Maj;
            C_ld = m;
            break;
        case 2 :  // A: Row B: Col C: Row
            A_get_index = Index::get_index_Row_Maj;
            A_ld = k;
            B_get_index = Index::get_index_Col_Maj;
            B_ld = k;
            C_get_i = Index::get_i_Row_Maj;
            C_get_j = Index::get_j_Row_Maj;
            C_ld = n;
            break;
        case 6 :  // A: Col B: Col C: Row
            A_get_index = Index::get_index_Col_Maj;
            A_ld = m;
            B_get_index = Index::get_index_Col_Maj;
            B_ld = k;
            C_get_i = Index::get_i_Row_Maj;
            C_get_j = Index::get_j_Row_Maj;
            C_ld = n;
            break;
        case 1 :  // A: Row B: Row C: Col
            A_get_index = Index::get_index_Row_Maj;
            A_ld = k;
            B_get_index = Index::get_index_Row_Maj;
            B_ld = n;
            C_get_i = Index::get_i_Col_Maj;
            C_get_j = Index::get_j_Col_Maj;
            C_ld = m;
            break;
        case 5 :  // A: Col B: Row C: Col
            A_get_index = Index::get_index_Col_Maj;
            A_ld = m;
            B_get_index = Index::get_index_Row_Maj;
            B_ld = n;
            C_get_i = Index::get_i_Col_Maj;
            C_get_j = Index::get_j_Col_Maj;
            C_ld = m;
            break;
        case 0 :  // A: Row B: Row C: Row
            A_get_index = Index::get_index_Row_Maj;
            A_ld = k;
            B_get_index = Index::get_index_Row_Maj;
            B_ld = n;
            C_get_i = Index::get_i_Row_Maj;
            C_get_j = Index::get_j_Row_Maj;
            C_ld = n;
            break;
        case 4 :  // A: Col B: Row C: Row
            A_get_index = Index::get_index_Col_Maj;
            A_ld = m;
            B_get_index = Index::get_index_Row_Maj;
            B_ld = n;
            C_get_i = Index::get_i_Row_Maj;
            C_get_j = Index::get_j_Row_Maj;
            C_ld = n;
            break;
        default :
            assert(0 && "The matrix should be Col(1) or Row(0) Major!");
    }
    int64_t length = m * n;
    #if (defined(USE_OPENMP_SIMD) && defined(USE_OPENMP))
    #pragma omp for simd schedule(static, (length - 1)/omp_get_num_threads() + 1) nowait
    #elif defined(USE_OPENMP)
    #pragma omp for schedule(static, (length - 1)/omp_get_num_threads() + 1) nowait
    #elif defined(USE_OPENMP_SIMD)
    #pragma omp simd
    #endif
    for (int64_t index_C = 0; index_C < length; index_C++) {
        T temp = ptr_C[index_C];
        ptr_C[index_C] = 0;
        int64_t i_C = C_get_i(index_C, C_ld);
        int64_t j_C = C_get_j(index_C, C_ld);
        for (int64_t j_A = 0, i_B = 0; j_A < k; j_A++, i_B++)
        {
            ptr_C[index_C] += ptr_A[A_get_index(i_C, j_A, A_ld)] * ptr_B[B_get_index(i_B, j_C, B_ld)];
        }
        ptr_C[index_C] = alpha * ptr_C[index_C] + beta * temp;
    }          
}
// template void matrix_product_general<int>(const int* ptr_A, const uint is_Col_Maj_A,
//                                                   const int* ptr_B, const uint is_Col_Maj_B,
//                                                   int* ptr_C, const uint is_Col_Maj_C,
//                                                   const uint m, const uint n, const uint k,
//                                                   const int alpha, const int beta);
// template void matrix_product_general<float>(const float* ptr_A, const uint is_Col_Maj_A,
//                                                     const float* ptr_B, const uint is_Col_Maj_B,
//                                                     float* ptr_C, const uint is_Col_Maj_C,
//                                                     const uint m, const uint n, const uint k,
//                                                     const float alpha, const float beta);
// template void matrix_product_general<double>(const double* ptr_A, const uint is_Col_Maj_A,
//                                                      const double* ptr_B, const uint is_Col_Maj_B,
//                                                      double* ptr_C, const uint is_Col_Maj_C,
//                                                      const uint m, const uint n, const uint k,
//                                                      const double alpha, const double beta);

template<typename T>
void matrix_product_general(const T* ptr_A, const uint is_Col_Maj_A,
                                    const T* ptr_B, const uint is_Col_Maj_B,
                                    T* ptr_C, const uint is_Col_Maj_C,
                                    const int64_t m, const int64_t n, const int64_t k,
                                    const T alpha, const T beta, const T gamma) {
    uint64_t (*A_get_index)(const uint64_t i, const uint64_t j, const uint64_t ld) = nullptr;
    uint64_t (*B_get_index)(const uint64_t i, const uint64_t j, const uint64_t ld) = nullptr;
    uint64_t (*C_get_i)(const uint64_t index, const uint64_t ld) = nullptr;
    uint64_t (*C_get_j)(const uint64_t index, const uint64_t ld) = nullptr;
    int64_t A_ld, B_ld, C_ld;
    switch(4 * is_Col_Maj_A + 2 * is_Col_Maj_B + is_Col_Maj_C) {
        case 3 :  // A: Row B: Col C: Col
            A_get_index = Index::get_index_Row_Maj;
            A_ld = k;
            B_get_index = Index::get_index_Col_Maj;
            B_ld = k;
            C_get_i = Index::get_i_Col_Maj;
            C_get_j = Index::get_j_Col_Maj;
            C_ld = m;
            break;
        case 7 :  // A: Col B: Col C: Col
            A_get_index = Index::get_index_Col_Maj;
            A_ld = m;
            B_get_index = Index::get_index_Col_Maj;
            B_ld = k;
            C_get_i = Index::get_i_Col_Maj;
            C_get_j = Index::get_j_Col_Maj;
            C_ld = m;
            break;
        case 2 :  // A: Row B: Col C: Row
            A_get_index = Index::get_index_Row_Maj;
            A_ld = k;
            B_get_index = Index::get_index_Col_Maj;
            B_ld = k;
            C_get_i = Index::get_i_Row_Maj;
            C_get_j = Index::get_j_Row_Maj;
            C_ld = n;
            break;
        case 6 :  // A: Col B: Col C: Row
            A_get_index = Index::get_index_Col_Maj;
            A_ld = m;
            B_get_index = Index::get_index_Col_Maj;
            B_ld = k;
            C_get_i = Index::get_i_Row_Maj;
            C_get_j = Index::get_j_Row_Maj;
            C_ld = n;
            break;
        case 1 :  // A: Row B: Row C: Col
            A_get_index = Index::get_index_Row_Maj;
            A_ld = k;
            B_get_index = Index::get_index_Row_Maj;
            B_ld = n;
            C_get_i = Index::get_i_Col_Maj;
            C_get_j = Index::get_j_Col_Maj;
            C_ld = m;
            break;
        case 5 :  // A: Col B: Row C: Col
            A_get_index = Index::get_index_Col_Maj;
            A_ld = m;
            B_get_index = Index::get_index_Row_Maj;
            B_ld = n;
            C_get_i = Index::get_i_Col_Maj;
            C_get_j = Index::get_j_Col_Maj;
            C_ld = m;
            break;
        case 0 :  // A: Row B: Row C: Row
            A_get_index = Index::get_index_Row_Maj;
            A_ld = k;
            B_get_index = Index::get_index_Row_Maj;
            B_ld = n;
            C_get_i = Index::get_i_Row_Maj;
            C_get_j = Index::get_j_Row_Maj;
            C_ld = n;
            break;
        case 4 :  // A: Col B: Row C: Row
            A_get_index = Index::get_index_Col_Maj;
            A_ld = m;
            B_get_index = Index::get_index_Row_Maj;
            B_ld = n;
            C_get_i = Index::get_i_Row_Maj;
            C_get_j = Index::get_j_Row_Maj;
            C_ld = n;
            break;
        default :
            assert(0 && "The matrix should be Col(1) or Row(0) Major!");
    }
    int64_t length = m * n;
    #if (defined(USE_OPENMP_SIMD) && defined(USE_OPENMP))
    #pragma omp for simd schedule(static, (length - 1)/omp_get_num_threads() + 1) nowait
    #elif defined(USE_OPENMP)
    #pragma omp for schedule(static, (length - 1)/omp_get_num_threads() + 1) nowait
    #elif defined(USE_OPENMP_SIMD)
    #pragma omp simd
    #endif
    for (int64_t index_C = 0; index_C < length; index_C++) {
        T temp = ptr_C[index_C];
        ptr_C[index_C] = 0;
        int64_t i_C = C_get_i(index_C, C_ld);
        int64_t j_C = C_get_j(index_C, C_ld);
        for (int64_t j_A = 0, i_B = 0; j_A < k; j_A++, i_B++)
        {
            ptr_C[index_C] += ptr_A[A_get_index(i_C, j_A, A_ld)] * ptr_B[B_get_index(i_B, j_C, B_ld)];
        }
        ptr_C[index_C] = alpha * ptr_C[index_C] + beta * temp + gamma;
    }          
}
// template void matrix_product_general<int>(const int* ptr_A, const uint is_Col_Maj_A,
//                                                   const int* ptr_B, const uint is_Col_Maj_B,
//                                                   int* ptr_C, const uint is_Col_Maj_C,
//                                                   const uint m, const uint n, const uint k,
//                                                   const int alpha, const int beta, const int gamma);
// template void matrix_product_general<float>(const float* ptr_A, const uint is_Col_Maj_A,
//                                                     const float* ptr_B, const uint is_Col_Maj_B,
//                                                     float* ptr_C, const uint is_Col_Maj_C,
//                                                     const uint m, const uint n, const uint k,
//                                                     const float alpha, const float beta, const float gamma);
// template void matrix_product_general<double>(const double* ptr_A, const uint is_Col_Maj_A,
//                                                      const double* ptr_B, const uint is_Col_Maj_B,
//                                                      double* ptr_C, const uint is_Col_Maj_C,
//                                                      const uint m, const uint n, const uint k,
//                                                      const double alpha, const double beta, const double gamma);

#ifdef USE_CBLAS
template<>
void cblas__gemv(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE trans,
                const int64_t m, const int64_t n,
                const float alpha, float const* const a, const int64_t lda,
                const float *x, const int64_t incx,
                const float beta, float* const y, const int64_t incy) {
    cblas_sgemv(Layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    return;
}
template<>
void cblas__gemv(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE trans,
                const int64_t m, const int64_t n,
                const double alpha, double const* const a, const int64_t lda,
                const double *x, const int64_t incx,
                const double beta, double* const y, const int64_t incy) {
    cblas_dgemv(Layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    return;
}
template<>
void cblas__gemv(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE trans,
                const int64_t m, const int64_t n,
                const std::complex<float> alpha, std::complex<float> const* const a, const int64_t lda,
                const std::complex<float> *x, const int64_t incx,
                const std::complex<float> beta, std::complex<float>* const y, const int64_t incy) {
    cblas_cgemv(Layout, trans, m, n, &alpha, a, lda, x, incx, &beta, y, incy);
    return;
}
template<>
void cblas__gemv(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE trans,
                const int64_t m, const int64_t n,
                const std::complex<double> alpha, std::complex<double> const* const a, const int64_t lda,
                const std::complex<double> *x, const int64_t incx,
                const std::complex<double> beta, std::complex<double>* const y, const int64_t incy) {
    cblas_zgemv(Layout, trans, m, n, &alpha, a, lda, x, incx, &beta, y, incy);
    return;
}

#ifdef USE_OPENMP
template<>
inline void cblas__gemv_omp_row_par(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE trans,
                const int64_t m, const int64_t n,
                const float alpha, float const* const a, const int64_t lda,
                const float *x, const int64_t incx,
                const float beta, float* const y, const int64_t incy) {
    int64_t chunksize = get_chunksize(omp_get_num_threads(), m);
    int64_t thread_id = omp_get_thread_num();
    if (thread_id * chunksize <= m - 1) {
        int64_t is = thread_id * chunksize;
        int64_t ie = (thread_id + 1) * chunksize - 1 <= m - 1 ? (thread_id + 1) * chunksize - 1 : m - 1;
        int64_t nloc = ie - is + 1;
        int64_t offset_A = Index::get_index(is, 0, lda, Layout, trans);
        cblas_sgemv(Layout, trans, nloc, n, alpha, a + offset_A, lda,
                    x, incx, beta, y + is, incy);
    }
    return;
}
template<>
inline void cblas__gemv_omp_row_par(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE trans,
                const int64_t m, const int64_t n,
                const double alpha, double const* const a, const int64_t lda,
                const double *x, const int64_t incx,
                const double beta, double* const y, const int64_t incy) {
    int64_t chunksize = get_chunksize(omp_get_num_threads(), m);
    int64_t thread_id = omp_get_thread_num();
    if (thread_id * chunksize <= m - 1) {
        int64_t is = thread_id * chunksize;
        int64_t ie = (thread_id + 1) * chunksize - 1 <= m - 1 ? (thread_id + 1) * chunksize - 1 : m - 1;
        int64_t nloc = ie - is + 1;
        int64_t offset_A = Index::get_index(is, 0, lda, Layout, trans);
        cblas_dgemv(Layout, trans, nloc, n, alpha, a + offset_A, lda,
                    x, incx, beta, y + is, incy);
    }
    return;
}
template<>
inline void cblas__gemv_omp_row_par(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE trans,
                const int64_t m, const int64_t n,
                const std::complex<float> alpha, std::complex<float> const* const a, const int64_t lda,
                const std::complex<float> *x, const int64_t incx,
                const std::complex<float> beta, std::complex<float>* const y, const int64_t incy) {
    int64_t chunksize = get_chunksize(omp_get_num_threads(), m);
    int64_t thread_id = omp_get_thread_num();
    if (thread_id * chunksize <= m - 1) {
        int64_t is = thread_id * chunksize;
        int64_t ie = (thread_id + 1) * chunksize - 1 <= m - 1 ? (thread_id + 1) * chunksize - 1 : m - 1;
        int64_t nloc = ie - is + 1;
        int64_t offset_A = Index::get_index(is, 0, lda, Layout, trans);
        cblas_cgemv(Layout, trans, nloc, n, &alpha, a + offset_A, lda,
                    x, incx, &beta, y + is, incy);
    }
    return;
}
template<>
inline void cblas__gemv_omp_row_par(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE trans,
                const int64_t m, const int64_t n,
                const std::complex<double> alpha, std::complex<double> const* const a, const int64_t lda,
                const std::complex<double> *x, const int64_t incx,
                const std::complex<double> beta, std::complex<double>* const y, const int64_t incy) {
    int64_t chunksize = get_chunksize(omp_get_num_threads(), m);
    int64_t thread_id = omp_get_thread_num();
    if (thread_id * chunksize <= m - 1) {
        int64_t is = thread_id * chunksize;
        int64_t ie = (thread_id + 1) * chunksize - 1 <= m - 1 ? (thread_id + 1) * chunksize - 1 : m - 1;
        int64_t nloc = ie - is + 1;
        int64_t offset_A = Index::get_index(is, 0, lda, Layout, trans);
        cblas_zgemv(Layout, trans, nloc, n, &alpha, a + offset_A, lda,
                    x, incx, &beta, y + is, incy);
    }
    return;
}
#endif //USE_OPENMP

template<>
void cblas__gemm(const CBLAS_ORDER C_LAYOUT, const CBLAS_TRANSPOSE A_TRANSPOSE, const CBLAS_TRANSPOSE B_TRANSPOSE,
                const int64_t m, const int64_t n, const int64_t k,
                const float alpha, float const* const ptr_A, const int64_t A_ld,
                float const* const ptr_B, const int64_t B_ld,
                const float beta, float* const ptr_C, const int64_t C_ld) {
    cblas_sgemm(C_LAYOUT, A_TRANSPOSE, B_TRANSPOSE, m, n, k, alpha,
        ptr_A, A_ld, ptr_B, B_ld, beta, ptr_C, C_ld);
    return;
}
template<>
void cblas__gemm(const CBLAS_ORDER C_LAYOUT, const CBLAS_TRANSPOSE A_TRANSPOSE, const CBLAS_TRANSPOSE B_TRANSPOSE,
                const int64_t m, const int64_t n, const int64_t k,
                const double alpha, double const* const ptr_A, const int64_t A_ld,
                double const* const ptr_B, const int64_t B_ld,
                const double beta, double* const ptr_C, const int64_t C_ld) {
    cblas_dgemm(C_LAYOUT, A_TRANSPOSE, B_TRANSPOSE, m, n, k, alpha,
        ptr_A, A_ld, ptr_B, B_ld, beta, ptr_C, C_ld);
    return;
}
template<>
void cblas__gemm(const CBLAS_ORDER C_LAYOUT, const CBLAS_TRANSPOSE A_TRANSPOSE, const CBLAS_TRANSPOSE B_TRANSPOSE,
                const int64_t m, const int64_t n, const int64_t k,
                const std::complex<float> alpha,  std::complex<float> const* const ptr_A, const int64_t A_ld,
                std::complex<float> const* const ptr_B, const int64_t B_ld,
                const std::complex<float> beta,  std::complex<float>* const ptr_C, const int64_t C_ld) {
    cblas_cgemm(C_LAYOUT, A_TRANSPOSE, B_TRANSPOSE, m, n, k, &alpha,
        ptr_A, A_ld, ptr_B, B_ld, &beta, ptr_C, C_ld);
    return;
}
template<>
void cblas__gemm(const CBLAS_ORDER C_LAYOUT, const CBLAS_TRANSPOSE A_TRANSPOSE, const CBLAS_TRANSPOSE B_TRANSPOSE,
                const int64_t m, const int64_t n, const int64_t k,
                const std::complex<double> alpha,  std::complex<double> const* const ptr_A, const int64_t A_ld,
                std::complex<double> const* const ptr_B, const int64_t B_ld,
                const std::complex<double> beta,  std::complex<double>* const ptr_C, const int64_t C_ld) {
    cblas_zgemm(C_LAYOUT, A_TRANSPOSE, B_TRANSPOSE, m, n, k, &alpha,
        ptr_A, A_ld, ptr_B, B_ld, &beta, ptr_C, C_ld);
    return;
}

#ifdef USE_OPENMP
template<>
inline void cblas__gemm_omp_col_par(const CBLAS_ORDER C_LAYOUT, const CBLAS_TRANSPOSE A_TRANSPOSE, const CBLAS_TRANSPOSE B_TRANSPOSE,
                const int64_t m, const int64_t n, const int64_t k,
                const float alpha, float const* const ptr_A, const int64_t A_ld,
                float const* const ptr_B, const int64_t B_ld,
                const float beta, float* const ptr_C, const int64_t C_ld) {
    int64_t chunksize = get_chunksize(omp_get_num_threads(), n);
    int64_t thread_id = omp_get_thread_num();
    if (thread_id * chunksize <= n - 1) {
        int64_t js = thread_id * chunksize;
        int64_t je = (thread_id + 1) * chunksize - 1 <= n - 1 ? (thread_id + 1) * chunksize - 1 : n - 1;
        int64_t nloc = je - js + 1;
        int64_t offset_B = Index::get_index(0, js, B_ld, C_LAYOUT, B_TRANSPOSE);
        int64_t offset_C = Index::get_index(0, js, C_ld, C_LAYOUT, CblasNoTrans);
        cblas_sgemm(C_LAYOUT, A_TRANSPOSE, B_TRANSPOSE, m, nloc, k, alpha,
                ptr_A, A_ld, ptr_B + offset_B, B_ld, beta, ptr_C + offset_C, C_ld);
    }
    return;
}
template<>
inline void cblas__gemm_omp_col_par(const CBLAS_ORDER C_LAYOUT, const CBLAS_TRANSPOSE A_TRANSPOSE, const CBLAS_TRANSPOSE B_TRANSPOSE,
                const int64_t m, const int64_t n, const int64_t k,
                const double alpha, double const* const ptr_A, const int64_t A_ld,
                double const* const ptr_B, const int64_t B_ld,
                const double beta, double* const ptr_C, const int64_t C_ld) {
    int64_t chunksize = get_chunksize(omp_get_num_threads(), n);
    int64_t thread_id = omp_get_thread_num();
    if (thread_id * chunksize <= n - 1) {
        int64_t js = thread_id * chunksize;
        int64_t je = (thread_id + 1) * chunksize - 1 <= n - 1 ? (thread_id + 1) * chunksize - 1 : n - 1;
        int64_t nloc = je - js + 1;
        int64_t offset_B = Index::get_index(0, js, B_ld, C_LAYOUT, B_TRANSPOSE);
        int64_t offset_C = Index::get_index(0, js, C_ld, C_LAYOUT, CblasNoTrans);
        cblas_dgemm(C_LAYOUT, A_TRANSPOSE, B_TRANSPOSE, m, nloc, k, alpha,
                ptr_A, A_ld, ptr_B + offset_B, B_ld, beta, ptr_C + offset_C, C_ld);
    }
    return;
}
template<>
inline void cblas__gemm_omp_col_par(const CBLAS_ORDER C_LAYOUT, const CBLAS_TRANSPOSE A_TRANSPOSE, const CBLAS_TRANSPOSE B_TRANSPOSE,
                const int64_t m, const int64_t n, const int64_t k,
                const std::complex<float> alpha,  std::complex<float> const* const ptr_A, const int64_t A_ld,
                std::complex<float> const* const ptr_B, const int64_t B_ld,
                const std::complex<float> beta,  std::complex<float>* const ptr_C, const int64_t C_ld) {
    int64_t chunksize = get_chunksize(omp_get_num_threads(), n);
    int64_t thread_id = omp_get_thread_num();
    if (thread_id * chunksize <= n - 1) {
        int64_t js = thread_id * chunksize;
        int64_t je = (thread_id + 1) * chunksize - 1 <= n - 1 ? (thread_id + 1) * chunksize - 1 : n - 1;
        int64_t nloc = je - js + 1;
        int64_t offset_B = Index::get_index(0, js, B_ld, C_LAYOUT, B_TRANSPOSE);
        int64_t offset_C = Index::get_index(0, js, C_ld, C_LAYOUT, CblasNoTrans);
        cblas_cgemm(C_LAYOUT, A_TRANSPOSE, B_TRANSPOSE, m, nloc, k, &alpha,
                ptr_A, A_ld, ptr_B + offset_B, B_ld, &beta, ptr_C + offset_C, C_ld);
    }
    return;
}
template<>
inline void cblas__gemm_omp_col_par(const CBLAS_ORDER C_LAYOUT, const CBLAS_TRANSPOSE A_TRANSPOSE, const CBLAS_TRANSPOSE B_TRANSPOSE,
                const int64_t m, const int64_t n, const int64_t k,
                const std::complex<double> alpha,  std::complex<double> const* const ptr_A, const int64_t A_ld,
                std::complex<double> const* const ptr_B, const int64_t B_ld,
                const std::complex<double> beta,  std::complex<double>* const ptr_C, const int64_t C_ld) {
    int64_t chunksize = get_chunksize(omp_get_num_threads(), n);
    int64_t thread_id = omp_get_thread_num();
    if (thread_id * chunksize <= n - 1) {
        int64_t js = thread_id * chunksize;
        int64_t je = (thread_id + 1) * chunksize - 1 <= n - 1 ? (thread_id + 1) * chunksize - 1 : n - 1;
        int64_t nloc = je - js + 1;
        int64_t offset_B = Index::get_index(0, js, B_ld, C_LAYOUT, B_TRANSPOSE);
        int64_t offset_C = Index::get_index(0, js, C_ld, C_LAYOUT, CblasNoTrans);
        cblas_zgemm(C_LAYOUT, A_TRANSPOSE, B_TRANSPOSE, m, nloc, k, &alpha,
                ptr_A, A_ld, ptr_B + offset_B, B_ld, &beta, ptr_C + offset_C, C_ld);
    }
    return;
}
#endif //USE_OPENMP

template<>
void cblas__syrk(const CBLAS_ORDER layout, const CBLAS_UPLO uplo, const CBLAS_TRANSPOSE trans,
                        const int64_t n, const int64_t k,
                        const float alpha, float const* const ptr_A, const int64_t A_ld,
                        const float beta, float* const ptr_C, const int64_t C_ld) {
    cblas_ssyrk(layout, uplo, trans, n, k, alpha, ptr_A, A_ld, beta, ptr_C, C_ld);
    return;
}
template<>
void cblas__syrk(const CBLAS_ORDER layout, const CBLAS_UPLO uplo, const CBLAS_TRANSPOSE trans,
                        const int64_t n, const int64_t k,
                        const double alpha, double const* const ptr_A, const int64_t A_ld,
                        const double beta, double* const ptr_C, const int64_t C_ld) {
    cblas_dsyrk(layout, uplo, trans, n, k, alpha, ptr_A, A_ld, beta, ptr_C, C_ld);
    return;
}
template<>
void cblas__syrk(const CBLAS_ORDER layout, const CBLAS_UPLO uplo, const CBLAS_TRANSPOSE trans,
                        const int64_t n, const int64_t k,
                        const std::complex<float> alpha, std::complex<float> const* const ptr_A, const int64_t A_ld,
                        const std::complex<float> beta, std::complex<float>* const ptr_C, const int64_t C_ld) {
    cblas_csyrk(layout, uplo, trans, n, k, &alpha, ptr_A, A_ld, &beta, ptr_C, C_ld);
    return;
}
template<>
void cblas__syrk(const CBLAS_ORDER layout, const CBLAS_UPLO uplo, const CBLAS_TRANSPOSE trans,
                        const int64_t n, const int64_t k,
                        const std::complex<double> alpha, std::complex<double> const* const ptr_A, const int64_t A_ld,
                        const std::complex<double> beta, std::complex<double>* const ptr_C, const int64_t C_ld) {
    cblas_zsyrk(layout, uplo, trans, n, k, &alpha, ptr_A, A_ld, &beta, ptr_C, C_ld);
    return;
}

template<typename T>
void matrix_vector_product_cblas(const T* ptr_A, const uint is_Col_Maj_A,
                                         const T* ptr_B, T* ptr_C,
                                         const int64_t m, const int64_t n,
                                         const T alpha, const T beta) {
    int64_t A_ld;
    CBLAS_ORDER A_LAYOUT;
    if (is_Col_Maj_A == 1) {
        A_LAYOUT = CblasColMajor;
        A_ld = m;
    } else {
        A_LAYOUT = CblasRowMajor;
        A_ld = n;
    }
    #ifdef USE_OPENMP
        #ifdef USE_MY_OPENMP_BLAS
            cblas__gemv_omp_row_par<T>(A_LAYOUT, CblasNoTrans, m, n, alpha, ptr_A, A_ld,
                                    ptr_B, 1, beta, ptr_C, 1);
        #elif defined(USE_BLAS_OPEMNP_NESTED)
            #pragma omp single nowait
            {
                omp_set_max_active_levels(2);
                cblas__gemv<T>(A_LAYOUT, CblasNoTrans, m, n, alpha, ptr_A, A_ld,
                            ptr_B, 1, beta, ptr_C, 1);
                omp_set_max_active_levels(1);
            }
        #else
            #pragma omp single nowait
            {
                cblas__gemv<T>(A_LAYOUT, CblasNoTrans, m, n, alpha, ptr_A, A_ld,
                            ptr_B, 1, beta, ptr_C, 1);
            }
        #endif //USE_MY_OPENMP_BLAS
    #else
        cblas__gemv<T>(A_LAYOUT, CblasNoTrans, m, n, alpha, ptr_A, A_ld,
                                ptr_B, 1, beta, ptr_C, 1);
    #endif //USE_OPENMP
    return;
}
template void matrix_vector_product_cblas<float>(const float* ptr_A, const uint is_Col_Maj_A,
                                                         const float* ptr_B, float* ptr_C,
                                                         const int64_t m, const int64_t n,
                                                         const float alpha, const float beta);
template void matrix_vector_product_cblas<double>(const double* ptr_A, const uint is_Col_Maj_A,
                                                          const double* ptr_B, double* ptr_C,
                                                          const int64_t m, const int64_t n,
                                                          const double alpha, const double beta);

template<typename T>
void matrix_vector_product_cblas(const T* ptr_A, const uint is_Col_Maj_A,
                                         const T* ptr_B, T* ptr_C,
                                         const int64_t m, const int64_t n) {
    matrix_vector_product_cblas(ptr_A, is_Col_Maj_A, ptr_B, ptr_C,
                                 m, n, static_cast<T>(1), static_cast<T>(0));
    return;
}
template void matrix_vector_product_cblas<float>(const float* ptr_A, const uint is_Col_Maj_A,
                                                         const float* ptr_B, float* ptr_C,
                                                         const int64_t m, const int64_t n);
template void matrix_vector_product_cblas<double>(const double* ptr_A, const uint is_Col_Maj_A,
                                                          const double* ptr_B, double* ptr_C,
                                                          const int64_t m, const int64_t n);

template<typename T>
void matrix_vector_product_cblas(const T* ptr_A, const uint is_Col_Maj_A,
                                         const T* ptr_B, T* ptr_C,
                                         const int64_t m, const int64_t n,
                                         const T alpha) {
    matrix_vector_product_cblas(ptr_A, is_Col_Maj_A, ptr_B, ptr_C,
                                 m, n, alpha, static_cast<T>(0));
    return;
}
template void matrix_vector_product_cblas<float>(const float* ptr_A, const uint is_Col_Maj_A,
                                                         const float* ptr_B, float* ptr_C,
                                                         const int64_t m, const int64_t n,
                                                         const float alpha);
template void matrix_vector_product_cblas<double>(const double* ptr_A, const uint is_Col_Maj_A,
                                                          const double* ptr_B, double* ptr_C,
                                                          const int64_t m, const int64_t n,
                                                          const double alpha);

template<typename T>
void matrix_vector_product_cblas(const T* ptr_A, const uint is_Col_Maj_A,
                                         const T* ptr_B, T* ptr_C,
                                         const int64_t m, const int64_t n,
                                         const T alpha, const T beta, const T gamma) {
    matrix_vector_product_cblas(ptr_A, is_Col_Maj_A, ptr_B, ptr_C,
                                 m, n, alpha, beta);
    scalar_plus_general(ptr_C, gamma, m);
    return;
}
template void matrix_vector_product_cblas<float>(const float* ptr_A, const uint is_Col_Maj_A,
                                                         const float* ptr_B, float* ptr_C,
                                                         const int64_t m, const int64_t n,
                                                         const float alpha, const float beta, const float gamma);
template void matrix_vector_product_cblas<double>(const double* ptr_A, const uint is_Col_Maj_A,
                                                          const double* ptr_B, double* ptr_C,
                                                          const int64_t m, const int64_t n,
                                                          const double alpha, const double beta, const double gamma);

template<typename T>
void matrix_product_cblas(const T* ptr_A, const uint is_Col_Maj_A,
                                  const T* ptr_B, const uint is_Col_Maj_B,
                                  T* ptr_C, const uint is_Col_Maj_C,
                                  const int64_t m, const int64_t n, const int64_t k,
                                  const T alpha, const T beta) {
    int64_t A_ld, B_ld, C_ld;
    CBLAS_ORDER C_LAYOUT;
    CBLAS_TRANSPOSE A_TRANSPOSE;
    CBLAS_TRANSPOSE B_TRANSPOSE;
    switch(4 * is_Col_Maj_A + 2 * is_Col_Maj_B + is_Col_Maj_C) {
        case 3 :  // A: Row B: Col C: Col
            A_TRANSPOSE = CblasTrans;
            A_ld = k;
            B_TRANSPOSE = CblasNoTrans;
            B_ld = k;
            C_LAYOUT = CblasColMajor;
            C_ld = m;
            break;
        case 7 :  // A: Col B: Col C: Col
            A_TRANSPOSE = CblasNoTrans;
            A_ld = m;
            B_TRANSPOSE = CblasNoTrans;
            B_ld = k;
            C_LAYOUT = CblasColMajor;
            C_ld = m;
            break;
        case 2 :  // A: Row B: Col C: Row
            A_TRANSPOSE = CblasNoTrans;
            A_ld = k;
            B_TRANSPOSE = CblasTrans;
            B_ld = k;
            C_LAYOUT = CblasRowMajor;
            C_ld = n;
            break;
        case 6 :  // A: Col B: Col C: Row
            A_TRANSPOSE = CblasTrans;
            A_ld = m;
            B_TRANSPOSE = CblasTrans;
            B_ld = k;
            C_LAYOUT = CblasRowMajor;
            C_ld = n;
            break;
        case 1 :  // A: Row B: Row C: Col
            A_TRANSPOSE = CblasTrans;
            A_ld = k;
            B_TRANSPOSE = CblasTrans;
            B_ld = n;
            C_LAYOUT = CblasColMajor;
            C_ld = m;
            break;
        case 5 :  // A: Col B: Row C: Col
            A_TRANSPOSE = CblasNoTrans;
            A_ld = m;
            B_TRANSPOSE = CblasTrans;
            B_ld = n;
            C_LAYOUT = CblasColMajor;
            C_ld = m;
            break;
        case 0 :  // A: Row B: Row C: Row
            A_TRANSPOSE = CblasNoTrans;
            A_ld = k;
            B_TRANSPOSE = CblasNoTrans;
            B_ld = n;
            C_LAYOUT = CblasRowMajor;
            C_ld = n;
            break;
        case 4 :  // A: Col B: Row C: Row
            A_TRANSPOSE = CblasTrans;
            A_ld = m;
            B_TRANSPOSE = CblasNoTrans;
            B_ld = n;
            C_LAYOUT = CblasRowMajor;
            C_ld = n;
            break;
        default :
            assert(0 && "The matrix should be Col(1) or Row(0) Major!");
    }
    #ifdef USE_OPENMP
        #ifdef USE_MY_OPENMP_BLAS
            cblas__gemm_omp_col_par<T>(C_LAYOUT, A_TRANSPOSE, B_TRANSPOSE, m, n, k, alpha,
                                        ptr_A, A_ld, ptr_B, B_ld, beta, ptr_C, C_ld);
        #elif defined(USE_BLAS_OPEMNP_NESTED)
            #pragma omp single nowait
            {
                omp_set_max_active_levels(2);
                cblas__gemm<T>(C_LAYOUT, A_TRANSPOSE, B_TRANSPOSE, m, n, k, alpha,
                                ptr_A, A_ld, ptr_B, B_ld, beta, ptr_C, C_ld);
                omp_set_max_active_levels(1);
            }
        #else
            #pragma omp single nowait
            {
                cblas__gemm<T>(C_LAYOUT, A_TRANSPOSE, B_TRANSPOSE, m, n, k, alpha,
                                ptr_A, A_ld, ptr_B, B_ld, beta, ptr_C, C_ld);
            }
        #endif //USE_MY_OPENMP_BLAS
    #else
        cblas__gemm<T>(C_LAYOUT, A_TRANSPOSE, B_TRANSPOSE, m, n, k, alpha,
                            ptr_A, A_ld, ptr_B, B_ld, beta, ptr_C, C_ld);
    #endif //USE_OPENMP
}
template void matrix_product_cblas<float>(const float* ptr_A, const uint is_Col_Maj_A,
                                                  const float* ptr_B, const uint is_Col_Maj_B,
                                                  float* ptr_C, const uint is_Col_Maj_C,
                                                  const int64_t m, const int64_t n, const int64_t k,
                                                  const float alpha, const float beta);
template void matrix_product_cblas<double>(const double* ptr_A, const uint is_Col_Maj_A,
                                                   const double* ptr_B, const uint is_Col_Maj_B,
                                                   double* ptr_C, const uint is_Col_Maj_C,
                                                   const int64_t m, const int64_t n, const int64_t k,
                                                   const double alpha, const double beta);

template<typename T>
void matrix_product_cblas(const T* ptr_A, const uint is_Col_Maj_A,
                                  const T* ptr_B, const uint is_Col_Maj_B,
                                  T* ptr_C, const uint is_Col_Maj_C,
                                  const int64_t m, const int64_t n, const int64_t k) {
    matrix_product_cblas(ptr_A, is_Col_Maj_A, ptr_B, is_Col_Maj_B,
                                 ptr_C, is_Col_Maj_C, m, n, k, static_cast<T>(1), static_cast<T>(0));
    return;
}
template void matrix_product_cblas<float>(const float* ptr_A, const uint is_Col_Maj_A,
                                                  const float* ptr_B, const uint is_Col_Maj_B,
                                                  float* ptr_C, const uint is_Col_Maj_C,
                                                  const int64_t m, const int64_t n, const int64_t k);
template void matrix_product_cblas<double>(const double* ptr_A, const uint is_Col_Maj_A,
                                                   const double* ptr_B, const uint is_Col_Maj_B,
                                                   double* ptr_C, const uint is_Col_Maj_C,
                                                   const int64_t m, const int64_t n, const int64_t k);

template<typename T>
void matrix_product_cblas(const T* ptr_A, const uint is_Col_Maj_A,
                                  const T* ptr_B, const uint is_Col_Maj_B,
                                  T* ptr_C, const uint is_Col_Maj_C,
                                  const int64_t m, const int64_t n, const int64_t k,
                                  const T alpha) {
    matrix_product_cblas(ptr_A, is_Col_Maj_A, ptr_B, is_Col_Maj_B,
                                 ptr_C, is_Col_Maj_C, m, n, k, alpha, static_cast<T>(0));
    return;
}
template void matrix_product_cblas<float>(const float* ptr_A, const uint is_Col_Maj_A,
                                                  const float* ptr_B, const uint is_Col_Maj_B,
                                                  float* ptr_C, const uint is_Col_Maj_C,
                                                  const int64_t m, const int64_t n, const int64_t k,
                                                  const float alpha);
template void matrix_product_cblas<double>(const double* ptr_A, const uint is_Col_Maj_A,
                                                   const double* ptr_B, const uint is_Col_Maj_B,
                                                   double* ptr_C, const uint is_Col_Maj_C,
                                                   const int64_t m, const int64_t n, const int64_t k,
                                                   const double alpha);

template<typename T>
void matrix_product_cblas(const T* ptr_A, const uint is_Col_Maj_A,
                                  const T* ptr_B, const uint is_Col_Maj_B,
                                  T* ptr_C, const uint is_Col_Maj_C,
                                  const int64_t m, const int64_t n, const int64_t k,
                                  const T alpha, const T beta, const T gamma) {
    matrix_product_cblas(ptr_A, is_Col_Maj_A, ptr_B, is_Col_Maj_B,
                                 ptr_C, is_Col_Maj_C, m, n, k, alpha, beta);
    scalar_plus_general(ptr_C, gamma, m * n);
    return;
}
template void matrix_product_cblas<float>(const float* ptr_A, const uint is_Col_Maj_A,
                                                  const float* ptr_B, const uint is_Col_Maj_B,
                                                  float* ptr_C, const uint is_Col_Maj_C,
                                                  const int64_t m, const int64_t n, const int64_t k,
                                                  const float alpha, const float beta, const float gamma);
template void matrix_product_cblas<double>(const double* ptr_A, const uint is_Col_Maj_A,
                                                   const double* ptr_B, const uint is_Col_Maj_B,
                                                   double* ptr_C, const uint is_Col_Maj_C,
                                                   const int64_t m, const int64_t n, const int64_t k,
                                                   const double alpha, const double beta, const double gamma);
#endif // USE_CBLAS

#ifdef USE_LAPACK
template<>
lapack_int LAPACKE__gelsd(const int matrix_layout, const lapack_int m, const lapack_int n,
                            const lapack_int nrhs, double *a, const lapack_int lda,
                            double *b, const lapack_int ldb, double *s, double rcond,
                            lapack_int* rank) {
    #ifdef USE_OPENMP
        lapack_int info = 0;
        #ifdef USE_LAPACK_OPEMNP_NESTED
            #pragma omp single
            {
                omp_set_max_active_levels(2);
                info = LAPACKE_dgelsd(matrix_layout, m, n, nrhs, a, lda, b, ldb, s, rcond, rank);
                omp_set_max_active_levels(1);
            }
        #else
            // #pragma omp single
            info = LAPACKE_dgelsd(matrix_layout, m, n, nrhs, a, lda, b, ldb, s, rcond, rank);
        #endif
        return info;
    #else
        return LAPACKE_dgelsd(matrix_layout, m, n, nrhs, a, lda, b, ldb, s, rcond, rank);
    #endif
}
template<>
lapack_int LAPACKE__gelsd(const int matrix_layout, const lapack_int m, const lapack_int n,
                            const lapack_int nrhs, float *a, const lapack_int lda,
                            float *b, const lapack_int ldb, float *s, float rcond,
                            lapack_int* rank) {
    #ifdef USE_OPENMP
        lapack_int info = 0;
        #ifdef USE_LAPACK_OPEMNP_NESTED
            #pragma omp single
            {
                omp_set_max_active_levels(2);
                info = LAPACKE_sgelsd(matrix_layout, m, n, nrhs, a, lda, b, ldb, s, rcond, rank);
                omp_set_max_active_levels(1);
            }
        #else
            // #pragma omp single
            info = LAPACKE_sgelsd(matrix_layout, m, n, nrhs, a, lda, b, ldb, s, rcond, rank);
        #endif
        return info;
    #else
        return LAPACKE_sgelsd(matrix_layout, m, n, nrhs, a, lda, b, ldb, s, rcond, rank);
    #endif
}
template<>
lapack_int LAPACKE__gelsd_org(const int matrix_layout, const lapack_int m, const lapack_int n,
                            const lapack_int nrhs, double *a, const lapack_int lda,
                            double *b, const lapack_int ldb, double *s, double rcond,
                            lapack_int* rank) {
    return LAPACKE_dgelsd(matrix_layout, m, n, nrhs, a, lda, b, ldb, s, rcond, rank);
}
template<>
lapack_int LAPACKE__gelsd_org(const int matrix_layout, const lapack_int m, const lapack_int n,
                            const lapack_int nrhs, float *a, const lapack_int lda,
                            float *b, const lapack_int ldb, float *s, float rcond,
                            lapack_int* rank) {
    return LAPACKE_sgelsd(matrix_layout, m, n, nrhs, a, lda, b, ldb, s, rcond, rank);
}
template<>
lapack_int LAPACKE__gelsd(const int matrix_layout, const lapack_int m, const lapack_int n,
                            const lapack_int nrhs, std::complex<float> *a,
                            const lapack_int lda, std::complex<float> *b,
                            const lapack_int ldb, float* s, float rcond,
                            lapack_int* rank) {
    #ifdef USE_OPENMP
        lapack_int info = 0;
        #ifdef USE_LAPACK_OPEMNP_NESTED
            #pragma omp single
            {
                omp_set_max_active_levels(2);
                info = LAPACKE_cgelsd(matrix_layout, m, n, nrhs,
                                reinterpret_cast<lapack_complex_float*>(a), lda,
                                reinterpret_cast<lapack_complex_float*>(b), ldb,
                                s, rcond, rank);
                omp_set_max_active_levels(1);
            }
        #else
            // #pragma omp single
            info = LAPACKE_cgelsd(matrix_layout, m, n, nrhs,
                                reinterpret_cast<lapack_complex_float*>(a), lda,
                                reinterpret_cast<lapack_complex_float*>(b), ldb,
                                s, rcond, rank);
        #endif
        return info;
    #else
        return LAPACKE_cgelsd(matrix_layout, m, n, nrhs,
                                reinterpret_cast<lapack_complex_float*>(a), lda,
                                reinterpret_cast<lapack_complex_float*>(b), ldb,
                                s, rcond, rank);
    #endif
}
template<>
lapack_int LAPACKE__gelsd(const int matrix_layout, const lapack_int m, const lapack_int n,
                            const lapack_int nrhs, std::complex<double> *a,
                            const lapack_int lda, std::complex<double> *b,
                            const lapack_int ldb, double* s, double rcond,
                            lapack_int* rank) {
    #ifdef USE_OPENMP
        lapack_int info = 0;
        #ifdef USE_LAPACK_OPEMNP_NESTED
            #pragma omp single
            {
                omp_set_max_active_levels(2);
                info = LAPACKE_zgelsd(matrix_layout, m, n, nrhs,
                            reinterpret_cast<lapack_complex_double*>(a), lda,
                            reinterpret_cast<lapack_complex_double*>(b), ldb,
                            s, rcond, rank);
                omp_set_max_active_levels(1);
            }
        #else
            // #pragma omp single
            info = LAPACKE_zgelsd(matrix_layout, m, n, nrhs,
                            reinterpret_cast<lapack_complex_double*>(a), lda,
                            reinterpret_cast<lapack_complex_double*>(b), ldb,
                            s, rcond, rank);
        #endif
        return info;
    #else
        return LAPACKE_zgelsd(matrix_layout, m, n, nrhs,
                            reinterpret_cast<lapack_complex_double*>(a), lda,
                            reinterpret_cast<lapack_complex_double*>(b), ldb,
                            s, rcond, rank);
    #endif
}

template<>
lapack_int LAPACKE__sygvd(const int matrix_layout, const lapack_int itype, char jobz,
                    char uplo, const lapack_int n, double* a, const lapack_int lda,
                    double* b, const lapack_int ldb, double* w) {
    #ifdef USE_OPENMP
        lapack_int info = 0;
        #ifdef USE_LAPACK_OPEMNP_NESTED
            #pragma omp single
            {
                omp_set_max_active_levels(2);
                info = LAPACKE_dsygvd(matrix_layout, itype, jobz, uplo, n, a, lda, b, ldb, w);
                omp_set_max_active_levels(1);
            }
        #else
            // #pragma omp single
            info = LAPACKE_dsygvd(matrix_layout, itype, jobz, uplo, n, a, lda, b, ldb, w);
        #endif
        return info;
    #else
        return LAPACKE_dsygvd(matrix_layout, itype, jobz, uplo, n, a, lda, b, ldb, w);
    #endif
}
template<>
lapack_int LAPACKE__sygvd(const int matrix_layout, const lapack_int itype, char jobz,
                            char uplo, const lapack_int n, float* a, const lapack_int lda,
                            float* b, const lapack_int ldb, float* w) {
    #ifdef USE_OPENMP
        lapack_int info = 0;
        #ifdef USE_LAPACK_OPEMNP_NESTED
            #pragma omp single
            {
                omp_set_max_active_levels(2);
                info = LAPACKE_ssygvd(matrix_layout, itype, jobz, uplo, n, a, lda, b, ldb, w);
                omp_set_max_active_levels(1);
            }
        #else
            // #pragma omp single
            info = LAPACKE_ssygvd(matrix_layout, itype, jobz, uplo, n, a, lda, b, ldb, w);
        #endif
        return info;
    #else
        return LAPACKE_ssygvd(matrix_layout, itype, jobz, uplo, n, a, lda, b, ldb, w);
    #endif
}

template<>
lapack_int LAPACKE__sygvd_org(const int matrix_layout, const lapack_int itype, char jobz,
                    char uplo, const lapack_int n, double* a, const lapack_int lda,
                    double* b, const lapack_int ldb, double* w) {
    return LAPACKE_dsygvd(matrix_layout, itype, jobz, uplo, n, a, lda, b, ldb, w);
}
template<>
lapack_int LAPACKE__sygvd_org(const int matrix_layout, const lapack_int itype, char jobz,
                            char uplo, const lapack_int n, float* a, const lapack_int lda,
                            float* b, const lapack_int ldb, float* w) {
    
    return LAPACKE_ssygvd(matrix_layout, itype, jobz, uplo, n, a, lda, b, ldb, w);
}
template<>
lapack_int LAPACKE__hygvd(const int matrix_layout, const lapack_int itype, char jobz,
                            char uplo, const lapack_int n, std::complex<float>* a,
                            const lapack_int lda, std::complex<float>* b,
                            const lapack_int ldb, float* w) {
    #ifdef USE_OPENMP
        lapack_int info = 0;
        #ifdef USE_LAPACK_OPEMNP_NESTED
            #pragma omp single
            {
                omp_set_max_active_levels(2);
                info = LAPACKE_chegvd(matrix_layout, itype, jobz, uplo, n,
                                reinterpret_cast<lapack_complex_float*>(a), lda,
                                reinterpret_cast<lapack_complex_float*>(b), ldb, w);
                omp_set_max_active_levels(1);
            }
        #else
            // #pragma omp single
            info = LAPACKE_chegvd(matrix_layout, itype, jobz, uplo, n,
                                reinterpret_cast<lapack_complex_float*>(a), lda,
                                reinterpret_cast<lapack_complex_float*>(b), ldb, w);
        #endif
        return info;
    #else
        return LAPACKE_chegvd(matrix_layout, itype, jobz, uplo, n,
                                reinterpret_cast<lapack_complex_float*>(a), lda,
                                reinterpret_cast<lapack_complex_float*>(b), ldb, w);
    #endif
}
template<>
lapack_int LAPACKE__hygvd(const int matrix_layout, const lapack_int itype, char jobz,
                            char uplo, const lapack_int n, std::complex<double>* a,
                            const lapack_int lda, std::complex<double>* b,
                            const lapack_int ldb, double* w) {
    #ifdef USE_OPENMP
        lapack_int info = 0;
        #ifdef USE_LAPACK_OPEMNP_NESTED
            #pragma omp single
            {
                omp_set_max_active_levels(2);
                info = LAPACKE_zhegvd(matrix_layout, itype, jobz, uplo, n,
                                reinterpret_cast<lapack_complex_double*>(a), lda,
                                reinterpret_cast<lapack_complex_double*>(b), ldb, w);
                omp_set_max_active_levels(1);
            }
        #else
            // #pragma omp single
            info = LAPACKE_zhegvd(matrix_layout, itype, jobz, uplo, n,
                                reinterpret_cast<lapack_complex_double*>(a), lda,
                                reinterpret_cast<lapack_complex_double*>(b), ldb, w);
        #endif
        return info;
    #else
        return LAPACKE_zhegvd(matrix_layout, itype, jobz, uplo, n,
                                reinterpret_cast<lapack_complex_double*>(a), lda,
                                reinterpret_cast<lapack_complex_double*>(b), ldb, w);
    #endif
}

template<>
lapack_int LAPACKE__sterf(const lapack_int order, float* d, float* e) {
    #ifdef USE_OPENMP
        lapack_int info = 0;
        #ifdef USE_LAPACK_OPEMNP_NESTED
            #pragma omp single
            {
                omp_set_max_active_levels(2);
                info = LAPACKE_ssterf(order, d, e);
                omp_set_max_active_levels(1);
            }
        #else
            // #pragma omp single
            info = LAPACKE_ssterf(order, d, e);
        #endif
        return info;
    #else
        return LAPACKE_ssterf(order, d, e);
    #endif
}
template<>
lapack_int LAPACKE__sterf(const lapack_int order, double* d, double* e) {
    #ifdef USE_OPENMP
        lapack_int info = 0;
        #ifdef USE_LAPACK_OPEMNP_NESTED
            #pragma omp single
            {
                omp_set_max_active_levels(2);
                info = LAPACKE_dsterf(order, d, e);
                omp_set_max_active_levels(1);
            }
        #else
            // #pragma omp single
            info = LAPACKE_dsterf(order, d, e);
        #endif
        return info;
    #else
        return LAPACKE_dsterf(order, d, e);
    #endif
}

template<>
lapack_int LAPACKE__sterf_org(const lapack_int order, float* d, float* e) {
    return LAPACKE_ssterf(order, d, e);
}
template<>
lapack_int LAPACKE__sterf_org(const lapack_int order, double* d, double* e) {
    return LAPACKE_dsterf(order, d, e);
}

#endif //USE_LAPACK

#if (defined(USE_MKL) || defined(USE_SCALAPACK))
template<>
void p_gemr2d_(const int *m, const int *n,
               		    const float *a, const int *ia, const int *ja, const int *desca,
               		    float *b, const int *ib, const int *jb, const int *descb,
               		    const int *ictxt) {
    #ifdef USE_OPENMP
        #ifdef USE_SCALALAPACK_OPEMNP_NESTED
            #pragma omp single nowait
            {
                omp_set_max_active_levels(2);
                psgemr2d_(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
                omp_set_max_active_levels(1);
            }
        #else
            #pragma omp single nowait
            psgemr2d_(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
        #endif
    #else
        psgemr2d_(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
    #endif
    return;
}
template<>
void p_gemr2d_(const int *m, const int *n,
                        const double *a, const int *ia, const int *ja, const int *desca,
                        double *b, const int *ib, const int *jb, const int *descb,
                        const int *ictxt) {
    #ifdef USE_OPENMP
        #ifdef USE_SCALALAPACK_OPEMNP_NESTED
            #pragma omp single nowait
            {
                omp_set_max_active_levels(2);
                pdgemr2d_(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
                omp_set_max_active_levels(1);
            }
        #else
            #pragma omp single nowait
            pdgemr2d_(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
        #endif
    #else
        pdgemr2d_(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
    #endif
    return;
}
template<>
void p_syrk_(const char *uplo, const char *trans, const int *n, const int *k,
				      const float *alpha, const float *a, const int *ia, const int *ja,
				      const int *desca, const float *beta, float *c, const int *ic,
				      const int *jc, const int *descc ) {
    #ifdef USE_OPENMP
        #ifdef USE_SCALALAPACK_OPEMNP_NESTED
            #pragma omp single nowait
            {
                omp_set_max_active_levels(2);
                pssyrk_(uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
                omp_set_max_active_levels(1);
            }
        #else
            #pragma omp single nowait
            pssyrk_(uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
        #endif
    #else
        pssyrk_(uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
    #endif
    return;
}
template<>
void p_syrk_(const char *uplo, const char *trans, const int *n, const int *k,
                      const double *alpha, const double *a, const int *ia, const int *ja,
                      const int *desca, const double *beta, double *c, const int *ic,
                      const int *jc, const int *descc ) {
    #ifdef USE_OPENMP
        #ifdef USE_SCALALAPACK_OPEMNP_NESTED
            #pragma omp single nowait
            {
                omp_set_max_active_levels(2);
                pdsyrk_(uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
                omp_set_max_active_levels(1);
            }
        #else
            #pragma omp single nowait
            pdsyrk_(uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
        #endif
    #else
        pdsyrk_(uplo, trans, n, k, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
    #endif
    return;
}
template<>
void p_gemm_(const char *transa, const char *transb, const int *m, const int *n,
                      const int *k, const float *alpha, const float *a, const int *ia,
                      const int *ja, const int *desca, const float *b, const int *ib,
                      const int *jb, const int *descb, const float *beta, float *c,
                      const int *ic, const int *jc, const int *descc ) {
    #ifdef USE_OPENMP
        #ifdef USE_SCALALAPACK_OPEMNP_NESTED
            #pragma omp single nowait
            {
                omp_set_max_active_levels(2);
                psgemm_(transa, transb, m, n, k, alpha, a, ia, ja, desca, b, ib,
                        jb, descb, beta, c, ic, jc, descc);
                omp_set_max_active_levels(1);
            }
        #else
            #pragma omp single nowait
            psgemm_(transa, transb, m, n, k, alpha, a, ia, ja, desca, b, ib,
                        jb, descb, beta, c, ic, jc, descc);
        #endif
    #else
        psgemm_(transa, transb, m, n, k, alpha, a, ia, ja, desca, b, ib,
            jb, descb, beta, c, ic, jc, descc);
    #endif
    return;
}
template<>
void p_gemm_(const char *transa, const char *transb, const int *m, const int *n,
                      const int *k, const double *alpha, const double *a, const int *ia,
                      const int *ja, const int *desca, const double *b, const int *ib,
                      const int *jb, const int *descb, const double *beta, double *c,
                      const int *ic, const int *jc, const int *descc ) {
    #ifdef USE_OPENMP
        #ifdef USE_SCALALAPACK_OPEMNP_NESTED
            #pragma omp single nowait
            {
                omp_set_max_active_levels(2);
                pdgemm_(transa, transb, m, n, k, alpha, a, ia, ja, desca, b, ib,
                            jb, descb, beta, c, ic, jc, descc);
                omp_set_max_active_levels(1);
            }
        #else
            #pragma omp single nowait
            pdgemm_(transa, transb, m, n, k, alpha, a, ia, ja, desca, b, ib,
                        jb, descb, beta, c, ic, jc, descc);
        #endif
    #else
        pdgemm_(transa, transb, m, n, k, alpha, a, ia, ja, desca, b, ib,
            jb, descb, beta, c, ic, jc, descc);
    #endif
    return;
}
template<>
void p_sygvx_(int const* const ibtype, char const* const jobz, char const* const range, char const* const uplo,
				      int const* const n, float* const a, int const* const ia, const int* const ja, const int* const desca,
				      float* const b, const int* const ib, const int* const jb, const int* const descb, const float* const vl,
				      const float* const vu, const int* const il, const int* const iu, const float* const abstol,
				      int* const m, int* const nz, float* const w, const float* const orfac, float* const z, const int* const iz,
				      const int* const jz, const int* const descz, float* const work, const int* const lwork, int* const iwork,
				      const int* const liwork, int* const ifail, int* const iclustr, float* const gap, int* const info) {
    #ifdef USE_OPENMP
        #ifdef USE_SCALALAPACK_OPEMNP_NESTED
            #pragma omp single nowait
            {
                omp_set_max_active_levels(2);
                pssygvx_(ibtype, jobz, range, uplo, n, a, ia, ja, desca, b, ib, jb, descb, vl,
                            vu, il, iu, abstol, m, nz, w, orfac, z, iz, jz, descz, work, lwork, iwork,
                            liwork, ifail, iclustr, gap, info);
                omp_set_max_active_levels(1);
            }
        #else
            #pragma omp single nowait
            pssygvx_(ibtype, jobz, range, uplo, n, a, ia, ja, desca, b, ib, jb, descb, vl,
                        vu, il, iu, abstol, m, nz, w, orfac, z, iz, jz, descz, work, lwork, iwork,
                        liwork, ifail, iclustr, gap, info);
        #endif
    #else
        pssygvx_(ibtype, jobz, range, uplo, n, a, ia, ja, desca, b, ib, jb, descb, vl,
             vu, il, iu, abstol, m, nz, w, orfac, z, iz, jz, descz, work, lwork, iwork,
             liwork, ifail, iclustr, gap, info);
    #endif
    return;
}
template<>
void p_sygvx_(int const* const ibtype, char const* const jobz, char const* const range, char const* const uplo,
				  int const* const n, double* const a, int const* const ia, const int* const ja, const int* const desca,
				  double* const b, const int* const ib, const int* const jb, const int* const descb, const double* const vl,
				  const double* const vu, const int* const il, const int* const iu, const double* const abstol,
				  int* const m, int* const nz, double* const w, const double* const orfac, double* const z, const int* const iz,
				  const int* const jz, const int* const descz, double* const work, const int* const lwork, int* const iwork,
				  const int* const liwork, int* const ifail, int* const iclustr, double* const gap, int* const info) {
    #ifdef USE_OPENMP
        #ifdef USE_SCALALAPACK_OPEMNP_NESTED
            #pragma omp single nowait
            {
                omp_set_max_active_levels(2);
                pdsygvx_(ibtype, jobz, range, uplo, n, a, ia, ja, desca, b, ib, jb, descb, vl,
                            vu, il, iu, abstol, m, nz, w, orfac, z, iz, jz, descz, work, lwork, iwork,
                            liwork, ifail, iclustr, gap, info);
                omp_set_max_active_levels(1);
            }
        #else
            #pragma omp single nowait
            pdsygvx_(ibtype, jobz, range, uplo, n, a, ia, ja, desca, b, ib, jb, descb, vl,
                        vu, il, iu, abstol, m, nz, w, orfac, z, iz, jz, descz, work, lwork, iwork,
                        liwork, ifail, iclustr, gap, info);
        #endif
    #else
        pdsygvx_(ibtype, jobz, range, uplo, n, a, ia, ja, desca, b, ib, jb, descb, vl,
             vu, il, iu, abstol, m, nz, w, orfac, z, iz, jz, descz, work, lwork, iwork,
             liwork, ifail, iclustr, gap, info);
    #endif
    return;
}
#endif //USE_MKL | USE_SCALAPACK

} // namespace Linalg

#undef FOR_WHILE
