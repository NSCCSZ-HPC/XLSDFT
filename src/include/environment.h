#ifndef _ENVIRONMENT_H_
#define _ENVIRONMENT_H_

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define RESET "\033[0m"

#if (defined(USE_OPENMP_SIMD) || defined(USE_OPENMP))
    #include <omp.h>
#endif

#ifdef USE_MKL
#include <mkl_dfti.h>
#endif

#if defined(USE_FFTW) || defined(USE_MKL) || defined(USE_KML)
#include <fftw3.h>
#endif

#ifdef USE_KML

    #ifdef USE_HBM
    #ifdef USE_HUGEPAGE_HBM
        #include <sys/mman.h>
        #include <numa.h>
        #include <numaif.h>
        #include <unistd.h>
    #else
        #include <hbwmalloc.h>
        #include <numaif.h>
        #include <unistd.h>
    #endif
    #endif

    #ifndef USE_CBLAS
        #define USE_CBLAS
    #endif
    #include <kblas.h>
    #ifndef USE_LAPACK
        #define USE_LAPACK
    #endif
    #include <lapacke.h>
    

    #ifdef USE_OPENMP
        #ifndef USE_MY_OPENMP_BLAS
            #define USE_MY_OPENMP_BLAS
        #endif
        #ifndef USE_OPENMP_NESTED
            #define USE_OPENMP_NESTED
        #endif
        // #ifndef USE_BLAS_OPEMNP_NESTED
        //     #define USE_BLAS_OPEMNP_NESTED
        // #endif
    #endif

#elif defined(USE_MKL)

    #ifdef USE_OPENMP
        #ifndef USE_MY_OPENMP_BLAS
            #define USE_MY_OPENMP_BLAS
        #endif
        #ifndef USE_OPENMP_NESTED
            #define USE_OPENMP_NESTED
        #endif
        // #ifndef USE_BLAS_OPEMNP_NESTED
        //     #define USE_BLAS_OPEMNP_NESTED
        // #endif
        // #ifndef USE_LAPACK_OPEMNP_NESTED
        //     #define USE_LAPACK_OPEMNP_NESTED
        // #endif
        // #ifndef USE_SCALALAPACK_OPEMNP_NESTED
        //     #define USE_SCALALAPACK_OPEMNP_NESTED
        // #endif
    #endif
    #include <mkl.h>
    #ifndef USE_CBLAS
        #define USE_CBLAS
    #endif
    #ifndef USE_LAPACK
        #define USE_LAPACK
    #endif
    #include "scalapack_extern.h"

#elif defined(USE_SCALAPACK)

    #ifdef USE_OPENBLAS
        #ifndef USE_CBLAS
            #define USE_CBLAS
        #endif
        #ifndef USE_LAPACK
            #define USE_LAPACK
        #endif
    #endif
    #ifdef USE_CBLAS
    #include <cblas.h>
    #endif
    #ifdef USE_LAPACK
    #include <lapacke.h>
    #endif
    #include "scalapack_extern.h"

#elif defined(USE_OPENBLAS)

    #ifndef USE_CBLAS
        #define USE_CBLAS
    #endif
    #ifndef USE_LAPACK
        #define USE_LAPACK
    #endif
    #ifdef USE_CBLAS
    #include <cblas.h>
    #endif
    #ifdef USE_LAPACK
    #include <lapacke.h>
    #endif

#else

    #ifdef USE_CBLAS
    #include <cblas.h>
    #endif
    #ifdef USE_LAPACK
    #include <lapacke.h>
    #endif

#endif // USE_MKL

#endif //_ENVIRONMENT_H_
