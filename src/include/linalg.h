#ifndef _LINALG_H_
#define _LINALG_H_

#include <iostream>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <climits>
#include <mpi.h>
#include <complex>
#include "linalg.h"
#include "environment.h"
#include "index.h"
#ifdef USE_MEMFUNS
#include <cstring>
#endif //USE_MEMFUNS
namespace Linalg {

    #ifdef USE_OPENMP
    void init_omp_env();
    #endif
    void set_kblas_nthread();
    void set_kblas_1();
    uint64_t get_chunksize(const uint64_t nthreads, const uint64_t njobs);
    template<typename T> uint64_t get_type_id();
    template<typename T1, typename T2> void convert_type(T1* const out_ptr, T2 const* const in_ptr, const uint64_t length);
    template<typename T1, typename T2> void convert_type(std::complex<T1>* const out_ptr, T2 const* const in_ptr, const uint64_t length);
    template<typename T> MPI_Datatype get_mpi_datatype();
    template<typename T> void print_vector(T const* const ptr, uint64_t const length);
    template<typename T> void print_vector(T const* const ptr, uint64_t const length, const std::string& fname);
    template<typename T> void print_vectorX(const T* ptr, const uint& length, const std::string& fname);

    inline uint64_t splitmix64(uint64_t x);
    template<typename T> void seededrand(T* const ptr, uint64_t const length, T const rand_min = 0, T const rand_max = RAND_MAX, uint64_t const seed = 1);
    template<typename T> void seededrand(std::complex<T>* const ptr, uint64_t const length, T const rand_min = 0, T const rand_max = RAND_MAX, uint64_t const seed = 1);
    template<typename T1, typename T2> void seededrand_sequential(T1* const ptr, uint64_t const length, T2 const rand_min = 0, T2 const rand_max = RAND_MAX, uint64_t const seed = 1);
    template<typename T1, typename T2> void seededrand_sequential(std::complex<T1>* const ptr, uint64_t const length, T2 const rand_min = 0, T2 const rand_max = RAND_MAX, uint64_t const seed = 1);

//Vertor functions
    template<typename T> T vector_sum(T const* const ptr, uint64_t const length, const MPI_Comm comm = MPI_COMM_NULL);
    template<typename T> std::complex<T> vector_sum(std::complex<T> const* const ptr, uint64_t const length, const MPI_Comm comm = MPI_COMM_NULL);
    template<typename T> T vector_norm_square_sum(T const* const ptr, uint64_t const length, const MPI_Comm comm = MPI_COMM_NULL);
    template<typename T> T vector_norm_square_sum(std::complex<T> const* const ptr, uint64_t const length, const MPI_Comm comm = MPI_COMM_NULL);
    template<typename T> T vector_dot_product(T const* const ptr_A, T const* const ptr_B, const uint64_t length, const MPI_Comm comm = MPI_COMM_NULL);
    template<typename T> std::complex<T> vector_dot_product(std::complex<T> const* const ptr_A, std::complex<T> const* const ptr_B, const uint64_t length, const MPI_Comm comm = MPI_COMM_NULL);
    template<typename T> T vector_2norm(T const* const ptr, const uint64_t length, const MPI_Comm comm = MPI_COMM_NULL);
    template<typename T> T vector_2norm(std::complex<T> const* const ptr, const uint64_t length, const MPI_Comm comm = MPI_COMM_NULL);
    template<typename T> void vector_normalize(T* const ptr, const uint64_t length, const MPI_Comm comm = MPI_COMM_NULL);

    template<typename T> void vector_orthogonalize(T* const inout_ptr, T const* const ptr_A, const uint64_t length, const MPI_Comm comm = MPI_COMM_NULL);
    template<typename T> void vector_norm_square(T* const out_ptr, const uint64_t length);
    template<typename T> void vector_norm_square(std::complex<T>* const out_ptr, const uint64_t length);
    template<typename T> void vector_norm_square(T* const out_ptr, T const* const ptr_A, const uint64_t length);
    template<typename T> void vector_norm_square(T* const out_ptr, std::complex<T> const* const ptr_A, const uint64_t length);
    template<typename T> void vector_norm_square(T* const out_ptr, T const* const ptr_A, const uint64_t length, const T alpha);
    template<typename T> void vector_norm_square(T const out_ptr, std::complex<T> const* const ptr_A, const uint64_t length, const T alpha);
    template<typename T> void accumulate_vector_norm_square(T* const out_ptr, T const* const ptr_A, const uint64_t length, const T alpha);
    template<typename T> void accumulate_vector_norm_square(T* const out_ptr, std::complex<T> const* const ptr_A, const uint64_t length, const T alpha);

    template<typename T> void set_value_general(T* const out_ptr, const T num, const uint64_t length);
    template<typename T> void set_value_general(T* const out_ptr, T const* const ptr_A, const uint64_t length);

    template<typename T> void vector_cbrt(T* const out_ptr, const uint64_t length);
    template<typename T> void vector_cbrt(T* const out_ptr, T const* const ptr_A, const uint64_t length);
    template<typename T> void conj_general(std::complex<T>* const out_ptr, const uint64_t length);
    template<typename T> void conj_general(std::complex<T>* const out_ptr, std::complex<T> const* const ptr_A, const uint64_t length);
    template<typename T> void conj_product(std::complex<T>* const out_ptr, std::complex<T> const* const ptr_A, std::complex<T> const* const ptr_B, const uint64_t length);

    template<typename T> void hadamard_plus_general(T* const out_ptr, T const* const ptr_A, const uint64_t length);
    template<typename T> void hadamard_minus_general(T* const out_ptr, T const* const ptr_A, const uint64_t length);
    template<typename T> void hadamard_product_general(T* const out_ptr, T const* const ptr_A, const uint64_t length);
    template<typename T> void hadamard_divide_general(T* const out_ptr, T const* const ptr_A, const uint64_t length);
    template<typename T> void hadamard_product_general(T* const out_ptr, T const* const ptr_A, const uint64_t length,
                                                       T const alpha);
    template<typename T> void hadamard_product_general(T* const out_ptr, T const* const ptr_A, const uint64_t length,
                                                       T const alpha, T const beta);
    template<typename T> void hadamard_product_general(T* const out_ptr, T const* const ptr_A, const uint64_t length,
                                                       T const alpha, T const beta, T const gamma);

    template<typename T> void hadamard_plus_general(T* const out_ptr, T const* const ptr_A, T const* const ptr_B, const uint64_t length);
    template<typename T> void hadamard_minus_general(T* const out_ptr, T const* const ptr_A, T const* const ptr_B, const uint64_t length);
    template<typename T> void hadamard_product_general(T* const out_ptr, T const* const ptr_A, T const* const ptr_B, const uint64_t length);
    template<typename T> void hadamard_divide_general(T* const out_ptr, T const* const ptr_A, T const* const ptr_B, const uint64_t length);
    template<typename T> void hadamard_product_general(T* const out_ptr, T const* const ptr_A, T const* const ptr_B, const uint64_t length,
                                                        T const alpha);
    template<typename T> void hadamard_product_general(T* const out_ptr, T const* const ptr_A, T const* const ptr_B, const uint64_t length,
                                                        T const alpha, T const beta);
    template<typename T> void hadamard_product_general(T* const out_ptr, T const* const ptr_A, T const* const ptr_B, const uint64_t length,
                                                        T const alpha, T const beta, T const gamma);

    template<typename T> void accumulate_hadamard_plus_general(T* const out_ptr, T const* const ptr_A, T const* const ptr_B, const uint64_t length);
    template<typename T> void accumulate_hadamard_minus_general(T* const out_ptr, T const* const ptr_A, T const* const ptr_B, const uint64_t length);
    template<typename T> void accumulate_hadamard_product_general(T* const out_ptr, T const* const ptr_A, T const* const ptr_B, const uint64_t length);
    template<typename T> void accumulate_hadamard_divide_general(T* const out_ptr, T const* const ptr_A, T const* const ptr_B, const uint64_t length);
    template<typename T> void accumulate_hadamard_product_general(T* const out_ptr, T const* const ptr_A, T const* const ptr_B, const uint64_t length,
                                                        T const alpha);
    template<typename T> void accumulate_hadamard_product_general(T* const out_ptr, T const* const ptr_A, T const* const ptr_B, const uint64_t length,
                                                        T const alpha, T const beta);
    template<typename T> void accumulate_hadamard_product_general(T* const out_ptr, T const* const ptr_A, T const* const ptr_B, const uint64_t length,
                                                        T const alpha, T const beta, T const gamma);

    template<typename T> void scalar_plus_general(T* const out_ptr, const T num, const uint64_t length);
    template<typename T> void scalar_minus_general(T* const out_ptr, const T num, const uint64_t length);
    template<typename T1, typename T2> void scalar_product_general(T1* const out_ptr, const T2 num, const uint64_t length);
    template<typename T> void scalar_divide_general(T* const out_ptr, const T num, const uint64_t length);
    template<typename T> void scalar_product_general(T* const out_ptr, const T num, const uint64_t length,
                                                        T const* const ptr_B);
    template<typename T1, typename T2> void scalar_product_general(T1* const out_ptr, const T2 num, const uint64_t length,
                                                        T1 const* const ptr_B, T2 const beta);
    template<typename T> void scalar_product_general(T* const out_ptr, const T num, const uint64_t length,
                                                        T const* const ptr_B, T const beta, T const gamma);

    template<typename T> void scalar_plus_general(T* const out_ptr, T const* const ptr_A, const T num, const uint64_t length);
    template<typename T> void scalar_minus_general(T* const out_ptr, T const* const ptr_A, const T num, const uint64_t length);
    template<typename T1, typename T2> void scalar_product_general(T1* const out_ptr, T1 const* const ptr_A, const T2 num, const uint64_t length);
    template<typename T> void scalar_divide_general(T* const out_ptr, T const* const ptr_A, const T num, const uint64_t length);
    template<typename T> void scalar_product_general(T* const out_ptr, T const* const ptr_A, const T num, const uint64_t length,
                                                     T const* const ptr_B);
    template<typename T> void scalar_product_general(T* const out_ptr, T const* const ptr_A, const T num, const uint64_t length,
                                                     T const* const ptr_B, T const beta);
    template<typename T> void scalar_product_general(T* const out_ptr, T const* const ptr_A, const T num, const uint64_t length,
                                                     T const* const ptr_B, T const beta, T const gamma);

    template<typename T> void accumulate_scalar_plus_general(T* const out_ptr, T const* const ptr_A, const T num, const uint64_t length);
    template<typename T> void accumulate_scalar_minus_general(T* const out_ptr, T const* const ptr_A, const T num, const uint64_t length);
    template<typename T1, typename T2> void accumulate_scalar_product_general(T1* const out_ptr, T1 const* const ptr_A, const T2 num, const uint64_t length);
    template<typename T> void accumulate_scalar_divide_general(T* const out_ptr, T const* const ptr_A, const T num, const uint64_t length);
    template<typename T> void accumulate_scalar_product_general(T* const out_ptr, T const* const ptr_A, const T num, const uint64_t length,
                                                    T const* const ptr_B);
    template<typename T> void accumulate_scalar_product_general(T* const out_ptr, T const* const ptr_A, const T num, const uint64_t length,
                                                    T const* const ptr_B, T const beta);
    template<typename T> void accumulate_scalar_product_general(T* const out_ptr, T const* const ptr_A, const T num, const uint64_t length,
                                                    T const* const ptr_B, T const beta, T const gamma);

//Matrix functions
    template<typename T> void matrix_vector_product(const T* ptr_A, const uint is_Col_Maj_A,
                                                    const T* ptr_B, T* ptr_C,
                                                    const int64_t m, const int64_t n);
    template<typename T> void matrix_vector_product(const T* ptr_A, const uint is_Col_Maj_A,
                                                    const T* ptr_B, T* ptr_C,
                                                    const int64_t m, const int64_t n,
                                                    const T alpha);
    template<typename T> void matrix_vector_product(const T* ptr_A, const uint is_Col_Maj_A,
                                                    const T* ptr_B, T* ptr_C,
                                                    const int64_t m, const int64_t n,
                                                    const T alpha, const T beta);
    template<typename T> void matrix_vector_product(const T* ptr_A, const uint is_Col_Maj_A,
                                                    const T* ptr_B, T* ptr_C,
                                                    const int64_t m, const int64_t n,
                                                    const T alpha, const T beta, const T gamma);
    template<typename T> void matrix_product(const T* ptr_A, const uint is_Col_Maj_A,
                                             const T* ptr_B, const uint is_Col_Maj_B,
                                             T* ptr_C, const uint is_Col_Maj_C,
                                             const int64_t m, const int64_t n, const int64_t k);
    template<typename T> void matrix_product(const T* ptr_A, const uint is_Col_Maj_A,
                                             const T* ptr_B, const uint is_Col_Maj_B,
                                             T* ptr_C, const uint is_Col_Maj_C,
                                             const int64_t m, const int64_t n, const int64_t k,
                                             const T alpha);
    template<typename T> void matrix_product(const T* ptr_A, const uint is_Col_Maj_A,
                                             const T* ptr_B, const uint is_Col_Maj_B,
                                             T* ptr_C, const uint is_Col_Maj_C,
                                             const int64_t m, const int64_t n, const int64_t k,
                                             const T alpha, const T beta);
    template<typename T> void matrix_product(const T* ptr_A, const uint is_Col_Maj_A,
                                             const T* ptr_B, const uint is_Col_Maj_B,
                                             T* ptr_C, const uint is_Col_Maj_C,
                                             const int64_t m, const int64_t n, const int64_t k,
                                             const T alpha, const T beta, const T gamma);
    template<typename T> void matrix_product_general(const T* ptr_A, const uint is_Col_Maj_A,
                                                     const T* ptr_B, const uint is_Col_Maj_B,
                                                     T* ptr_C, const uint is_Col_Maj_C,
                                                     const int64_t m, const int64_t n, const int64_t k);
    template<typename T> void matrix_product_general(const T* ptr_A, const uint is_Col_Maj_A,
                                                     const T* ptr_B, const uint is_Col_Maj_B,
                                                     T* ptr_C, const uint is_Col_Maj_C,
                                                     const int64_t m, const int64_t n, const int64_t k,
                                                     const T alpha);
    template<typename T> void matrix_product_general(const T* ptr_A, const uint is_Col_Maj_A,
                                                     const T* ptr_B, const uint is_Col_Maj_B,
                                                     T* ptr_C, const uint is_Col_Maj_C,
                                                     const int64_t m, const int64_t n, const int64_t k,
                                                     const T alpha, const T beta);
    template<typename T> void matrix_product_general(const T* ptr_A, const uint is_Col_Maj_A,
                                                     const T* ptr_B, const uint is_Col_Maj_B,
                                                     T* ptr_C, const uint is_Col_Maj_C,
                                                     const int64_t m, const int64_t n, const int64_t k,
                                                     const T alpha, const T beta, const T gamma);
    #ifdef USE_CBLAS
    template<typename T>
    void cblas__gemv(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE trans,
                      const int64_t m, const int64_t n,
                      const T alpha, T const* const a, const int64_t lda,
                      const T *x, const int64_t incx,
                      const T beta, T* const y, const int64_t incy);
    #ifdef USE_OPENMP
    template<typename T>
    void cblas__gemv_omp_row_par(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE trans,
                      const int64_t m, const int64_t n,
                      const T alpha, T const* const a, const int64_t lda,
                      const T *x, const int64_t incx,
                      const T beta, T* const y, const int64_t incy);
    #endif //USE_OPENMP
    template<typename T>
    void cblas__gemm(const CBLAS_ORDER C_LAYOUT, const CBLAS_TRANSPOSE A_TRANSPOSE, const CBLAS_TRANSPOSE B_TRANSPOSE,
                      const int64_t m, const int64_t n, const int64_t k,
                      const T alpha, T const* const ptr_A, const int64_t A_ld,
                      T const* const ptr_B, const int64_t B_ld,
                      const T beta, T* const ptr_C, const int64_t C_ld);
    #ifdef USE_OPENMP
    template<typename T>
    inline void cblas__gemm_omp_col_par(const CBLAS_ORDER C_LAYOUT, const CBLAS_TRANSPOSE A_TRANSPOSE, const CBLAS_TRANSPOSE B_TRANSPOSE,
                      const int64_t m, const int64_t n, const int64_t k,
                      const T alpha, T const* const ptr_A, const int64_t A_ld,
                      T const* const ptr_B, const int64_t B_ld,
                      const T beta, T* const ptr_C, const int64_t C_ld);
    #endif
    template<typename T>
    void cblas__syrk(const CBLAS_ORDER layout, const CBLAS_UPLO uplo, const CBLAS_TRANSPOSE trans,
            const int64_t n, const int64_t k,
            const T alpha, T const* const ptr_A, const int64_t A_ld,
            const T beta, T* const ptr_C, const int64_t C_ld);
    template<typename T> void matrix_vector_product_cblas(const T* ptr_A, const uint is_Col_Maj_A,
                                                          const T* ptr_B, T* ptr_C,
                                                          const int64_t m, const int64_t n,
                                                          const T alpha, const T beta);
    template<typename T> void matrix_vector_product_cblas(const T* ptr_A, const uint is_Col_Maj_A,
                                                          const T* ptr_B, T* ptr_C,
                                                          const int64_t m, const int64_t n);
    template<typename T> void matrix_vector_product_cblas(const T* ptr_A, const uint is_Col_Maj_A,
                                                          const T* ptr_B, T* ptr_C,
                                                          const int64_t m, const int64_t n,
                                                          const T alpha);
    template<typename T> void matrix_vector_product_cblas(const T* ptr_A, const uint is_Col_Maj_A,
                                                          const T* ptr_B, T* ptr_C,
                                                          const int64_t m, const int64_t n,
                                                          const T alpha, const T beta, const T gamma);
    template<typename T> void matrix_product_cblas(const T* ptr_A, const uint is_Col_Maj_A,
                                                   const T* ptr_B, const uint is_Col_Maj_B,
                                                   T* ptr_C, const uint is_Col_Maj_C,
                                                   const int64_t m, const int64_t n, const int64_t k,
                                                   const T alpha, const T beta);
    template<typename T> void matrix_product_cblas(const T* ptr_A, const uint is_Col_Maj_A,
                                                   const T* ptr_B, const uint is_Col_Maj_B,
                                                   T* ptr_C, const uint is_Col_Maj_C,
                                                   const int64_t m, const int64_t n, const int64_t k);
    template<typename T> void matrix_product_cblas(const T* ptr_A, const uint is_Col_Maj_A,
                                                   const T* ptr_B, const uint is_Col_Maj_B,
                                                   T* ptr_C, const uint is_Col_Maj_C,
                                                   const int64_t m, const int64_t n, const int64_t k,
                                                   const T alpha);
    template<typename T> void matrix_product_cblas(const T* ptr_A, const uint is_Col_Maj_A,
                                                   const T* ptr_B, const uint is_Col_Maj_B,
                                                   T* ptr_C, const uint is_Col_Maj_C,
                                                   const int64_t m, const int64_t n, const int64_t k,
                                                   const T alpha, const T beta, const T gamma);
    #endif //USE_CBLAS

    #ifdef USE_LAPACK
    template<typename T>
    lapack_int LAPACKE__gelsd(const int matrix_layout, const lapack_int m, const lapack_int n,
                                const lapack_int nrhs, T *a, const lapack_int lda,
                                T *b, const lapack_int ldb, T *s, T rcond,
                                lapack_int* rank);
    template<typename T>
    lapack_int LAPACKE__gelsd_org(const int matrix_layout, const lapack_int m, const lapack_int n,
                                const lapack_int nrhs, T *a, const lapack_int lda,
                                T *b, const lapack_int ldb, T *s, T rcond,
                                lapack_int* rank);
    template<typename T>
    lapack_int LAPACKE__gelsd(const int matrix_layout, const lapack_int m, const lapack_int n,
                                const lapack_int nrhs, std::complex<T> *a,
                                const lapack_int lda, std::complex<T> *b,
                                const lapack_int ldb, T *s, T rcond,
                                lapack_int* rank);
    template<typename T>
    lapack_int LAPACKE__sygvd(const int matrix_layout, const lapack_int itype, char jobz,
                        char uplo, const lapack_int n, T* a, const lapack_int lda,
                        T* b, const lapack_int ldb, T* w);
    template<typename T>
    lapack_int LAPACKE__sygvd_org(const int matrix_layout, const lapack_int itype, char jobz,
                        char uplo, const lapack_int n, T* a, const lapack_int lda,
                        T* b, const lapack_int ldb, T* w);
    template<typename T>
    lapack_int LAPACKE__hygvd(const int matrix_layout, const lapack_int itype, char jobz,
                                char uplo, const lapack_int n, std::complex<T>* a,
                                const lapack_int lda, std::complex<T>* b,
                                const lapack_int ldb, T* w);
    template<typename T>
    lapack_int LAPACKE__sterf(const lapack_int order, T* d, T* e);
    template<typename T>
    lapack_int LAPACKE__sterf_org(const lapack_int order, T* d, T* e);
    #endif //USE_LAPACK

    #if (defined(USE_MKL) || defined(USE_SCALAPACK))
    template<typename T>
    void p_gemr2d_(const int *m, const int *n,
                    const T *a, const int *ia, const int *ja, const int *desca,
                    T *b, const int *ib, const int *jb, const int *descb,
                    const int *ictxt);
    template<typename T>
    void p_syrk_(const char *uplo, const char *trans, const int *n, const int *k,
                  const T *alpha, const T *a, const int *ia, const int *ja,
                  const int *desca, const T *beta, T *c, const int *ic,
                  const int *jc, const int *descc );
    template<typename T>
    void p_gemm_(const char *transa, const char *transb, const int *m, const int *n,
                  const int *k, const T *alpha, const T *a, const int *ia,
                  const int *ja, const int *desca, const T *b, const int *ib,
                  const int *jb, const int *descb, const T *beta, T *c,
                  const int *ic, const int *jc, const int *descc );
    template<typename T>
    void p_sygvx_(int const* const ibtype, char const* const jobz, char const* const range, char const* const uplo,
				  int const* const n, T* const a, int const* const ia, const int* const ja, const int* const desca,
				  T* const b, const int* const ib, const int* const jb, const int* const descb, const T* const vl,
				  const T* const vu, const int* const il, const int* const iu, const T* const abstol,
				  int* const m, int* const nz, T* const w, const T* const orfac, T* const z, const int* const iz,
				  const int* const jz, const int* const descz, T* const work, const int* const lwork, int* const iwork,
				  const int* const liwork, int* const ifail, int* const iclustr, T* const gap, int* const info);
    #endif

}

#endif //_LINALG_H_