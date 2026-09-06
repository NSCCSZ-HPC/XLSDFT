#ifndef _TOOLS_H_
#define _TOOLS_H_

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <string>
#include <chrono>
#include <mpi.h>
#include <stdexcept>
#include <functional>

namespace Tools {
    uint get_node_number();
    void print_rankfile();
    int group_of(const int& i, const int& n);
    template<typename T> void sort(T const* const& X, const int& len, T* const& Xsorted, int* const& Ind);
    template<typename T1, typename T2> int binary_interval_search(T1 const* const& list, const int len, const T2 x);
    template<typename T> void tridiag_gen(T const* const& A, T const* const& B,  T const* const& C, T* const D, const int& len);
    template<typename T> void getYD_gen(T const* const& X, T const* const& Y, T* const& YD, const int& len);
    template<typename T1, typename T2> void SplineInterp(T1 const* const& X1, T1 const* const& Y1, const int& len1,
                                                         T2 const* const& X2, T2* const& Y2, const int& len2, T1 const* const& YD);
    template<typename T1, typename T2> void SortSplineInterp(T1 const* const& X1, T1 const* const& Y1, const int& len1,
                                                             T2 const* const& X2, T2* const& Y2, const int& len2, T1 const* const& YD);
    template<typename T1, typename T2> void SplineInterpUniform(T1 const* const& X1, T1 const* const& Y1, const int& len1,
                                                                T2 const* const& X2, T2* const& Y2, const int& len2, T1 const* const& YD);
    template<typename T1, typename T2> void SplineInterpNonuniform(T1 const* const& X1, T1 const* const& Y1, const int& len1,
                                                                   T2 const* const& X2, T2* const& Y2, const int& len2, T1 const* const& YD);
    template<typename T> void RealSphericalHarmonic(T* const& Ylm, const int& l, const int& m, const int& length,
                                                    T const* const& x, T const* const& y, T const* const& z,
                                                    T const* const& r);
    template<typename T> void RealSphericalHarmonic2(T* const& Ylm, const int& l, const int& m, const int& length,
                                                     T const* const& x_r, T const* const& y_r, T const* const& z_r,
                                                     T const* const& r);
    template<typename T> bool is_uniform(T const* const& x, const int& length);
    template<typename T> void simpson_antideriv(T* const y, T const* const x, T const h, const uint64_t length, const bool if_periodic);
    template<typename T> void simpson_double_antideriv(T* const y, T const* const x, T const h, const uint64_t length, const bool if_periodic);
    template<typename T> void second_deriv_fd(T* const y, T const* const f, T const* const coef, const uint64_t FDn, const uint64_t length, const bool if_periodic);
    std::string time_cost(const std::chrono::steady_clock::time_point& begin, const std::chrono::steady_clock::time_point& end);
    template<typename T>
    T BrentsFun(std::function<T(const T)> func, const T lower_bound, const T upper_bound,
                const uint max_iter, const T tol);
    template<typename T>
    T BatchedBisection(std::function<void(const T* const in, T* const out, const uint n)> func, 
                    const T lower_bound, const T upper_bound,
                    const uint max_iter, const T tol, const uint n);
    template<typename T>
    T BatchedBrent(std::function<void(const T* const, T* const, const unsigned int)> func, 
                    const T lower_bound, const T upper_bound,
                    const unsigned int max_iter, const T tol, const unsigned int n);
    template<typename T>
    T SpectralHistogramRoot(
        std::function<void(const T L, const T R, const uint num_bins, T* const out_hist)> hist_func,
        std::function<void(const T* const in, T* const out_f, T* const out_df, const uint n)> exact_func_df,
        std::function<void(const T* const in, T* const out_f, const uint n)> exact_func_f,
        std::function<T(const T delta_E)> local_smear_eval,
        const T lower_bound, const T upper_bound,
        const uint max_iter, const T tol, const T target_charge,
        const uint num_bins, const uint fallback_n,
        std::function<T(const T chemical_potential)> local_extra_eval = {});
    void calc_mpi_statistics(double const* const local_values,
                            const int nitems,
                            const MPI_Comm comm,
                            double *const max_values,
                            double *const min_values,
                            double *const mean_values,
                            double *const stddev_values,
                            double *const reciprocal_mean_values);
    void calc_mpi_statistics(double const* const local_values,
                            const int nitems,
                            const MPI_Comm comm,
                            double* const max_values,
                            double* const min_values,
                            double* const mean_values,
                            double* const stddev_values,
                            double* const reciprocal_mean_values,
                            int* const max_ranks);
    void calc_mpi_statistics(double const* const local_values,
                            const int nitems,
                            const MPI_Comm comm,
                            double* const max_values,
                            double* const min_values,
                            double* const mean_values,
                            double* const stddev_values,
                            double* const reciprocal_mean_values,
                            int* const max_ranks,
                            int* const min_ranks);
}

#endif //_TOOLS_H_