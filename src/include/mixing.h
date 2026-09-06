#ifndef _MIXING_H_
#define _MIXING_H_
#include <iostream>
#include <mpi.h>
#include "linalg.h"
#include "arr.h"
#include "aar.h"
#include "control.h"
#include "spin.h"

template<typename T>
class Mixing
{
public:
    const Mixing_control& mixing_control;
    const Spin& spin;
    const Stencil<T>& stencil;
    const Exarr_3D_mpi_package& exarr_mpi_package;
    const uint& mixing_variable;
    Array_3D<T> x_km1;
    Array_3D<T> f_km1;
    Array_3D<T> pf;
    Array_4D<T> R;
    Array_4D<T> F;
    Mixing(const Mixing_control& mixing_control,
           const Spin& spin,
           const Stencil<T>& stencil,
           const Exarr_3D_mpi_package& exarr_mpi_package,
           const uint& mixing_variable);
    ~Mixing();
    void set_x_km1(const Array_3D<T>& x_km1);
    void set_electron_densities(const std::vector<Array_3D<T>>& electron_densities);
    void record_history(const Array_3D<T>& x_k_out, const Array_3D<T>& f_k, const uint& iter);
    void record_F_history(const Array_3D<T>& f_k, const uint iter);
    void record_F_history(T const* const f_k, const uint iter, uint const nd, const uint ncol);
    void record_R_history(const Array_3D<T>& x_k_out, const uint iter);
    void record_R_history(T const* const x_k_out, const uint iter, uint const nd, const uint ncol);
    void generate_weight_average(Array_3D<T>& x_wavg, Array_3D<T>& f_wavg,
                                 const Array_3D<T>& x_k, const Array_3D<T>& f_k,
                                 const uint iter, const MPI_Comm comm);
    void generate_weight_average_mp(T* const x_wavg, T* const f_wavg,
                                    T const* const x_k, T const* const f_k,
                                    const uint iter, const MPI_Comm comm,
                                    uint const nd, const uint ncol,
                                    Memory_pool<T, Fast_memory>& pool_fast,
                                    Memory_pool<T, Capacity_memory>& pool_cap);
    void apply_precondition(const Array_3D<T>& f_wavg, Array_3D<T>& pf_wavg, const uint iter);
    void apply_precondition(T const* const f_wavg, T* const pf_wavg, const Vertices_3D& vertices, const uint iter, const uint ncol);
    void apply_precondition_mp(T const* const f_wavg, T* const pf_wavg, const Vertices_3D& vertices, const uint iter, const uint ncol,
                                    Memory_pool<T, Fast_memory>& pool_fast,
                                    Memory_pool<T, Capacity_memory>& pool_cap);
    void shift_pf(const Array_3D<T>& f_wavg, Array_3D<T>& pf_wavg, const MPI_Comm comm);
    void shift_pf(T const* const f_wavg, T* const pf_wavg, const Vertices_3D& vertices, const uint ncol, const MPI_Comm comm);
    void scale_density(const Array_3D<T>& x_k, Array_3D<T>& x_k_out, const MPI_Comm comm);
    void scale_density(T const* const x_k, T* const x_k_out, const MPI_Comm comm, uint const nd);
    void run(Array_3D<T>& x_k, const uint iter, const MPI_Comm comm);
    void run_mp(T* const x_k, const uint iter, const MPI_Comm comm,
                Memory_pool<T, Fast_memory>& pool_fast,
                Memory_pool<T, Capacity_memory>& pool_cap);
    void init(const Array_3D<T>& x_km1, char const* const& header = "SCF");
    template<typename T2> void init(const Mixing<T2>& mixing);
    void destructor();
    void show() const;
};

namespace Mixing_method {

    template<typename T>
    void AndersonExtrapolation(const int N, const int m, T* const x_kp1, T const* const x_k, 
                               T const* const f_k, T const* const X, T const* const F, 
                               const T beta, const MPI_Comm comm);
    template<typename T>
    void AndersonExtrapolation_ompunnested(const int N, const int m, T* const x_kp1, T const* const x_k, 
                                            T const* const f_k, T const* const X, T const* const F, 
                                            const T beta, const MPI_Comm comm);
    template<typename T>
    void AndersonExtrapolation_ompunnested_mp(const int N, const int m, T* const x_kp1, T const* const x_k, 
                                            T const* const f_k, T const* const X, T const* const F, 
                                            const T beta, const MPI_Comm comm,
                                            Memory_pool<T, Fast_memory>& pool_fast,
                                            Memory_pool<T, Capacity_memory>& pool_cap);

    template<typename T>
    void Anderson_extrapolation_weighted_averaged_vectors(const int N, const int m,
                                                          T* const x_wavg, T* const f_wavg,
                                                          T const* const x_k, T const* const f_k,
                                                          T const* const X, T const* const F, 
                                                          const MPI_Comm comm);
    template<typename T>
    void Anderson_extrapolation_weighted_averaged_vectors_mp(const int N, const int m,
                                                            T* const x_wavg, T* const f_wavg,
                                                            T const* const x_k, T const* const f_k,
                                                            T const* const X, T const* const F, 
                                                            const MPI_Comm comm,
                                                            Memory_pool<T, Fast_memory>& pool_fast,
                                                            Memory_pool<T, Capacity_memory>& pool_cap);

    template<typename T>
    void Anderson_history_vector(const uint N, const uint m, const T beta, T const* const f_k,
                                        T const* const X, T const* const F, T* const history_v, const MPI_Comm comm);
    template<typename T>
    void Anderson_history_vector_ompunnested(const uint N, const uint m, const T beta, T const* const f_k,
                                        T const* const X, T const* const F, T* const history_v, const MPI_Comm comm);
    template<typename T>
    void Anderson_history_vector_ompunnested_mp(const uint N, const uint m, const T beta, T const* const f_k,
                                        T const* const X, T const* const F, T* const history_v, const MPI_Comm comm,
                                        Memory_pool<T, Fast_memory>& pool_fast,
                                        Memory_pool<T, Capacity_memory>& pool_cap);
    template<typename T>
    void cal_Gamma(const uint N, const uint m, T const* const  F, T const* const  f,
                   T* const Gamma, const MPI_Comm comm);
    template<typename T>
    void cal_Gamma_ompunnested(const uint N, const uint m, T const* const  F, T const* const  f,
                   T* const Gamma, const MPI_Comm comm);
    template<typename T>
    void cal_Gamma_ompunnested_mp(const uint N, const uint m, T const* const  F, T const* const  f,
                                    T* const Gamma, const MPI_Comm comm,
                                    Memory_pool<T, Fast_memory>& pool_fast,
                                    Memory_pool<T, Capacity_memory>& pool_cap);
    template<typename T>
    void kerker_precondition(const Array_3D<T>& f_wavg, Array_3D<T>& pf_wavg, const T& alpha,
                             const Stencil<T> stencil, const Kerker_control& kerker_control,
                             const Exarr_3D_mpi_package& exarr_mpi_package);
    template<typename T>
    void kerker_precondition(T const* const f_wavg, T* const pf_wavg, const Vertices_3D& vertices,
                             const T alpha, const Stencil<T> stencil, const Kerker_control& kerker_control,
                             const Exarr_3D_mpi_package& exarr_mpi_package);
    template<typename T>
    void kerker_precondition_mp(T const* const f_wavg, T* const pf_wavg, const Vertices_3D& vertices,
                                const T alpha, const Stencil<T> stencil, const Kerker_control& kerker_control,
                                const Exarr_3D_mpi_package& exarr_mpi_package,
                                Memory_pool<T, Fast_memory>& pool_fast,
                                Memory_pool<T, Capacity_memory>& pool_cap);

}


#endif //_MIXING_H_