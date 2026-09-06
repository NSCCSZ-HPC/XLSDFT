#ifndef _LANCZOS_H_
#define _LANCZOS_H_

#include "parallel_vertices.h"
#include "hamiltonian.h"
#include "control.h"

template<typename T>
class Lanczos
{
public:
    T eig_min = (T)0.0;
    T eig_max = (T)0.0;
    T lambda_cutoff = (T)0.0;
    uint max_iter = 100;
    double tolerance = 1e-2;
    Array_3D<T> V;
    #ifdef USE_OPENMP
        T* a = nullptr;
        T* b = nullptr;
        T* d = nullptr;
        T* e = nullptr;
        T* V_j = nullptr;
        T* V_jp1 = nullptr;
    #endif //USE_OPENMP
    Lanczos();
    ~Lanczos();
    void set_eig_min(const T& eig_min);
    void set_eig_max(const T& eig_max);
    void set_lambda_cutoff(const T& lambda_cutoff);
    void set_max_iter(const uint& max_iter);
    void set_tolerance(const double& tolerance);
    void cal_max_and_min(const Array_3D<T>& Vloc, const Effective_potential_nloc<T>& Vnloc,
                         const Stencil<T>& stencil, const T& dv, const Exarr_3D_mpi_package& exarr_mpi_package);
    void cal_max_and_min(T const* const& Vloc, const Vertices_3D& vertices, const Effective_potential_nloc<T>& Vnloc,
                         const Stencil<T>& stencil,  const T& dv, const Exarr_3D_mpi_package& exarr_mpi_package);
    void cal_max_and_min_without_nonlocal(T const* const& Vloc, const Vertices_3D& vertices, const Effective_potential_nloc<T>& Vnloc,
                         const Stencil<T>& stencil,  const T& dv, const Exarr_3D_mpi_package& exarr_mpi_package);
    void cal_max_and_min_without_nonlocal_mp(T const* const Vloc, const Vertices_3D& vertices, const Effective_potential_nloc<T>& Vnloc,
                         const Stencil<T>& stencil,  const T dv, const Exarr_3D_mpi_package& exarr_mpi_package,
                        Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap);
    void run(const MPI_Comm& comm, const Array_3D<T>& Vloc, const Effective_potential_nloc<T>& Vnloc,
             const Stencil<T>& stencil, const T& dv, const Exarr_3D_mpi_package& exarr_mpi_package,
             const bool& calculate_flag, const T& eig_max, const T& eig_min,
             const bool& print_flag = true);
    void run(const MPI_Comm& comm, T const* const Vloc, const Vertices_3D vertices_3d, const Effective_potential_nloc<T>& Vnloc,
             const Stencil<T>& stencil, const T& dv, const Exarr_3D_mpi_package& exarr_mpi_package,
             const bool& calculate_flag, const T& eig_max, const T& eig_min,
             const bool& print_flag = true);
    void run_mp(const MPI_Comm comm, T const* const Vloc, const Vertices_3D vertices_3d, const Effective_potential_nloc<T>& Vnloc,
             const Stencil<T>& stencil, const T dv, const Exarr_3D_mpi_package& exarr_mpi_package,
             const bool calculate_flag, const T eig_max, const T eig_min,
             const bool print_flag, Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap);
    void init(const Domain_parallel_vertices_3D& domain_vertices, const bool& is_rand_fixed);
    template<typename T2> void init(const Lanczos<T2>& lanczos);
    void destructor();
    void show() const;
};

#endif //_LANCZOS_H_
