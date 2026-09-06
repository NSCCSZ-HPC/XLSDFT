#ifndef _AAR_H_
#define _AAR_H_
#include <iostream>
#include <mpi.h>
#include "linalg.h"
#include "arr.h"
#include "control.h"
#include "mixing.h"
#include "wrapper_sstructmg.h"

#ifndef M_1_PI
#define M_1_PI		0.31830988618379067154	/* 1/pi */
#endif

template<typename T>
class Aar
{
public:
    Aar_control aar_control;
    const Domain_parallel_vertices_3D& domain_vertice;
    //res
    Stencil<T> res_stencil;
    Exarr_3D_mpi_package res_exarr_mpi_package;
    //pre
    Stencil<T> pre_stencil;
    // DST preconditioner data (only used when precondition_method == 1)
    Stencil_method::DST_Preconditioner_Data<T> dst_pre_data;
    #ifdef USE_OPENMP
        T* r = nullptr;
        T* x_old = nullptr;
        T* f = nullptr;
        T* f_old = nullptr;
        T* X = nullptr;
        T* F = nullptr;
        T* ex_result = nullptr;
    #endif
    #ifdef USE_SSTRUCTMG
        #ifdef MG_FLOAT32
            SStructMG_wrapper<int, T, float, float> sstruct_mg;  // only used when aar_control.precondition_method == 2
        #else
            SStructMG_wrapper<int, T, T, T> sstruct_mg;  // only used when aar_control.precondition_method == 2
        #endif
        // TODO: Release working vectors of MG after solving.
    #endif
    Aar(const Domain_parallel_vertices_3D& domain_vertice);
    ~Aar();
    // void run(Array_3D<T>& result, const Array_3D<T>& rhs, const MPI_Comm& comm = MPI_COMM_NULL);
    void run(T* const result, T const* const rhs, const Vertices_3D& local_vertices, const MPI_Comm comm = MPI_COMM_NULL);
    void run_ompunnested(T* const result, T const* const rhs, const Vertices_3D& local_vertices, const MPI_Comm comm = MPI_COMM_NULL);
    void run_ompunnested_mp(T* const result, T const* const rhs,
                        const Vertices_3D& local_vertices, const MPI_Comm comm,
                        Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap);
    void print_runtime_result(const uint iter, const T norm2);
    void init();
    template<typename T2> void init(const Aar<T2>& aar);
    void init_res_method0(const Stencil<T>& stencil, const Exarr_3D_mpi_package& exarr_mpi_package,
                          const double& ratio = 1.0, const double& shfit = 0.0);
    void init_pre_method0(const Stencil<T>& stencil, const double& ratio = 1.0, const double& shfit = 0.0);
    void init_pre_mg(const Mesh_control& mesh_control);
    void init_pre_dst();
    void des_pre_dst();
    void destructor();
    void show() const;
};


#endif //_AAR_H_