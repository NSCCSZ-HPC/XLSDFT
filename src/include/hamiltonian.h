#ifndef _HAMILTONIAN_H_
#define _HAMILTONIAN_H_

#include "parallel_vertices.h"
#include "effective_potential_nloc.h"

namespace Hamiltonian {
    
template<typename T> void hamiltonian_product_vectors(Array_3D<T>& result,
                                                      const Array_3D<T>& eigen_vector,
                                                      const Array_3D<T>& Vloc,
                                                      const Effective_potential_nloc<T>& Vnloc,
                                                      const Stencil<T>& stencil, 
                                                      const T& dv,
                                                      const Exarr_3D_mpi_package& exarr_mpi_package);

template<typename T> void hamiltonian_product_vectors(T* const& result,
                                                      const Vertices_3D& vertices,
                                                      T const* const& eigen_vector,
                                                      T const* const& Vloc,
                                                      const Effective_potential_nloc<T>& Vnloc,
                                                      const Stencil<T>& stencil, 
                                                      const T& dv,
                                                      const Exarr_3D_mpi_package& exarr_mpi_package);
    
template<typename T> void hamiltonian_product_vectors(Array_4D<T>& result,
                                                      const Array_4D<T>& eigen_vectors,
                                                      const Array_3D<T>& Vloc,
                                                      const Effective_potential_nloc<T>& Vnloc,
                                                      const Stencil<T>& stencil,
                                                      const T& dv,
                                                      const Exarr_4D_mpi_package& exarr_mpi_package);

template<typename T> void hamiltonian_product_vectors(T* const& result,
                                                      const Vertices_4D& vertices,
                                                      T const* const& eigen_vectors,
                                                      T const* const& Vloc,
                                                      const Effective_potential_nloc<T>& Vnloc,
                                                      const Stencil<T>& stencil, 
                                                      const T& dv,
                                                      const Exarr_4D_mpi_package& exarr_mpi_package);

template<typename T> void hamiltonian_product_vectors_column_wise(T* const& result,
                                                                  const Vertices_4D& vertices,
                                                                  T const* const& eigen_vectors,
                                                                  T const* const& Vloc,
                                                                  const Effective_potential_nloc<T>& Vnloc,
                                                                  const Stencil<T>& stencil, 
                                                                  const T& dv,
                                                                  const Exarr_3D_mpi_package& exarr_mpi_package);

template<typename T> void hamiltonian_product_vectors_column_wise2(T* const& result,
                                                                   const Vertices_4D& vertices,
                                                                   T const* const& eigen_vectors,
                                                                   T const* const& Vloc,
                                                                   const Effective_potential_nloc<T>& Vnloc,
                                                                   const Stencil<T>& stencil,
                                                                   const T& dv,
                                                                   const Exarr_3D_mpi_package& exarr_mpi_package,
                                                                   const bool& print_flag = true);

template<typename T> void hamiltonian_product_vectors_column_wise2_specialization(T* const& result,
                                                                   const Vertices_4D& vertices,
                                                                   T const* const& eigen_vectors,
                                                                   T const* const& Vloc,
                                                                   const Effective_potential_nloc<T>& Vnloc,
                                                                   const Stencil<T>& stencil,
                                                                   const T& dv,
                                                                   const Exarr_3D_mpi_package& exarr_mpi_package,
                                                                   const bool& print_flag = true);

#ifdef USE_OPENMP
template<typename T> inline void hamiltonian_product_vectors_column_wise2_omp(T* const& result,
                                                                   const Vertices_4D& vertices,
                                                                   T const* const& eigen_vectors,
                                                                   T const* const& Vloc,
                                                                   const Effective_potential_nloc<T>& Vnloc,
                                                                   const Stencil<T>& stencil,
                                                                   const T& dv,
                                                                   const Exarr_3D_mpi_package& exarr_mpi_package,
                                                                   const bool& print_flag = true);
template<typename T> inline void hamiltonian_product_vectors_column_wise2_omp_comm_self(T* const& result,
                                                                   const Vertices_4D& vertices,
                                                                   T const* const& eigen_vectors,
                                                                   T const* const& Vloc,
                                                                   const Effective_potential_nloc<T>& Vnloc,
                                                                   const Stencil<T>& stencil,
                                                                   const T& dv,
                                                                   const Exarr_3D_mpi_package& exarr_mpi_package,
                                                                   const bool& print_flag = true);
#endif //USE_OPENMP
template<typename T> void nloc_project_vectors(Array_3D<T>& result, const Array_3D<T>& eigen_vector,
                                               const Effective_potential_nloc<T>& effective_potential_nloc,
                                               const T& dv, const MPI_Comm& comm);

template<typename T> void nloc_project_vectors(T* const& result, T const* const& eigen_vector,
                                               const Effective_potential_nloc<T>& effective_potential_nloc,
                                               const T& dv, const MPI_Comm& comm);

template<typename T> void nloc_project_vectors(Array_4D<T>& result, const Array_4D<T>& eigen_vectors,
                                               const Effective_potential_nloc<T>& effective_potential_nloc,
                                               const T& dv, const MPI_Comm& comm);

template<typename T> void nloc_project_vectors(T* const& result, const Vertices_4D& vertices, T const* const& eigen_vectors,
                                               const Effective_potential_nloc<T>& effective_potential_nloc,
                                               const T& dv, const MPI_Comm& comm);

#ifdef USE_OPENMP
template<typename T> inline void nloc_project_vectors_omp_domain(T* const& result, const Vertices_4D& vertices, T const* const& eigen_vectors,
                                                            const Effective_potential_nloc<T>& effective_potential_nloc,
                                                            const T& dv, const MPI_Comm& comm);
template<typename T> inline void nloc_project_vectors_omp_for(T* const& result, const Vertices_4D& vertices, T const* const& eigen_vectors,
                                                                const Effective_potential_nloc<T>& effective_potential_nloc,
                                                                const T& dv, const MPI_Comm& comm);
template<typename T> inline void nloc_project_vectors_omp_for_comm_self_with_chunk(T* const& result, const Vertices_4D& vertices, T const* const& eigen_vectors,
                                                                const Effective_potential_nloc<T>& effective_potential_nloc,
                                                                const T& dv, const MPI_Comm& comm);
template<typename T> inline void nloc_project_vectors_omp_task(T* const& result, const Vertices_4D& vertices, T const* const& eigen_vectors,
                                                                const Effective_potential_nloc<T>& effective_potential_nloc,
                                                                const T& dv, const MPI_Comm& comm);
#endif //USE_OPENMP

template<typename T> void nloc_project_vectors_omp_for_comm_self_with_chunk_mp(T* const result, const Vertices_4D& vertices, T const* const eigen_vectors,
                                                                const Effective_potential_nloc<T>& effective_potential_nloc,
                                                                const T dv, const MPI_Comm comm,
                                                                Memory_pool<T, Fast_memory>& pool_fast,
                                                                Memory_pool<T, Capacity_memory>& pool_cap);

template<typename T> void hamiltonian_product_vectors_column_wise2_specialization_mp(T* const result,
                                                                   const Vertices_4D& vertices,
                                                                   T const* const eigen_vectors,
                                                                   T const* const Vloc,
                                                                   const Effective_potential_nloc<T>& Vnloc,
                                                                   const Stencil<T>& stencil,
                                                                   const T dv,
                                                                   const Exarr_3D_mpi_package& exarr_mpi_package,
                                                                   const bool print_flag,
                                                                    Memory_pool<T, Fast_memory>& pool_fast,
                                                                    Memory_pool<T, Capacity_memory>& pool_cap);

template<typename T> void hamiltonian_product_vectors_column_wise2_compute_fusion_mp(T* const result,
                                                                   const Vertices_4D& vertices,
                                                                   T const* const eigen_vectors,
                                                                   T const* const Vloc,
                                                                   const Effective_potential_nloc<T>& Vnloc,
                                                                   const Stencil<T>& stencil,
                                                                   const T shift,
                                                                   const T dv,
                                                                   const Exarr_3D_mpi_package& exarr_mpi_package,
                                                                   const bool print_flag,
                                                                    Memory_pool<T, Fast_memory>& pool_fast,
                                                                    Memory_pool<T, Capacity_memory>& pool_cap);

}

#endif //_HAMILTONIAN_H_
