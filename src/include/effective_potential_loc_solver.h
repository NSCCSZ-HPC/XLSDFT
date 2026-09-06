#ifndef _EFFECTIVE_POTENTIAL_LOC_SOLVER_H_
#define _EFFECTIVE_POTENTIAL_LOC_SOLVER_H_

#include "poisson_solver.h"
#include "exchange_correlation_solver.h"
#include "pseudo_charge_solver.h"

template<typename T>
class Effective_potential_loc_solver
{
public:
    const Poisson_solver_control& poisson_solver_control;
    const Exchange_correlation_solver_control& exchange_correlation_solver_control;
    const Geometry& geometry;
    const Spin& spin;
    const Stencil<T>& stencil;
    const Mesh_control& mesh_control;
    const std::vector<Psp8_file>& psp8_files;
    const Domain_parallel_vertices_3D& domain_vertices;
    const Exarr_3D_mpi_package& exarr_mpi_package;
    std::vector<Array_3D<T>> effective_potentail_locs;
    Pseudo_charge_solver<T> pseudo_charge_solver;
    Poisson_solver<T> poisson_solver;
    Exchange_correlation_solver<T> exchange_correlation_solver;
    Effective_potential_loc_solver(const Poisson_solver_control& poisson_solver_control,
                                   const Exchange_correlation_solver_control& exchange_correlation_solver_control,
                                   const Geometry& geometry,
                                   const Spin& spin,
                                   const Stencil<T>& stencil,
                                   const Mesh_control& mesh_control,
                                   const std::vector<Psp8_file>& psp8_files,
                                   const Domain_parallel_vertices_3D& domain_vertices,
                                   const Exarr_3D_mpi_package& exarr_mpi_package);
    ~Effective_potential_loc_solver();
    std::vector<Array_3D<T>>& cal_effective_potential_loc(const std::vector<Array_3D<T>>& electron_densities,
                                                          const Array_3D<T>& electron_density_core);
    std::vector<Array_3D<T>>& cal_effective_potential_loc_mp(const std::vector<Array_3D<T>>& electron_densities,
                                                            const Array_3D<T>& electron_density_core,
                                                            Memory_pool<T, Fast_memory>& pool_fast,
                                                            Memory_pool<T, Capacity_memory>& pool_cap);
    T& evaluate_exchange_correlation_energy(const std::vector<Array_3D<T>>& electron_densities,
                                            const Array_3D<T>& electron_density_core);
    void init();
    template<typename T2> void init(const Effective_potential_loc_solver<T2>& effective_potential_loc_solver);
    void destructor();
    void show() const;
};

#endif //_EFFECTIVE_POTENTIAL_LOC_SOLVER_H_