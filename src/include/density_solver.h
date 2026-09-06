#ifndef _DENSITY_SOLVER_H_
#define _DENSITY_SOLVER_H_
#include <iostream>
#include <vector>
#include "parallel_vertices.h"
#include "control.h"
#include "atom.h"
#include "eigen_solver.h"
#include "density_matrix_solver.h"
#include "spin.h"

#if !defined(ENABLE_DENSITY_SOLVER_TIMER) && defined(ENABLE_TIMER)
#define ENABLE_DENSITY_SOLVER_TIMER
#endif

#ifdef ENABLE_DENSITY_SOLVER_TIMER
#include "timer.h"
class Density_solver_timer
{
public:
    Timer density_solver;
    Timer density_solver_kernel;
    Timer chemical_potential;
    Timer update_density;
    Density_solver_timer();
    ~Density_solver_timer();
    void reset();
    void show(std::ostream& output = std::cout) const;
};
#endif //ENABLE_DENSITY_SOLVER_TIMER

template<typename T>
class Density_solver
{
public:
    const Density_solver_control& density_solver_control;
    const Mesh_control& mesh_control;
    const Geometry& geometry;
    const Spin& spin;
    const Stencil<T>& stencil;
    const std::vector<Psp8_file>& psp8_files;
    const Domain_parallel_vertices_3D& domain_vertices;
    #ifdef ENABLE_DENSITY_SOLVER_TIMER
    Density_solver_timer density_solver_timer;
    #endif //ENABLE_DENSITY_SOLVER_TIMER
    T chemical_potential = (T)0.0;
    T electron_charge;
    T band_energy;
    T entropy_energy;
    bool NLCC_flag = false;
    Array_3D<T> electron_density_init;
    Array_3D<T> magnetization_z;
    Array_3D<T> electron_density_core;
    std::vector<Array_3D<T>> electron_densities;                                               // 3D local_vertices electron density(\rho(r))
    std::vector<Array_3D<T>> electron_densities_in;                                               // 3D local_vertices electron density(\rho(r))
    Smearing smearing;
    Domain_parallel_vertices_4D domain_vertices_with_band;
    Domain_3D_to_4D_mpi_package domain_band_mpi_package;
    Eigen_solver<T> eigen_solver;
    Density_matrix_solver<T> density_matrix_solver;
    Density_solver(const Density_solver_control& density_solver_control,
                   const Mesh_control& mesh_control,
                   const Geometry& geometry,
                   const Spin& spin,
                   const Stencil<T>& stencil,
                   const std::vector<Psp8_file>& psp8_files,
                   const Domain_parallel_vertices_3D& domain_vertices);
    ~Density_solver();
    void generate_initial_electron_density();
    void cal_magnetization_z();
    T& generate_electron_charge();
    void solve_eigen_problem(const Array_3D<T>& effective_potentail_loc);
    void solve_density_matrix(const Array_3D<T>& effective_potentail_loc);
    void solve_density_matrix_mp(T const* const effective_potentail_loc, const Vertices_3D& effective_potentail_loc_vertices);
    void solve_density_matrix_mp(T const* const effective_potentail_loc, const Vertices_3D& effective_potentail_loc_vertices,
                                Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap);
    T& evaluate_chemical_potential();
    T& evaluate_chemical_potential_mp(Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap);
    T& evaluate_band_energy();
    T& evaluate_entropy_energy();
    void update_electron_density();
    void update_electron_density_mp(Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap);
    void cal_nonlocal_forces(Array_2D<T>& nonlocal_forces) const;
    void print_eigens(const std::string& fname = "EIGENS") const;
    void print_pdos(const std::string& fname = "PDOS") const;
    double evalutate_flops();
    void run(const std::vector<Array_3D<T>>& effective_potentail_locs);
    void run_mp(T const* const* const effective_potentail_locs);
    void run_mp(T const* const* const effective_potentail_locs,
                Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap);
    #if defined(ENABLE_TIMER)
    void print_timer_statistics(const bool if_print, std::ostream& output) const;
    #endif
    void init();
    template<typename T2> void init(const Density_solver<T2>& density_solver);
    void destructor();
    void show() const;
    void density_dump(const std::string& dir_name = "DENSITY_OUT") const;
    void density_load(const std::string& dir_name = "DENSITY_IN");
};



#endif //_DENSITY_SOLVER_H_