#ifndef _DENSITY_MATRIX_SOLVER_H_
#define _DENSITY_MATRIX_SOLVER_H_

#include "parallel_vertices.h"
#include "control.h"
#include "xlsdft.h"

#if !defined(ENABLE_DENSITY_MATRIX_TIMER) && defined(ENABLE_TIMER)
#define ENABLE_DENSITY_MATRIX_TIMER
#endif

#ifdef ENABLE_DENSITY_MATRIX_TIMER
#include "timer.h"
class Density_matrix_solver_timer
{
public:
    Timer density_matrix_solver;
    Timer density_matrix_solver_distribute_veff;
    Timer density_matrix_solver_kernel;
    Density_matrix_solver_timer();
    ~Density_matrix_solver_timer();
    void reset();
    void show(std::ostream& output = std::cout) const;
};
#endif //ENABLE_DENSITY_MATRIX_TIMER

template<typename T>
class Density_matrix_solver
{
public:
    const Density_matrix_solver_control& density_matrix_solver_control;
    const Mesh_control& mesh_control;
    const Geometry& geometry;
    const Stencil<T>& stencil;
    const Domain_parallel_vertices_4D& domain_vertices;
    #ifdef ENABLE_DENSITY_MATRIX_TIMER
    Density_matrix_solver_timer density_matrix_solver_timer;
    #endif //ENABLE_DENSITY_MATRIX_TIMER
    // Array_3D<T> dia_density_matrix; //diagonal part of density matrix
    Exarr_3D_mpi_package exarr_mpi_package;
    Vertices_3D ex_vertices;
    Xlsdft<T> xlsdft;
    Density_matrix_solver(const Density_matrix_solver_control& density_matrix_solver_control,
                          const Mesh_control& mesh_control,
                          const Geometry& geometry,
                          const Stencil<T>& stencil,
                          const Domain_parallel_vertices_4D& domain_vertices);
    ~Density_matrix_solver();
    T evaluate_chemical_potential(const Smearing& smearing, const T electron_charge,
                                  const T smearing_coef, const T trial_chemical_potential) const;
    T evaluate_chemical_potential_mp(const Smearing& smearing, const T electron_charge,
                                    const T smearing_coef, const T trial_chemical_potential,
                                    Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap) const;
    T evaluate_band_energy(const Smearing& smearing, const T& chemical_potential, const T& smearing_coef) const;
    T evaluate_entropy_energy(const Smearing& smearing, const T& chemical_potential, const T& smearing_coef) const;
    void update_electron_density(Array_3D<T>& electron_density, const Smearing& smearing,
                                 const T chemical_potential, const T smearing_coef, const T dV,
                                 const Domain_3D_to_4D_mpi_package& domain_band_mpi_package);
    void update_electron_density_mp(Array_3D<T>& electron_density, const Smearing& smearing,
                                 const T chemical_potential, const T smearing_coef, const T dV,
                                 const Domain_3D_to_4D_mpi_package& domain_band_mpi_package,
                                Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap);
    void cal_nonlocal_forces(Array_2D<T>& nonlocal_forces, const T& chemical_potential, const Smearing& smearing,
                             const Spin& spin, const std::vector<Psp8_file>& psp8_files) const;
    void print_eigens(const T& chemical_potential, const Smearing& smearing, const Spin& spin,
                      const std::string& fname = "EIGENS") const;
    void print_pdos(const T& chemical_potential, const Spin& spin, const std::string& dir_name = "PDOS") const;
    double evalutate_flops();
    void run(const Array_3D<T>& effective_potentail_loc);
    void run_mp(T const* const effective_potentail_loc);
    void run_mp(T const* const effective_potentail_loc,
                Memory_pool<T, Fast_memory>& pool_fast,
                Memory_pool<T, Capacity_memory>& pool_cap);
    #if defined(ENABLE_TIMER)
    void print_timer_statistics(const bool if_print, std::ostream& output) const;
    #endif
    void init(const std::vector<Psp8_file>& psp8_files);
    template<typename T2> void init(const Density_matrix_solver<T2>& other);
    void destructor();
    void show() const;
};


#endif //_DENSITY_MATRIX_SOLVER_H_
