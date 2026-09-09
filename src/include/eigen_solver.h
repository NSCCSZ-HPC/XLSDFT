#ifndef _EIGEN_SOLVER_H_
#define _EIGEN_SOLVER_H_

#include "parallel_vertices.h"
#include "control.h"
#include "chefsi.h"
#include "xlsdft_nchi_diag.hpp"

#if !defined(ENABLE_EIGEN_SOLVER_TIMER) && defined(ENABLE_TIMER)
#define ENABLE_EIGEN_SOLVER_TIMER
#endif

#ifdef ENABLE_EIGEN_SOLVER_TIMER
#include "timer.h"
class Eigen_solver_timer
{
public:
    Timer eigen_solver;
    #if (defined(LOW_MEMORY))
    Timer low_memory_chi;
    #endif
    Timer eigen_solver_kernel;
    Eigen_solver_timer();
    ~Eigen_solver_timer();
    void reset();
    void show(std::ostream& output = std::cout) const;
    Eigen_solver_timer& operator+=(const Eigen_solver_timer& other) {
        eigen_solver += other.eigen_solver;
#if defined(LOW_MEMORY)
        low_memory_chi += other.low_memory_chi;
#endif
        eigen_solver_kernel += other.eigen_solver_kernel;
        return *this;
    }
    Eigen_solver_timer operator+(const Eigen_solver_timer& other) const {
        Eigen_solver_timer result = *this;
        result += other;
        return result;
    }
};
#endif //ENABLE_EIGEN_SOLVER_TIMER

template<typename T>
class Eigen_solver
{
public:
    const Eigen_solver_control& eigen_solver_control;
    const Mesh_control& mesh_control;
    const Geometry& geometry;
    const Stencil<T>& stencil;
    const std::vector<Psp8_file>& psp8_files;
    const Domain_parallel_vertices_4D& domain_vertices;
    #ifdef ENABLE_EIGEN_SOLVER_TIMER
    Eigen_solver_timer eigen_solver_timer;
    #endif //ENABLE_EIGEN_SOLVER_TIMER
    Array_0D<T> eigen_values;   // global length = nstates
    Array_4D<T> eigen_vectors;
    Effective_potential_nloc<T> effective_potential_nloc;
    Exarr_4D_mpi_package exarr_mpi_package;
    Domain_parallel_vertices_4D intra_band_domain_vertices;
    Exarr_4D_mpi_package intra_band_exarr_mpi_package;
    Chefsi<T> chefsi;
    Xlsdft_element_workload_diag element_workload_diag;
    Eigen_solver(const Eigen_solver_control& eigen_solver_control,
                 const Mesh_control& mesh_control,
                 const Geometry& geometry,
                 const Stencil<T>& stencil,
                 const std::vector<Psp8_file>& psp8_files,
                 const Domain_parallel_vertices_4D& domain_vertices);
    ~Eigen_solver();
    void cal_int_density_per_band(const Vertices_3D& region, std::vector<T>& eigen_value_per_band,
                                  std::vector<T>& int_density_per_band) const;
    Array_3D<T> cal_electron_charge_density(const Smearing& smearing, const T& smearing_coef,
                                  const T& chemical_potential, const Vertices_3D& region) const;
    T cal_electron_charge(const Smearing& smearing, const T& smearing_coef,
                          const T& chemical_potential, const Vertices_3D& region) const;
    T evaluate_chemical_potential(const Smearing& smearing, const T& electron_charge, const T& smearing_coef) const;
    T evaluate_band_energy(const Smearing& smearing, const T& chemical_potential, const T& smearing_coef) const;
    T evaluate_band_energy(const Smearing& smearing, const T& chemical_potential, const T& smearing_coef, const Vertices_3D& region) const;
    T evaluate_entropy_energy(const Smearing& smearing, const T& chemical_potential, const T& smearing_coef) const;
    T evaluate_entropy_energy(const Smearing& smearing, const T& chemical_potential, const T& smearing_coef, const Vertices_3D& region) const;
    void update_electron_density(Array_3D<T>& electron_density, const Smearing& smearing,
                                 const T& chemical_potential, const T& smearing_coef, const T& dV,
                                 const Domain_3D_to_4D_mpi_package& domain_band_mpi_package);
    void cal_nonlocal_forces(Array_2D<T>& nonlocal_forces, const T& chemical_potential, const Smearing& smearing, const Spin& spin) const;
    void cal_nonlocal_forces(Array_2D<T>& nonlocal_forces, const T& chemical_potential, const Smearing& smearing,
                             const Spin& spin, const Effective_potential_nloc<T>& effective_potential_nloc) const;
    #if (defined(LOW_MEMORY))
    template<typename T2>
    void cal_nonlocal_forces(Array_2D<T>& nonlocal_forces, const T& chemical_potential, const Smearing& smearing,
                             const Spin& spin, const Effective_potential_nloc<T>& effective_potential_nloc,
                             const Array_4D<T2> eigen_vectors_other_type) const;
    #endif
    void print_eigens(const T& chemical_potential, const Smearing& smearing, const Spin& spin,
                      const std::string& fname = "EIGENS") const;
    void print_eigens(const T& chemical_potential, const Smearing& smearing, const Spin& spin,
                      std::ostream &file) const;
    void print_eigens(const T& chemical_potential, const Smearing& smearing, const Spin& spin,
                      const Vertices_3D& region, const std::string& fname) const;
    void print_eigens(const T& chemical_potential, const Smearing& smearing, const Spin& spin,
                      T const* const fracs, std::ostream &file) const;
    void print_pdos(const T& chemical_potential, const Spin& spin, const std::string& fname = "PDOS") const;
    void print_psi_sparc(const Spin& spin, const std::string& fname = "PSI") const;
    std::vector<Array_2D<T>> cal_pdos() const;
    std::vector<Array_2D<T>> cal_pdos(const std::vector<Atom>& atoms, const Array_4D<T>& eigen_vectors) const;
    void get_region_fracs(const Vertices_3D& region, T* const fracs) const;
    void get_eigens(T* const eigens) const;
    void run(const Array_3D<T>& effective_potentail_loc, const bool print_flag = true);
    void run_mp(T const* const effective_potentail_loc, const bool print_flag);
    void run_mp(T*& eigen_vectors_in, T*& eigen_vectors_out, T const* const effective_potentail_loc,
            const bool print_flag, Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap);
    void run_mp_opt(T const* const effective_potentail_loc, const bool print_flag);
    void run_mp_opt(T*& eigen_vectors_in, T*& eigen_vectors_out, T const* const effective_potentail_loc,
            const bool print_flag, Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap);
    void print_runtime_result();
    double evalutate_flops();
    void init();
    template<typename T2> void init(const Eigen_solver<T2>& eigen_solver);
    void destructor();
    void show() const;
};

namespace Eigen_solver_method {
    template<typename T>
    T evaluate_chemical_potential(T const * const eigen_values, const uint length, const T lower_bound, const T upper_bound,
                                const Smearing& smearing, const T electron_charge, const T smearing_coef, const uint max_iter, const T tol);
}

#endif //_EIGEN_SOLVER_H_
