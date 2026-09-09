#ifndef _POISSON_SOLVER_H_
#define _POISSON_SOLVER_H_

#include "control.h"
#include "arr.h"
#include "mixing.h"
#include "parallel_vertices.h"
#include "aar.h"
#ifdef USE_HYPRE
#include "wrapper_hypre.h"
#endif
#include "wrapper_sstructmg.h"

template<typename T>
class Poisson_solver
{
#ifdef USE_HYPRE
private:
    #define idx_t long long int
    static constexpr idx_t nvars = 1;
    static constexpr idx_t ndim = 3;
    static constexpr idx_t my_part = 0;
    static constexpr idx_t num_diag = 37;
    static constexpr idx_t nparts = 1;    
    static constexpr idx_t radius = 6;
    static constexpr idx_t obj_type = HYPRE_PARCSR;
    idx_t my_pid;
    idx_t hypre_ilower[ndim], hypre_iupper[ndim]; // {inner, ..., outer}
    idx_t glb_dims[ndim];  // {outer, mid, inner}
    HYPRE_SStructGrid ssgrid;
    HYPRE_SStructStencil stencils[nvars];
    HYPRE_SStructGraph ssgraph;
    HYPRE_SStructMatrix   A;
    HYPRE_SStructVector   b, x, y;
    HYPRE_ParCSRMatrix par_A;
    HYPRE_ParVector par_b, par_x, par_y;
    HYPRE_Solver par_solver, par_precond;
#endif
#ifdef USE_SSTRUCTMG
private:
#ifdef MG_FLOAT32
    SStructMG_wrapper<int, T, float, float> sstruct_mg;  // only used when poisson_solver_control.method == 2(CG) / 3(GMRES)
#else
    SStructMG_wrapper<int, T, T, T> sstruct_mg;  // only used when poisson_solver_control.method == 2(CG) / 3(GMRES)
#endif
#endif

public:
    const Poisson_solver_control& poisson_solver_control;
    const Stencil<T>& stencil;
    const Mesh_control& mesh_control;
    const Domain_parallel_vertices_3D& domain_vertices;
    const Exarr_3D_mpi_package& exarr_mpi_package;
    Array_3D<T> electrostatic_potential;
    Aar<T> aar;
    Poisson_solver(const Poisson_solver_control& poisson_solver_control,
                   const Stencil<T>& stencil,
                   const Mesh_control& mesh_control,
                   const Domain_parallel_vertices_3D& domain_vertices,
                   const Exarr_3D_mpi_package& exarr_mpi_package);
    ~Poisson_solver();
    Array_3D<T>& cal_electrostatic_potential(const std::vector<Array_3D<T>>& electron_densities, const Array_3D<T>& pseudo_charge_density);
    Array_3D<T>& cal_electrostatic_potential_mp(T const* const* const electron_densities, T const* const pseudo_charge_density,
                                                Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap);
    Array_3D<T>& cal_electrostatic_potential(const Array_3D<T>& electron_density, const Array_3D<T>& pseudo_charge_density);
    Array_3D<T>& cal_electrostatic_potential_mp(T const* const electron_density,
                                                T const* const pseudo_charge_density,
                                                Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap);
    void aar_backup(const Array_3D<T>& electron_density, const Array_3D<T>& pseudo_charge_density);
    void init();
    template<typename T2> void init(const Poisson_solver<T2>& poisson_solver);
    void destructor();
    void show() const;
};

namespace Poisson_solver_method {

template<typename T>
void MultipoleExpansion_phi(T const* const& f, T* const& d_cor,
                            const Stencil<T>& stencil,
                            const Domain_parallel_vertices_3D& domain_vertices,
                            const Mesh_control& mesh_control);
template<typename T>
void PartrialDipole_surface(T const* const& f, T* const& d_cor, const T& NetCharge,
                            const Stencil<T>& stencil,
                            const Domain_parallel_vertices_3D& domain_vertices,
                            const Mesh_control& mesh_control);
template<typename T>
void PartrialDipole_wire(T const* const& f, T* const& d_cor,
                            const Stencil<T>& stencil,
                            const Domain_parallel_vertices_3D& domain_vertices,
                            const Mesh_control& mesh_control);


}

#endif //_POISSON_SOLVER_H_
