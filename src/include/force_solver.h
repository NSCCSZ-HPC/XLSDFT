#ifndef _FORCE_SOLVER_H_
#define _FORCE_SOLVER_H_

#include "parallel_vertices.h"
#include "filesys.h"
#include "nloc_projector.h"
#include "atom.h"
#include "control.h"
#include "density_solver.h"


template<typename T>
class Force_solver
{
public:
    const Domain_parallel_vertices_3D& domain_vertices;
    const Exarr_3D_mpi_package& exarr_mpi_package;
    const Mesh_control& mesh_control;
    const Geometry& geometry;
    const Stencil<T>& stencil;
    const Density_solver<T>& density_solver;
    const std::vector<Psp8_file>& psp8_files;
    Array_2D<T> forces;
    Force_solver(const Domain_parallel_vertices_3D& domain_vertices,
                 const Exarr_3D_mpi_package& exarr_mpi_package,
                 const Mesh_control& mesh_control,
                 const Geometry& geometry,
                 const Stencil<T>& stencil,
                 const Density_solver<T>& density_solver,
                 const std::vector<Psp8_file>& psp8_files);
    ~Force_solver();
    void cal_local_forces(const Array_3D<T>& electrostatic_potential,
                          const Array_3D<T>& pseudo_charge_density,
                          const Array_3D<T>& pseudo_charge_density_ref,
                          const Array_3D<T>& pseudo_charge_density_potiential_correction);
    void cal_xc_forces(const std::vector<Array_3D<T>>& xc_potentials);
    void cal_xc_forces(const Array_3D<T>& xc_potential);
    void cal_nonlocal_forces();
    void balance_forces();
    void init();
    void run();
    void show() const;
};

#endif //_FORCE_SOLVER_H_
