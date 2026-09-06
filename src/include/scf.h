#ifndef _SCF_H_
#define _SCF_H_

#include "geometry.h"
#include "control.h"
#include "linalg.h"
#include "effective_potential_loc_solver.h"
#include "density_solver.h"
#include "preparation.h"
#include "spin.h"
#include "force_solver.h"

class Scf
{
public:
    const MPI_Comm& comm;
    const Scf_control& scf_control;
    const Misc_control& misc_control;
    const Mesh_control& mesh_control;
    const std::vector<Psp8_file>& psp8_files;
    const Density_solver_control& density_solver_control;
    const Spin_control& spin_control;
    const Stencil_control& stencil_control;
    const Geometry& geometry;
    Domain_parallel_vertices_3D global_domain_vertices;
    Domain_3D_to_3D_mpi_package global_domain_mpi_package;
    Domain_parallel_vertices_3D domain_vertices;
    Exarr_3D_mpi_package exarr_mpi_package;
    Spin spin;
    Stencil<float> sstencil;
    Stencil<double> dstencil;
    Effective_potential_loc_solver<float> seffective_potential_loc_solver;
    Effective_potential_loc_solver<double> deffective_potential_loc_solver;
    Density_solver<float> sdensity_solver;
    Density_solver<double> ddensity_solver;
    Mixing<float> smixing;
    Mixing<double> dmixing;
    double error;
    bool double_precision_flag = true;
    double ion_elecst_energy;
    double electron_elecst_energy;
    double electron_exchange_correlation_energy;
    double free_energy;
    Scf(const Preparation& preparation, const MPI_Comm& comm = MPI_COMM_WORLD);
    Scf(const Control& control, const Geometry& geometry,
        const std::vector<Psp8_file>& psp8_files, const MPI_Comm& comm = MPI_COMM_WORLD);
    ~Scf();
    void scale_electron_density();
    void electron_density_pretreatment();
    void set_electron_density(const Array_3D<double>& electron_density);
    double& evaluate_error(const uint& iter);
    double& evaluate_ion_elecst_energy();
    double& evaluate_electron_elecst_energy();
    double& evaluate_electron_exchange_correlation_energy();
    double evaluate_exchange_correlation_energy();
    double& evaluate_free_energy();
    void density_mixing(const uint iter);
    void density_mixing_mp(const uint iter,
                            Memory_pool<double, Fast_memory>& pool_fast,
                            Memory_pool<double, Capacity_memory>& pool_cap);
    void potential_mixing(const uint& iter);
    void precision_conversion();
    void cal_forces();
    void init();
    void run();
    void run_mp();
    void run_mp(Memory_pool<double, Fast_memory>& pool_fast, Memory_pool<double, Capacity_memory>& pool_cap);
    void destructor();
    void show() const;
};

#endif //_SCF_H_
