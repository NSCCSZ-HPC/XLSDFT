#ifndef _CONTROL_H_
#define _CONTROL_H_

#include <iostream>
#include "filesys.h"
#include "geometry.h"

// Boltzmann constant in Ha/K
#define CONST_KB 3.1668115634556e-6

class Mesh_control
{
public:
    double delta_x = 0.0;         // mesh size in x-direction
    double delta_y = 0.0;         // mesh size in y-direction
    double delta_z = 0.0;         // mesh size in z-direction
    double delta_V = 0.0;         // volumn of one mesh cubic
    uint nx = 0;                // number of nodes in x direction
    uint ny = 0;                // number of nodes in y direction
    uint nz = 0;                // number of nodes in z direction 
    uint nd = 0;                // total number of grid nodes
    bool is_periodic[3] = {true, true, true};
    Mesh_control();
    Mesh_control(const Mesh_control& mesh_control);
    ~Mesh_control();
    Mesh_control& operator=(const Mesh_control& other);
    void set_deltas(const double& delta_x, const double& delta_y, const double& delta_z);
    void set_delta_x(const double& delta_x);
    void set_delta_y(const double& delta_y);
    void set_delta_z(const double& delta_z);
    void set_ns(const uint& nx, const uint& ny, const uint& nz);
    void set_nx(const uint& nx);
    void set_ny(const uint& ny);
    void set_nz(const uint& nz);
    void set_is_PBCs(const bool& is_x_PBC, const bool& is_y_PBC, const bool& is_z_PBC);
    void set_is_PBCs(const bool& is_PBC);
    void set_is_x_PBC(const bool& is_x_PBC);
    void set_is_y_PBC(const bool& is_y_PBC);
    void set_is_z_PBC(const bool& is_z_PBC);
    void init(const Input_file& input_file, const Geometry& geometry);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
    void print(std::ostream& output = std::cout) const;
};

class Kerker_control
{
public:
    double kerker_ktf = 1.0;
    double kerker_thresh = 0.1;
    double tolerance = -1.0;
    Kerker_control();
    ~Kerker_control();
    void set_tolerance(const double& tolerance);
    void set_kerker_ktf(const double& kerker_ktf);
    void set_kerker_thresh(const double& kerker_thresh);
    void init(const Input_file& input_file);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
    void print(std::ostream& output = std::cout) const;
};

class Pulay_control
{
public:
    uint pulay_frequency = 1;
    bool pulay_restart = false;
    double beta = 0.3;
    Pulay_control();
    ~Pulay_control();
    void set_pulay_frequency(const uint& pulay_frequency);
    void set_pulay_restart(const bool& pulay_restart);
    void set_beta(const double& beta);
    void init(const Input_file& input_file);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
    void print(std::ostream& output = std::cout) const;
};

class Mixing_control
{
public:
    uint method = 1;
    uint mixing_history = 7;
    uint precondition_method = 0;
    uint simple_precondition_method = 0;
    double alpha = 0.3;    // MixingParameter
    double alpha_mag = 4.0;    // MAG_MixingParameter
    Pulay_control pulay_control;
    Kerker_control kerker_control;
    Mixing_control();
    ~Mixing_control();
    void set_method(const uint method);
    void set_mixing_history(const uint mixing_history);
    void set_precondition_method(const uint precondition_method);
    void set_simple_precondition_method(const uint simple_precondition_method);
    void set_alpha(const double alpha);
    void set_alpha_mag(const double alpha);
    void init(const Input_file& input_file);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
    void print(std::ostream& output = std::cout) const;
};

class Scf_control
{
public:
    uint min_iter = 1;
    uint max_iter = 100;
    uint max_single_precision_iter = 0;
    uint mixing_variable = 0;
    double tolerance = 1e-5;
    double single_precision_tolerance = 1e-2;
    Mixing_control mixing_control;
    Scf_control();
    ~Scf_control();
    void set_min_iter(const uint& min_iter);
    void set_max_iter(const uint& max_iter);
    void set_max_single_precision_iter(const uint& max_single_precision_iter);
    void set_mixing_variable(const uint& mixing_variable);
    void set_tolerance(const double& tolerance);
    void set_single_precision_tolerance(const double& single_precision_tolerance);
    double get_tolerance() const;
    void init(const Input_file& input_file);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
    void print(std::ostream& output = std::cout) const;
};

class K_sample_control
{
public:
    uint method = 0;
    int nkpts[3] = {1, 1, 1};
    double k_shifts[3] = {0.0, 0.0, 0.0};
    K_sample_control();
    ~K_sample_control();
    void set_method(const uint& method);
    void set_nkts(const int& nkpt0, const int& nkpt1, const int& nkpt2);
    void set_k_shifts(const double& k_shift0, const double& k_shift1, const double& k_shift2);
    void init(const Input_file& input_file);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
    void print(std::ostream& output = std::cout) const;
};

class Spin_control
{
public:
    uint spin_type = 0;
    Spin_control();
    ~Spin_control();
    void set_spin_type(const uint& spin_type);
    void init(const Input_file& input_file);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
    void print(std::ostream& output = std::cout) const;
};

class Exchange_correlation_solver_control
{
public:
    uint exchange_method = 0;
    uint correlation_method = 0;
    double xc_rhotol = 1e-10;
    Exchange_correlation_solver_control();
    ~Exchange_correlation_solver_control();
    void set_exchange_method(const uint& exchange_method);
    void set_correlation_method(const uint& correlation_method);
    void set_xc_rhotol(const double& xc_rhotol);
    void init(const Input_file& input_file);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
    void print(std::ostream& output = std::cout) const;
};

class Poisson_solver_control
{
public:
    bool is_rand_fixed = 0;
    uint method = 0;
    uint precondition_method = 0;
    uint max_iter = 3000;
    double tolerance = 1e-7;
    Poisson_solver_control();
    ~Poisson_solver_control();
    void set_is_rand_fixed(const bool& is_rand_fixed);
    void set_method(const uint& method);
    void set_precondition_method(const uint& precondition_method);
    void set_max_iter(const uint& max_iter);
    void set_tolerance(const double& tolerance);
    void init(const Input_file& input_file, const Scf_control& scf_control);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
    void print(std::ostream& output = std::cout) const;
};

class Chefsi_control
{
public:
    uint max_iter = 1;
    uint chebyshev_filter_degree = 25;
    uint rho_trigger = 4;
    uint projection_method = 0;
    double tolerance_lanczos = 1e-7;
    double abstol = -1.0;
    double orfac = 1e-3;
    Chefsi_control();
    Chefsi_control(const Chefsi_control& other);
    ~Chefsi_control();
    Chefsi_control& operator=(const Chefsi_control& other);
    void set_max_iter(const uint& max_iter);
    void set_chebyshev_filter_degree(const uint& chebyshev_filter_degree);
    void set_rho_trigger(const uint& rho_trigger);
    void set_projection_method(const uint& projection_method);
    void set_tolerance_lanczos(const double& tolerance_lanczos);
    void init(const Input_file& input_file);
    void init(const Chefsi_control& chefsi_control);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
    void print(std::ostream& output = std::cout) const;
};

class Eigen_solver_control
{
public:
    uint method = 0;
    uint nstates = 100;
    bool is_rand_fixed = false;
    Chefsi_control chefsi_control;
    Eigen_solver_control();
    Eigen_solver_control(const Eigen_solver_control& other);
    ~Eigen_solver_control();
    Eigen_solver_control& operator=(const Eigen_solver_control& other);
    void set_method(const uint& method);
    void set_nstates(const uint& nstates);
    void set_is_rand_fixed(const bool& is_rand_fixed);
    void init(const Input_file& input_file);
    void init(const Eigen_solver_control& eigen_solver_control);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
    void print(std::ostream& output = std::cout) const;
};

class Spinor_eigen_solver_control
{
public:
    uint method = 0;
    double delta = 0.005;
    double gamma = 0.0069088671;
    Spinor_eigen_solver_control();
    ~Spinor_eigen_solver_control();
    void set_method(const uint& method);
    void set_delta(const double& delta);
    void init(const Input_file& input_file);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
};

class Xlsdft_control
{
public:
    bool is_periodic[3] = {false, false, false};
    bool is_rand_fixed = false;
    uint element_numbers[3] = {1, 1, 1};
    uint element_comm_numbers[3] = {1, 1, 1};
    uint element_nstates = 0;
    double basis_per_atom = 1.3;
    double buffers[3] = {0.0, 0.0, 0.0};
    Chefsi_control chefsi_control;
    Xlsdft_control();
    ~Xlsdft_control();
    void set_element_comm_numbers(const uint& comm_ni, const uint& comm_nj, const uint& comm_nk);
    void set_element_numbers(const uint& ni, const uint& nj, const uint& nk);
    void set_basis_per_atom(const double& basis_per_atom);
    void set_buffers(const double& buffer_x, const double& buffer_y, const double& buffer_z);
    void set_is_PBCs(const bool& is_x_PBC, const bool& is_y_PBC, const bool& is_z_PBC);
    void set_is_rand_fixed(const bool& is_rand_fixed);
    void set_element_nstates(const uint& element_nstates);
    void init(const Input_file& input_file, const std::string& prefix = "XLSDFT");
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
    void print(std::ostream& output = std::cout, const std::string& prefix = "XLSDFT") const;
};

class Density_matrix_solver_control
{
public:
    uint method = 0;
    bool is_rand_fixed = false;
    Xlsdft_control xlsdft_control;
    Density_matrix_solver_control();
    ~Density_matrix_solver_control();
    void set_method(const uint& method);
    void set_is_rand_fixed(const bool& is_rand_fixed);
    void init(const Input_file& input_file);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
    void print(std::ostream& output = std::cout) const;
};

class Smearing_control
{
public:
    uint method = 0;
    double beta = 0.001;      // 1.0/(k_B*T)
    Smearing_control();
    ~Smearing_control();
    void set_method(const uint& method);
    void set_beta(const double& beta);
    const double& get_beta() const;
    void init(const Input_file& input_file);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
};

class Density_solver_control
{
public:
    uint method = 0;
    bool if_cal_spinor = false;
    bool if_cal_oneshot = false;
    Eigen_solver_control eigen_solver_control;
    Spinor_eigen_solver_control spinor_eigen_solver_control;
    Density_matrix_solver_control density_matrix_solver_control;
    Smearing_control smearing_control;
    Density_solver_control();
    ~Density_solver_control();
    void set_method(const uint& method);
    void set_if_cal_spinor(const bool& if_cal_spinor);
    void set_if_cal_oneshot(const bool& if_cal_oneshot);
    void init(const Input_file& input_file);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
    void print(std::ostream& output = std::cout) const;
};

class Misc_control
{
public:
    bool if_print_forces = 0;
    bool if_print_eigen = 0;
    bool if_print_density = 0;
    bool if_print_atoms = 0;
    bool if_print_pdos = 0;
    bool if_read_density = 0;
    Misc_control();
    ~Misc_control();
    void set_if_print_forces(const bool& if_print_forces);
    void set_if_print_eigen(const bool& if_print_eigen);
    void set_if_print_density(const bool& if_print_density);
    void set_if_print_atoms(const bool& if_print_atoms);
    void set_if_print_pdos(const bool& if_print_pdos);
    void set_if_read_density(const bool& if_read_density);
    void init(const Input_file& input_file);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
    void print(std::ostream& output = std::cout) const;
};

class Flow_control
{
public:
    bool scf_flag = 1;
    bool md_flag = 0;
    bool relax_flag = 0;
    Flow_control();
    ~Flow_control();
    void set_scf_flag(const bool& scf_flag);
    void set_md_flag(const bool& md_flag);
    void set_relax_flag(const bool& relax_flag);
    void init(const Input_file& input_file);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
    void print(std::ostream& output = std::cout) const;
};

class Stencil_control
{
public:
    int order = 12;
    Stencil_control();
    ~Stencil_control();
    void set_order(const int& set_order);
    void init(const Input_file& input_file);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
    void print(std::ostream& output = std::cout) const;
};

class Aar_control
{
public:
    uint precondition_method = 0;
    uint residual_method = 0;
    double anderson_beta = 0.6;
    double richardson_omega = 0.6;
    int mixing_history = 7;
    int anderson_frequency = 6;
    double tolerance = 1e-7;
    uint max_iter = 3000;
    Aar_control();
    ~Aar_control();
    void set_precondition_method(const uint& precondition_method);
    void set_residual_method(const uint& residual_method);
    void set_anderson_beta(const double& anderson_beta);
    void set_richardson_omega(const double& richardson_omega);
    void set_mixing_history(const int& mixing_history);
    void set_anderson_frequency(const int& anderson_frequency);
    void set_tolerance(const double& tolerance);
    void set_max_iter(const uint& max_iter);
    void init();
    // void init(const Input_file& input_file);
    // MPI_Datatype register_mpi_type() const;
    // void deregister_mpi_type(MPI_Datatype type) const;
    // void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
};

class Control
{
public:
    Mesh_control mesh_control;
    Scf_control scf_control;
    Poisson_solver_control poisson_solver_control;
    Exchange_correlation_solver_control exchange_correlation_solver_control;
    Density_solver_control density_solver_control;
    Misc_control misc_control;
    Flow_control flow_control;
    K_sample_control k_sample_control;
    Spin_control spin_control;
    Stencil_control stencil_control;
    Control();
    ~Control();
    void init_addition();
    void init(const Input_file& input_file, const Geometry& geometry);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
    void print(std::ostream& output = std::cout) const;
};

#endif //_CONTROL_H_
