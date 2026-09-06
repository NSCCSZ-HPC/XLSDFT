#include "control.h"

Mesh_control::Mesh_control() {}

Mesh_control::Mesh_control(const Mesh_control& mesh_control) {
    *this = mesh_control;
}

Mesh_control::~Mesh_control() {}

Mesh_control& Mesh_control::operator=(const Mesh_control& other) {
    this->delta_x = other.delta_x;         // mesh size in x-direction
    this->delta_y = other.delta_y;         // mesh size in y-direction
    this->delta_z = other.delta_z;         // mesh size in z-direction
    this->delta_V = other.delta_V;         // volumn of one mesh cubic
    this->nx = other.nx;                // number of nodes in x direction
    this->ny = other.ny;                // number of nodes in y direction
    this->nz = other.nz;                // number of nodes in z direction 
    this->nd = other.nd;                // total number of grid nodes
    this->is_periodic[0] = other.is_periodic[0];
    this->is_periodic[1] = other.is_periodic[1];
    this->is_periodic[2] = other.is_periodic[2];
    return *this;
}

void Mesh_control::set_deltas(const double& delta_x, const double& delta_y, const double& delta_z) {
    assert(delta_x >= 0.0 && delta_y >= 0.0 && delta_z >= 0.0);
    this->delta_x = delta_x;
    this->delta_y = delta_y;
    this->delta_z = delta_z;
    this->delta_V = delta_x * delta_y * delta_z;
    return;
}

void Mesh_control::set_delta_x(const double& delta_x) {
    this->set_deltas(delta_x, this->delta_y, this->delta_z);
    return;
}

void Mesh_control::set_delta_y(const double& delta_y) {
    this->set_deltas(this->delta_x, delta_y, this->delta_z);
    return;
}

void Mesh_control::set_delta_z(const double& delta_z) {
    this->set_deltas(this->delta_x, this->delta_y, delta_z);
    return;
}

void Mesh_control::set_ns(const uint& nx, const uint& ny, const uint& nz) {
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
    this->nd = nx * ny * nz;
    return;
}

void Mesh_control::set_nx(const uint& nx) {
    this->set_ns(nx, this->ny, this->nz);
    return;
}

void Mesh_control::set_ny(const uint& ny) {
    this->set_ns(this->nx, ny, this->nz);
    return;
}

void Mesh_control::set_nz(const uint& nz) {
    this->set_ns(this->nx, this->ny, nz);
    return;
}

void Mesh_control::set_is_PBCs(const bool& is_x_PBC, const bool& is_y_PBC, const bool& is_z_PBC) {
    this->is_periodic[0] = is_x_PBC;
    this->is_periodic[1] = is_y_PBC;
    this->is_periodic[2] = is_z_PBC;
    return;
}

void Mesh_control::set_is_PBCs(const bool& is_PBC) {
    this->is_periodic[0] = is_PBC;
    this->is_periodic[1] = is_PBC;
    this->is_periodic[2] = is_PBC;
    return;
}

void Mesh_control::set_is_x_PBC(const bool& is_x_PBC) {
    this->is_periodic[0] = is_x_PBC;
    return;
}

void Mesh_control::set_is_y_PBC(const bool& is_y_PBC) {
    this->is_periodic[1] = is_y_PBC;
    return;
}

void Mesh_control::set_is_z_PBC(const bool& is_z_PBC) {
    this->is_periodic[2] = is_z_PBC;
    return;
}

void Mesh_control::init(const Input_file& input_file, const Geometry& geometry) {
    assert(geometry.cell_type <= 2);
    std::stringstream ss;
    if (input_file.map.find("BC") != input_file.map.end()) {
        ss.str(input_file.get_value("BC"));
        char BCx, BCy, BCz;
        ss >> BCx >> BCy >> BCz;
        ss.clear();
        if (BCx == 'P' || BCx == 'p' || BCx == '1') {
            this->is_periodic[0] = true;
        } else if (BCx == 'D' || BCx == 'd' || BCx == '0') {
            this->is_periodic[0] = false;
        } else {
            assert(!"BC should be P or D");
        }
        if (BCy == 'P' || BCy == 'p' || BCy == '1') {
            this->is_periodic[1] = true;
        } else if (BCy == 'D' || BCy == 'd' || BCy == '0') {
            this->is_periodic[1] = false;
        } else {
            assert(!"BC should be P or D");
        }
        if (BCz == 'P' || BCz == 'p' || BCz == '1') {
            this->is_periodic[2] = true;
        } else if (BCz == 'D' || BCz == 'd' || BCz == '0') {
            this->is_periodic[2] = false;
        } else {
            assert(!"BC should be P or D");
        }
    } else {
        assert(0 && "ERROR:: There should be a BC in input file!");
    }

    if (input_file.map.find("FD_GRID") != input_file.map.end()) {
        ss.str(input_file.get_value("FD_GRID"));
        uint ni, nj, nk;
        ss >> ni >> nj >> nk;
        this->set_ns(ni, nj, nk);
        ss.clear();
        double delta_x = (this->is_periodic[0]) ? geometry.a/(double)ni : geometry.a/(double)(ni - 1);
        double delta_y = (this->is_periodic[1]) ? geometry.b/(double)nj : geometry.b/(double)(nj - 1);
        double delta_z = (this->is_periodic[2]) ? geometry.c/(double)nk : geometry.c/(double)(nk - 1);
        this->set_deltas(delta_x, delta_y, delta_z);
    } else if (input_file.map.find("MESH_SPACING") != input_file.map.end()) {
        double mesh_spacing = std::stod(input_file.get_value("MESH_SPACING"));
        uint nx = (this->is_periodic[0]) ? std::ceil(geometry.a/mesh_spacing) + 1e-12 
                                         : std::ceil(geometry.a/mesh_spacing) + 1e-12 + 1;
        uint ny = (this->is_periodic[1]) ? std::ceil(geometry.b/mesh_spacing) + 1e-12 
                                         : std::ceil(geometry.b/mesh_spacing) + 1e-12 + 1;
        uint nz = (this->is_periodic[2]) ? std::ceil(geometry.c/mesh_spacing) + 1e-12 
                                         : std::ceil(geometry.c/mesh_spacing) + 1e-12 + 1;
        this->set_ns(nx, ny, nz);
        double delta_x = (this->is_periodic[0]) ? geometry.a/(double)nx : geometry.a/(double)(nx - 1);
        double delta_y = (this->is_periodic[1]) ? geometry.b/(double)ny : geometry.b/(double)(ny - 1);
        double delta_z = (this->is_periodic[2]) ? geometry.c/(double)nz : geometry.c/(double)(nz - 1);
        this->set_deltas(delta_x, delta_y, delta_z);
    } else {
        assert(0 && "ERROR:: There should be a MESH_SPACING in input file!");
    }
    return;
}

MPI_Datatype Mesh_control::register_mpi_type() const {
    constexpr std::size_t num_members = 9;
    int lengths[num_members] = { 1, 1, 1, 1, 1, 1, 1, 1, 3};
    MPI_Aint offsets[num_members] = {offsetof(Mesh_control, delta_x), offsetof(Mesh_control, delta_y),
                                     offsetof(Mesh_control, delta_z), offsetof(Mesh_control, delta_V),
                                     offsetof(Mesh_control, nx), offsetof(Mesh_control, ny),
                                     offsetof(Mesh_control, nz), offsetof(Mesh_control, nd),
                                     offsetof(Mesh_control, is_periodic)};
    MPI_Datatype types[num_members] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                                       MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED, 
                                       MPI_C_BOOL};
    MPI_Datatype type;
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    return type;
}

void Mesh_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Mesh_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype Mesh_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Mesh_control_type, root, comm);
    this->deregister_mpi_type(Mesh_control_type);
    return;
}

void Mesh_control::show() const {
    std::cout << "[delta_x, delta_y, delta_z, delta_V] = [" 
            << std::setw(12) << std::setprecision(8) << std::fixed << this->delta_x << ", "
            << std::setw(12) << std::setprecision(8) << std::fixed << this->delta_y << ", " 
            << std::setw(12) << std::setprecision(8) << std::fixed << this->delta_z << ", " 
            << std::setw(12) << std::setprecision(8) << std::fixed << this->delta_V << "]"
            << std::endl;
    std::cout << "[nx, ny, nz, nd] = [" 
            << std::setw(8) << this->nx << " , "
            << std::setw(8) << this->ny << " , " 
            << std::setw(8) << this->nz << " , " 
            << std::setw(8) << this->nd << " ]"
            << std::endl;
    std::cout << "is_periodic = [" 
            << std::setw(8) << this->is_periodic[0] << " , "
            << std::setw(8) << this->is_periodic[1] << " , " 
            << std::setw(8) << this->is_periodic[2] << " ]"
            << std::endl;
    return;
}

void Mesh_control::print(std::ostream& output) const {
    output << "BC: ";
    if (this->is_periodic[0]) {
        output << "P ";
    } else {
        output << "D ";
    }
    if (this->is_periodic[1]) {
        output << "P ";
    } else {
        output << "D ";
    }
    if (this->is_periodic[2]) {
        output << "P";
    } else {
        output << "D";
    }
    output << std::endl;
    output << "FD_GRID: " << std::fixed << this->nx << " " << this->ny << " " << this->nz << std::endl;
    return;
}


Kerker_control::Kerker_control() {}

Kerker_control::~Kerker_control() {}

void Kerker_control::set_tolerance(const double& tolerance) {
    this->tolerance = tolerance;
    return;
}

void Kerker_control::set_kerker_ktf(const double& kerker_ktf) {
    this->kerker_ktf = kerker_ktf;
    return;
}

void Kerker_control::set_kerker_thresh(const double& kerker_thresh) {
    this->kerker_thresh = kerker_thresh;
    return;
}

void Kerker_control::init(const Input_file& input_file) {
    if (input_file.map.find("PRECOND_KERKER_KTF") != input_file.map.end()) {
        this->set_kerker_ktf(std::stod(input_file.get_value("PRECOND_KERKER_KTF")));
    }
    if (input_file.map.find("PRECOND_KERKER_THRESH") != input_file.map.end()) {
        this->set_kerker_thresh(std::stod(input_file.get_value("PRECOND_KERKER_THRESH")));
    }
    if (input_file.map.find("TOL_PRECOND") != input_file.map.end()) {
        this->set_tolerance(std::stod(input_file.get_value("TOL_PRECOND")));
    }
    return;
}

MPI_Datatype Kerker_control::register_mpi_type() const {
    MPI_Datatype type;
    constexpr std::size_t num_members = 3;
    int lengths[num_members] = {1, 1, 1};
    MPI_Aint offsets[num_members] = {offsetof(Kerker_control, tolerance),
                                     offsetof(Kerker_control, kerker_ktf),
                                     offsetof(Kerker_control, kerker_thresh)};
    MPI_Datatype types[num_members] = {MPI_DOUBLE, MPI_DOUBLE,
                                       MPI_DOUBLE};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    return type;
}

void Kerker_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Kerker_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype Kerker_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Kerker_control_type, root, comm);
    this->deregister_mpi_type(Kerker_control_type);
    return;
}

void Kerker_control::show() const {
    std::cout << "tolerance = " << this->tolerance << std::endl;
    std::cout << "kerker_ktf = " << this->kerker_ktf << std::endl;
    std::cout << "kerker_thresh = " << this->kerker_thresh << std::endl;
    return;
}

void Kerker_control::print(std::ostream& output) const {
    output << "TOL_PRECOND: " << this->tolerance << std::endl;
    output << "PRECOND_KERKER_KTF: " << this->kerker_ktf << std::endl;
    output << "PRECOND_KERKER_THRESH: " << this->kerker_thresh << std::endl;
    return;
}

Pulay_control::Pulay_control() {}

Pulay_control::~Pulay_control() {}

void Pulay_control::set_pulay_frequency(const uint& pulay_frequency) {
    this->pulay_frequency = pulay_frequency;
    return;
}

void Pulay_control::set_pulay_restart(const bool& pulay_restart) {
    this->pulay_restart = pulay_restart;
    return;
}

void Pulay_control::set_beta(const double& beta) {
    this->beta = beta;
    return;
}

void Pulay_control::init(const Input_file& input_file) {
    if (input_file.map.find("PULAY_FREQUENCY") != input_file.map.end()) {
        this->set_pulay_frequency(std::stoi(input_file.get_value("PULAY_FREQUENCY")));
    }
    if (input_file.map.find("PULAY_RESTART") != input_file.map.end()) {
        this->set_pulay_restart(std::stoi(input_file.get_value("PULAY_RESTART")));
    }
    if (input_file.map.find("MIXING_PARAMETER") != input_file.map.end()) {
        this->set_beta(std::stod(input_file.get_value("MIXING_PARAMETER")));
    }
    return;
}

MPI_Datatype Pulay_control::register_mpi_type() const {
    MPI_Datatype type;
    constexpr std::size_t num_members = 3;
    int lengths[num_members] = {1, 1, 1};
    MPI_Aint offsets[num_members] = {offsetof(Pulay_control, pulay_frequency),
                                     offsetof(Pulay_control, pulay_restart),
                                     offsetof(Pulay_control, beta)};
    MPI_Datatype types[num_members] = {MPI_UNSIGNED, MPI_C_BOOL,
                                       MPI_DOUBLE};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    return type;
}

void Pulay_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Pulay_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype Pulay_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Pulay_control_type, root, comm);
    this->deregister_mpi_type(Pulay_control_type);
    return;
}

void Pulay_control::show() const {
    std::cout << "pulay_frequency = " << this->pulay_frequency << std::endl;
    std::cout << "pulay_restart = " << this->pulay_restart << std::endl;
    std::cout << "beta = " << this->beta << std::endl;
    return;
}

void Pulay_control::print(std::ostream& output) const {
    output << "PULAY_FREQUENCY: " << this->pulay_frequency << std::endl;
    output << "PULAY_RESTART: " << this->pulay_restart << std::endl;
    output << "MIXING_PARAMETER: " << this->beta << std::endl;
    return;
}

Mixing_control::Mixing_control() {}

Mixing_control::~Mixing_control() {}

void Mixing_control::set_method(const uint method) {
    this->method = method;
    return;
}

void Mixing_control::set_mixing_history(const uint mixing_history) {
    this->mixing_history = mixing_history;
    return;
}

void Mixing_control::set_precondition_method(const uint precondition_method) {
    this->precondition_method = precondition_method;
    return;
}

void Mixing_control::set_simple_precondition_method(const uint simple_precondition_method) {
    this->simple_precondition_method = simple_precondition_method;
    return;
}

void Mixing_control::set_alpha(const double alpha) {
    this->alpha = alpha;
    return;
}

void Mixing_control::set_alpha_mag(const double alpha_mag) {
    this->alpha_mag = alpha_mag;
    return;
}

void Mixing_control::init(const Input_file& input_file) {
    if (input_file.map.find("MIXING_METHOD") != input_file.map.end()) {
        std::string str = Filesys::toupper(input_file.get_value("MIXING_METHOD"));
        if (str == "LINEAR") {
            this->set_method(0);
        } else if (str == "PULAY") {
            this->set_method(1);
        } else {
            assert(!"ERROR:: EIGEN_SOLVER should be LINEAR or PULAY!");
        }
    }
    if (input_file.map.find("MIXING_HISTORY") != input_file.map.end()) {
        this->set_mixing_history(std::stoi(input_file.get_value("MIXING_HISTORY")));
    }
    if (input_file.map.find("MIXING_PARAMETER_SIMPLE") != input_file.map.end()) {
        this->set_alpha(std::stod(input_file.get_value("MIXING_PARAMETER_SIMPLE")));
    } else if (input_file.map.find("MIXING_PARAMETER") != input_file.map.end()) {
        this->set_alpha(std::stod(input_file.get_value("MIXING_PARAMETER")));
    }
    if (input_file.map.find("MIXING_PARAMETER_SIMPLE_MAG") != input_file.map.end()) {
        this->set_alpha_mag(std::stod(input_file.get_value("MIXING_PARAMETER_SIMPLE_MAG")));
    } else if (input_file.map.find("MIXING_PARAMETER_MAG") != input_file.map.end()) {
        this->set_alpha_mag(std::stod(input_file.get_value("MIXING_PARAMETER_MAG")));
    }
    if (input_file.map.find("MIXING_PRECOND") != input_file.map.end()) {
        std::string str = Filesys::toupper(input_file.get_value("MIXING_PRECOND"));
        if (str == "NONE") {
            this->set_precondition_method(0);
        } else if (str == "KERKER") {
            this->set_precondition_method(1);
        } else {
            assert(!"ERROR:: MIXING_PRECOND should be NONE or KERKER!");
        }
    }
    if (input_file.map.find("MIXING_PRECOND_SIMPLE") != input_file.map.end()) {
        std::string str = Filesys::toupper(input_file.get_value("MIXING_PRECOND_SIMPLE"));
        if (str == "NONE") {
            this->set_simple_precondition_method(0);
        } else if (str == "KERKER") {
            this->set_simple_precondition_method(1);
        } else {
            assert(!"ERROR:: MIXING_PRECOND_SIMPLE should be NONE or KERKER!");
        }
    } else {
        this->set_simple_precondition_method(this->precondition_method);
    }
    if (this->method == 1) {
        this->pulay_control.init(input_file);
    }
    if (this->precondition_method == 1) {
        this->kerker_control.init(input_file);
    }
    return;
}

MPI_Datatype Mixing_control::register_mpi_type() const {
    MPI_Datatype type;
    MPI_Datatype Pulay_control_type = this->pulay_control.register_mpi_type();
    MPI_Datatype Kerker_control_type = this->kerker_control.register_mpi_type();

    constexpr std::size_t num_members = 8;
    int lengths[num_members] = {1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Aint offsets[num_members] = {offsetof(Mixing_control, method),
                                     offsetof(Mixing_control, mixing_history),
                                     offsetof(Mixing_control, precondition_method),
                                     offsetof(Mixing_control, simple_precondition_method),
                                     offsetof(Mixing_control, alpha),
                                     offsetof(Mixing_control, alpha_mag),
                                     offsetof(Mixing_control, pulay_control),
                                     offsetof(Mixing_control, kerker_control)};
    MPI_Datatype types[num_members] = {MPI_UNSIGNED, MPI_UNSIGNED,
                                       MPI_UNSIGNED, MPI_UNSIGNED,
                                       MPI_DOUBLE, MPI_DOUBLE,
                                       Pulay_control_type,
                                       Kerker_control_type};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);

    this->pulay_control.deregister_mpi_type(Pulay_control_type);

    return type;
}

void Mixing_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Mixing_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype Mixing_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Mixing_control_type, root, comm);
    this->deregister_mpi_type(Mixing_control_type);
    return;
}

void Mixing_control::show() const {
    if (this->method == 0) std::cout << "method = 0, which means linear mixing." << std::endl;
    if (this->method == 1) {
        std::cout << "method = 1, which means Pulay mixing." << std::endl;
        this->pulay_control.show();
    }
    if (this->precondition_method == 0) std::cout << "precondition_method = 0, which means no precondition." << std::endl;
    if (this->precondition_method == 1) {
        std::cout << "precondition_method = 1, which means kerker precondition." << std::endl;
        this->kerker_control.show();
    }
    if (this->simple_precondition_method == 0) std::cout << "simple_precondition_method = 0, which means no precondition." << std::endl;
    if (this->simple_precondition_method == 1) {
        std::cout << "simple_precondition_method = 1, which means kerker precondition." << std::endl;
        this->kerker_control.show();
    }
    std::cout << "mixing_history = " << this->mixing_history << std::endl;
    std::cout << "alpha = " << this->alpha << std::endl;
    return;
}

void Mixing_control::print(std::ostream& output) const {
    output << "MIXING_METHOD: ";
    if (this->method == 0) {
        output << "LINEAR" << std::endl;
    } else if (this->method == 1) {
        output << "PULAY" << std::endl;
        this->pulay_control.print(output);
    } else {
        assert(!"ERROR:: EIGEN_SOLVER should be LINEAR or PULAY!");
    }
    output << "MIXING_HISTORY: " << this->mixing_history << std::endl;
    output << "MIXING_PARAMETER: " << std::scientific << std::setprecision(3) << this->alpha << std::endl;
    output << "MIXING_PARAMETER_MAG: " << std::scientific << std::setprecision(3) << this->alpha_mag << std::endl;
    output << "MIXING_PRECOND: ";
    if (this->precondition_method == 0) {
        output << "NONE" << std::endl;
    } else if (this->precondition_method == 1) {
        output << "KERKER" << std::endl;
        this->kerker_control.print(output);
    } else {
        assert(!"ERROR:: MIXING_PRECOND should be NONE or KERKER!");
    }
    output << "MIXING_PRECOND_SIMPLE: ";
    if (this->simple_precondition_method == 0) {
        output << "NONE" << std::endl;
    } else if (this->simple_precondition_method == 1) {
        output << "KERKER" << std::endl;
    } else {
        assert(!"ERROR:: MIXING_PRECOND_SIMPLE should be NONE or KERKER!");
    }
    return;
}

Scf_control::Scf_control() {}

Scf_control::~Scf_control() {}

void Scf_control::set_min_iter(const uint& min_iter) {
    this->min_iter = min_iter;
    return;
}

void Scf_control::set_max_iter(const uint& max_iter) {
    this->max_iter = max_iter;
    return;
}

void Scf_control::set_max_single_precision_iter(const uint& max_single_precision_iter) {
    this->max_single_precision_iter = max_single_precision_iter;
    return;
}

void Scf_control::set_mixing_variable(const uint& mixing_variable) {
    this->mixing_variable = mixing_variable;
    return;
}

void Scf_control::set_tolerance(const double& tolerance) {
    this->tolerance = tolerance;
    return;
}

void Scf_control::set_single_precision_tolerance(const double& single_precision_tolerance) {
    this->single_precision_tolerance = single_precision_tolerance;
    return;
}

double Scf_control::get_tolerance() const {
    return this->tolerance;
}

void Scf_control::init(const Input_file& input_file) {
    if (input_file.map.find("MINIT_SCF") != input_file.map.end()) {
        this->set_min_iter(std::stoi(input_file.get_value("MINIT_SCF")));
    }
    if (input_file.map.find("MAXIT_SCF") != input_file.map.end()) {
        this->set_max_iter(std::stoi(input_file.get_value("MAXIT_SCF")));
    }
    if (input_file.map.find("MAXIT_SINGLE_PRECISION") != input_file.map.end()) {
        this->set_max_single_precision_iter(std::stoi(input_file.get_value("MAXIT_SINGLE_PRECISION")));
    }
    if (input_file.map.find("MIXING_VARIABLE") != input_file.map.end()) {
        std::string str = Filesys::toupper(input_file.get_value("MIXING_VARIABLE"));
        if (str == "DENSITY") {
            this->set_mixing_variable(0);
        } else if (str == "POTENTIAL") {
            this->set_mixing_variable(1);
        } else {
            assert(!"ERROR:: EIGEN_SOLVER should be DENSITY or POTENTIAL!");
        }
    }
    if (input_file.map.find("TOL_SCF") != input_file.map.end()) {
        this->set_tolerance(std::stod(input_file.get_value("TOL_SCF")));
    }
    if (input_file.map.find("TOL_SINGLE_PRECISION_SCF") != input_file.map.end()) {
        this->set_single_precision_tolerance(std::stod(input_file.get_value("TOL_SINGLE_PRECISION_SCF")));
    }
    this->mixing_control.init(input_file);
    return;
}

MPI_Datatype Scf_control::register_mpi_type() const {
    MPI_Datatype type;
    MPI_Datatype Mixing_control_type = this->mixing_control.register_mpi_type();
    constexpr std::size_t num_members = 7;
    int lengths[num_members] = {1, 1, 1, 1, 1, 1, 1};
    MPI_Aint offsets[num_members] = {offsetof(Scf_control, min_iter),
                                     offsetof(Scf_control, max_iter),
                                     offsetof(Scf_control, max_single_precision_iter),
                                     offsetof(Scf_control, mixing_variable),
                                     offsetof(Scf_control, tolerance),
                                     offsetof(Scf_control, single_precision_tolerance),
                                     offsetof(Scf_control, mixing_control)};
    MPI_Datatype types[num_members] = {MPI_UNSIGNED, MPI_UNSIGNED,
                                       MPI_UNSIGNED, MPI_UNSIGNED,
                                       MPI_DOUBLE, MPI_DOUBLE,
                                       Mixing_control_type};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);

    this->mixing_control.deregister_mpi_type(Mixing_control_type);

    return type;
}

void Scf_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Scf_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype Scf_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Scf_control_type, root, comm);
    this->deregister_mpi_type(Scf_control_type);
    return;
}

void Scf_control::show() const {
    std::cout << "min_iter = " << this->min_iter << std::endl;
    std::cout << "max_iter = " << this->max_iter << std::endl;
    std::cout << "max_single_precision_iter = " << this->max_single_precision_iter << std::endl;
    std::cout << "mixing_variable = " << this->mixing_variable << std::endl;
    std::cout << "tolerance = " << std::setw(12) << std::setprecision(3) << std::scientific << std::left
              << this->tolerance << std::endl;
    std::cout << "single_precision_tolerance = " << std::setw(12) << std::setprecision(3) << std::scientific << std::left
              << this->single_precision_tolerance << std::endl;
    this->mixing_control.show();
    return;
}

void Scf_control::print(std::ostream& output) const {
    output << "MINIT_SCF: " << this->min_iter << std::endl;
    output << "MAXIT_SCF: " << this->max_iter << std::endl;
    output << "MAXIT_SINGLE_PRECISION: " << this->max_single_precision_iter << std::endl;
    output << "MIXING_VARIABLE: " << ((this->mixing_variable == 0) ? "DENSITY" : "POTENTIAL") << std::endl;
    output << "TOL_SCF: " << std::scientific << std::setprecision(3) << this->tolerance << std::endl;
    output << "TOL_SINGLE_PRECISION_SCF: " << std::scientific << std::setprecision(3) << this->single_precision_tolerance << std::endl;
    this->mixing_control.print(output);
    return;
}

K_sample_control::K_sample_control() {}

K_sample_control::~K_sample_control() {}

void K_sample_control::set_method(const uint& method) {
    this->method = method;
    return;
}

void K_sample_control::set_nkts(const int& nkpt0, const int& nkpt1, const int& nkpt2) {
    this->nkpts[0] = nkpt0;
    this->nkpts[1] = nkpt1;
    this->nkpts[2] = nkpt2;
    return;
}

void K_sample_control::set_k_shifts(const double& k_shift0, const double& k_shift1, const double& k_shift2) {
    this->k_shifts[0] = k_shift0;
    this->k_shifts[1] = k_shift1;
    this->k_shifts[2] = k_shift2;
    return;
}

void K_sample_control::init(const Input_file& input_file) {
    std::stringstream ss;
    if (input_file.map.find("KPOINT_METHOD") != input_file.map.end()) {
        ss.str(input_file.get_value("KPOINT_METHOD"));
        uint method;
        ss >> method;
        this->set_method(method);
        ss.clear();
    }
    if (input_file.map.find("KPOINT_GRID") != input_file.map.end()) {
        ss.str(input_file.get_value("KPOINT_GRID"));
        int nkpt0, nkpt1, nkpt2;
        ss >> nkpt0 >> nkpt1 >> nkpt2;
        this->set_nkts(nkpt0, nkpt1, nkpt2);
        ss.clear();
    }
    if (input_file.map.find("KPOINT_SHIFT") != input_file.map.end()) {
        ss.str(input_file.get_value("KPOINT_SHIFT"));
        double k_shift0, k_shift1, k_shift2;
        ss >> k_shift0 >> k_shift1 >> k_shift2;
        this->set_k_shifts(k_shift0, k_shift1, k_shift2);
        ss.clear();
    }
    return;
}

MPI_Datatype K_sample_control::register_mpi_type() const {
    MPI_Datatype type;
    constexpr std::size_t num_members = 3;
    int lengths[num_members] = {1, 3, 3};
    MPI_Aint offsets[num_members] = {offsetof(K_sample_control, method),
                                     offsetof(K_sample_control, nkpts),
                                     offsetof(K_sample_control, k_shifts)};
    MPI_Datatype types[num_members] = {MPI_UNSIGNED, MPI_INT, MPI_DOUBLE};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    return type;
}

void K_sample_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void K_sample_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype K_sample_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, K_sample_control_type, root, comm);
    this->deregister_mpi_type(K_sample_control_type);
    return;
}

void K_sample_control::show() const {
    std::cout << "K_method = " << this->method << std::endl;
    std::cout << "nkpts[0, 1, 2] = ["
        << std::fixed << this->nkpts[0] << " , "
        << std::fixed << this->nkpts[1] << " , "
        << std::fixed << this->nkpts[2] << " ]"
        << std::endl;
    std::cout << "k_shifts[0, 1, 2] = [" 
        << std::fixed << this->k_shifts[0] << " , "
        << std::fixed << this->k_shifts[1] << " , "
        << std::fixed << this->k_shifts[2] << " ]"
        << std::endl;
    return;
}

void K_sample_control::print(std::ostream& output) const {
    output << "KPOINT_METHOD: " << this->method << std::endl;
    output << "KPOINT_GRID: " << this->nkpts[0] << " " << this->nkpts[1] << " " << this->nkpts[2] << std::endl;
    output << "KPOINT_SHIFT: " << this->k_shifts[0] << " " << this->k_shifts[1] << " " << this->k_shifts[2] << std::endl;
    return;
}

Spin_control::Spin_control() {}

Spin_control::~Spin_control() {}

void Spin_control::set_spin_type(const uint& spin_type) {
    this->spin_type = spin_type;
    return;
}

void Spin_control::init(const Input_file& input_file) {
    if (input_file.map.find("SPIN_TYP") != input_file.map.end()) {
        this->set_spin_type(std::stoi(input_file.get_value("SPIN_TYP")));
    }
    assert(this->spin_type <= 0);
    return;
}

MPI_Datatype Spin_control::register_mpi_type() const {
    MPI_Datatype type;
    constexpr std::size_t num_members = 1;
    int lengths[num_members] = {1};
    MPI_Aint offsets[num_members] = {offsetof(Spin_control, spin_type)};
    MPI_Datatype types[num_members] = {MPI_UNSIGNED};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    return type;
}

void Spin_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Spin_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype Spin_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Spin_control_type, root, comm);
    this->deregister_mpi_type(Spin_control_type);
    return;
}

void Spin_control::show() const {
    std::cout << "spin_type = " << this->spin_type << std::endl;
    return;
}

void Spin_control::print(std::ostream& output) const {
    output << "SPIN_TYP: " << this->spin_type << std::endl;
    return;
}

Exchange_correlation_solver_control::Exchange_correlation_solver_control() {}

Exchange_correlation_solver_control::~Exchange_correlation_solver_control() {}

void Exchange_correlation_solver_control::set_exchange_method(const uint& exchange_method) {
    this->exchange_method = exchange_method;
    return;
}

void Exchange_correlation_solver_control::set_correlation_method(const uint& correlation_method) {
    this->correlation_method = correlation_method;
    return;
}

void Exchange_correlation_solver_control::set_xc_rhotol(const double& xc_rhotol) {
    this->xc_rhotol = xc_rhotol;
    return;
}

void Exchange_correlation_solver_control::init(const Input_file& input_file) {
    if (input_file.map.find("EXCHANGE_CORRELATION") != input_file.map.end()) {
        std::string str = Filesys::toupper(input_file.get_value("EXCHANGE_CORRELATION"));
        if (str == "LDA_PZ") {
            this->set_exchange_method(0);  // slater_exchange
            this->set_correlation_method(0); // pz_correlation
        } else if (str == "GGA_PBE") {
            this->set_exchange_method(1);
            this->set_correlation_method(1);
        } else {
            assert(!"ERROR:: EXCHANGE_CORRELATION should be LDA_PZ or GGA_PBE!");
        }
    }
    if (input_file.map.find("XC_RHOTOL") != input_file.map.end()) {
        this->set_xc_rhotol(std::stod(input_file.get_value("XC_RHOTOL")));
    }
    return;
}

MPI_Datatype Exchange_correlation_solver_control::register_mpi_type() const {
    MPI_Datatype type;
    constexpr std::size_t num_members = 3;
    int lengths[num_members] = {1, 1, 1};
    MPI_Aint offsets[num_members] = {offsetof(Exchange_correlation_solver_control, exchange_method),
                                     offsetof(Exchange_correlation_solver_control, correlation_method),
                                     offsetof(Exchange_correlation_solver_control, xc_rhotol)};
    MPI_Datatype types[num_members] = {MPI_UNSIGNED, MPI_UNSIGNED,
                                       MPI_DOUBLE};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    return type;
}

void Exchange_correlation_solver_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Exchange_correlation_solver_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype Exchange_correlation_solver_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Exchange_correlation_solver_control_type, root, comm);
    this->deregister_mpi_type(Exchange_correlation_solver_control_type);
    return;
}

void Exchange_correlation_solver_control::show() const {
    if (this->exchange_method == 0) std::cout << "exchange_method = 0, which means slater_exchange." << std::endl;
    if (this->exchange_method == 1) std::cout << "exchange_method = 1, which means pbe_exchange." << std::endl;
    if (this->correlation_method == 0) std::cout << "correlation_method = 0, which means pz_correlation." << std::endl;
    if (this->correlation_method == 1) std::cout << "correlation_method = 1, which means pbe_correlation." << std::endl;
    std::cout << "xc_rhotol = " << std::setw(12) << std::setprecision(3) << std::scientific << std::left
              << this->xc_rhotol << std::endl;
    return;
}

void Exchange_correlation_solver_control::print(std::ostream& output) const {
    output << "EXCHANGE_CORRELATION: ";
    if (this->exchange_method == 0 && this->correlation_method == 0) {
        output << "LDA_PZ" << std::endl;
    } else if (this->exchange_method == 1 && this->correlation_method == 1) {
        output << "GGA_PBE" << std::endl;
    } else {
        assert(!"ERROR:: EXCHANGE_CORRELATION should be LDA_PZ or GGA_PBE!");
    }
    output << "XC_RHOTOL: " << std::scientific << std::setprecision(3) << this->xc_rhotol << std::endl;
    return;
}

Poisson_solver_control::Poisson_solver_control() {}

Poisson_solver_control::~Poisson_solver_control() {}

void Poisson_solver_control::set_is_rand_fixed(const bool& is_rand_fixed) {
    this->is_rand_fixed = is_rand_fixed;
    return;
}

void Poisson_solver_control::set_method(const uint& method) {
    this->method = method;
    return;
}

void Poisson_solver_control::set_precondition_method(const uint& precondition_method) {
    this->precondition_method = precondition_method;
    return;
}

void Poisson_solver_control::set_max_iter(const uint& max_iter) {
    this->max_iter = max_iter;
    return;
}

void Poisson_solver_control::set_tolerance(const double& tolerance) {
    this->tolerance = tolerance;
    return;
}

void Poisson_solver_control::init(const Input_file& input_file, const Scf_control& scf_control) {
    if (input_file.map.find("POISSON_SOLVER") != input_file.map.end()) {
        std::string str = Filesys::toupper(input_file.get_value("POISSON_SOLVER"));
        if (str == "AAR") {
            this->set_method(0);
        } else if (str == "HYPRE") {
            this->set_method(1);
        } else if (str == "CG") {
            this->set_method(2);
        } else if (str == "GMRES") {
            this->set_method(3);
        } else {
            assert(!"ERROR:: POISSON_SOLVER should be AAR, HYPRE, CG or GMRES!");
        }
    }
    if (input_file.map.find("POISSON_SOLVER_PRECOND") != input_file.map.end()) {
        std::string str = Filesys::toupper(input_file.get_value("POISSON_SOLVER_PRECOND"));
        if (str == "JACOBI") {
            this->set_precondition_method(0);
        } else if (str == "DST") {
            this->set_precondition_method(1);
        } else if (str == "MG") {
            this->set_precondition_method(2);
        } else {
            assert(!"ERROR:: POISSON_SOLVER_PRECOND should be JACOBI, DST or MG!");
        }
    }
    if (input_file.map.find("MAXIT_POISSON") != input_file.map.end()) {
        this->set_max_iter(std::stoi(input_file.get_value("MAXIT_POISSON")));
    }
    if (input_file.map.find("TOL_POISSON") != input_file.map.end()) {
        this->set_tolerance(std::stod(input_file.get_value("TOL_POISSON")));
    } else {
        this->set_tolerance(0.01 * scf_control.get_tolerance());
    }
    if (input_file.map.find("POISSON_FIX_RAND") != input_file.map.end()) {
        this->set_is_rand_fixed(std::stoi(input_file.get_value("POISSON_FIX_RAND")));
    } else if (input_file.map.find("FIX_RAND") != input_file.map.end()) {
        this->set_is_rand_fixed(std::stoi(input_file.get_value("FIX_RAND")));
    }
    return;
}

MPI_Datatype Poisson_solver_control::register_mpi_type() const {
    MPI_Datatype type;
    constexpr std::size_t num_members = 5;
    int lengths[num_members] = {1, 1, 1, 1, 1};
    MPI_Aint offsets[num_members] = {offsetof(Poisson_solver_control, is_rand_fixed),
                                     offsetof(Poisson_solver_control, method),
                                     offsetof(Poisson_solver_control, precondition_method),
                                     offsetof(Poisson_solver_control, max_iter),
                                     offsetof(Poisson_solver_control, tolerance)};
    MPI_Datatype types[num_members] = {MPI_C_BOOL, MPI_UNSIGNED,
                                        MPI_UNSIGNED, MPI_UNSIGNED,
                                        MPI_DOUBLE};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    return type;
}

void Poisson_solver_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Poisson_solver_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype Poisson_solver_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Poisson_solver_control_type, root, comm);
    this->deregister_mpi_type(Poisson_solver_control_type);
    return;
}

void Poisson_solver_control::show() const {
    if (this->method == 0) std::cout << "method = 0, which means AAR method." << std::endl;
    if (this->precondition_method == 0) std::cout << "precondition_method = o, which means jacobi method." << std::endl;
    if (this->precondition_method == 1) std::cout << "precondition_method = 1, which means DST method." << std::endl;
    std::cout << "max_iter = " << this->max_iter << std::endl;
    std::cout << "tolerance = " << std::setw(12) << std::setprecision(3) << std::scientific << std::left
              << this->tolerance << std::endl;
    std::cout << "is_rand_fixed = " << this->is_rand_fixed << std::endl;
    return;
}

void Poisson_solver_control::print(std::ostream& output) const {
    output << "POISSON_SOLVER: ";
    if (this->method == 0) {
        output << "AAR" << std::endl;
    } else if (this->method == 1) {
        output << "HYPRE" << std::endl;
    } else if (this->method == 2) {
        output << "CG" << std::endl;
    } else if (this->method == 3) {
        output << "GMRES" << std::endl;
    } else {
        assert(!"ERROR:: POISSON_SOLVER should be AAR, HYPRE, CG or GMRES!");
    }
    output << "POISSON_SOLVER_PRECOND: ";
    if (this->precondition_method == 0) {
        output << "JACOBI" << std::endl;
    } else if (this->precondition_method == 1) {
        output << "DST" << std::endl;
    } else if (this->precondition_method == 2) {
        output << "MG" << std::endl;
    } else {
        assert(!"ERROR:: POISSON_SOLVER_PRECOND should be JACOBI, DST or MG!");
    }
    output << "MAXIT_POISSON: " << this->max_iter << std::endl;
    output << "TOL_POISSON: " << std::scientific << std::setprecision(3) << this->tolerance << std::endl;
    output << "POISSON_FIX_RAND: " << this->is_rand_fixed << std::endl;
    return;
}

Chefsi_control::Chefsi_control() {}

Chefsi_control::Chefsi_control(const Chefsi_control& other) {
    *this = other;
}

Chefsi_control::~Chefsi_control() {}

Chefsi_control& Chefsi_control::operator=(const Chefsi_control& other) {
    this->max_iter = other.max_iter;
    this->chebyshev_filter_degree = other.chebyshev_filter_degree;
    this->rho_trigger = other.rho_trigger;
    this->projection_method = other.projection_method;
    this->tolerance_lanczos = other.tolerance_lanczos;
    return *this;
}

void Chefsi_control::set_max_iter(const uint& max_iter) {
    this->max_iter = max_iter;
    return;
}

void Chefsi_control::set_chebyshev_filter_degree(const uint& chebyshev_filter_degree) {
    this->chebyshev_filter_degree = chebyshev_filter_degree;
    return;
}

void Chefsi_control::set_rho_trigger(const uint& rho_trigger) {
    this->rho_trigger = rho_trigger;
    return;
}

void Chefsi_control::set_projection_method(const uint& projection_method) {
    this->projection_method = projection_method;
    return;
}

void Chefsi_control::set_tolerance_lanczos(const double& tolerance_lanczos) {
    this->tolerance_lanczos = tolerance_lanczos;
    return;
}

void Chefsi_control::init(const Input_file& input_file) {
    if (input_file.map.find("MAXIT_CHEFSI") != input_file.map.end()) {
        this->set_max_iter(std::stoi(input_file.get_value("MAXIT_CHEFSI")));
    }
    if (input_file.map.find("CHEB_DEGREE") != input_file.map.end()) {
        this->set_chebyshev_filter_degree(std::stoi(input_file.get_value("CHEB_DEGREE")));
    }
    if (input_file.map.find("RHO_TRIGGER") != input_file.map.end()) {
        this->set_rho_trigger(std::stoi(input_file.get_value("RHO_TRIGGER")));
    }
    if (input_file.map.find("TOL_LANCZOS") != input_file.map.end()) {
        this->set_tolerance_lanczos(std::stod(input_file.get_value("TOL_LANCZOS")));
    }
    if (input_file.map.find("CHEFSI_PROJECTION_METHOD") != input_file.map.end()) {
        this->set_projection_method(std::stoi(input_file.get_value("CHEFSI_PROJECTION_METHOD")));
        #if !(defined(USE_MKL) || defined(USE_SCALAPACK))
            if (this->projection_method == 0
             || this->projection_method == 1) {
                assert(!"To use scalapack functions should turn on USE_MKL or USE_SCALAPACK, when CHEFSI_PROJECTION_METHOD == 0 || CHEFSI_PROJECTION_METHOD == 1");
            }
        #endif
    }
    return;
}

void Chefsi_control::init(const Chefsi_control& chefsi_control) {
    this->max_iter = chefsi_control.max_iter;
    this->chebyshev_filter_degree = chefsi_control.chebyshev_filter_degree;
    this->rho_trigger = chefsi_control.rho_trigger;
    this->projection_method = chefsi_control.projection_method;
    this->tolerance_lanczos = chefsi_control.tolerance_lanczos;
    this->abstol = chefsi_control.abstol;
    this->orfac = chefsi_control.orfac;
    return;
}

MPI_Datatype Chefsi_control::register_mpi_type() const {
    MPI_Datatype type;
    constexpr std::size_t num_members = 7;
    int lengths[num_members] = {1, 1, 1, 1, 1, 1, 1};
    MPI_Aint offsets[num_members] = {offsetof(Chefsi_control, max_iter),
                                     offsetof(Chefsi_control, chebyshev_filter_degree),
                                     offsetof(Chefsi_control, rho_trigger),
                                     offsetof(Chefsi_control, projection_method),
                                     offsetof(Chefsi_control, tolerance_lanczos),
                                     offsetof(Chefsi_control, abstol),
                                     offsetof(Chefsi_control, orfac)};
    MPI_Datatype types[num_members] = {MPI_UNSIGNED, MPI_UNSIGNED,
                                       MPI_UNSIGNED, MPI_UNSIGNED,
                                       MPI_DOUBLE, MPI_DOUBLE,
                                       MPI_DOUBLE};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    return type;
}

void Chefsi_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Chefsi_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype Chefsi_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Chefsi_control_type, root, comm);
    this->deregister_mpi_type(Chefsi_control_type);
    return;
}

void Chefsi_control::show() const {
    std::cout << "max_iter = " << this->max_iter << std::endl;
    std::cout << "chebyshev_filter_degree = " << this->chebyshev_filter_degree << std::endl;
    std::cout << "rho_trigger = " << this->rho_trigger << std::endl;
    if (this->projection_method == 0) std::cout << "projection_method = 0, which means DP_SUBEIG method." << std::endl;
    if (this->projection_method == 1) std::cout << "projection_method = 1, which means scalapack for projection and diagonalization." << std::endl;
    if (this->projection_method == 2) std::cout << "projection_method = 2, which means scalapack diagonalization only, projection use DP_SUBEIG." << std::endl;
    std::cout << "tolerance_lanczos = " << std::setw(12) << std::setprecision(3) << std::scientific << std::left
              << this->tolerance_lanczos << std::endl;
    std::cout << "abstol = " << std::setw(12) << std::setprecision(3) << std::scientific << std::left
              << this->abstol << std::endl;
    std::cout << "orfac = " << std::setw(12) << std::setprecision(3) << std::scientific << std::left
              << this->orfac << std::endl;
    return;
}

void Chefsi_control::print(std::ostream& output) const {
    output << "MAXIT_CHEFSI: " << this->max_iter << std::endl;
    output << "CHEB_DEGREE: " << this->chebyshev_filter_degree << std::endl;
    output << "RHO_TRIGGER: " << this->rho_trigger << std::endl;
    output << "TOL_LANCZOS: " << std::scientific << std::setprecision(3) << this->tolerance_lanczos << std::endl;
    output << "CHEFSI_PROJECTION_METHOD: " << this->projection_method << std::endl;
    return;
}

Eigen_solver_control::Eigen_solver_control() {}

Eigen_solver_control::Eigen_solver_control(const Eigen_solver_control& other) {
    *this = other;
}

Eigen_solver_control::~Eigen_solver_control() {}

Eigen_solver_control& Eigen_solver_control::operator=(const Eigen_solver_control& other) {
    this->method = other.method;
    this->nstates = other.nstates;
    this->is_rand_fixed = other.is_rand_fixed;
    this->chefsi_control = other.chefsi_control;
    return *this;
}

void Eigen_solver_control::set_method(const uint& method) {
    this->method = method;
    return;
}

void Eigen_solver_control::set_nstates(const uint& nstates) {
    this->nstates = nstates;
    return;
}

void Eigen_solver_control::set_is_rand_fixed(const bool& is_rand_fixed) {
    this->is_rand_fixed = is_rand_fixed;
    return;
}

void Eigen_solver_control::init(const Input_file& input_file) {
    if (input_file.map.find("EIGEN_SOLVER") != input_file.map.end()) {
        std::string str = Filesys::toupper(input_file.get_value("EIGEN_SOLVER"));
        if (str == "CHEFSI") {
            this->set_method(0);
        } else {
            assert(!"ERROR:: EIGEN_SOLVER should be CHEFSI!");
        }
    }
    if (input_file.map.find("NSTATES") != input_file.map.end()) {
        this->set_nstates(std::stoi(input_file.get_value("NSTATES")));
    }
    if (input_file.map.find("EIGEN_FIX_RAND") != input_file.map.end()) {
        this->set_is_rand_fixed(std::stoi(input_file.get_value("POISSON_FIX_RAND")));
    } else if (input_file.map.find("FIX_RAND") != input_file.map.end()) {
        this->set_is_rand_fixed(std::stoi(input_file.get_value("FIX_RAND")));
    }
    if (this->method == 0) {
        this->chefsi_control.init(input_file);
    }
    return;
}

void Eigen_solver_control::init(const Eigen_solver_control& eigen_solver_control) {
    this->method = eigen_solver_control.method;
    this->nstates = eigen_solver_control.nstates;
    this->is_rand_fixed = eigen_solver_control.is_rand_fixed;
    this->chefsi_control.init(eigen_solver_control.chefsi_control);
    return;
}

MPI_Datatype Eigen_solver_control::register_mpi_type() const {
    MPI_Datatype type;
    MPI_Datatype Chefsi_control_type = this->chefsi_control.register_mpi_type();
    constexpr std::size_t num_members = 4;
    int lengths[num_members] = {1, 1, 1, 1};
    MPI_Aint offsets[num_members] = {offsetof(Eigen_solver_control, method),
                                     offsetof(Eigen_solver_control, nstates),
                                     offsetof(Eigen_solver_control, is_rand_fixed),
                                     offsetof(Eigen_solver_control, chefsi_control)};
    MPI_Datatype types[num_members] = {MPI_UNSIGNED, MPI_UNSIGNED,
                                       MPI_C_BOOL, Chefsi_control_type};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    this->chefsi_control.deregister_mpi_type(Chefsi_control_type);
    return type;
}

void Eigen_solver_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Eigen_solver_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype Eigen_solver_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Eigen_solver_control_type, root, comm);
    this->deregister_mpi_type(Eigen_solver_control_type);
    return;
}

void Eigen_solver_control::show() const {
    if (this->method == 0) std::cout << "method = 0, which means CHEFSI method." << std::endl;
    std::cout << "nstates = " << this->nstates << std::endl;
    std::cout << "is_rand_fixed = " << this->is_rand_fixed << std::endl;
    this->chefsi_control.show();
    return;
}

void Eigen_solver_control::print(std::ostream& output) const {
    output << "EIGEN_SOLVER: ";
    if (this->method == 0) {
        output << "CHEFSI" << std::endl;
        this->chefsi_control.print(output);
    } else {
        assert(!"ERROR:: EIGEN_SOLVER should be CHEFSI!");
    }
    output << "NSTATES: " << this->nstates << std::endl;
    output << "EIGEN_FIX_RAND: " << this->is_rand_fixed << std::endl;
    return;
}

Spinor_eigen_solver_control::Spinor_eigen_solver_control() {}

Spinor_eigen_solver_control::~Spinor_eigen_solver_control() {}

void Spinor_eigen_solver_control::set_method(const uint& method) {
    this->method = method;
    return;
}

void Spinor_eigen_solver_control::set_delta(const double& delta) {
    this->delta = delta;
    return;
}

void Spinor_eigen_solver_control::init(const Input_file& input_file) {
    if (input_file.map.find("SPINOR_EIGEN_METHOD") != input_file.map.end()) {
        this->set_method(std::stoi(input_file.get_value("SPINOR_EIGEN_METHOD")));
    }
    if (input_file.map.find("SPINOR_DELTA") != input_file.map.end()) {
        this->set_delta(std::stod(input_file.get_value("SPINOR_DELTA")));
    }
    return;
}

MPI_Datatype Spinor_eigen_solver_control::register_mpi_type() const {
    MPI_Datatype type;
    constexpr std::size_t num_members = 2;
    int lengths[num_members] = {1, 1};
    MPI_Aint offsets[num_members] = {offsetof(Spinor_eigen_solver_control, method),
                                     offsetof(Spinor_eigen_solver_control, delta)};
    MPI_Datatype types[num_members] = {MPI_UNSIGNED, MPI_DOUBLE};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    return type;
}

void Spinor_eigen_solver_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Spinor_eigen_solver_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype Spinor_eigen_solver_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Spinor_eigen_solver_control_type, root, comm);
    this->deregister_mpi_type(Spinor_eigen_solver_control_type);
    return;
}

void Spinor_eigen_solver_control::show() const {
    if (this->method == 0) std::cout << "Spinor_eigen_solver_method = 0, which means subspace diagonalization method." << std::endl;
    if (this->method != 0) std::cout << "Spinor_eigen_solver_method != 0, which means XXXX method." << std::endl;
    std::cout << "Spinor_eigen_solver delta = " << this->delta << std::endl;
    return;
}

Xlsdft_control::Xlsdft_control() {}

Xlsdft_control::~Xlsdft_control() {}

void Xlsdft_control::set_element_comm_numbers(const uint& comm_ni, const uint& comm_nj, const uint& comm_nk) {
    this->element_comm_numbers[0] = comm_ni;
    this->element_comm_numbers[1] = comm_nj;
    this->element_comm_numbers[2] = comm_nk;
    return;
}

void Xlsdft_control::set_element_numbers(const uint& ni, const uint& nj, const uint& nk) {
    this->element_numbers[0] = ni;
    this->element_numbers[1] = nj;
    this->element_numbers[2] = nk;
    return;
}

void Xlsdft_control::set_basis_per_atom(const double& basis_per_atom) {
    this->basis_per_atom = basis_per_atom;
    return;
}

void Xlsdft_control::set_buffers(const double& buffer_x, const double& buffer_y, const double& buffer_z) {
    this->buffers[0] = buffer_x;
    this->buffers[1] = buffer_y;
    this->buffers[2] = buffer_z;
    return;
}

void Xlsdft_control::set_is_PBCs(const bool& is_x_PBC, const bool& is_y_PBC, const bool& is_z_PBC) {
    this->is_periodic[0] = is_x_PBC;
    this->is_periodic[1] = is_y_PBC;
    this->is_periodic[2] = is_z_PBC;
    return;
}

void Xlsdft_control::set_is_rand_fixed(const bool& is_rand_fixed) {
    this->is_rand_fixed = is_rand_fixed;
    return;
}

void Xlsdft_control:: set_element_nstates(const uint& element_nstates) {
    this->element_nstates = element_nstates;
    return;
}

void Xlsdft_control::init(const Input_file& input_file, const std::string& prefix) {
    std::stringstream ss;
    if (input_file.map.find(prefix + "_NELEMS") != input_file.map.end()) {
        ss.str(input_file.get_value(prefix + "_NELEMS"));
        uint ni, nj, nk;
        ss >> ni >> nj >> nk;
        this->set_element_numbers(ni, nj, nk);
        ss.clear();
    }
    if (input_file.map.find(prefix + "_NCOMMS") != input_file.map.end()) {
        ss.str(input_file.get_value(prefix + "_NCOMMS"));
        uint temp;
        std::vector<uint> temps;
        while (ss >> temp) {
            temps.emplace_back(temp);
        }
        ss.clear();
        if (temps.size() == 3) {
            assert(temps[0] <= this->element_numbers[0]);
            assert(temps[1] <= this->element_numbers[1]);
            assert(temps[2] <= this->element_numbers[2]);
            this->set_element_comm_numbers(temps[0], temps[1], temps[2]);
        } else if (temps.size() == 1) {
            int np = (int)temps[0];
            assert(np <= (int)(this->element_numbers[0] * this->element_numbers[1] * this->element_numbers[2]));
            assert((int)(this->element_numbers[0] * this->element_numbers[1] * this->element_numbers[2]) % np == 0);
            int element_number_per_comm = (int)(this->element_numbers[0] * this->element_numbers[1] * this->element_numbers[2]) / np;
            int comm_ni = 1;
            int comm_nj = 1;
            int comm_nk = 1;
            if (np == 1) {
            } else if (element_number_per_comm == 1) {
                comm_ni = this->element_numbers[0];
                comm_nj = this->element_numbers[1];
                comm_nk = this->element_numbers[2];
            } else if (this->element_numbers[0] % element_number_per_comm == 0) {
                comm_ni = this->element_numbers[0] / element_number_per_comm;
            } else if (this->element_numbers[1] % element_number_per_comm == 0) {
                comm_nj = this->element_numbers[1] / element_number_per_comm;
            } else if (this->element_numbers[2] % element_number_per_comm == 0) {
                comm_nk = this->element_numbers[2] / element_number_per_comm;
            }  else {
                assert(!"Need to distribute the element_numbers manually!");
            }
            assert(comm_ni * comm_nj * comm_nk == np);
            this->set_element_comm_numbers(comm_ni, comm_nj, comm_nk);
        } else {
            assert(temps.size() == 3 || temps.size() == 1);
        }
    }
    if (input_file.map.find(prefix + "_BUFFERS") != input_file.map.end()) {
        ss.str(input_file.get_value(prefix + "_BUFFERS"));
        double temp;
        std::vector<double> temps;
        while (ss >> temp) {
            temps.emplace_back(temp);
        }
        ss.clear();
        if (temps.size() == 3) {
            this->set_buffers(temps[0], temps[1], temps[2]);
        } else if (temps.size() == 1) {
            this->set_buffers(temps[0], temps[0], temps[0]);
        } else {
            assert(temps.size() == 3 || temps.size() == 1);
        }
    }
    if (input_file.map.find(prefix + "_BC") != input_file.map.end()) {
        ss.str(input_file.get_value(prefix + "_BC"));
        char BCx, BCy, BCz;
        ss >> BCx >> BCy >> BCz;
        ss.clear();
        if (BCx == 'P' || BCx == 'p' || BCx == '1') {
            this->is_periodic[0] = true;
        } else if (BCx == 'D' || BCx == 'd' || BCx == '0') {
            this->is_periodic[0] = false;
        } else {
            assert(!"_BC should be P or D");
        }
        if (BCy == 'P' || BCy == 'p' || BCy == '1') {
            this->is_periodic[1] = true;
        } else if (BCy == 'D' || BCy == 'd' || BCy == '0') {
            this->is_periodic[1] = false;
        } else {
            assert(!"_BC should be P or D");
        }
        if (BCz == 'P' || BCz == 'p' || BCz == '1') {
            this->is_periodic[2] = true;
        } else if (BCz == 'D' || BCz == 'd' || BCz == '0') {
            this->is_periodic[2] = false;
        } else {
            assert(!"_BC should be P or D");
        }
    }
    if (input_file.map.find(prefix + "_NSTATES") != input_file.map.end()) {
        this->set_element_nstates(std::stoi(input_file.get_value(prefix + "_NSTATES")));
    }
    if (input_file.map.find(prefix + "_BASIS_PER_ELECTRON") != input_file.map.end()) {
        double basis_per_atom;
        basis_per_atom = std::stod(input_file.get_value(prefix + "_BASIS_PER_ELECTRON"));
        this->set_basis_per_atom(basis_per_atom);
    }
    if (input_file.map.find(prefix + "_FIX_RAND") != input_file.map.end()) {
        this->set_is_rand_fixed(std::stoi(input_file.get_value(prefix + "_FIX_RAND")));
    } else if (input_file.map.find("FIX_RAND") != input_file.map.end()) {
        this->set_is_rand_fixed(std::stoi(input_file.get_value("FIX_RAND")));
    }
    this->chefsi_control.init(input_file);
    return;
}

MPI_Datatype Xlsdft_control::register_mpi_type() const {
    MPI_Datatype type;
    MPI_Datatype Chefsi_control_type = this->chefsi_control.register_mpi_type();
    constexpr std::size_t num_members = 8;
    int lengths[num_members] = {3, 1, 3, 3, 1, 1, 3, 1};
    MPI_Aint offsets[num_members] = {offsetof(Xlsdft_control, is_periodic),
                                     offsetof(Xlsdft_control, is_rand_fixed),
                                     offsetof(Xlsdft_control, element_numbers),
                                     offsetof(Xlsdft_control, element_comm_numbers),
                                     offsetof(Xlsdft_control, element_nstates),
                                     offsetof(Xlsdft_control, basis_per_atom),
                                     offsetof(Xlsdft_control, buffers),
                                     offsetof(Xlsdft_control, chefsi_control)};
    MPI_Datatype types[num_members] = {MPI_C_BOOL, MPI_C_BOOL,
                                        MPI_UNSIGNED, MPI_UNSIGNED,
                                        MPI_UNSIGNED, MPI_DOUBLE,
                                        MPI_DOUBLE, Chefsi_control_type};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    this->chefsi_control.deregister_mpi_type(Chefsi_control_type);
    return type;
}

void Xlsdft_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Xlsdft_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype Xlsdft_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Xlsdft_control_type, root, comm);
    this->deregister_mpi_type(Xlsdft_control_type);
    return;
}

void Xlsdft_control::show() const {
    std::cout << "is_periodic = [" 
            << std::setw(8) << std::right << this->is_periodic[0] << " , "
            << std::setw(8) << std::right << this->is_periodic[1] << " , " 
            << std::setw(8) << std::right << this->is_periodic[2] << " ]"
            << std::endl;
    std::cout << "is_rand_fixed = " << std::fixed << this->is_rand_fixed << std::endl;
    std::cout << "element_numbers = ["
            << std::setw(8) << std::right << this->element_numbers[0] << " , "
            << std::setw(8) << std::right << this->element_numbers[1] << " , "
            << std::setw(8) << std::right << this->element_numbers[2] << " ]"
            << std::endl;
    std::cout << "element_comm_numbers = ["
            << std::setw(8) << std::right << this->element_comm_numbers[0] << " , "
            << std::setw(8) << std::right << this->element_comm_numbers[1] << " , "
            << std::setw(8) << std::right << this->element_comm_numbers[2] << " ]"
            << std::endl;
    std::cout << "element_nstates = " << std::fixed << this->element_nstates << std::endl;
    std::cout << "basis_per_atom = " << std::fixed << this->basis_per_atom << std::endl;
    std::cout << "buffers = [" 
            << std::setw(8) << std::right << this->buffers[0] << " , "
            << std::setw(8) << std::right << this->buffers[1] << " , " 
            << std::setw(8) << std::right << this->buffers[2] << " ]"
            << std::endl;
    this->chefsi_control.show();
    return;
}

void Xlsdft_control::print(std::ostream& output, const std::string& prefix) const {
    output << prefix + "_NELEMS: " << this->element_numbers[0] << " " << this->element_numbers[1] << " " << this->element_numbers[2] << std::endl;
    output << prefix + "_NCOMMS: " << this->element_comm_numbers[0] << " " << this->element_comm_numbers[1] << " " << this->element_comm_numbers[2] << std::endl;
    output << prefix + "_BUFFERS: " << std::fixed << std::setprecision(3) << this->buffers[0] << " " << this->buffers[1] << " " << this->buffers[2] << std::endl;
    output << prefix + "_BC: " 
           << (this->is_periodic[0] ? "P" : "D") << " "
           << (this->is_periodic[1] ? "P" : "D") << " "
           << (this->is_periodic[2] ? "P" : "D") << std::endl;
    if (this->element_nstates > 0) {
        output << prefix + "_NSTATES: " << this->element_nstates << std::endl;
    } else {
        output << prefix + "_BASIS_PER_ELECTRON: " << this->basis_per_atom << std::endl;
    }
    output << prefix + "_FIX_RAND: " << this->is_rand_fixed << std::endl;
    this->chefsi_control.print(output);
    return;
}

Density_matrix_solver_control::Density_matrix_solver_control() {}

Density_matrix_solver_control::~Density_matrix_solver_control() {}

void Density_matrix_solver_control::set_method(const uint& method) {
    this->method = method;
    return;
}

void Density_matrix_solver_control::set_is_rand_fixed(const bool& is_rand_fixed) {
    this->is_rand_fixed = is_rand_fixed;
    return;
}

void Density_matrix_solver_control::init(const Input_file& input_file) {
    if (input_file.map.find("DENSITY_MATRIX_SOLVER") != input_file.map.end()) {
        std::string str = Filesys::toupper(input_file.get_value("DENSITY_MATRIX_SOLVER"));
        if (str == "XLS" || str == "XLSDFT") {
            this->set_method(0);
        } else {
            assert(!"ERROR:: DENSITY_MATRIX_SOLVER should be XLSDFT!");
        }
    }
    if (input_file.map.find("DENSITY_MATRIX_FIX_RAND") != input_file.map.end()) {
        this->set_is_rand_fixed(std::stoi(input_file.get_value("DENSITY_MATRIX_FIX_RAND")));
    } else if (input_file.map.find("FIX_RAND") != input_file.map.end()) {
        this->set_is_rand_fixed(std::stoi(input_file.get_value("FIX_RAND")));
    }
    if (this->method == 0) {
        this->xlsdft_control.init(input_file, "XLSDFT");
    }
    return;
}

MPI_Datatype Density_matrix_solver_control::register_mpi_type() const {
    MPI_Datatype type;
    MPI_Datatype Xlsdft_control_type = this->xlsdft_control.register_mpi_type();
    constexpr std::size_t num_members = 3;
    int lengths[num_members] = {1, 1, 1};
    MPI_Aint offsets[num_members] = {offsetof(Density_matrix_solver_control, method),
                                     offsetof(Density_matrix_solver_control, is_rand_fixed),
                                     offsetof(Density_matrix_solver_control, xlsdft_control)};
    MPI_Datatype types[num_members] = {MPI_UNSIGNED,  MPI_C_BOOL,
                                       Xlsdft_control_type};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    this->xlsdft_control.deregister_mpi_type(Xlsdft_control_type);
    return type;
}

void Density_matrix_solver_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Density_matrix_solver_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype Density_matrix_solver_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Density_matrix_solver_control_type, root, comm);
    this->deregister_mpi_type(Density_matrix_solver_control_type);
    return;
}

void Density_matrix_solver_control::show() const {
    if (this->method == 0) std::cout << "method = 0, which means XLSDFT method." << std::endl;
    std::cout << "is_rand_fixed = " << this->is_rand_fixed << std::endl;
    this->xlsdft_control.show();
    return;
}

void Density_matrix_solver_control::print(std::ostream& output) const {
    output << "DENSITY_MATRIX_SOLVER: ";
    if (this->method == 0) {
        output << "XLSDFT" << std::endl;
        this->xlsdft_control.print(output);
    } else {
        assert(!"ERROR:: DENSITY_MATRIX_SOLVER should be XLSDFT!");
    }
    output << "DENSITY_MATRIX_FIX_RAND: " << this->is_rand_fixed << std::endl;
    return;
}

Smearing_control::Smearing_control() {}

Smearing_control::~Smearing_control() {}

void Smearing_control::set_method(const uint& method) {
    this->method = method;
    return;
}

void Smearing_control::set_beta(const double& beta) {
    this->beta = beta;
    return;
}

const double& Smearing_control::get_beta() const {
    return this->beta;
}

void Smearing_control::init(const Input_file& input_file) {
    if (input_file.map.find("ELEC_TEMP_TYPE") != input_file.map.end()) {
        std::string str = Filesys::toupper(input_file.get_value("ELEC_TEMP_TYPE"));
        if (str == "FERMI-DIRAC" || str == "FD") {
            this->set_method(0);
        } else if (str == "GAUSSIAN") {
            this->set_method(1);
        } else {
            assert(!"ERROR:: ELEC_TEMP_TYPE should be Fermi-Dirac or gaussian!");
        }
    }
    if (input_file.map.find("SMEARING") != input_file.map.end()) {
        this->set_beta(1.0 / std::stod(input_file.get_value("SMEARING")));
    } else if (input_file.map.find("ELEC_TEMP") != input_file.map.end()) {
        this->set_beta(1.0 / (CONST_KB * std::stod(input_file.get_value("ELEC_TEMP"))));
    }
    return;
}

MPI_Datatype Smearing_control::register_mpi_type() const {
    MPI_Datatype type;
    constexpr std::size_t num_members = 2;
    int lengths[num_members] = {1, 1};
    MPI_Aint offsets[num_members] = {offsetof(Smearing_control, method),
                                     offsetof(Smearing_control, beta)};
    MPI_Datatype types[num_members] = {MPI_UNSIGNED, MPI_DOUBLE};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    return type;
}

void Smearing_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Smearing_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype smearing_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, smearing_control_type, root, comm);
    this->deregister_mpi_type(smearing_control_type);
    return;
}

void Smearing_control::show() const {
    if (this->method == 0) {
        std::cout << "method = 0, which means Fermi-Dirac smearing method." << std::endl;
    } else if (this->method == 1) {
        std::cout << "method = 1, which means Gaussain smearing method." << std::endl;
    }  else {
        assert(!"ERROR:: ELEC_TEMP_TYPE should be Fermi-Dirac or gaussian!");
    }
    std::cout << "beta = " << this->beta << std::endl;
    return;
}

Density_solver_control::Density_solver_control() {}

Density_solver_control::~Density_solver_control() {}

void Density_solver_control::set_method(const uint& method) {
    this->method = method;
    return;
}

void Density_solver_control::set_if_cal_spinor(const bool& if_cal_spinor) {
    this->if_cal_spinor = if_cal_spinor;
    return;
}

void Density_solver_control::set_if_cal_oneshot(const bool& if_cal_oneshot) {
    this->if_cal_oneshot = if_cal_oneshot;
    return;
}

void Density_solver_control::init(const Input_file& input_file) {
    if (input_file.map.find("XLSDFT_FLAG") != input_file.map.end()) {
        if (std::stoi(input_file.get_value("XLSDFT_FLAG")) == 0) {
            this->set_method(0);
        } else if (std::stoi(input_file.get_value("XLSDFT_FLAG")) == 1) {
            this->set_method(1);
        } else {
            assert(!"XLSDFT_FLAG should be 0 or 1");
        }
    } else if (input_file.map.find("DENSITY_SOLVER") != input_file.map.end()) {
        std::string str = Filesys::toupper(input_file.get_value("DENSITY_SOLVER"));
        if (str == "EIGEN") {
            this->set_method(0);
        } else if (str == "DENSITY_MATRIX") {
            this->set_method(1);
        } else {
            assert(!"DENSITY_SOLVER should be EIGEN or DENSITY_MATRIX");
        }
    }
    if (input_file.map.find("CAL_SPINOR") != input_file.map.end()) {
        this->set_if_cal_spinor(std::stoi(input_file.get_value("CAL_SPINOR")));
    }
    if (input_file.map.find("CAL_ONESHOT") != input_file.map.end()) {
        this->set_if_cal_oneshot(std::stoi(input_file.get_value("CAL_ONESHOT")));
    }
    if (this->method == 0) {
        this->eigen_solver_control.init(input_file);
    } else if (this->method == 1) {
        this->density_matrix_solver_control.init(input_file);
    }
    this->smearing_control.init(input_file);
    if (this->if_cal_spinor) {
        this->spinor_eigen_solver_control.init(input_file);
    }
    return;
}

MPI_Datatype Density_solver_control::register_mpi_type() const {
    MPI_Datatype type;
    MPI_Datatype Eigen_solver_control_type = this->eigen_solver_control.register_mpi_type();
    MPI_Datatype Spinor_eigen_solver_control_type = this->spinor_eigen_solver_control.register_mpi_type();
    MPI_Datatype Density_matrix_solver_control_type = this->density_matrix_solver_control.register_mpi_type();
    MPI_Datatype Smearing_control_type = this->smearing_control.register_mpi_type();
    constexpr std::size_t num_members = 7;
    int lengths[num_members] = {1, 1, 1, 1, 1, 1, 1};
    MPI_Aint offsets[num_members] = {offsetof(Density_solver_control, method),
                                     offsetof(Density_solver_control, if_cal_spinor),
                                     offsetof(Density_solver_control, if_cal_oneshot),
                                     offsetof(Density_solver_control, eigen_solver_control),
                                     offsetof(Density_solver_control, spinor_eigen_solver_control),
                                     offsetof(Density_solver_control, density_matrix_solver_control),
                                     offsetof(Density_solver_control, smearing_control)};
    MPI_Datatype types[num_members] = {MPI_UNSIGNED,
                                       MPI_C_BOOL,
                                       MPI_C_BOOL,
                                       Eigen_solver_control_type,
                                       Spinor_eigen_solver_control_type,
                                       Density_matrix_solver_control_type,
                                       Smearing_control_type};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    this->eigen_solver_control.deregister_mpi_type(Eigen_solver_control_type);
    this->spinor_eigen_solver_control.deregister_mpi_type(Spinor_eigen_solver_control_type);
    this->density_matrix_solver_control.deregister_mpi_type(Density_matrix_solver_control_type);
    this->smearing_control.deregister_mpi_type(Smearing_control_type);
    return type;
}

void Density_solver_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Density_solver_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype Density_solver_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Density_solver_control_type, root, comm);
    this->deregister_mpi_type(Density_solver_control_type);
    return;
}

void Density_solver_control::show() const {
    if (this->method == 0) {
        std::cout << "method = 0, which means eigensolver method." << std::endl;
        this->eigen_solver_control.show();
    } else if (this->method == 1) {
        std::cout << "method = 1, which means density matrix method." << std::endl;
        this->density_matrix_solver_control.show();
    }
    this->smearing_control.show();
    if (this->if_cal_spinor) {
        std::cout << "if_cal_spinor = 1, which means calculate spinor." << std::endl;
        this->spinor_eigen_solver_control.show();
    }
    return;
}

void Density_solver_control::print(std::ostream& output) const {
    output << "DENSITY_SOLVER: ";
    if (this->method == 0) {
        output << "EIGEN" << std::endl;
        this->eigen_solver_control.print(output);
    } else if (this->method == 1) {
        output << "DENSITY_MATRIX" << std::endl;
        this->density_matrix_solver_control.print(output);
    }
    // output << "CAL_SPINOR = " << this->if_cal_spinor << std::endl;
    // output << "CAL_ONESHOT = " << this->if_cal_oneshot << std::endl;
    //  if (this->if_cal_spinor) {
    //     this->spinor_eigen_solver_control.print(output);
    // }
    return;
}

Misc_control::Misc_control() {}

Misc_control::~Misc_control() {}

void Misc_control::set_if_print_forces(const bool& if_print_forces) {
    this->if_print_forces = if_print_forces;
    return;
}

void Misc_control::set_if_print_eigen(const bool& if_print_eigen) {
    this->if_print_eigen = if_print_eigen;
    return;
}

void Misc_control::set_if_print_density(const bool& if_print_density) {
    this->if_print_density = if_print_density;
    return;
}

void Misc_control::set_if_read_density(const bool& if_read_density) {
    this->if_read_density = if_read_density;
    return;
}

void Misc_control::set_if_print_pdos(const bool& if_print_pdos) {
    this->if_print_pdos = if_print_pdos;
    return;
}

void Misc_control::set_if_print_atoms(const bool& if_print_atoms) {
    this->if_print_atoms = if_print_atoms;
    return;
}

void Misc_control::init(const Input_file& input_file) {
    if (input_file.map.find("PRINT_FORCES") != input_file.map.end()) {
        this->set_if_print_forces(std::stoi(input_file.get_value("PRINT_FORCES")));
    }
    if (input_file.map.find("PRINT_EIGEN") != input_file.map.end()) {
        this->set_if_print_eigen(std::stoi(input_file.get_value("PRINT_EIGEN")));
    }
    if (input_file.map.find("PRINT_DENSITY") != input_file.map.end()) {
        this->set_if_print_density(std::stoi(input_file.get_value("PRINT_DENSITY")));
    }
    if (input_file.map.find("PRINT_ATOMS") != input_file.map.end()) {
        this->set_if_print_atoms(std::stoi(input_file.get_value("PRINT_ATOMS")));
    }
    if (input_file.map.find("PRINT_PDOS") != input_file.map.end()) {
        this->set_if_print_pdos(std::stoi(input_file.get_value("PRINT_PDOS")));
    }
    if (input_file.map.find("READ_DENSITY") != input_file.map.end()) {
        this->set_if_read_density(std::stoi(input_file.get_value("READ_DENSITY")));
    }
    return;
}

MPI_Datatype Misc_control::register_mpi_type() const {
    MPI_Datatype type;
    constexpr std::size_t num_members = 6;
    int lengths[num_members] = {1, 1, 1, 1, 1, 1};
    MPI_Aint offsets[num_members] = {offsetof(Misc_control, if_print_forces),
                                     offsetof(Misc_control, if_print_eigen),
                                     offsetof(Misc_control, if_print_density),
                                     offsetof(Misc_control, if_print_atoms),
                                     offsetof(Misc_control, if_print_pdos),
                                     offsetof(Misc_control, if_read_density)};
    MPI_Datatype types[num_members] = {MPI_C_BOOL, MPI_C_BOOL,
                                       MPI_C_BOOL, MPI_C_BOOL,
                                       MPI_C_BOOL, MPI_C_BOOL};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    return type;
}

void Misc_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Misc_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype Misc_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Misc_control_type, root, comm);
    this->deregister_mpi_type(Misc_control_type);
    return;
}

void Misc_control::show() const {
    std::cout << "if_print_forces = " << this->if_print_forces << std::endl;
    std::cout << "if_print_eigen = " << this->if_print_eigen << std::endl;
    std::cout << "if_print_density = " << this->if_print_density << std::endl;
    std::cout << "if_print_atoms = " << this->if_print_atoms << std::endl;
    std::cout << "if_print_pdos = " << this->if_print_atoms << std::endl;
    std::cout << "if_read_density = " << this->if_read_density << std::endl;
    return;
}

void Misc_control::print(std::ostream& output) const {
    output << "PRINT_FORCES: " << this->if_print_forces << std::endl;
    output << "PRINT_EIGEN: " << this->if_print_eigen << std::endl;
    output << "PRINT_DENSITY: " << this->if_print_density << std::endl;
    output << "PRINT_ATOMS: " << this->if_print_atoms << std::endl;
    output << "PRINT_PDOS: " << this->if_print_pdos << std::endl;
    output << "READ_DENSITY: " << this->if_read_density << std::endl;
    return;
}

Flow_control::Flow_control() {}

Flow_control::~Flow_control() {}

void Flow_control::set_scf_flag(const bool& scf_flag) {
    this->scf_flag = scf_flag;
    return;
}

void Flow_control::set_md_flag(const bool& md_flag) {
    this->md_flag = md_flag;
    return;
}

void Flow_control::set_relax_flag(const bool& relax_flag) {
    this->relax_flag = relax_flag;
    return;
}

void Flow_control::init(const Input_file& input_file) {
    if (input_file.map.find("SCF_FLAG") != input_file.map.end()) {
        this->set_scf_flag(std::stoi(input_file.get_value("SCF_FLAG")));
    }
    if (input_file.map.find("MD_FLAG") != input_file.map.end()) {
        this->set_md_flag(std::stoi(input_file.get_value("MD_FLAG")));
    }
    if (input_file.map.find("RELAX_FLAG") != input_file.map.end()) {
        this->set_relax_flag(std::stoi(input_file.get_value("RELAX_FLAG")));
    }
    return;
}

MPI_Datatype Flow_control::register_mpi_type() const {
    MPI_Datatype type;
    constexpr std::size_t num_members = 3;
    int lengths[num_members] = {1, 1, 1};
    MPI_Aint offsets[num_members] = {offsetof(Flow_control, scf_flag),
                                     offsetof(Flow_control, md_flag),
                                     offsetof(Flow_control, relax_flag)};
    MPI_Datatype types[num_members] = {MPI_C_BOOL, MPI_C_BOOL, MPI_C_BOOL};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    return type;
}

void Flow_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Flow_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype Flow_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Flow_control_type, root, comm);
    this->deregister_mpi_type(Flow_control_type);
    return;
}

void Flow_control::show() const {
    std::cout << "scf_flag = " << this->scf_flag << std::endl;
    std::cout << "md_flag = " << this->md_flag << std::endl;
    std::cout << "relax_flag = " << this->relax_flag << std::endl;
    return;
}

void Flow_control::print(std::ostream& output) const {
    output << "SCF_FLAG: " << this->scf_flag << std::endl;
    output << "MD_FLAG: " << this->md_flag << std::endl;
    output << "RELAX_FLAG: " << this->relax_flag << std::endl;
    return;
}

Stencil_control::Stencil_control() {}

Stencil_control::~Stencil_control() {}

void Stencil_control::set_order(const int& order) {
    this->order = order;
    return;
}

void Stencil_control::init(const Input_file& input_file) {
    if (input_file.map.find("FD_ORDER") != input_file.map.end()) {
        this->set_order(std::stoi(input_file.get_value("FD_ORDER")));
    }
    return;
}

MPI_Datatype Stencil_control::register_mpi_type() const {
    MPI_Datatype type;
    constexpr std::size_t num_members = 1;
    int lengths[num_members] = {1};
    MPI_Aint offsets[num_members] = {offsetof(Stencil_control, order)};
    MPI_Datatype types[num_members] = {MPI_INT};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    return type;
}

void Stencil_control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Stencil_control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype Stencil_control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Stencil_control_type, root, comm);
    this->deregister_mpi_type(Stencil_control_type);
    return;
}

void Stencil_control::show() const {
    std::cout << "order = " << this->order << std::endl;
    return;
}

void Stencil_control::print(std::ostream& output) const {
    output << "FD_ORDER: " << this->order << std::endl;
    return;
}

Aar_control::Aar_control() {}

Aar_control::~Aar_control() {}

void Aar_control::set_precondition_method(const uint& precondition_method) {
    this->precondition_method = precondition_method;
    return;
}

void Aar_control::set_residual_method(const uint& residual_method) {
    this->residual_method = residual_method;
    return;
}

void Aar_control::set_anderson_beta(const double& anderson_beta) {
    this->anderson_beta = anderson_beta;
    return;
}

void Aar_control::set_richardson_omega(const double& richardson_omega) {
    this->richardson_omega = richardson_omega;
    return;
}

void Aar_control::set_mixing_history(const int& mixing_history) {
    this->mixing_history = mixing_history;
    return;
}

void Aar_control::set_anderson_frequency(const int& anderson_frequency) {
    this->anderson_frequency = anderson_frequency;
    return;
}

void Aar_control::set_tolerance(const double& tolerance) {
    this->tolerance = tolerance;
    return;
}

void Aar_control::set_max_iter(const uint& max_iter) {
    this->max_iter = max_iter;
    return;
}

void Aar_control::init() {
    this->precondition_method = 0;
    this->residual_method = 0;
    this->anderson_beta = 0.6;
    this->richardson_omega = 0.6;
    this->mixing_history = 7;
    this->anderson_frequency = 6;
    this->tolerance = 1e-7;
    this->max_iter = 3000;
    return;
}

void Aar_control::show() const {
    std::cout << "precondition_method = " << this->precondition_method << std::endl;
    std::cout << "residual_method = " << this->residual_method << std::endl;
    std::cout << "anderson_beta = " << this->anderson_beta << std::endl;
    std::cout << "richardson_omega = " << this->richardson_omega << std::endl;
    std::cout << "mixing_history = " << this->mixing_history << std::endl;
    std::cout << "anderson_frequency = " << this->anderson_frequency << std::endl;
    std::cout << "tolerance = " << this->tolerance << std::endl;
    std::cout << "max_iter = " << this->max_iter << std::endl;
    return;
}

Control::Control() {}

Control::~Control() {}

void Control::init_addition() {
    // default Kerker tolerance
    if (this->scf_control.mixing_control.precondition_method == 1 && this->scf_control.mixing_control.kerker_control.tolerance < 0) {
        double h_eff = 0.0;
        if (fabs(this->mesh_control.delta_x - this->mesh_control.delta_y) < 1e-12 &&
            fabs(this->mesh_control.delta_y - this->mesh_control.delta_z) < 1e-12) {
            h_eff = this->mesh_control.delta_x;
        } else {
            // 2nd derivative weights including mesh
            double dx2_inv, dy2_inv, dz2_inv;
            dx2_inv = 1.0 / (this->mesh_control.delta_x * this->mesh_control.delta_x);
            dy2_inv = 1.0 / (this->mesh_control.delta_y * this->mesh_control.delta_y);
            dz2_inv = 1.0 / (this->mesh_control.delta_z * this->mesh_control.delta_z);
            h_eff = sqrt(3.0 / (dx2_inv + dy2_inv + dz2_inv));
        }
        this->scf_control.mixing_control.kerker_control.tolerance = (h_eff * h_eff) * 1e-3;
    }
    return;
}

void Control::init(const Input_file& input_file, const Geometry& geometry) {
    this->mesh_control.init(input_file, geometry);
    this->scf_control.init(input_file);
    this->poisson_solver_control.init(input_file, this->scf_control);
    this->exchange_correlation_solver_control.init(input_file);
    this->density_solver_control.init(input_file);
    this->misc_control.init(input_file);
    this->flow_control.init(input_file);
    this->stencil_control.init(input_file);
    this->k_sample_control.init(input_file);
    this->spin_control.init(input_file);
    this->init_addition();
    return;
}

MPI_Datatype Control::register_mpi_type() const {
    MPI_Datatype type;

    MPI_Datatype Mesh_control_type = this->mesh_control.register_mpi_type();
    MPI_Datatype Scf_control_type = this->scf_control.register_mpi_type();
    MPI_Datatype Poisson_solver_control_type = this->poisson_solver_control.register_mpi_type();
    MPI_Datatype Exchange_correlation_solver_control_type = this->exchange_correlation_solver_control.register_mpi_type();
    MPI_Datatype Density_solver_control_type = this->density_solver_control.register_mpi_type();
    MPI_Datatype Misc_control_type = this->misc_control.register_mpi_type();
    MPI_Datatype Flow_control_type = this->flow_control.register_mpi_type();
    MPI_Datatype K_sample_control_type = this->k_sample_control.register_mpi_type();
    MPI_Datatype Spin_control_type = this->spin_control.register_mpi_type();
    MPI_Datatype Stencil_control_type = this->stencil_control.register_mpi_type();

    constexpr std::size_t num_members = 10;
    int lengths[num_members] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Aint offsets[num_members] = {offsetof(Control, mesh_control),
                                     offsetof(Control, scf_control),
                                     offsetof(Control, poisson_solver_control),
                                     offsetof(Control, exchange_correlation_solver_control),
                                     offsetof(Control, density_solver_control),
                                     offsetof(Control, misc_control),
                                     offsetof(Control, flow_control),
                                     offsetof(Control, k_sample_control),
                                     offsetof(Control, spin_control),
                                     offsetof(Control, stencil_control)};
    MPI_Datatype types[num_members] = {Mesh_control_type,
                                       Scf_control_type,
                                       Poisson_solver_control_type,
                                       Exchange_correlation_solver_control_type,
                                       Density_solver_control_type,
                                       Misc_control_type,
                                       Flow_control_type,
                                       K_sample_control_type,
                                       Spin_control_type,
                                       Stencil_control_type};
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);

    this->mesh_control.deregister_mpi_type(Mesh_control_type);
    this->scf_control.deregister_mpi_type(Scf_control_type);
    this->poisson_solver_control.deregister_mpi_type(Poisson_solver_control_type);
    this->exchange_correlation_solver_control.deregister_mpi_type(Exchange_correlation_solver_control_type);
    this->density_solver_control.deregister_mpi_type(Density_solver_control_type);
    this->misc_control.deregister_mpi_type(Misc_control_type);
    this->flow_control.deregister_mpi_type(Flow_control_type);
    this->spin_control.deregister_mpi_type(K_sample_control_type);
    this->spin_control.deregister_mpi_type(Spin_control_type);
    this->stencil_control.deregister_mpi_type(Stencil_control_type);

    return type;
}

void Control::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Control::bcast(const MPI_Comm& comm, const int& root) {
    MPI_Datatype Control_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Control_type, root, comm);
    this->deregister_mpi_type(Control_type);
    return;
}

void Control::show() const {
    this->mesh_control.show();
    this->scf_control.show();
    this->poisson_solver_control.show();
    this->exchange_correlation_solver_control.show();
    this->density_solver_control.show();
    this->misc_control.show();
    this->flow_control.show();
    this->k_sample_control.show();
    this->spin_control.show();
    this->stencil_control.show();
    return;
}

void Control::print(std::ostream& output) const {
    this->mesh_control.print(output);
    this->spin_control.print(output);
    this->stencil_control.print(output);
    this->exchange_correlation_solver_control.print(output);
    this->scf_control.print(output);
    this->poisson_solver_control.print(output);
    this->density_solver_control.print(output);
    this->misc_control.print(output);
    this->flow_control.print(output);
    this->k_sample_control.print(output);
}
