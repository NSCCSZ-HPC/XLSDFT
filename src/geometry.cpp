#include "geometry.h"

Geometry::Geometry() {}

Geometry::Geometry(const Geometry& other) {
    *this = other;
}

Geometry::Geometry(Geometry&& other) {
    *this = std::move(other);
}

Geometry::~Geometry() {}

Geometry& Geometry::operator=(const Geometry& other) {
    this->cell_type = other.cell_type;
    this->origin[0] = other.origin[0];
    this->origin[1] = other.origin[1];
    this->origin[2] = other.origin[2];
    this->v1[0] = other.v1[0];
    this->v1[1] = other.v1[1];
    this->v1[2] = other.v1[2];
    this->v2[0] = other.v2[0];
    this->v2[1] = other.v2[1];
    this->v2[2] = other.v2[2];
    this->v3[0] = other.v3[0];
    this->v3[1] = other.v3[1];
    this->v3[2] = other.v3[2];
    this->a = other.a;
    this->b = other.b;
    this->c = other.c;
    this->alpha = other.alpha;
    this->beta = other.beta;
    this->gamma = other.gamma;
    this->volume = other.volume;
    this->natom = other.natom;
    this->natom_type = other.natom_type;
    this->atoms = other.atoms;
    return *this;
}

Geometry& Geometry::operator=(Geometry&& other) {
    this->cell_type = other.cell_type;
    this->origin[0] = other.origin[0];
    this->origin[1] = other.origin[1];
    this->origin[2] = other.origin[2];
    this->v1[0] = other.v1[0];
    this->v1[1] = other.v1[1];
    this->v1[2] = other.v1[2];
    this->v2[0] = other.v2[0];
    this->v2[1] = other.v2[1];
    this->v2[2] = other.v2[2];
    this->v3[0] = other.v3[0];
    this->v3[1] = other.v3[1];
    this->v3[2] = other.v3[2];
    this->a = other.a;
    this->b = other.b;
    this->c = other.c;
    this->alpha = other.alpha;
    this->beta = other.beta;
    this->gamma = other.gamma;
    this->volume = other.volume;
    this->natom = other.natom;
    this->natom_type = other.natom_type;
    this->atoms.swap(other.atoms);
    return *this;
}

void Geometry::set_cell_type(const uint& cell_type) {
    assert(cell_type <= 2 && "ERROR:: only Orthorhombi is supported~");
    this->cell_type = cell_type;
    return;
}

void Geometry::set_v1(const double& v1_0) {
    this->set_v1(v1_0, 0.0, 0.0);
    return;
}

void Geometry::set_v1(const double* v1) {
    this->set_v1(v1[0], v1[1], v1[2]);
    return;
}

void Geometry::set_v1(const double& v1_0, const double& v1_1, const double& v1_2) {
    this->v1[0] = v1_0;
    this->v1[1] = v1_1;
    this->v1[2] = v1_2;
    this->a = std::sqrt(v1_0 * v1_0 + v1_1 * v1_1 + v1_2 * v1_2);
    return;
}

void Geometry::set_v2(const double& av_1) {
    this->set_v2(0.0, av_1, 0.0);
    return;
}

void Geometry::set_v2(const double* v2) {
    this->set_v2(v2[0], v2[1], v2[2]);
    return;
}

void Geometry::set_v2(const double& v2_0, const double& v2_1, const double& v2_2) {
    this->v2[0] = v2_0;
    this->v2[1] = v2_1;
    this->v2[2] = v2_2;
    this->b = std::sqrt(v2_0 * v2_0 + v2_1 * v2_1 + v2_2 * v2_2);
    return;
}

void Geometry::set_v3(const double& v3_2) {
    this->set_v3(0.0, 0.0, v3_2);
    return;
}

void Geometry::set_v3(const double* v3) {
    this->set_v3(v3[0], v3[1], v3[2]);
    return;
}

void Geometry::set_v3(const double& v3_0, const double& v3_1, const double& v3_2) {
    this->v3[0] = v3_0;
    this->v3[1] = v3_1;
    this->v3[2] = v3_2;
    this->c = std::sqrt(v3_0 * v3_0 + v3_1 * v3_1 + v3_2 * v3_2);
    return;
}

void Geometry::set_a(const double& a) {
    assert(a > -1e-14);
    this->a = a;
    return;
}

void Geometry::set_b(const double& b) {
    assert(b > -1e-14);
    this->b = b;
    return;
}

void Geometry::set_c(const double& c) {
    assert(c > -1e-14);
    this->c = c;
    return;
}

void Geometry::set_edge_lengths(const double* lengths) {
    this->set_edge_lengths(lengths[0], lengths[1], lengths[2]);
    return;
}

void Geometry::set_edge_lengths(const double& a, const double& b, const double& c) {
    this->set_a(a);
    this->set_b(b);
    this->set_c(c);
    return;
}

void Geometry::set_alpha(const double& alpha) {
    assert(alpha > -1e-14 && alpha < M_PI + 1e-14);
    this->alpha = alpha;
    return;
}

void Geometry::set_beta(const double& beta) {
    assert(beta > -1e-14 && beta < M_PI + 1e-14);
    this->beta = beta;
    return;
}

void Geometry::set_gamma(const double& gamma) {
    assert(gamma > -1e-14 && gamma < M_PI + 1e-14);
    this->gamma = gamma;
    return;
}

void Geometry::set_angles(const double& alpha, const double& beta, const double& gamma) {
    this->set_alpha(alpha);
    this->set_beta(beta);
    this->set_gamma(gamma);
    return;
}

void Geometry::sync_from_vectors() {
    this->set_v1(this->v1);
    this->set_v2(this->v2);
    this->set_v3(this->v3);
    if (fabs(this->v1[1]) < 1e-14 && fabs(this->v1[2]) < 1e-14
     && fabs(this->v2[0]) < 1e-14 && fabs(this->v2[2]) < 1e-14
     && fabs(this->v3[0]) < 1e-14 && fabs(this->v3[1]) < 1e-14) { // Orthorhombic at least
        this->set_angles(M_PI_2, M_PI_2, M_PI_2);
        this->volume = this->a * this->b * this->c;
        if( fabs(this->a - this->b) < 1e-14
         && fabs(this->a - this->c) < 1e-14
         && fabs(this->b - this->c) < 1e-14 ) {         // Cubic
            this->set_cell_type(0);
         } else if (fabs(this->a - this->b) < 1e-14
                 || fabs(this->a - this->c) < 1e-14
                 || fabs(this->b - this->c) < 1e-14) {  // Tetragonal
            this->set_cell_type(1);
        } else {                                        // Orthorhombic
            this->set_cell_type(2);
        }
    } else {
        this->show();
        assert(0 && "ERROR:: only Orthorhombi is supported~");
    }
    return;
}

void Geometry::sync_from_angles() {
    if (fabs(this->alpha - M_PI_2) < 1e-14
     && fabs(this->beta - M_PI_2) < 1e-14
     && fabs(this->gamma - M_PI_2) < 1e-14) {           // Orthorhombic at least
        this->set_angles(M_PI_2, M_PI_2, M_PI_2);
        this->set_v1(this->a);
        this->set_v2(this->b);
        this->set_v3(this->c);
        this->volume = this->a * this->b * this->c;
        if( fabs(this->a - this->b) < 1e-14
         && fabs(this->a - this->c) < 1e-14
         && fabs(this->b - this->c) < 1e-14 ) {         // Cubic
            this->set_cell_type(0);
         } else if (fabs(this->a - this->b) < 1e-14
                 || fabs(this->a - this->c) < 1e-14
                 || fabs(this->b - this->c) < 1e-14) {  // Tetragonal
            this->set_cell_type(1);
        } else {                                        // Orthorhombic
            this->set_cell_type(2);
        }
     } else {
        this->show();
        assert(0 && "ERROR:: only Orthorhombi is supported~");
     }
     return;
}

void Geometry::atom_fractional_coordinates(double* const& frac_xs, double* const& frac_ys, double* const& frac_zs) const {
    double lattice[3][3] = {
        {v1[0], v2[0], v3[0]},
        {v1[1], v2[1], v3[1]},
        {v1[2], v2[2], v3[2]}
    };
    double det = lattice[0][0]*(lattice[1][1]*lattice[2][2]-lattice[1][2]*lattice[2][1]) -
                 lattice[0][1]*(lattice[1][0]*lattice[2][2]-lattice[1][2]*lattice[2][0]) +
                 lattice[0][2]*(lattice[1][0]*lattice[2][1]-lattice[1][1]*lattice[2][0]);
    double inv_lattice[3][3];
    inv_lattice[0][0] = (lattice[1][1]*lattice[2][2]-lattice[1][2]*lattice[2][1])/det;
    inv_lattice[0][1] = (lattice[0][2]*lattice[2][1]-lattice[0][1]*lattice[2][2])/det;
    inv_lattice[0][2] = (lattice[0][1]*lattice[1][2]-lattice[0][2]*lattice[1][1])/det;
    inv_lattice[1][0] = (lattice[1][2]*lattice[2][0]-lattice[1][0]*lattice[2][2])/det;
    inv_lattice[1][1] = (lattice[0][0]*lattice[2][2]-lattice[0][2]*lattice[2][0])/det;
    inv_lattice[1][2] = (lattice[0][2]*lattice[1][0]-lattice[0][0]*lattice[1][2])/det;
    inv_lattice[2][0] = (lattice[1][0]*lattice[2][1]-lattice[1][1]*lattice[2][0])/det;
    inv_lattice[2][1] = (lattice[0][1]*lattice[2][0]-lattice[0][0]*lattice[2][1])/det;
    inv_lattice[2][2] = (lattice[0][0]*lattice[1][1]-lattice[0][1]*lattice[1][0])/det;
    for (uint iatom = 0; iatom < natom; iatom++) {
        double x = this->atoms[iatom].x - this->origin[0];
        double y = this->atoms[iatom].y - this->origin[1];
        double z = this->atoms[iatom].z - this->origin[2];
        frac_xs[iatom] = inv_lattice[0][0]*x + inv_lattice[0][1]*y + inv_lattice[0][2]*z;
        frac_ys[iatom] = inv_lattice[1][0]*x + inv_lattice[1][1]*y + inv_lattice[1][2]*z;
        frac_zs[iatom] = inv_lattice[2][0]*x + inv_lattice[2][1]*y + inv_lattice[2][2]*z;
    }
    return;
}

// Geometry Geometry::generate_extended_geometry(const Vertices_3D& my_vertices, const Vertices_3D& other_vertices,
//                                               const double& dx, const double& dy, const double& dz,
//                                               bool const* const& is_periodic, double const* const& buffers) const {
//     assert(this->cell_type <= 2);
//     Geometry other;
//     other.cell_type = this->cell_type;
//     // double dx = is_periodic[0] ? (this->a / (double)my_vertices.ni) : (this->a / (double)((int)my_vertices.ni - 1));
//     // double dy = is_periodic[1] ? (this->b / (double)my_vertices.nj) : (this->b / (double)((int)my_vertices.nj - 1));
//     // double dz = is_periodic[2] ? (this->c / (double)my_vertices.nk) : (this->c / (double)((int)my_vertices.nk - 1));
//     other.origin[0] = other_vertices.is * dx;
//     other.origin[1] = other_vertices.js * dy;
//     other.origin[2] = other_vertices.ks * dz;
//     double cell_x_start = other.origin[0];
//     double cell_y_start = other.origin[1];
//     double cell_z_start = other.origin[2];
//     // double cell_x_end = is_periodic[0] || other_vertices.get_ie() < my_vertices.get_ie()
//     //                   ? (other_vertices.get_ie() + 1) * dx : other_vertices.get_ie() * dx;
//     // double cell_y_end = is_periodic[1] || other_vertices.get_je() < my_vertices.get_je()
//     //                   ? (other_vertices.get_je() + 1) * dy : other_vertices.get_je() * dy;
//     // double cell_z_end = is_periodic[2] || other_vertices.get_ke() < my_vertices.get_ke()
//     //                   ? (other_vertices.get_ke() + 1) * dz : other_vertices.get_ke() * dz;
//     double cell_x_end = other_vertices.get_ie() * dx;
//     double cell_y_end = other_vertices.get_je() * dy;
//     double cell_z_end = other_vertices.get_ke() * dz;
//     other.a = cell_x_end - cell_x_start;
//     other.b = cell_y_end - cell_y_start;
//     other.c = cell_z_end - cell_z_start;
//     other.sync_from_angles();
//     // double excell_x_start = cell_x_start - buffers[0];
//     // double excell_y_start = cell_y_start - buffers[1];
//     // double excell_z_start = cell_z_start - buffers[2];
//     // double excell_x_end = cell_x_end + buffers[0];
//     // double excell_y_end = cell_y_end + buffers[1];
//     // double excell_z_end = cell_z_end + buffers[2];
//     other.atoms.resize(0);
//     uint natom = 0;
//     for (std::vector<Atom>::const_iterator it = this->atoms.cbegin(); it != this->atoms.cend(); ++it) {
//         natom += it->generate_valid_images(other.atoms, this->cell_type, my_vertices, other_vertices,
//                                            is_periodic, dx, dy, dz, this->a, this->b, this->c,
//                                            buffers[0], buffers[1], buffers[2]);
//     }
//     other.natom = natom;
//     other.natom_type = this->natom_type;
//     return other;
// }

Geometry Geometry::generate_extended_geometry(const Vertices_3D& my_vertices, const Vertices_3D& other_vertices,
                                              const double& dx, const double& dy, const double& dz,
                                              bool const* const& is_periodic, bool const* const& if_one_more_grid,
                                              const std::vector<Psp8_file>& psp8_files) const {
    assert(this->cell_type <= 2);
    Geometry other;
    other.cell_type = this->cell_type;
    // double dx = is_periodic[0] ? (this->a / (double)my_vertices.ni) : (this->a / (double)((int)my_vertices.ni - 1));
    // double dy = is_periodic[1] ? (this->b / (double)my_vertices.nj) : (this->b / (double)((int)my_vertices.nj - 1));
    // double dz = is_periodic[2] ? (this->c / (double)my_vertices.nk) : (this->c / (double)((int)my_vertices.nk - 1));
    other.origin[0] = other_vertices.is * dx;
    other.origin[1] = other_vertices.js * dy;
    other.origin[2] = other_vertices.ks * dz;
    double cell_x_start = other.origin[0];
    double cell_y_start = other.origin[1];
    double cell_z_start = other.origin[2];
    // double cell_x_end = is_periodic[0] || other_vertices.get_ie() < my_vertices.get_ie()
    //                   ? (other_vertices.get_ie() + 1) * dx : other_vertices.get_ie() * dx;
    // double cell_y_end = is_periodic[1] || other_vertices.get_je() < my_vertices.get_je()
    //                   ? (other_vertices.get_je() + 1) * dy : other_vertices.get_je() * dy;
    // double cell_z_end = is_periodic[2] || other_vertices.get_ke() < my_vertices.get_ke()
    //                   ? (other_vertices.get_ke() + 1) * dz : other_vertices.get_ke() * dz;
    double cell_x_end = (if_one_more_grid[0] ? other_vertices.get_ie() + 1 : other_vertices.get_ie()) * dx;
    double cell_y_end = (if_one_more_grid[1] ? other_vertices.get_je() + 1 : other_vertices.get_je()) * dy;
    double cell_z_end = (if_one_more_grid[2] ? other_vertices.get_ke() + 1 : other_vertices.get_ke()) * dz;
    other.a = cell_x_end - cell_x_start;
    other.b = cell_y_end - cell_y_start;
    other.c = cell_z_end - cell_z_start;
    other.sync_from_angles();
    other.atoms.resize(0);
    uint natom = 0;
    for (std::vector<Atom>::const_iterator it = this->atoms.cbegin(); it != this->atoms.cend(); ++it) {
        const Psp8_file& psp8_file = psp8_files[it->type];
        double rc = psp8_file.r_core[0];
        double rcbox_x = 0.0;
        double rcbox_y = 0.0;
        double rcbox_z = 0.0;
        if (this->cell_type <= 2) {
            for (int i = 1; i <= psp8_file.lmax; i++) {
                if (rc < psp8_file.r_core[i]) rc = psp8_file.r_core[i];
            }
            rcbox_x = rc;
            rcbox_y = rc;
            rcbox_z = rc;
        } else {
            assert(this->cell_type <= 2);
        }
        natom += it->generate_valid_images(other.atoms, this->cell_type, my_vertices, other_vertices,
                                           is_periodic, dx, dy, dz, this->a, this->b, this->c, rcbox_x, rcbox_y, rcbox_z);
    }
    other.natom = natom;
    other.natom_type = this->natom_type;
    return other;
}

void Geometry::init(Input_file& input_file, const Ion_file& ion_file, const bool& is_map_into_cell) {
    std::stringstream ss;
    if(input_file.map.find("CELL") != input_file.map.end()) {
        ss.str(input_file.map["CELL"]);
        ss >> this->a >> this->b >> this->c;
        ss.clear();
        this->sync_from_angles();
    } else {
        assert(0 && "ERROR::there should be a CELL in input file!");
    }

    this->natom_type = ion_file.atom_types.size();
    this->natom = 0;
    double x;
    double y;
    double z;
    uint index = 0;
    for (uint type = 0; type < this->natom_type; type++)
    {
        uint natoms = ion_file.atom_numbers[type];
        this->natom += natoms;
        this->atoms.reserve(this->natom);
        // bool is_spin = ion_file.atom_spins[type].size() > 0;
        for (uint iatom = 0; iatom < natoms; iatom++)
        {
            if (this->cell_type <= 2) {         // Orthorhombi
                if (ion_file.is_abs_coord[type]) {
                    x = ion_file.atom_coords[type][iatom*3];
                    y = ion_file.atom_coords[type][iatom*3 + 1];
                    z = ion_file.atom_coords[type][iatom*3 + 2];
                } else {
                    x = ion_file.atom_coords[type][iatom*3] * this->a;
                    y = ion_file.atom_coords[type][iatom*3 + 1] * this->b;
                    z = ion_file.atom_coords[type][iatom*3 + 2] * this->b;
                }
                if (is_map_into_cell) {
                    if (x < -1e-14 || x > this->a + 1e-14) x = x > 0.0 ? fmod(x, this->a) : fmod(x, this->a) + this->a;
                    if (y < -1e-14 || y > this->b + 1e-14) y = y > 0.0 ? fmod(y, this->b) : fmod(y, this->b) + this->b;
                    if (z < -1e-14 || z > this->c + 1e-14) z = z > 0.0 ? fmod(z, this->c) : fmod(z, this->c) + this->c;
                }
                this->atoms.emplace_back(x, y, z, type, index);
            } else {
                assert(0 && "ERROR:: only Orthorhombi is supported~");
            }
            // this->atoms.back().is_spin = is_spin;
            // if (is_spin) {
            //     if (ion_file.atom_spins[type].size() == natoms * 3) {
            //         this->atoms.back().atom_spin[0] = ion_file.atom_spins[type][iatom*3];
            //         this->atoms.back().atom_spin[1] = ion_file.atom_spins[type][iatom*3 + 1];
            //         this->atoms.back().atom_spin[2] = ion_file.atom_spins[type][iatom*3 + 2];
            //     } else if (ion_file.atom_spins[type].size() == natoms) {
            //         this->atoms.back().atom_spin[2] = ion_file.atom_spins[type][iatom];
            //     } else {
            //         assert(0 && "ERROR::the size of atom spin in .ion is not correct!");
            //     }
            // }
            ++index;
        }
    }
    return;
}

MPI_Datatype Geometry::register_mpi_type() const {
    MPI_Datatype type;
    if (this->natom > 0) {
        MPI_Datatype Atom_type = this->atoms[0].register_mpi_type();
        constexpr std::size_t num_members = 6;
        int lengths[num_members] = {3, 3, 3, 3, 1, (int) this->natom};
        // MPI_Aint geometry_base;
        // MPI_Get_address(this, &geometry_base);
        // MPI_Aint atom_addr;
        // MPI_Get_address(this->atoms.data(), &atom_addr);
        // MPI_Aint offsets[num_members] = {offsetof(Geometry, origin), offsetof(Geometry, v1),
        //     offsetof(Geometry, v2), offsetof(Geometry, v3),
        //     offsetof(Geometry, natom_type), (atom_addr - geometry_base)};
        MPI_Aint geometry_base;
        MPI_Get_address(this, &geometry_base);
        MPI_Aint offsets[num_members];
        int count = 0;
        MPI_Aint address;
        MPI_Get_address(&(this->origin), &address);
        offsets[count++] = address - geometry_base;
        MPI_Get_address(&(this->v1), &address);
        offsets[count++] = address - geometry_base;
        MPI_Get_address(&(this->v2), &address);
        offsets[count++] = address - geometry_base;
        MPI_Get_address(&(this->v3), &address);
        offsets[count++] = address - geometry_base;
        MPI_Get_address(&(this->natom_type), &address);
        offsets[count++] = address - geometry_base;
        MPI_Get_address(this->atoms.data(), &address);
        offsets[count++] = address - geometry_base;
        assert(count == num_members);
        MPI_Datatype types[num_members] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                            MPI_DOUBLE, MPI_UNSIGNED, Atom_type};
        MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
        MPI_Type_commit(&type);
        this->atoms[0].deregister_mpi_type(Atom_type);
    } else {
        constexpr std::size_t num_members = 5;
        int lengths[num_members] = {3, 3, 3, 3, 1};
        // MPI_Aint offsets[num_members] = {offsetof(Geometry, origin), offsetof(Geometry, v1),
        //     offsetof(Geometry, v2), offsetof(Geometry, v3),
        //     offsetof(Geometry, natom_type)};
        MPI_Aint geometry_base;
        MPI_Get_address(this, &geometry_base);
        MPI_Aint offsets[num_members];
        int count = 0;
        MPI_Aint address;
        MPI_Get_address(&(this->origin), &address);
        offsets[count++] = address - geometry_base;
        MPI_Get_address(&(this->v1), &address);
        offsets[count++] = address - geometry_base;
        MPI_Get_address(&(this->v2), &address);
        offsets[count++] = address - geometry_base;
        MPI_Get_address(&(this->v3), &address);
        offsets[count++] = address - geometry_base;
        MPI_Get_address(&(this->natom_type), &address);
        offsets[count++] = address - geometry_base;
        assert(count == num_members);
        MPI_Datatype types[num_members] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                                           MPI_DOUBLE, MPI_UNSIGNED};
        MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
        MPI_Type_commit(&type);
    }
    return type;
}

void Geometry::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Geometry::bcast(const MPI_Comm& comm, const int& root) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Bcast(&(this->natom), 1, MPI_UNSIGNED, root, comm);
    if (rank != root) {
        this->atoms.resize(this->natom);
    }
    MPI_Datatype Geometry_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Geometry_type, root, comm);
    if (rank != root) {
        this->sync_from_vectors();
    }
    this->deregister_mpi_type(Geometry_type);
    return;
}

void Geometry::show() const {
    if (this->cell_type == 0) std::cout << "cell_type = " << this->cell_type << ", which means Cubic" << std::endl;
    if (this->cell_type == 1) std::cout << "cell_type = " << this->cell_type << ", which means Tetragonal" << std::endl;
    if (this->cell_type == 2) std::cout << "cell_type = " << this->cell_type << ", which means Orthorhombic" << std::endl;
    std::cout << "v1 = [" << std::setw(14) << std::setprecision(8) << std::fixed << this->v1[0] << ", "
                          << std::setw(14) << std::setprecision(8) << std::fixed << this->v1[1] << ", " 
                          << std::setw(14) << std::setprecision(8) << std::fixed << this->v1[2] << "]" << std::endl;
    std::cout << "v2 = [" << std::setw(14) << std::setprecision(8) << std::fixed << this->v2[0] << ", "
                          << std::setw(14) << std::setprecision(8) << std::fixed << this->v2[1] << ", " 
                          << std::setw(14) << std::setprecision(8) << std::fixed << this->v2[2] << "]" << std::endl;
    std::cout << "v3 = [" << std::setw(14) << std::setprecision(8) << std::fixed << this->v3[0] << ", "
                          << std::setw(14) << std::setprecision(8) << std::fixed << this->v3[1] << ", " 
                          << std::setw(14) << std::setprecision(8) << std::fixed << this->v3[2] << "]" << std::endl;
    std::cout << "[a, b, c] = [" << std::setw(14) << std::setprecision(8) << std::fixed << this->a << ", "
                                 << std::setw(14) << std::setprecision(8) << std::fixed << this->b << ", " 
                                 << std::setw(14) << std::setprecision(8) << std::fixed << this->c << "]" << std::endl;
    std::cout << "[alpha, beta, gamma] = [" << std::setw(12) << std::setprecision(10) << std::fixed << this->alpha << ", "
                                            << std::setw(12) << std::setprecision(10) << std::fixed << this->beta << ", " 
                                            << std::setw(12) << std::setprecision(10) << std::fixed << this->gamma << "]" << std::endl;
    std::cout << "volume = " << std::setw(15) << std::setprecision(10) << std::fixed << std::left << this->volume << std::endl;
    std::cout << "natom = " << std::setw(10) << std::fixed << std::left << this->natom << std::endl;
    std::cout << "natom_type = " << std::setw(10) << std::fixed << std::left << this->natom_type << std::endl;
    for (uint iatom = 0; iatom < this->natom; ++iatom)
    {
        this->atoms[iatom].show();
    }
    return;
}