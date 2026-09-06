#include "force_solver.h"

template<typename T>
Force_solver<T>::Force_solver(const Domain_parallel_vertices_3D& domain_vertices,
                              const Exarr_3D_mpi_package& exarr_mpi_package,
                              const Mesh_control& mesh_control,
                              const Geometry& geometry,
                              const Stencil<T>& stencil,
                              const Density_solver<T>& density_solver,
                              const std::vector<Psp8_file>& psp8_files)
                            : domain_vertices(domain_vertices),
                              exarr_mpi_package(exarr_mpi_package),
                              mesh_control(mesh_control),
                              geometry(geometry),
                              stencil(stencil),
                              density_solver(density_solver),
                              psp8_files(psp8_files) {}

template<typename T>
Force_solver<T>::~Force_solver(){}

template<typename T>
void Force_solver<T>::cal_local_forces(const Array_3D<T>& electrostatic_potential,
                                       const Array_3D<T>& pseudo_charge_density,
                                       const Array_3D<T>& pseudo_charge_density_ref,
                                       const Array_3D<T>& pseudo_charge_density_potiential_correction) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    Vertices_3D local_vertice = this->domain_vertices.get_3D_local_vertices();
    Vertices_3D ex_local_vertices = local_vertice.generate_ex_vertices(this->stencil.FDn);

    Array_3D<T> temp(ex_local_vertices, 0);

    Array_3D<T> DelecstPotential_x(local_vertice);
    Array_3D<T> DelecstPotential_y(local_vertice);
    Array_3D<T> DelecstPotential_z(local_vertice);
    this->exarr_mpi_package.fill_domain_par_ex_arr(electrostatic_potential, temp);
    Stencil_method::calc_gradient(temp.data, ex_local_vertices, 0, this->stencil,
                                  local_vertice, DelecstPotential_x.data, local_vertice);
    Stencil_method::calc_gradient(temp.data, ex_local_vertices, 1, this->stencil,
                                  local_vertice, DelecstPotential_y.data, local_vertice);
    Stencil_method::calc_gradient(temp.data, ex_local_vertices, 2, this->stencil,
                                  local_vertice, DelecstPotential_z.data, local_vertice);
    T const* const& DelecstPotential_x_data = DelecstPotential_x.data;
    T const* const& DelecstPotential_y_data = DelecstPotential_y.data;
    T const* const& DelecstPotential_z_data = DelecstPotential_z.data;

    Array_3D<T> DVc_x(local_vertice);
    Array_3D<T> DVc_y(local_vertice);
    Array_3D<T> DVc_z(local_vertice);
    this->exarr_mpi_package.fill_domain_par_ex_arr(pseudo_charge_density_potiential_correction, temp);
    Stencil_method::calc_gradient(temp.data, ex_local_vertices, 0, this->stencil,
                                  local_vertice, DVc_x.data, local_vertice);
    Stencil_method::calc_gradient(temp.data, ex_local_vertices, 1, this->stencil,
                                  local_vertice, DVc_y.data, local_vertice);
    Stencil_method::calc_gradient(temp.data, ex_local_vertices, 2, this->stencil,
                                  local_vertice, DVc_z.data, local_vertice);
    T const* const& DVc_x_data = DVc_x.data;
    T const* const& DVc_y_data = DVc_y.data;
    T const* const& DVc_z_data = DVc_z.data;

    Array_3D<T> b_plus_b_ref = pseudo_charge_density + pseudo_charge_density_ref;
    T const* const& b_plus_b_ref_data = b_plus_b_ref.data;

    this->forces = (T)0.0;
    std::vector<Atom> image_atoms;
    Atom_method::generate_valid_images(this->geometry.atoms, image_atoms, this->geometry.cell_type,
                                       this->domain_vertices, this->mesh_control.is_periodic,
                                       this->mesh_control.delta_x, this->mesh_control.delta_y,
                                       this->mesh_control.delta_z,
                                       this->geometry.a, this->geometry.b, this->geometry.c,
                                       this->psp8_files, 1);
     for (std::vector<Atom>::iterator it = image_atoms.begin(); it != image_atoms.end(); ++it) {
        Vertices_3D atom_vertice =
                            local_vertice.get_overlap_vertices(it->generate_rc_vertices(
                            this->geometry.cell_type,
                            this->mesh_control.delta_x,
                            this->mesh_control.delta_y,
                            this->mesh_control.delta_z,
                            this->psp8_files[it->type].charge_cut_x,
                            this->psp8_files[it->type].charge_cut_y,
                            this->psp8_files[it->type].charge_cut_z));
        if (atom_vertice.get_size() == 0) continue;
        Array_3D<T> atom_psoducharge_density(atom_vertice);
        Array_3D<T> atom_psoducharge_density_ref(atom_vertice);
        Array_3D<T> atom_Dpotiential_correction_x(atom_vertice);
        Array_3D<T> atom_Dpotiential_correction_y(atom_vertice);
        Array_3D<T> atom_Dpotiential_correction_z(atom_vertice);
        it->generate_pseudo_charge_density_local_force_related<T>(atom_psoducharge_density,
                                                                  atom_psoducharge_density_ref,
                                                                  atom_Dpotiential_correction_x,
                                                                  atom_Dpotiential_correction_y,
                                                                  atom_Dpotiential_correction_z,
                                                                  this->geometry.cell_type, atom_vertice,
                                                                  this->psp8_files[it->type],
                                                                  this->stencil, this->mesh_control.delta_x,
                                                                  this->mesh_control.delta_y, this->mesh_control.delta_z);
        T const* const& atom_psoducharge_density_data = atom_psoducharge_density.data;
        T const* const& atom_psoducharge_density_ref_data = atom_psoducharge_density_ref.data;
        T const* const& atom_Dpotiential_correction_x_data = atom_Dpotiential_correction_x.data;
        T const* const& atom_Dpotiential_correction_y_data = atom_Dpotiential_correction_y.data;
        T const* const& atom_Dpotiential_correction_z_data = atom_Dpotiential_correction_z.data;

        T force_x = (T)0.0;
        T force_y = (T)0.0;
        T force_z = (T)0.0;
        T force_corr_x = (T)0.0;
        T force_corr_y = (T)0.0;
        T force_corr_z = (T)0.0;

        assert(local_vertice.is_ex_vertices(atom_vertice));
        const int local_index_origin = local_vertice.get_index_nocheck(atom_vertice.is, atom_vertice.js, atom_vertice.ks);
        const int local_ni = local_vertice.ni;
        const int local_ninj = local_vertice.ni * local_vertice.nj;
        const int atom_offset_origin = atom_vertice.get_index_nocheck(atom_vertice.is, atom_vertice.js, atom_vertice.ks);;
        const int atom_ni = atom_vertice.ni;
        const int atom_ninj = atom_vertice.ni * atom_vertice.nj;
        for (int k = 0; k < (int)atom_vertice.nk; ++k) {
            int local_offset_k = local_index_origin + k * local_ninj;
            int atom_offset_k = atom_offset_origin + k * atom_ninj;
            int local_offset_j = local_offset_k;
            int atom_offset_j = atom_offset_k;
            for (int j = 0; j < (int)atom_vertice.nj; ++j) {
                int local_offset_i = local_offset_j;
                int atom_offset_i = atom_offset_j;
                for (int i = 0; i < (int)atom_vertice.ni; ++i) {
                    force_x -= atom_psoducharge_density_data[atom_offset_i] * DelecstPotential_x_data[local_offset_i];
                    force_y -= atom_psoducharge_density_data[atom_offset_i] * DelecstPotential_y_data[local_offset_i];
                    force_z -= atom_psoducharge_density_data[atom_offset_i] * DelecstPotential_z_data[local_offset_i];
                    T atom_b_plus_b_ref = atom_psoducharge_density_data[atom_offset_i]
                                        + atom_psoducharge_density_ref_data[atom_offset_i];
                    force_corr_x += atom_Dpotiential_correction_x_data[atom_offset_i] * b_plus_b_ref_data[local_offset_i]
                                  - DVc_x_data[local_offset_i] * atom_b_plus_b_ref;
                    force_corr_y += atom_Dpotiential_correction_y_data[atom_offset_i] * b_plus_b_ref_data[local_offset_i]
                                  - DVc_y_data[local_offset_i] * atom_b_plus_b_ref;
                    force_corr_z += atom_Dpotiential_correction_z_data[atom_offset_i] * b_plus_b_ref_data[local_offset_i]
                                  - DVc_z_data[local_offset_i] * atom_b_plus_b_ref;
                    local_offset_i++;
                    atom_offset_i++;
                }
                local_offset_j += local_ni;
                atom_offset_j += atom_ni;
            }
        }
        this->forces[it->index*3    ] += (force_x + 0.5 * force_corr_x) * this->mesh_control.delta_V;
        this->forces[it->index*3 + 1] += (force_y + 0.5 * force_corr_y) * this->mesh_control.delta_V;
        this->forces[it->index*3 + 2] += (force_z + 0.5 * force_corr_z) * this->mesh_control.delta_V;
    }
    if (this->domain_vertices.get_comm_size() > 0) {
        MPI_Datatype mpi_data_type = Linalg::get_mpi_datatype<T>();
        MPI_Allreduce(MPI_IN_PLACE, this->forces.data, this->forces.length, mpi_data_type, MPI_SUM, this->domain_vertices.comm);
    }
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "local_forces = " << std::endl;
        this->forces.print();
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The cal_local_forces run took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

template<typename T>
void Force_solver<T>::cal_xc_forces(const std::vector<Array_3D<T>>& xc_potentials) {
    if (xc_potentials.size() == 1) {
        this->cal_xc_forces(xc_potentials[0]);
    } else if (xc_potentials.size() == 2) {
        // spin polarized case, not implemented yet
        assert(false);
    } else {
        assert(false);
    }
    return;
}

template<typename T>
void Force_solver<T>::cal_xc_forces(const Array_3D<T>& xc_potential) {

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    const Vertices_3D local_vertices = this->domain_vertices.get_3D_local_vertices();

    Array_2D<T> xc_forces(this->forces.get_vertices(), 0);
    std::vector<Atom> image_atoms;
    Atom_method::generate_valid_images(this->geometry.atoms, image_atoms, this->geometry.cell_type,
                                       this->domain_vertices, this->mesh_control.is_periodic,
                                       this->mesh_control.delta_x, this->mesh_control.delta_y,
                                       this->mesh_control.delta_z,
                                       this->geometry.a, this->geometry.b, this->geometry.c,
                                       this->psp8_files, 1);
    bool if_xc_forces = false;
    for (std::vector<Atom>::iterator it = image_atoms.begin(); it != image_atoms.end(); ++it) {
        const Psp8_file& psp8_file = this->psp8_files[it->type];
        if (psp8_file.fchrg < 1e-12) {
            continue;
        }
        if_xc_forces = true;
        Vertices_3D atom_vertice =
                            local_vertices.get_overlap_vertices(it->generate_rc_vertices(
                            this->geometry.cell_type,
                            this->mesh_control.delta_x,
                            this->mesh_control.delta_y,
                            this->mesh_control.delta_z,
                            this->psp8_files[it->type].charge_cut_x,
                            this->psp8_files[it->type].charge_cut_y,
                            this->psp8_files[it->type].charge_cut_z));
        if (atom_vertice.get_size() == 0) continue;
        const Vertices_3D ex_atom_vertice = atom_vertice.generate_ex_vertices(this->stencil.FDn);
        assert(this->stencil.cell_type <= 2);
        Array_3D<T> rhocJ = it->generate_initial_electron_density_core_array<T>(this->geometry.cell_type, ex_atom_vertice,
                                                                psp8_file, this->mesh_control.delta_x,
                                                                this->mesh_control.delta_y, this->mesh_control.delta_z);

        const int atom_vertice_len = atom_vertice.get_size();
        Array_0D<T> drhocJ_x(atom_vertice_len);
        Array_0D<T> drhocJ_y(atom_vertice_len);
        Array_0D<T> drhocJ_z(atom_vertice_len);
        Array_0D<T> xc_potential_atomic(atom_vertice_len);
        Stencil_method::calc_gradient(rhocJ.data, ex_atom_vertice, 0, this->stencil,
                                  atom_vertice, drhocJ_x.data, atom_vertice);
        Stencil_method::calc_gradient(rhocJ.data, ex_atom_vertice, 1, this->stencil,
                                  atom_vertice, drhocJ_y.data, atom_vertice);
        Stencil_method::calc_gradient(rhocJ.data, ex_atom_vertice, 2, this->stencil,
                                  atom_vertice, drhocJ_z.data, atom_vertice);

        Vertices_method::fill_region<T>(xc_potential.data, local_vertices, xc_potential_atomic.data, atom_vertice, atom_vertice);
        xc_forces[it->index*3    ] += this->mesh_control.delta_V * Linalg::vector_dot_product(drhocJ_x.data, xc_potential_atomic.data, atom_vertice_len);
        xc_forces[it->index*3 + 1] += this->mesh_control.delta_V * Linalg::vector_dot_product(drhocJ_y.data, xc_potential_atomic.data, atom_vertice_len);
        xc_forces[it->index*3 + 2] += this->mesh_control.delta_V * Linalg::vector_dot_product(drhocJ_z.data, xc_potential_atomic.data, atom_vertice_len);
    }
    MPI_Allreduce(MPI_IN_PLACE, &if_xc_forces, 1, MPI_C_BOOL, MPI_LOR, this->domain_vertices.comm);
    if (if_xc_forces) {
        if (this->domain_vertices.get_comm_size() > 0) {
            MPI_Datatype mpi_data_type = Linalg::get_mpi_datatype<T>();
            MPI_Allreduce(MPI_IN_PLACE, xc_forces.data, xc_forces.length, mpi_data_type, MPI_SUM, this->domain_vertices.comm);
        }
        if (this->domain_vertices.get_comm_rank() == 0) {
            std::cout << "xc_forces = " << std::endl;
            xc_forces.print();
        }
        Linalg::hadamard_plus_general(this->forces.data, xc_forces.data, this->forces.length);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (this->domain_vertices.get_comm_rank() == 0) {
            std::cout << "The cal_xc_forces run took " << Tools::time_cost(begin, end) << "." << std::endl;
        }
    }
    return;
}

template<typename T>
void Force_solver<T>::cal_nonlocal_forces() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->density_solver.density_solver_control.method == 0     // eigen_solver
     || this->density_solver.density_solver_control.method == 1) {  // density_matrix_solver
        Array_2D<T> nonlocal_forces(this->forces.get_vertices(), 0);
        this->density_solver.cal_nonlocal_forces(nonlocal_forces);
        this->forces += nonlocal_forces;
        if (this->domain_vertices.get_comm_rank() == 0) {
            std::cout << "nonlocal_forces = " << std::endl;
            nonlocal_forces.print();
        }
    } else {
        assert(this->density_solver.density_solver_control.method == 0
            || this->density_solver.density_solver_control.method == 1);
    }
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "The cal_nonlocal_forces run took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

template<typename T>
void Force_solver<T>::balance_forces() {
    uint natom = this->geometry.atoms.size();
    T sum_force_x = (T)0.0;
    T sum_force_y = (T)0.0;
    T sum_force_z = (T)0.0;
    for (uint iatom = 0; iatom < natom; iatom++) {
        sum_force_x += this->forces[iatom * 3];
        sum_force_y += this->forces[iatom * 3 + 1];
        sum_force_z += this->forces[iatom * 3 + 2];
    }
    T shift_force_x = -sum_force_x/(T)natom;
    T shift_force_y = -sum_force_y/(T)natom;
    T shift_force_z = -sum_force_z/(T)natom;
    for (uint iatom = 0; iatom < natom; iatom++) {
        this->forces[iatom * 3] += shift_force_x;
        this->forces[iatom * 3 + 1] += shift_force_y;
        this->forces[iatom * 3 + 2] += shift_force_z;
    }
    return;
}

template<typename T>
void Force_solver<T>::init() {
    this->forces.reconstructor(Vertices_2D(3, this->geometry.atoms.size()));
    return;
}

template<typename T>
void Force_solver<T>::show() const {
    std::cout << "Cartesian force = " << std::endl;
    this->forces.print();
    return;
}

template class Force_solver<float>;
template class Force_solver<double>;
