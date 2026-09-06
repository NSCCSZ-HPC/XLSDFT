#include <iostream>
#include "pseudo_charge_solver.h"

template<typename T>
Pseudo_charge_solver<T>::Pseudo_charge_solver(const Geometry& geometry,
                                              const Mesh_control& mesh_control,
                                              const std::vector<Psp8_file>& psp8_files,
                                              const Stencil<T>& stencil,
                                              const Domain_parallel_vertices_3D& domain_vertices)
                                            : geometry(geometry),
                                              mesh_control(mesh_control),
                                              psp8_files(psp8_files),
                                              stencil(stencil),
                                              domain_vertices(domain_vertices) {}

template<typename T>
Pseudo_charge_solver<T>::~Pseudo_charge_solver() {}

template<typename T>
Array_3D<T>& Pseudo_charge_solver<T>::generate_pseudo_charge_density() {
    std::vector<Atom> image_atoms;
    Atom_method::generate_valid_images(this->geometry.atoms, image_atoms, this->geometry.cell_type,
                                       this->domain_vertices, this->mesh_control.is_periodic,
                                       this->mesh_control.delta_x, this->mesh_control.delta_y,
                                       this->mesh_control.delta_z,
                                       this->geometry.a, this->geometry.b, this->geometry.c,
                                       this->psp8_files, 1);
    for (std::vector<Atom>::iterator it = image_atoms.begin(); it != image_atoms.end(); ++it) {
        Vertices_3D atom_vertice =
                            this->domain_vertices.get_3D_local_vertices().get_overlap_vertices(it->generate_rc_vertices(
                            this->geometry.cell_type,
                            this->mesh_control.delta_x,
                            this->mesh_control.delta_y,
                            this->mesh_control.delta_z,
                            this->psp8_files[it->type].charge_cut_x,
                            this->psp8_files[it->type].charge_cut_y,
                            this->psp8_files[it->type].charge_cut_z));
        if (atom_vertice.get_size() == 0) continue;
        Array_3D<T> atom_psoducharge_density =
                                    it->generate_pseudo_charge_density(this->geometry.cell_type, atom_vertice,
                                    this->psp8_files[it->type], this->stencil, this->mesh_control.delta_x,
                                    this->mesh_control.delta_y, this->mesh_control.delta_z);
        atom_psoducharge_density.accumulate_overlap(this->pseudo_charge_density);
    }
    return this->pseudo_charge_density;
}

template<typename T>
Array_3D<T>& Pseudo_charge_solver<T>::generate_pseudo_charge_density_ref() {
    std::vector<Atom> image_atoms;
    Atom_method::generate_valid_images(this->geometry.atoms, image_atoms, this->geometry.cell_type,
                                       this->domain_vertices, this->mesh_control.is_periodic,
                                       this->mesh_control.delta_x, this->mesh_control.delta_y,
                                       this->mesh_control.delta_z,
                                       this->geometry.a, this->geometry.b, this->geometry.c,
                                       this->psp8_files, 1);
    for (std::vector<Atom>::iterator it = image_atoms.begin(); it != image_atoms.end(); ++it) {
        Vertices_3D atom_vertice =
                            this->domain_vertices.get_3D_local_vertices().get_overlap_vertices(it->generate_rc_vertices(
                            this->geometry.cell_type,
                            this->mesh_control.delta_x,
                            this->mesh_control.delta_y,
                            this->mesh_control.delta_z,
                            this->psp8_files[it->type].charge_cut_x,
                            this->psp8_files[it->type].charge_cut_y,
                            this->psp8_files[it->type].charge_cut_z));
        if (atom_vertice.get_size() == 0) continue;
        Array_3D<T> atom_psoducharge_density_ref =
                                    it->generate_pseudo_charge_density_ref(this->geometry.cell_type, atom_vertice,
                                    this->psp8_files[it->type], this->stencil, this->mesh_control.delta_x,
                                    this->mesh_control.delta_y, this->mesh_control.delta_z);
        atom_psoducharge_density_ref.accumulate_overlap(this->pseudo_charge_density_ref);
    }
    return this->pseudo_charge_density_ref;
}

template<typename T>
void Pseudo_charge_solver<T>::generate_pseudo_charge_density_related() {
    std::vector<Atom> image_atoms;
    Atom_method::generate_valid_images(this->geometry.atoms, image_atoms, this->geometry.cell_type,
                                       this->domain_vertices, this->mesh_control.is_periodic,
                                       this->mesh_control.delta_x, this->mesh_control.delta_y,
                                       this->mesh_control.delta_z,
                                       this->geometry.a, this->geometry.b, this->geometry.c,
                                       this->psp8_files, 1);
    this->self_and_correction_energy = (T) 0.0;
    for (std::vector<Atom>::iterator it = image_atoms.begin(); it != image_atoms.end(); ++it) {
        Vertices_3D atom_vertice =
                            this->domain_vertices.get_3D_local_vertices().get_overlap_vertices(it->generate_rc_vertices(
                            this->geometry.cell_type,
                            this->mesh_control.delta_x,
                            this->mesh_control.delta_y,
                            this->mesh_control.delta_z,
                            this->psp8_files[it->type].charge_cut_x,
                            this->psp8_files[it->type].charge_cut_y,
                            this->psp8_files[it->type].charge_cut_z));
        if (atom_vertice.get_size() == 0) continue;
        Array_3D<T> atom_pseudo_charge_density(atom_vertice);
        Array_3D<T> atom_pseudo_charge_density_ref(atom_vertice);
        Array_3D<T> atom_pseudo_charge_density_potiential_correction(atom_vertice);
        this->self_and_correction_energy +=
                it->generate_pseudo_charge_density_related<T>(atom_pseudo_charge_density,
                                                              atom_pseudo_charge_density_ref,
                                                              atom_pseudo_charge_density_potiential_correction,
                                                              this->geometry.cell_type, atom_vertice,
                                                              this->psp8_files[it->type],
                                                              this->stencil, this->mesh_control.delta_x,
                                                              this->mesh_control.delta_y, this->mesh_control.delta_z);
        atom_pseudo_charge_density.accumulate_overlap(this->pseudo_charge_density);
        atom_pseudo_charge_density_ref.accumulate_overlap(this->pseudo_charge_density_ref);
        atom_pseudo_charge_density_potiential_correction.accumulate_overlap(this->pseudo_charge_density_potiential_correction);
    }
    this->cal_self_and_correction_energy();
    this->generate_pseudo_charge();
    return;
}

template<typename T>
T& Pseudo_charge_solver<T>::generate_pseudo_charge() {
    T int_b = this->pseudo_charge_density.vector_sum();
    MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
    MPI_Allreduce(MPI_IN_PLACE, &int_b, 1, mpi_datatype, MPI_SUM, this->domain_vertices.comm);
    this->pseudo_charge = int_b * this->mesh_control.delta_V;
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "Pseudo charge of system = " << std::setprecision(13) << this->pseudo_charge << std::endl;
    }
    return this->pseudo_charge;
}

template<typename T>
T& Pseudo_charge_solver<T>::cal_self_and_correction_energy() {
    uint length = this->domain_vertices.get_3D_local_vertices().get_size();
    Array_3D<T> temp(this->domain_vertices.get_3D_local_vertices());
    Linalg::hadamard_plus_general(temp.data, this->pseudo_charge_density.data, this->pseudo_charge_density_ref.data, length);
    Linalg::hadamard_product_general(temp.data, this->pseudo_charge_density_potiential_correction.data, length);
    this->self_and_correction_energy += Linalg::vector_sum(temp.data, length);
    this->self_and_correction_energy *= (T)(this->mesh_control.delta_V * 0.5);
    MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
    MPI_Allreduce(MPI_IN_PLACE, &(this->self_and_correction_energy), 1, mpi_datatype, MPI_SUM, this->domain_vertices.comm);
    return this->self_and_correction_energy;
}

template<typename T>
void Pseudo_charge_solver<T>::init() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    this->pseudo_charge_density.reconstructor(this->domain_vertices.local_vertices, 0);
    this->pseudo_charge_density_ref.reconstructor(this->domain_vertices.local_vertices, 0);
    this->pseudo_charge_density_potiential_correction.reconstructor(this->domain_vertices.local_vertices, 0);
    // this->generate_pseudo_charge_density();
    // this->generate_pseudo_charge_density_ref();
    this->generate_pseudo_charge_density_related();
    // this->generate_pseudo_charge();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The Pseudo_charge_solver init took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

template<typename T>
template<typename T2>
void Pseudo_charge_solver<T>::init(const Pseudo_charge_solver<T2>& pseudo_charge_solver) {
    this->pseudo_charge = (T)pseudo_charge_solver.pseudo_charge;
    this->self_and_correction_energy = (T)pseudo_charge_solver.self_and_correction_energy;
    this->pseudo_charge_density.deepcopy(
        std::move(pseudo_charge_solver.pseudo_charge_density.as_type(this->pseudo_charge_density.data)));
    this->pseudo_charge_density_ref.deepcopy(
        std::move(pseudo_charge_solver.pseudo_charge_density_ref.as_type(this->pseudo_charge_density_ref.data)));
    this->pseudo_charge_density_potiential_correction.deepcopy(
        std::move(pseudo_charge_solver.pseudo_charge_density_potiential_correction.as_type(
            this->pseudo_charge_density_potiential_correction.data)));
    return;
}
template void Pseudo_charge_solver<float>::init(const Pseudo_charge_solver<float>& pseudo_charge_solver);
template void Pseudo_charge_solver<double>::init(const Pseudo_charge_solver<double>& pseudo_charge_solver);
template void Pseudo_charge_solver<float>::init(const Pseudo_charge_solver<double>& pseudo_charge_solver);
template void Pseudo_charge_solver<double>::init(const Pseudo_charge_solver<float>& pseudo_charge_solver);

template<typename T>
void Pseudo_charge_solver<T>::destructor() {
    this->pseudo_charge_density.destructor();
    this->pseudo_charge_density_ref.destructor();
    this->pseudo_charge_density_potiential_correction.destructor();
    return;
}

template<typename T>
void Pseudo_charge_solver<T>::show() const {
    int block_size = 20;
    int precision = 13;
    int width = 18;
    std::cout << std::right << std::setw(block_size) << "pseudo_charge" << " = "
              << std::left << std::setprecision(precision) << std::setw(width) << std::fixed << this->pseudo_charge << std::endl;
    std::cout << std::right << std::setw(block_size) << "self_and_correction_energy" << " = "
              << std::left << std::setprecision(precision) << std::setw(width) << std::fixed << this->self_and_correction_energy << std::endl;
    this->pseudo_charge_density.show();
    this->pseudo_charge_density_ref.show();
    this->pseudo_charge_density_potiential_correction.show();
    return;
}

template class Pseudo_charge_solver<float>;
template class Pseudo_charge_solver<double>;
