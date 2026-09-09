#include "density_solver.h"
#include <filesystem>
#include <unistd.h>
#include <string>

#ifdef ENABLE_DENSITY_SOLVER_TIMER
#pragma message("Building with ENABLE_DENSITY_SOLVER_TIMER.")
Density_solver_timer::Density_solver_timer() {}
Density_solver_timer::~Density_solver_timer() {}
void Density_solver_timer::reset() {
    this->density_solver.reset();
    this->density_solver_kernel.reset();
    this->chemical_potential.reset();
    this->update_density.reset();
    return;
}
void Density_solver_timer::show(std::ostream& output) const {
    output << std::left << std::setw(20) << "density_solver"             << ": " << this->density_solver.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "  density_solver_kernel"    << ": " << this->density_solver_kernel.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "  chemical_potential"       << ": " << this->chemical_potential.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "  update_density"           << ": " << this->update_density.time_cost_millisecond() << " [ms]" << std::endl;
    return;
}
#endif //ENABLE_DENSITY_SOLVER_TIMER

template<typename T>
Density_solver<T>::Density_solver(const Density_solver_control& density_solver_control,
                                  const Mesh_control& mesh_control,
                                  const Geometry& geometry,
                                  const Spin& spin,
                                  const Stencil<T>& stencil,
                                  const std::vector<Psp8_file>& psp8_files,
                                  const Domain_parallel_vertices_3D& domain_vertices) 
                                : density_solver_control(density_solver_control),
                                  mesh_control(mesh_control),
                                  geometry(geometry),
                                  spin(spin),
                                  stencil(stencil),
                                  psp8_files(psp8_files),
                                  domain_vertices(domain_vertices),
                                  smearing(this->density_solver_control.smearing_control),
                                  domain_band_mpi_package(this->domain_vertices, this->domain_vertices_with_band),
                                  eigen_solver(this->density_solver_control.eigen_solver_control,
                                               this->mesh_control,
                                               this->geometry,
                                               this->stencil,
                                               this->psp8_files,
                                               this->domain_vertices_with_band),
                                  density_matrix_solver(this->density_solver_control.density_matrix_solver_control,
                                                        this->mesh_control,
                                                        this->geometry,
                                                        this->stencil,
                                                        this->domain_vertices_with_band) {}

template<typename T>
Density_solver<T>::~Density_solver() {}

template<typename T>
void Density_solver<T>::generate_initial_electron_density() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::vector<Atom> image_atoms;
    Atom_method::generate_valid_images(this->geometry.atoms, image_atoms, this->geometry.cell_type,
                                       this->domain_vertices, this->mesh_control.is_periodic,
                                       this->mesh_control.delta_x, this->mesh_control.delta_y, this->mesh_control.delta_z,
                                       this->geometry.a, this->geometry.b, this->geometry.c,
                                       this->psp8_files, 0);
    for (std::vector<Atom>::iterator it = image_atoms.begin(); it != image_atoms.end(); ++it) {
        Vertices_3D atom_vertice =
                            this->domain_vertices.local_vertices.get_overlap_vertices(it->generate_rc_vertices(
                            this->geometry.cell_type, this->mesh_control.delta_x, this->mesh_control.delta_y,
                            this->mesh_control.delta_z, this->psp8_files[it->type].charge_cut_x,
                            this->psp8_files[it->type].charge_cut_y, this->psp8_files[it->type].charge_cut_z));
        if (atom_vertice.get_size() == 0) continue;
        Array_3D<T> atom_initial_electron_density =
                                    it->generate_initial_electron_density_array<T>(this->geometry.cell_type,
                                    atom_vertice.generate_ex_vertices(this->stencil.FDn),
                                    this->psp8_files[it->type], this->mesh_control.delta_x, this->mesh_control.delta_y,
                                    this->mesh_control.delta_z);
        atom_initial_electron_density.accumulate_overlap(this->electron_density_init);
        if (this->psp8_files[it->type].fchrg > 1e-12) {
            Array_3D<T> atom_initial_electron_density_core =
                                    it->generate_initial_electron_density_core_array<T>(this->geometry.cell_type, atom_vertice,
                                    this->psp8_files[it->type], this->mesh_control.delta_x, this->mesh_control.delta_y,
                                    this->mesh_control.delta_z);
            atom_initial_electron_density_core.accumulate_overlap(this->electron_density_core);
        }
        if (this->spin.get_spin_type() == 1) {
            assert(false);
            // Array_3D<T> atom_initial_magnetization_z = atom_initial_electron_density * (it->atom_spin[2] / this->psp8_files[it->type].zion);
            // atom_initial_magnetization_z.accumulate_overlap(this->magnetization_z);
        } else if (this->spin.get_spin_type() > 1) {
            assert(this->spin.get_spin_type() <= 1);
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The generate_initial_electron_density took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

template<typename T>
void Density_solver<T>::cal_magnetization_z() {
    assert(this->spin.get_spin_type() <= 1);
    if (this->spin.get_spin_type() == 1) {
        Linalg::hadamard_minus_general(this->magnetization_z.data,
                                       this->electron_densities[0].data,
                                       this->electron_densities[1].data,
                                       this->magnetization_z.length);
        T total_magz = this->mesh_control.delta_V * Linalg::vector_sum(this->magnetization_z.data,
            this->magnetization_z.length, this->domain_vertices.comm);
        if (this->domain_vertices.get_comm_rank() == 0) {
            std::cout << "total_magz = "
                    << std::fixed << std::setprecision(6) << std::setw(10) << total_magz
                    << std::endl;
        }
    }
    return;
}

template<typename T>
T& Density_solver<T>::generate_electron_charge() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    T int_rho = this->electron_density_init.vector_sum();
    MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
    MPI_Allreduce(MPI_IN_PLACE, &int_rho, 1, mpi_datatype, MPI_SUM, this->domain_vertices.comm);
    this->electron_charge = int_rho * this->mesh_control.delta_V;
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "Electron charge of system = " << std::setprecision(13) << this->electron_charge << std::endl;
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The generate_initial_electron_charge took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return this->electron_charge;
}

template<typename T>
void Density_solver<T>::solve_eigen_problem(const Array_3D<T>& effective_potentail_loc) {
    Array_3D<T> effective_potentail_loc_reshape(this->domain_vertices_with_band.get_3D_local_vertices());
    this->domain_band_mpi_package.send_data(effective_potentail_loc, effective_potentail_loc_reshape);
    bool print_flag = this->domain_vertices.get_domain_3d_comm_rank() == 0 ? true : false;
    this->eigen_solver.run(effective_potentail_loc_reshape, print_flag);
    return;
}

template<typename T>
void Density_solver<T>::solve_density_matrix(const Array_3D<T>& effective_potentail_loc) {
    this->solve_density_matrix_mp(effective_potentail_loc.data, effective_potentail_loc.get_vertices());
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // Array_3D<T> effective_potentail_loc_reshape(this->domain_vertices_with_band.get_3D_local_vertices());
    // this->domain_band_mpi_package.send_data(effective_potentail_loc, effective_potentail_loc_reshape);
    // this->density_matrix_solver.run(effective_potentail_loc_reshape);
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // if (this->domain_vertices.get_comm_rank() == 0) {
    //     std::cout << "The solve_density_matrix run took " << Tools::time_cost(begin, end) << "." << std::endl;
    // }
    return;
}

template<typename T>
void Density_solver<T>::solve_density_matrix_mp(T const* const effective_potentail_loc,
                                                const Vertices_3D& effective_potentail_loc_vertices) {
    constexpr size_t GB = 1024 * 1024 * 1024 / sizeof(T);
    Memory_pool<T, Fast_memory> pool_fast(3.75 * GB);
    Memory_pool<T, Capacity_memory> pool_cap(3.75 * GB);
    this->solve_density_matrix_mp(effective_potentail_loc, effective_potentail_loc_vertices, pool_fast, pool_cap);
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // Array_3D<T> effective_potentail_loc_reshape(this->domain_vertices_with_band.get_3D_local_vertices());
    // this->domain_band_mpi_package.send_data(effective_potentail_loc, effective_potentail_loc_vertices,
    //                                         effective_potentail_loc_reshape.data,
    //                                         this->domain_vertices_with_band.get_3D_local_vertices());
    // this->density_matrix_solver.run(effective_potentail_loc_reshape);
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // if (this->domain_vertices.get_comm_rank() == 0) {
    //     std::cout << "The solve_density_matrix run took " << Tools::time_cost(begin, end) << "." << std::endl;
    // }
    return;
}

template<typename T>
void Density_solver<T>::solve_density_matrix_mp(T const* const effective_potentail_loc,
                                                const Vertices_3D& effective_potentail_loc_vertices,
                                                Memory_pool<T, Fast_memory>& pool_fast,
                                                Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // Array_3D<T> effective_potentail_loc_reshape(this->domain_vertices_with_band.get_3D_local_vertices());
    if (this->domain_band_mpi_package.need_comm) {
        T* effective_potentail_loc_reshape = pool_cap.allocate(this->domain_vertices_with_band.get_3D_local_vertices().get_size());;
        this->domain_band_mpi_package.send_data_mp(effective_potentail_loc, effective_potentail_loc_vertices,
                                                effective_potentail_loc_reshape,
                                                this->domain_vertices_with_band.get_3D_local_vertices(),
                                                pool_fast, pool_cap);
        this->density_matrix_solver.run_mp(effective_potentail_loc_reshape, pool_fast, pool_cap);
    } else {
        this->density_matrix_solver.run_mp(effective_potentail_loc, pool_fast, pool_cap);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The solve_density_matrix run took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

template<typename T>
T& Density_solver<T>::evaluate_chemical_potential() {
    #ifdef ENABLE_DENSITY_SOLVER_TIMER
        this->density_solver_timer.chemical_potential.start();
    #endif //ENABLE_DENSITY_SOLVER_TIMER
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->density_solver_control.method == 0) {  // eigen_solver
        this->chemical_potential = this->eigen_solver.evaluate_chemical_potential(
                this->smearing, this->electron_charge, (T)this->spin.generate_smearing_coef());
    } else if (this->density_solver_control.method == 1) {  // density_matrix_solver
        this->chemical_potential = this->density_matrix_solver.evaluate_chemical_potential(
                this->smearing, this->electron_charge, (T)this->spin.generate_smearing_coef(),
                this->chemical_potential);
    } else {
        assert(this->density_solver_control.method == 0
            || this->density_solver_control.method == 1);
    }
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "Chemical potential is " << this->chemical_potential << std::endl;
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The evaluate_chemical_potential took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    #ifdef ENABLE_DENSITY_SOLVER_TIMER
        this->density_solver_timer.chemical_potential.stop();
    #endif //ENABLE_DENSITY_SOLVER_TIMER
    return this->chemical_potential;
}

template<typename T>
T& Density_solver<T>::evaluate_chemical_potential_mp(
                Memory_pool<T, Fast_memory>& pool_fast,
                Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    #ifdef ENABLE_DENSITY_SOLVER_TIMER
        this->density_solver_timer.chemical_potential.start();
    #endif //ENABLE_DENSITY_SOLVER_TIMER
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->density_solver_control.method == 0) {  // eigen_solver
        this->chemical_potential = this->eigen_solver.evaluate_chemical_potential(
                this->smearing, this->electron_charge, (T)this->spin.generate_smearing_coef());
    } else if (this->density_solver_control.method == 1) {  // density_matrix_solver
        this->chemical_potential = this->density_matrix_solver.evaluate_chemical_potential_mp(
                this->smearing, this->electron_charge, (T)this->spin.generate_smearing_coef(),
                this->chemical_potential, pool_fast, pool_cap);
    } else {
        assert(this->density_solver_control.method == 0
            || this->density_solver_control.method == 1);
    }
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "Chemical potential is " << this->chemical_potential << std::endl;
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The evaluate_chemical_potential took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    #ifdef ENABLE_DENSITY_SOLVER_TIMER
        this->density_solver_timer.chemical_potential.stop();
    #endif //ENABLE_DENSITY_SOLVER_TIMER
    return this->chemical_potential;
}

template<typename T>
T& Density_solver<T>::evaluate_band_energy() {
    if (this->density_solver_control.method == 0) {  // eigen_solver
        this->band_energy = this->eigen_solver.evaluate_band_energy(this->smearing, this->chemical_potential, (T)this->spin.generate_smearing_coef());
    } else if (this->density_solver_control.method == 1) { // density matrix solver
        this->band_energy = this->density_matrix_solver.evaluate_band_energy(this->smearing, this->chemical_potential, (T)this->spin.generate_smearing_coef());
    } else {
        assert(this->density_solver_control.method == 0
            || this->density_solver_control.method == 1);
    }
    return this->band_energy;
}

template<typename T>
T& Density_solver<T>::evaluate_entropy_energy() {
    if (this->density_solver_control.method == 0) {  // eigen_solver
        this->entropy_energy = this->eigen_solver.evaluate_entropy_energy(this->smearing, this->chemical_potential, (T)this->spin.generate_smearing_coef());
    } else if (this->density_solver_control.method == 1) { // density matrix solver
        this->entropy_energy = this->density_matrix_solver.evaluate_entropy_energy(this->smearing, this->chemical_potential, (T)this->spin.generate_smearing_coef());
    } else {
        assert(this->density_solver_control.method == 0);
    }
    return this->entropy_energy;
}

template<typename T>
void Density_solver<T>::update_electron_density() {
    #ifdef ENABLE_DENSITY_SOLVER_TIMER
        this->density_solver_timer.update_density.start();
    #endif //ENABLE_DENSITY_SOLVER_TIMER
    assert(this->spin.get_spin_type() == 0);
    if (this->density_solver_control.method == 1) {  // density_matrix_solver
        this->density_matrix_solver.update_electron_density(this->electron_densities[0], this->smearing,
                                                            this->chemical_potential, (T)this->spin.generate_smearing_coef(),
                                                            this->mesh_control.delta_V, this->domain_band_mpi_package);
    } else if (this->density_solver_control.method == 0) {  // eigen_solver
        this->eigen_solver.update_electron_density(this->electron_densities[0], this->smearing,
                                                   this->chemical_potential, (T)this->spin.generate_smearing_coef(),
                                                   this->mesh_control.delta_V, this->domain_band_mpi_package);
    } else {
        assert(this->density_solver_control.method == 0
            || this->density_solver_control.method == 1);
    }
    #ifdef ENABLE_DENSITY_SOLVER_TIMER
        this->density_solver_timer.update_density.stop();
    #endif //ENABLE_DENSITY_SOLVER_TIMER
    return;
}

template<typename T>
void Density_solver<T>::update_electron_density_mp(
                Memory_pool<T, Fast_memory>& pool_fast, 
                Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    #ifdef ENABLE_DENSITY_SOLVER_TIMER
        this->density_solver_timer.update_density.start();
    #endif //ENABLE_DENSITY_SOLVER_TIMER
    assert(this->spin.get_spin_type() == 0);
    if (this->density_solver_control.method == 1) {  // density_matrix_solver
        this->density_matrix_solver.update_electron_density_mp(this->electron_densities[0], this->smearing,
                                                            this->chemical_potential, (T)this->spin.generate_smearing_coef(),
                                                            this->mesh_control.delta_V, this->domain_band_mpi_package,
                                                            pool_fast, pool_cap);
    } else if (this->density_solver_control.method == 0) {  // eigen_solver
        this->eigen_solver.update_electron_density(this->electron_densities[0], this->smearing,
                                                   this->chemical_potential, (T)this->spin.generate_smearing_coef(),
                                                   this->mesh_control.delta_V, this->domain_band_mpi_package);
    } else {
        assert(this->density_solver_control.method == 0
            || this->density_solver_control.method == 1);
    }
    #ifdef ENABLE_DENSITY_SOLVER_TIMER
        this->density_solver_timer.update_density.stop();
    #endif //ENABLE_DENSITY_SOLVER_TIMER
    return;
}

template<typename T>
void Density_solver<T>::cal_nonlocal_forces(Array_2D<T>& nonlocal_forces) const {
    if (this->density_solver_control.method == 0) { //eigen_solver
        this->eigen_solver.cal_nonlocal_forces(nonlocal_forces, this->chemical_potential, this->smearing, this->spin);
    } else if (this->density_solver_control.method == 1) { //density_matrix_solver
        this->density_matrix_solver.cal_nonlocal_forces(nonlocal_forces, this->chemical_potential, this->smearing,
                                                        this->spin, this->psp8_files);
    } else {
        assert(this->density_solver_control.eigen_solver_control.method == 0
            || this->density_solver_control.eigen_solver_control.method == 1);
    }
    return;
}

template<typename T>
void Density_solver<T>::print_eigens(const std::string& fname) const {
    if (this->density_solver_control.method == 0) {  // eigen_solver
        this->eigen_solver.print_eigens(this->chemical_potential, this->smearing,
                                   this->spin, fname);
    } else if (this->density_solver_control.method == 1) { // density_matrix_solver
        this->density_matrix_solver.print_eigens(this->chemical_potential, this->smearing,
                                   this->spin, fname);
    } else {
        assert(this->density_solver_control.method == 0 || this->density_solver_control.method == 1);
    }
    return;
}

template<typename T>
void Density_solver<T>::print_pdos(const std::string& fname) const {
    if (this->density_solver_control.method == 0) {  // eigen_solver
        this->eigen_solver.print_pdos(this->chemical_potential, this->spin, fname);
    } else if (this->density_solver_control.method == 1) { // density_matrix_solver
        this->density_matrix_solver.print_pdos(this->chemical_potential, this->spin, fname);
    } else {
        assert(this->density_solver_control.method == 0 || this->density_solver_control.method == 1);
    }
    return;
}

template<typename T>
double Density_solver<T>::evalutate_flops() {
    double flops = 0.0;
    if (this->density_solver_control.method == 0) {  // eigen_solver
        flops += this->eigen_solver.evalutate_flops();
    } else if (this->density_solver_control.method == 1) { // density_matrix_solver
        flops += this->density_matrix_solver.evalutate_flops();
    } else {
        assert(this->density_solver_control.method == 0 || this->density_solver_control.method == 1);
    }
    return flops;
}

template<typename T>
void Density_solver<T>::run(const std::vector<Array_3D<T>>& effective_potentail_locs) {
    T const* effective_potentail_locs_data[2];
    for (uint ispin = 0; ispin < this->spin.generate_nspin(); ispin++) {
        effective_potentail_locs_data[ispin] = effective_potentail_locs[ispin].data;
    }
    this->run_mp(effective_potentail_locs_data);
    // #ifdef ENABLE_DENSITY_SOLVER_TIMER
    //     this->density_solver_timer.reset();
    //     this->density_solver_timer.density_solver.start();
    // #endif //ENABLE_DENSITY_SOLVER_TIMER
    // assert(this->spin.get_spin_type() == 0);
    // const Array_3D<T>& effective_potentail_loc = effective_potentail_locs[0];
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // #ifdef ENABLE_DENSITY_SOLVER_TIMER
    //     this->density_solver_timer.density_solver_kernel.start();
    // #endif //ENABLE_DENSITY_SOLVER_TIMER
    // if (this->density_solver_control.method == 0) {  // eigen_solver
    //     this->solve_eigen_problem(effective_potentail_loc);
    // } else if (this->density_solver_control.method == 1) { // density_matrix_solver
    //     this->solve_density_matrix(effective_potentail_loc);
    // } else {
    //     assert(this->density_solver_control.method == 0 || this->density_solver_control.method == 1);
    // }
    // #ifdef ENABLE_DENSITY_SOLVER_TIMER
    //     this->density_solver_timer.density_solver_kernel.stop();
    // #endif //ENABLE_DENSITY_SOLVER_TIMER
    // this->evaluate_chemical_potential();
    // // this->evaluate_band_energy();
    // // this->evaluate_entropy_energy();
    // this->update_electron_density();
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // if (this->domain_vertices.get_comm_rank() == 0) {
    //     std::cout << "The density_solver run took " << Tools::time_cost(begin, end) << "." << std::endl;
    // }
    // #ifdef ENABLE_DENSITY_SOLVER_TIMER
    //     this->density_solver_timer.density_solver.stop();
    //     if (this->domain_vertices.get_comm_rank() == 0) this->density_solver_timer.show();
    // #endif //ENABLE_DENSITY_SOLVER_TIMER
    // // #ifdef ENABLE_TIMER
    // //     MPI_Barrier(this->domain_vertices.comm);
    // //     const std::string dir_name = "TIMER";
    // //     bool if_has_directory_root = false;
    // //     std::filesystem::path dir_root(dir_name);
    // //     if (this->domain_vertices.get_comm_rank() == 0) {
    // //         if (!std::filesystem::exists(dir_root)) {
    // //             if (!std::filesystem::create_directories(dir_root)) {
    // //                 std::cerr << "Failed to create directory: " << dir_name << std::endl;
    // //             } else {
    // //                 if_has_directory_root = true;
    // //             }
    // //         } else {
    // //             if_has_directory_root = true;
    // //         }
    // //     }
    // //     MPI_Allreduce(MPI_IN_PLACE, &if_has_directory_root, 1, MPI_C_BOOL, MPI_LOR, this->domain_vertices.comm);
    // //     if (!if_has_directory_root) return;
    // //     uint comm_i = this->domain_vertices.get_active_comm_i();
    // //     uint comm_j = this->domain_vertices.get_active_comm_j();
    // //     uint comm_k = this->domain_vertices.get_active_comm_k();
    // //     uint comm_b = this->domain_vertices.get_active_comm_b();
    // //     char subdir_k[20];
    // //     char subdir_j[20];
    // //     char subdir_i[20];
    // //     std::snprintf(subdir_k, sizeof(subdir_k), "comm_k_%d", comm_k);
    // //     std::snprintf(subdir_j, sizeof(subdir_j), "comm_j_%d", comm_j);
    // //     std::snprintf(subdir_i, sizeof(subdir_i), "comm_i_%d", comm_i);
    // //     std::filesystem::path dir_k = dir_root / subdir_k;
    // //     std::filesystem::path dir_j = dir_k / subdir_j;
    // //     std::filesystem::path dir_i = dir_j / subdir_i;

    // //     bool if_has_directory_k = false;
    // //     if (comm_b == 0 && comm_i == 0 && comm_j == 0) {
    // //         if (!std::filesystem::exists(dir_k)) {
    // //             if (!std::filesystem::create_directories(dir_k)) {
    // //                 std::cerr << "Failed to create directory: " << dir_k << std::endl;
    // //             } else {
    // //                 if_has_directory_k = true;
    // //             }
    // //         } else {
    // //             if_has_directory_k = true;
    // //         }
    // //     }
    // //     MPI_Allreduce(MPI_IN_PLACE, &if_has_directory_k, 1, MPI_C_BOOL, MPI_LOR, this->domain_vertices.get_domain_3d_comm());
    // //     if (!if_has_directory_k) return;

    // //     bool if_has_directory_j = false;
    // //     if (comm_b == 0 && comm_i == 0) {
    // //         if (!std::filesystem::exists(dir_j)) {
    // //             if (!std::filesystem::create_directories(dir_j)) {
    // //                 std::cerr << "Failed to create directory: " << dir_j << std::endl;
    // //             } else {
    // //                 if_has_directory_j = true;
    // //             }
    // //         } else {
    // //             if_has_directory_j = true;
    // //         }
    // //     }
    // //     MPI_Allreduce(MPI_IN_PLACE, &if_has_directory_j, 1, MPI_C_BOOL, MPI_LOR, this->domain_vertices.get_domain_3d_comm());
    // //     if (!if_has_directory_j) return;

    // //     bool if_has_directory_i = false;
    // //     if (comm_b == 0) {
    // //         if (!std::filesystem::exists(dir_i)) {
    // //             if (!std::filesystem::create_directories(dir_i)) {
    // //                 std::cerr << "Failed to create directory: " << dir_i << std::endl;
    // //             } else {
    // //                 if_has_directory_i = true;
    // //             }
    // //         } else {
    // //             if_has_directory_i = true;
    // //         }
    // //     }
    // //     MPI_Allreduce(MPI_IN_PLACE, &if_has_directory_i, 1, MPI_C_BOOL, MPI_LOR, this->domain_vertices.get_domain_3d_comm());
    // //     if (!if_has_directory_i) return;
    // //     // dir_i the final directory to write timer files
    // //     std::string filename = "timer_rank_" + std::to_string(this->domain_vertices.get_comm_rank()) + ".txt";
    // //     std::filesystem::path full_path = std::filesystem::path(dir_i) / filename;
    // //     std::ofstream ofs(full_path, std::ios::app);
    // //     if (!ofs.is_open()) {
    // //         std::cerr << "Failed to open file: " << full_path << std::endl;
    // //         return;
    // //     }
    // //     std::string hostname(256, '\0');
    // //     gethostname(hostname.data(), hostname.size());
    // //     ofs << "Hostname: " << hostname.c_str() << std::endl;
    // //     for (uint ielement = 0; ielement < this->density_matrix_solver.xlsdft.local_element_num; ielement++) {
    // //         ofs << "Element " << ielement << " in " << this->density_matrix_solver.xlsdft.local_element_num << ":" << std::endl;
    // //         this->density_matrix_solver.xlsdft.eigen_solvers[ielement].chefsi.chefsi_timer.show(ofs);
    // //         this->density_matrix_solver.xlsdft.eigen_solvers[ielement].eigen_solver_timer.show(ofs);
    // //     }
    // //     ofs << "Xlsdft::run" << std::endl;
    // //     this->density_matrix_solver.xlsdft.xlsdft_timer.show(ofs);
    // //     ofs << "Density_matrix_solver::run" << std::endl;
    // //     this->density_matrix_solver.density_matrix_solver_timer.show(ofs);
    // //     ofs << "Density_solver::run" << std::endl;
    // //     this->density_solver_timer.show(ofs);
    // //     ofs.close();
    // //     MPI_Barrier(this->domain_vertices.comm);
    // // #endif //ENABLE_TIMER
    return;
}

template<typename T>
void Density_solver<T>::run_mp(T const* const* const effective_potentail_locs) {
    constexpr size_t GB = 1024 * 1024 * 1024 / sizeof(T);
    Memory_pool<T, Fast_memory> pool_fast(3.75 * GB);
    Memory_pool<T, Capacity_memory> pool_cap(3.75 * GB);
    this->run_mp(effective_potentail_locs, pool_fast, pool_cap);
    // #ifdef ENABLE_DENSITY_SOLVER_TIMER
    //     this->density_solver_timer.reset();
    //     this->density_solver_timer.density_solver.start();
    // #endif //ENABLE_DENSITY_SOLVER_TIMER
    // assert(this->spin.get_spin_type() == 0);
    // // const Array_3D<T>& effective_potentail_loc = effective_potentail_locs[0];
    // T const* const effective_potentail_loc = effective_potentail_locs[0];
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // #ifdef ENABLE_DENSITY_SOLVER_TIMER
    //     this->density_solver_timer.density_solver_kernel.start();
    // #endif //ENABLE_DENSITY_SOLVER_TIMER
    // if (this->density_solver_control.method == 1) { // density_matrix_solver
    //     this->solve_density_matrix_mp(effective_potentail_loc, this->domain_vertices.get_3D_local_vertices());
    // } else if (this->density_solver_control.method == 0) {  // eigen_solver
    //     // this->solve_eigen_problem(effective_potentail_loc);
    //     assert(false);
    // } else {
    //     assert(this->density_solver_control.method == 0 || this->density_solver_control.method == 1);
    // }
    // #ifdef ENABLE_DENSITY_SOLVER_TIMER
    //     this->density_solver_timer.density_solver_kernel.stop();
    // #endif //ENABLE_DENSITY_SOLVER_TIMER
    // this->evaluate_chemical_potential();
    // // this->evaluate_band_energy();
    // // this->evaluate_entropy_energy();
    // this->update_electron_density();
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // if (this->domain_vertices.get_comm_rank() == 0) {
    //     std::cout << "The density_solver run took " << Tools::time_cost(begin, end) << "." << std::endl;
    // }
    // #ifdef ENABLE_DENSITY_SOLVER_TIMER
    //     this->density_solver_timer.density_solver.stop();
    //     if (this->domain_vertices.get_comm_rank() == 0) this->density_solver_timer.show();
    // #endif //ENABLE_DENSITY_SOLVER_TIMER
    // #ifdef ENABLE_TIMER
    //     MPI_Barrier(this->domain_vertices.comm);
    //     const std::string dir_name = "TIMER";
    //     bool if_has_directory_root = false;
    //     std::filesystem::path dir_root(dir_name);
    //     if (this->domain_vertices.get_comm_rank() == 0) {
    //         if (!std::filesystem::exists(dir_root)) {
    //             if (!std::filesystem::create_directories(dir_root)) {
    //                 std::cerr << "Failed to create directory: " << dir_name << std::endl;
    //             } else {
    //                 if_has_directory_root = true;
    //             }
    //         } else {
    //             if_has_directory_root = true;
    //         }
    //     }
    //     MPI_Allreduce(MPI_IN_PLACE, &if_has_directory_root, 1, MPI_C_BOOL, MPI_LOR, this->domain_vertices.comm);
    //     if (!if_has_directory_root) return;
    //     uint comm_i = this->domain_vertices.get_active_comm_i();
    //     uint comm_j = this->domain_vertices.get_active_comm_j();
    //     uint comm_k = this->domain_vertices.get_active_comm_k();
    //     uint comm_b = this->domain_vertices.get_active_comm_b();
    //     char subdir_k[20];
    //     char subdir_j[20];
    //     char subdir_i[20];
    //     std::snprintf(subdir_k, sizeof(subdir_k), "comm_k_%d", comm_k);
    //     std::snprintf(subdir_j, sizeof(subdir_j), "comm_j_%d", comm_j);
    //     std::snprintf(subdir_i, sizeof(subdir_i), "comm_i_%d", comm_i);
    //     std::filesystem::path dir_k = dir_root / subdir_k;
    //     std::filesystem::path dir_j = dir_k / subdir_j;
    //     std::filesystem::path dir_i = dir_j / subdir_i;

    //     bool if_has_directory_k = false;
    //     if (comm_b == 0 && comm_i == 0 && comm_j == 0) {
    //         if (!std::filesystem::exists(dir_k)) {
    //             if (!std::filesystem::create_directories(dir_k)) {
    //                 std::cerr << "Failed to create directory: " << dir_k << std::endl;
    //             } else {
    //                 if_has_directory_k = true;
    //             }
    //         } else {
    //             if_has_directory_k = true;
    //         }
    //     }
    //     MPI_Allreduce(MPI_IN_PLACE, &if_has_directory_k, 1, MPI_C_BOOL, MPI_LOR, this->domain_vertices.get_domain_3d_comm());
    //     if (!if_has_directory_k) return;

    //     bool if_has_directory_j = false;
    //     if (comm_b == 0 && comm_i == 0) {
    //         if (!std::filesystem::exists(dir_j)) {
    //             if (!std::filesystem::create_directories(dir_j)) {
    //                 std::cerr << "Failed to create directory: " << dir_j << std::endl;
    //             } else {
    //                 if_has_directory_j = true;
    //             }
    //         } else {
    //             if_has_directory_j = true;
    //         }
    //     }
    //     MPI_Allreduce(MPI_IN_PLACE, &if_has_directory_j, 1, MPI_C_BOOL, MPI_LOR, this->domain_vertices.get_domain_3d_comm());
    //     if (!if_has_directory_j) return;

    //     bool if_has_directory_i = false;
    //     if (comm_b == 0) {
    //         if (!std::filesystem::exists(dir_i)) {
    //             if (!std::filesystem::create_directories(dir_i)) {
    //                 std::cerr << "Failed to create directory: " << dir_i << std::endl;
    //             } else {
    //                 if_has_directory_i = true;
    //             }
    //         } else {
    //             if_has_directory_i = true;
    //         }
    //     }
    //     MPI_Allreduce(MPI_IN_PLACE, &if_has_directory_i, 1, MPI_C_BOOL, MPI_LOR, this->domain_vertices.get_domain_3d_comm());
    //     if (!if_has_directory_i) return;
    //     // dir_i the final directory to write timer files
    //     std::string filename = "timer_rank_" + std::to_string(this->domain_vertices.get_comm_rank()) + ".txt";
    //     std::filesystem::path full_path = std::filesystem::path(dir_i) / filename;
    //     std::ofstream ofs(full_path, std::ios::app);
    //     if (!ofs.is_open()) {
    //         std::cerr << "Failed to open file: " << full_path << std::endl;
    //         return;
    //     }
    //     std::string hostname(256, '\0');
    //     gethostname(hostname.data(), hostname.size());
    //     ofs << "Hostname: " << hostname.c_str() << std::endl;
    //     for (uint ielement = 0; ielement < this->density_matrix_solver.xlsdft.local_element_num; ielement++) {
    //         ofs << "Element " << ielement << " in " << this->density_matrix_solver.xlsdft.local_element_num << ":" << std::endl;
    //         this->density_matrix_solver.xlsdft.eigen_solvers[ielement].chefsi.chefsi_timer.show(ofs);
    //         this->density_matrix_solver.xlsdft.eigen_solvers[ielement].eigen_solver_timer.show(ofs);
    //     }
    //     ofs << "Xlsdft::run" << std::endl;
    //     this->density_matrix_solver.xlsdft.xlsdft_timer.show(ofs);
    //     ofs << "Density_matrix_solver::run" << std::endl;
    //     this->density_matrix_solver.density_matrix_solver_timer.show(ofs);
    //     ofs << "Density_solver::run" << std::endl;
    //     this->density_solver_timer.show(ofs);
    //     ofs.close();
    //     MPI_Barrier(this->domain_vertices.comm);
    // #endif //ENABLE_TIMER
    return;
}

template<typename T>
void Density_solver<T>::reserve_retained_packed_pool(Memory_pool<T, Fast_memory>& pool_fast) {
    if (this->density_solver_control.method == 1) {
        this->density_matrix_solver.reserve_retained_packed_pool(pool_fast);
    }
    return;
}

template<typename T>
void Density_solver<T>::run_mp(T const* const* const effective_potentail_locs,
                                    Memory_pool<T, Fast_memory>& pool_fast,
                                    Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    #ifdef ENABLE_DENSITY_SOLVER_TIMER
        this->density_solver_timer.reset();
        this->density_solver_timer.density_solver.start();
    #endif //ENABLE_DENSITY_SOLVER_TIMER
    assert(this->spin.get_spin_type() == 0);
    // const Array_3D<T>& effective_potentail_loc = effective_potentail_locs[0];
    T const* const effective_potentail_loc = effective_potentail_locs[0];
    #ifdef ENABLE_DENSITY_SOLVER_TIMER
        this->density_solver_timer.density_solver_kernel.start();
    #endif //ENABLE_DENSITY_SOLVER_TIMER
    if (this->density_solver_control.method == 1) { // density_matrix_solver
        this->solve_density_matrix_mp(effective_potentail_loc, this->domain_vertices.get_3D_local_vertices(),
                                        pool_fast, pool_cap);
    } else if (this->density_solver_control.method == 0) {  // eigen_solver
        // this->solve_eigen_problem(effective_potentail_loc);
        assert(false);
    } else {
        assert(this->density_solver_control.method == 0 || this->density_solver_control.method == 1);
    }
    #ifdef ENABLE_DENSITY_SOLVER_TIMER
        this->density_solver_timer.density_solver_kernel.stop();
    #endif //ENABLE_DENSITY_SOLVER_TIMER
    this->evaluate_chemical_potential_mp(pool_fast, pool_cap);
    // this->evaluate_band_energy();
    // this->evaluate_entropy_energy();
    this->update_electron_density_mp(pool_fast, pool_cap);
    #ifdef ENABLE_DENSITY_SOLVER_TIMER
        this->density_solver_timer.density_solver.stop();
        if (this->domain_vertices.get_comm_rank() == 0) this->density_solver_timer.show();
        // #if defined(ENABLE_TIMER)
        // this->print_timer_statistics(this->domain_vertices.get_comm_rank() == 0, std::cout);
        // #endif
    #endif //ENABLE_DENSITY_SOLVER_TIMER
    // #ifdef ENABLE_TIMER
    //     MPI_Barrier(this->domain_vertices.comm);
    //     const std::string dir_name = "TIMER";
    //     bool if_has_directory_root = false;
    //     std::filesystem::path dir_root(dir_name);
    //     if (this->domain_vertices.get_comm_rank() == 0) {
    //         if (!std::filesystem::exists(dir_root)) {
    //             if (!std::filesystem::create_directories(dir_root)) {
    //                 std::cerr << "Failed to create directory: " << dir_name << std::endl;
    //             } else {
    //                 if_has_directory_root = true;
    //             }
    //         } else {
    //             if_has_directory_root = true;
    //         }
    //     }
    //     MPI_Allreduce(MPI_IN_PLACE, &if_has_directory_root, 1, MPI_C_BOOL, MPI_LOR, this->domain_vertices.comm);
    //     if (!if_has_directory_root) return;
    //     uint comm_i = this->domain_vertices.get_active_comm_i();
    //     uint comm_j = this->domain_vertices.get_active_comm_j();
    //     uint comm_k = this->domain_vertices.get_active_comm_k();
    //     uint comm_b = this->domain_vertices.get_active_comm_b();
    //     char subdir_k[20];
    //     char subdir_j[20];
    //     char subdir_i[20];
    //     std::snprintf(subdir_k, sizeof(subdir_k), "comm_k_%d", comm_k);
    //     std::snprintf(subdir_j, sizeof(subdir_j), "comm_j_%d", comm_j);
    //     std::snprintf(subdir_i, sizeof(subdir_i), "comm_i_%d", comm_i);
    //     std::filesystem::path dir_k = dir_root / subdir_k;
    //     std::filesystem::path dir_j = dir_k / subdir_j;
    //     std::filesystem::path dir_i = dir_j / subdir_i;

    //     bool if_has_directory_k = false;
    //     if (comm_b == 0 && comm_i == 0 && comm_j == 0) {
    //         if (!std::filesystem::exists(dir_k)) {
    //             if (!std::filesystem::create_directories(dir_k)) {
    //                 std::cerr << "Failed to create directory: " << dir_k << std::endl;
    //             } else {
    //                 if_has_directory_k = true;
    //             }
    //         } else {
    //             if_has_directory_k = true;
    //         }
    //     }
    //     MPI_Allreduce(MPI_IN_PLACE, &if_has_directory_k, 1, MPI_C_BOOL, MPI_LOR, this->domain_vertices.get_domain_3d_comm());
    //     if (!if_has_directory_k) return;

    //     bool if_has_directory_j = false;
    //     if (comm_b == 0 && comm_i == 0) {
    //         if (!std::filesystem::exists(dir_j)) {
    //             if (!std::filesystem::create_directories(dir_j)) {
    //                 std::cerr << "Failed to create directory: " << dir_j << std::endl;
    //             } else {
    //                 if_has_directory_j = true;
    //             }
    //         } else {
    //             if_has_directory_j = true;
    //         }
    //     }
    //     MPI_Allreduce(MPI_IN_PLACE, &if_has_directory_j, 1, MPI_C_BOOL, MPI_LOR, this->domain_vertices.get_domain_3d_comm());
    //     if (!if_has_directory_j) return;

    //     bool if_has_directory_i = false;
    //     if (comm_b == 0) {
    //         if (!std::filesystem::exists(dir_i)) {
    //             if (!std::filesystem::create_directories(dir_i)) {
    //                 std::cerr << "Failed to create directory: " << dir_i << std::endl;
    //             } else {
    //                 if_has_directory_i = true;
    //             }
    //         } else {
    //             if_has_directory_i = true;
    //         }
    //     }
    //     MPI_Allreduce(MPI_IN_PLACE, &if_has_directory_i, 1, MPI_C_BOOL, MPI_LOR, this->domain_vertices.get_domain_3d_comm());
    //     if (!if_has_directory_i) return;
    //     // dir_i the final directory to write timer files
    //     std::string filename = "timer_rank_" + std::to_string(this->domain_vertices.get_comm_rank()) + ".txt";
    //     std::filesystem::path full_path = std::filesystem::path(dir_i) / filename;
    //     std::ofstream ofs(full_path, std::ios::app);
    //     if (!ofs.is_open()) {
    //         std::cerr << "Failed to open file: " << full_path << std::endl;
    //         return;
    //     }
    //     std::string hostname(256, '\0');
    //     gethostname(hostname.data(), hostname.size());
    //     ofs << "Hostname: " << hostname.c_str() << std::endl;
    //     for (uint ielement = 0; ielement < this->density_matrix_solver.xlsdft.local_element_num; ielement++) {
    //         ofs << "Element " << ielement << " in " << this->density_matrix_solver.xlsdft.local_element_num << ":" << std::endl;
    //         this->density_matrix_solver.xlsdft.eigen_solvers[ielement].chefsi.chefsi_timer.show(ofs);
    //         this->density_matrix_solver.xlsdft.eigen_solvers[ielement].eigen_solver_timer.show(ofs);
    //     }
    //     ofs << "Xlsdft::run" << std::endl;
    //     this->density_matrix_solver.xlsdft.xlsdft_timer.show(ofs);
    //     ofs << "Density_matrix_solver::run" << std::endl;
    //     this->density_matrix_solver.density_matrix_solver_timer.show(ofs);
    //     ofs << "Density_solver::run" << std::endl;
    //     this->density_solver_timer.show(ofs);
    //     ofs.close();
    //     MPI_Barrier(this->domain_vertices.comm);
    // #endif //ENABLE_TIMER
    return;
}

#if defined(ENABLE_TIMER)

template<typename T>
void Density_solver<T>::print_timer_statistics(
    const bool if_print,
    std::ostream& output) const
{
#if defined(LOW_MEMORY)
    constexpr int chefsi_timer_nitem          = 11;
    constexpr int eigen_solver_nitem          = 3;
    constexpr int xlsdft_nitem                = 7;
    constexpr int density_matrix_solver_nitem = 3;
    constexpr int density_solver_nitem        = 4;
#else
    constexpr int chefsi_timer_nitem          = 11;
    constexpr int eigen_solver_nitem          = 2;
    constexpr int xlsdft_nitem                = 6;
    constexpr int density_matrix_solver_nitem = 3;
    constexpr int density_solver_nitem        = 4;
#endif

    constexpr int nitems =
        chefsi_timer_nitem
        + eigen_solver_nitem
        + xlsdft_nitem
        + density_matrix_solver_nitem
        + density_solver_nitem;

    const MPI_Comm domain_comm =
        this->domain_vertices.get_domain_3d_comm();

    if (this->domain_vertices.get_active_comm_b() == 0 &&
        this->domain_vertices.is_active)
    {
        std::vector<double> t_locals(nitems, 0.0);

        std::vector<double> max_values(nitems, 0.0);
        std::vector<double> min_values(nitems, 0.0);
        std::vector<double> mean_values(nitems, 0.0);
        std::vector<double> stddev_values(nitems, 0.0);
        std::vector<double> reciprocal_mean_values(nitems, 0.0);

        /*
         * CheFSI + eigen solver timers are accumulated
         * over all local elements.
         */
        const uint nelement =
            this->density_matrix_solver.xlsdft.local_element_num;

        for (uint ielement = 0;
             ielement < nelement;
             ielement++)
        {
            uint count = 0;

            const Chefsi_timer& chefsi_timer =
                this->density_matrix_solver.xlsdft
                    .eigen_solvers[ielement]
                    .chefsi
                    .chefsi_timer;

            /*
             * CheFSI
             */
            t_locals[count++] +=
                chefsi_timer
                    .chefsi
                    .time_cost_millisecond_double();

            t_locals[count++] +=
                chefsi_timer
                    .lanczos
                    .time_cost_millisecond_double();

            t_locals[count++] +=
                chefsi_timer
                    .filter
                    .time_cost_millisecond_double();

            t_locals[count++] +=
                chefsi_timer
                    .filter_copy
                    .time_cost_millisecond_double();

            t_locals[count++] +=
                chefsi_timer
                    .filter_product
                    .time_cost_millisecond_double();

            t_locals[count++] +=
                chefsi_timer
                    .filter_lap
                    .time_cost_millisecond_double();

            t_locals[count++] +=
                chefsi_timer
                    .filter_nloc
                    .time_cost_millisecond_double();

            t_locals[count++] +=
                chefsi_timer
                    .H_psi
                    .time_cost_millisecond_double();

            t_locals[count++] +=
                chefsi_timer
                    .projection_time_cost_millisecond();

            t_locals[count++] +=
                chefsi_timer
                    .diagonalization
                    .time_cost_millisecond_double();

            t_locals[count++] +=
                chefsi_timer
                    .rotation
                    .time_cost_millisecond_double();

            assert(count == chefsi_timer_nitem);

            /*
             * Eigen solver
             */
            const Eigen_solver_timer& eigen_solver_timer =
                this->density_matrix_solver.xlsdft
                    .eigen_solvers[ielement]
                    .eigen_solver_timer;

            t_locals[count++] +=
                eigen_solver_timer
                    .eigen_solver
                    .time_cost_millisecond_double();

#if defined(LOW_MEMORY)
            t_locals[count++] +=
                eigen_solver_timer
                    .low_memory_chi
                    .time_cost_millisecond_double();
#endif

            t_locals[count++] +=
                eigen_solver_timer
                    .eigen_solver_kernel
                    .time_cost_millisecond_double();

            assert(
                count ==
                chefsi_timer_nitem +
                eigen_solver_nitem);
        }

        /*
         * XLSDFT
         */
        int count =
            chefsi_timer_nitem +
            eigen_solver_nitem;

        const Xlsdft_timer& xlsdft_timer =
            this->density_matrix_solver
                .xlsdft
                .xlsdft_timer;

        t_locals[count++] +=
            xlsdft_timer
                .xlsdft
                .time_cost_millisecond_double();

        t_locals[count++] +=
            xlsdft_timer
                .xlsdft_copy_veff
                .time_cost_millisecond_double();

#if defined(LOW_MEMORY)
        t_locals[count++] +=
            xlsdft_timer
                .xlsdft_low_memory_psi
                .time_cost_millisecond_double();
#endif

        t_locals[count++] +=
            xlsdft_timer
                .xlsdft_eigen_solver
                .time_cost_millisecond_double();

        t_locals[count++] +=
            xlsdft_timer
                .xlsdft_barrier1
                .time_cost_millisecond_double();

        t_locals[count++] +=
            xlsdft_timer
                .xlsdft_barrier2
                .time_cost_millisecond_double();

        t_locals[count++] +=
            xlsdft_timer
                .xlsdft_barrier3
                .time_cost_millisecond_double();

        assert(
            count ==
            chefsi_timer_nitem +
            eigen_solver_nitem +
            xlsdft_nitem);

        /*
         * Density matrix solver
         */
        const Density_matrix_solver_timer&
            density_matrix_solver_timer =
                this->density_matrix_solver
                    .density_matrix_solver_timer;

        t_locals[count++] +=
            density_matrix_solver_timer
                .density_matrix_solver
                .time_cost_millisecond_double();

        t_locals[count++] +=
            density_matrix_solver_timer
                .density_matrix_solver_distribute_veff
                .time_cost_millisecond_double();

        t_locals[count++] +=
            density_matrix_solver_timer
                .density_matrix_solver_kernel
                .time_cost_millisecond_double();

        assert(
            count ==
            chefsi_timer_nitem +
            eigen_solver_nitem +
            xlsdft_nitem +
            density_matrix_solver_nitem);

        /*
         * Density solver
         */
        const Density_solver_timer& density_solver_timer =
            this->density_solver_timer;

        t_locals[count++] +=
            density_solver_timer
                .density_solver
                .time_cost_millisecond_double();

        t_locals[count++] +=
            density_solver_timer
                .density_solver_kernel
                .time_cost_millisecond_double();

        t_locals[count++] +=
            density_solver_timer
                .chemical_potential
                .time_cost_millisecond_double();

        t_locals[count++] +=
            density_solver_timer
                .update_density
                .time_cost_millisecond_double();

        assert(count == nitems);

        /*
         * MPI statistics
         */
        Tools::calc_mpi_statistics(
            t_locals.data(),
            nitems,
            domain_comm,
            max_values.data(),
            min_values.data(),
            mean_values.data(),
            stddev_values.data(),
            reciprocal_mean_values.data());

        /*
         * Find ranks corresponding to maximum/minimum time.
         */
        std::vector<int> max_ranks(nitems, 0);
        std::vector<int> min_ranks(nitems, 0);

        int domain_rank = 0;

        MPI_Comm_rank(
            domain_comm,
            &domain_rank);

        struct Value_rank
        {
            double value;
            int rank;
        };

        for (int i = 0;
             i < nitems;
             i++)
        {
            Value_rank local_pair;
            Value_rank max_pair;
            Value_rank min_pair;

            local_pair.value = t_locals[i];
            local_pair.rank  = domain_rank;

            MPI_Allreduce(
                &local_pair,
                &max_pair,
                1,
                MPI_DOUBLE_INT,
                MPI_MAXLOC,
                domain_comm);

            MPI_Allreduce(
                &local_pair,
                &min_pair,
                1,
                MPI_DOUBLE_INT,
                MPI_MINLOC,
                domain_comm);

            max_ranks[i] = max_pair.rank;
            min_ranks[i] = min_pair.rank;
        }

        /*
         * Print
         */
        if (if_print)
        {
            /*
             * ====================================================
             * Original storage indices
             * ====================================================
             */

            constexpr int chefsi_begin = 0;

            constexpr int eigen_solver_begin =
                chefsi_begin +
                chefsi_timer_nitem;

            constexpr int xlsdft_begin =
                eigen_solver_begin +
                eigen_solver_nitem;

            constexpr int density_matrix_solver_begin =
                xlsdft_begin +
                xlsdft_nitem;

            constexpr int density_solver_begin =
                density_matrix_solver_begin +
                density_matrix_solver_nitem;

            /*
             * CheFSI
             */
            constexpr int i_chefsi =
                chefsi_begin + 0;

            constexpr int i_lanczos =
                chefsi_begin + 1;

            constexpr int i_filter =
                chefsi_begin + 2;

            constexpr int i_filter_copy =
                chefsi_begin + 3;

            constexpr int i_filter_product =
                chefsi_begin + 4;

            constexpr int i_filter_lap =
                chefsi_begin + 5;

            constexpr int i_filter_nloc =
                chefsi_begin + 6;

            constexpr int i_H_psi =
                chefsi_begin + 7;

            constexpr int i_projection =
                chefsi_begin + 8;

            constexpr int i_diagonalization =
                chefsi_begin + 9;

            constexpr int i_rotation =
                chefsi_begin + 10;

            /*
             * Eigen solver
             */
            constexpr int i_eigen_solver =
                eigen_solver_begin + 0;

#if defined(LOW_MEMORY)
            constexpr int i_eigen_solver_low_memory_chi =
                eigen_solver_begin + 1;

            [[maybe_unused]] constexpr int i_eigen_solver_kernel =
                eigen_solver_begin + 2;
#else
            [[maybe_unused]] constexpr int i_eigen_solver_kernel =
                eigen_solver_begin + 1;
#endif

            /*
             * XLSDFT
             */
            constexpr int i_xlsdft =
                xlsdft_begin + 0;

            constexpr int i_xlsdft_copy_veff =
                xlsdft_begin + 1;

#if defined(LOW_MEMORY)
            constexpr int i_xlsdft_low_memory_psi =
                xlsdft_begin + 2;

            [[maybe_unused]] constexpr int i_xlsdft_eigen_solver =
                xlsdft_begin + 3;

            [[maybe_unused]] constexpr int i_xlsdft_barrier1 =
                xlsdft_begin + 4;

            constexpr int i_xlsdft_barrier2 =
                xlsdft_begin + 5;

            [[maybe_unused]] constexpr int i_xlsdft_barrier3 =
                xlsdft_begin + 6;
#else
            [[maybe_unused]] constexpr int i_xlsdft_eigen_solver =
                xlsdft_begin + 2;

            [[maybe_unused]] constexpr int i_xlsdft_barrier1 =
                xlsdft_begin + 3;

            constexpr int i_xlsdft_barrier2 =
                xlsdft_begin + 4;

            [[maybe_unused]] constexpr int i_xlsdft_barrier3 =
                xlsdft_begin + 5;
#endif

            /*
             * Density matrix solver
             */
            constexpr int i_density_matrix_solver =
                density_matrix_solver_begin + 0;

            constexpr int i_density_matrix_distribute_veff =
                density_matrix_solver_begin + 1;

            [[maybe_unused]] constexpr int i_density_matrix_kernel =
                density_matrix_solver_begin + 2;

            /*
             * Density solver
             */
            constexpr int i_density_solver =
                density_solver_begin + 0;

            [[maybe_unused]] constexpr int i_density_solver_kernel =
                density_solver_begin + 1;

            constexpr int i_chemical_potential =
                density_solver_begin + 2;

            constexpr int i_update_density =
                density_solver_begin + 3;

            /*
             * ====================================================
             * Select which items to print here.
             *
             * Comment out any line you do not want.
             * Reorder the lines freely.
             *
             * One leading space = one hierarchy level.
             * ====================================================
             */

            struct Timer_print_item
            {
                int index;
                const char* name;
            };

            const std::vector<Timer_print_item> print_items = {

                /*
                 * Density solver
                 */
                {i_density_solver,
                 "Density solver"},

                // {i_density_solver_kernel,
                //  " Density solver kernel"},

                /*
                 * Density matrix solver
                 */
                {i_density_matrix_solver,
                 " Density matrix solver"},

                {i_density_matrix_distribute_veff,
                 "  Distribute Veff"},

                // {i_density_matrix_kernel,
                //  "  Density matrix kernel"},

                /*
                 * XLSDFT
                 */
                {i_xlsdft,
                 "  XLSDFT"},

                {i_xlsdft_copy_veff,
                 "   Copy Veff"},

#if defined(LOW_MEMORY)
                {i_xlsdft_low_memory_psi,
                 "   Low memory psi"},
#endif

                // {i_xlsdft_eigen_solver,
                //  "   XLSDFT eigen solver"},

                /*
                 * Eigen solver
                 */
                {i_eigen_solver,
                 "   Eigen solver"},

#if defined(LOW_MEMORY)
                {i_eigen_solver_low_memory_chi,
                 "    Low memory chi"},
#endif

                // {i_eigen_solver_kernel,
                //  "    Eigen solver kernel"},

                /*
                 * CheFSI
                 */
                {i_chefsi,
                 "    CheFSI"},

                {i_lanczos,
                 "     Lanczos"},

                {i_filter,
                 "     Filter"},

                {i_filter_copy,
                 "      Filter copy"},

                {i_filter_product,
                 "      Filter product"},

                {i_filter_lap,
                 "      Filter lap"},

                {i_filter_nloc,
                 "      Filter nloc"},

                {i_H_psi,
                 "     H_psi"},

                {i_projection,
                 "     Projection"},

                {i_diagonalization,
                 "     Diagonalization"},

                {i_rotation,
                 "     Rotation"},

                /*
                 * XLSDFT barrier
                 */
                // {i_xlsdft_barrier1,
                //  "   Barrier 1"},

                {i_xlsdft_barrier2,
                 "   Barrier"},

                // {i_xlsdft_barrier3,
                //  "   Barrier 3"},

                /*
                 * Other density solver parts
                 */
                {i_chemical_potential,
                 " Chemical potential"},

                {i_update_density,
                 " Update density"}
            };

            /*
             * ====================================================
             * Table
             * ====================================================
             */

            constexpr int name_width  = 30;
            constexpr int value_width = 15;
            constexpr int rank_width  = 12;

            /*
             * Seven numeric columns:
             *
             * Root
             * Mean
             * Max
             * Min
             * Diff
             * Std
             * Mean(1/t)
             */
            constexpr int table_width =
                name_width +
                value_width * 7 +
                rank_width * 2;

            output
                << '\n'
                << std::string(table_width, '=')
                << '\n';

            output
                << std::left
                << std::setw(name_width)
                << "Timer"

                << std::right
                << std::setw(value_width)
                << "Root(ms)"

                << std::setw(value_width)
                << "Mean(ms)"

                << std::setw(value_width)
                << "Max(ms)"

                << std::setw(value_width)
                << "Min(ms)"

                << std::setw(value_width)
                << "Diff(ms)"

                << std::setw(value_width)
                << "Std(ms)"

                << std::setw(value_width)
                << "Mean(1/t)"

                << std::setw(rank_width)
                << "Max rank"

                << std::setw(rank_width)
                << "Min rank"

                << '\n';

            output
                << std::string(table_width, '-')
                << '\n';

            for (const Timer_print_item& item :
                 print_items)
            {
                const int i = item.index;

                assert(i >= 0);
                assert(i < nitems);

                const double diff =
                    max_values[i] -
                    min_values[i];

                output
                    << std::left
                    << std::setw(name_width)
                    << item.name

                    << std::right
                    << std::fixed
                    << std::setprecision(3)

                    << std::setw(value_width)
                    << t_locals[i]

                    << std::setw(value_width)
                    << mean_values[i]

                    << std::setw(value_width)
                    << max_values[i]

                    << std::setw(value_width)
                    << min_values[i]

                    << std::setw(value_width)
                    << diff

                    << std::setw(value_width)
                    << stddev_values[i]

                    << std::fixed
                    << std::setprecision(6)

                    << std::setw(value_width)
                    << reciprocal_mean_values[i]

                    << std::setw(rank_width)
                    << max_ranks[i]

                    << std::setw(rank_width)
                    << min_ranks[i]

                    << '\n';
            }

            output
                << std::string(table_width, '-')
                << '\n'
                << "Mean(1/t) unit: 1/ms"
                << '\n'
                << std::string(table_width, '=')
                << '\n';
        }
    }
}

#endif // ENABLE_TIMER

template<typename T>
void Density_solver<T>::init() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    uint nspin = this->spin.generate_nspin();

    this->electron_density_init.reconstructor(this->domain_vertices.local_vertices, 0);
    if (nspin != 1) this->magnetization_z.reconstructor(this->domain_vertices.local_vertices, 0);
    const uint ntype = this->psp8_files.size();
    for (uint itype = 0; itype < ntype; itype++) {
        if(this->psp8_files[itype].fchrg > 1e-12) this->NLCC_flag = true;
    }
    if (this->NLCC_flag) {
        this->electron_density_core.reconstructor(this->domain_vertices.local_vertices, 0);
    }
    this->electron_densities.reserve(nspin);
    this->electron_densities_in.reserve(nspin);
    for (uint i = 0; i < nspin; i++) {
        this->electron_densities.emplace_back(this->domain_vertices.local_vertices, 0);
        this->electron_densities_in.emplace_back(this->domain_vertices.local_vertices, 0);
    }
    // method control
    if (this->density_solver_control.method == 0) { //EIGEN SOLVER METHOD
        // domain_vertices_with_band
        Vertices_4D shared_vertices_4d(this->domain_vertices.get_3D_shared_vertices(),
                                       this->density_solver_control.eigen_solver_control.nstates);
        this->domain_vertices_with_band.init(shared_vertices_4d, this->domain_vertices.comm);
        // domain_band_mpi_package
        this->domain_band_mpi_package.init();
        // exarr_mpi_package_with_band
        // bool is_periodic[4] = {this->mesh_control.is_periodic[0],
        //                        this->mesh_control.is_periodic[1],
        //                        this->mesh_control.is_periodic[2],
        //                        false};
        // int FDn[4] = {this->stencil.FDn, this->stencil.FDn, this->stencil.FDn, 0};
        // this->exarr_mpi_package_with_band.init(this->domain_vertices_with_band, is_periodic, FDn);
        this->eigen_solver.init();
        // this->effective_potential_nloc.init();
    } else if (this->density_solver_control.method == 1) { //density matrix METHOD
        // domain_vertices_with_band
        uint comm_ni = this->density_solver_control.density_matrix_solver_control.xlsdft_control.element_comm_numbers[0];
        uint comm_nj = this->density_solver_control.density_matrix_solver_control.xlsdft_control.element_comm_numbers[1];
        uint comm_nk = this->density_solver_control.density_matrix_solver_control.xlsdft_control.element_comm_numbers[2];
        uint comm_nijk = comm_ni * comm_nj * comm_nk;
        if ((int)comm_nijk > this->domain_vertices.get_comm_size()) {
            if (this->domain_vertices.get_comm_rank() == 0) {
                printf("ERROR::The xlsdft comm_[ni, nj, nk] = [%d, %d, %d].\n", comm_ni, comm_nj, comm_nk);
                printf("ERROR::The xlsdft comm size is %d which is bigger than the comm_world size %d.\n",
                        comm_nijk, this->domain_vertices.get_comm_size());
            }
            exit(EXIT_FAILURE);
        }
        uint comm_nb = this->domain_vertices.get_comm_size() / comm_nijk;
        Vertices_4D shared_vertices_4d(this->domain_vertices.get_3D_shared_vertices(), comm_nb);
        // this->domain_vertices_with_band.set_is_Col_Maj(false);
        this->domain_vertices_with_band.set_shared_vertices(shared_vertices_4d);
        this->domain_vertices_with_band.set_comm_ni(comm_ni);
        this->domain_vertices_with_band.set_comm_nj(comm_nj);
        this->domain_vertices_with_band.set_comm_nk(comm_nk);
        this->domain_vertices_with_band.set_comm_nb(comm_nb);
        this->domain_vertices_with_band.init(shared_vertices_4d, this->domain_vertices.comm);
        // domain_band_mpi_package
        this->domain_band_mpi_package.init();
        // density_matrix_solver
        this->density_matrix_solver.init(this->psp8_files);
    } else {
        assert(!"ONLY EIGEN SOLVER or DENSITY MATRIX METHOD SUPPORTED!");
    }
    this->generate_initial_electron_density();
    this->generate_electron_charge();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The density_solver init took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

template<typename T>
template<typename T2>
void Density_solver<T>::init(const Density_solver<T2>& density_solver) {
    this->chemical_potential = (T)density_solver.chemical_potential;
    this->electron_charge = (T)density_solver.electron_charge;
    this->band_energy = (T)density_solver.band_energy;
    this->entropy_energy = (T)density_solver.entropy_energy;
    this->electron_density_init.deepcopy(std::move(density_solver.electron_density_init.as_type(this->electron_density_init.data)));
    this->magnetization_z.deepcopy(density_solver.magnetization_z.as_type(this->magnetization_z.data));
    this->NLCC_flag = density_solver.NLCC_flag;
    if (this->NLCC_flag) {
        this->electron_density_core.deepcopy(
            std::move(density_solver.electron_density_core.as_type(this->electron_density_core.data)));
    }
    uint nspin = this->spin.generate_nspin();
    std::vector<Array_3D<T>>().swap(this->electron_densities);
    std::vector<Array_3D<T>>().swap(this->electron_densities_in);
    this->electron_densities.reserve(nspin);
    this->electron_densities_in.reserve(nspin);
    for (uint i = 0; i < nspin; i++) {
        this->electron_densities.emplace_back(density_solver.electron_densities[i].as_type(&(this->chemical_potential)));
        this->electron_densities_in.emplace_back(density_solver.electron_densities_in[i].as_type(&(this->chemical_potential)));
    }
    // this->smearing.init(density_solver.smearing);
    // this->effective_potential_nloc.init(density_solver.effective_potential_nloc);
    this->domain_vertices_with_band.init(density_solver.domain_vertices_with_band);
    this->domain_band_mpi_package.init(density_solver.domain_band_mpi_package);
    if (this->density_solver_control.method == 0) { //EIGEN SOLVER METHOD
        this->eigen_solver.init(density_solver.eigen_solver);
    } else if (this->density_solver_control.method == 1) { //density matrix METHOD
        this->density_matrix_solver.init(density_solver.density_matrix_solver);
    } else {
        assert(!"ONLY EIGEN SOLVER or DENSITY MATRIX METHOD SUPPORTED!");
    }
    // this->exarr_mpi_package_with_band.init(density_solver.exarr_mpi_package_with_band);
    return;
}
template void Density_solver<float>::init(const Density_solver<float>& density_solver);
template void Density_solver<double>::init(const Density_solver<double>& density_solver);
template void Density_solver<float>::init(const Density_solver<double>& density_solver);
template void Density_solver<double>::init(const Density_solver<float>& density_solver);

template<typename T>
void Density_solver<T>::destructor() {
    this->electron_density_init.destructor();
    this->magnetization_z.destructor();
    std::vector<Array_3D<T>>().swap(this->electron_densities);
    std::vector<Array_3D<T>>().swap(this->electron_densities_in);
    if (this->NLCC_flag) {
        this->electron_density_core.destructor();
    }
    this->eigen_solver.destructor();
    // this->effective_potential_nloc.destructor();
    this->domain_band_mpi_package.destructor();
    // this->exarr_mpi_package_with_band.destructor();
    return;
}

template<typename T>
void Density_solver<T>::show() const {
    int block_size = 20;
    int precision = 13;
    int width = 18;
    std::cout << std::right << std::setw(block_size) << "chemical_potential" << " = "
              << std::left << std::setprecision(precision) << std::setw(width) << std::fixed << this->chemical_potential << std::endl;
    std::cout << std::right << std::setw(block_size) << "electron_charge" << " = "
              << std::left << std::setprecision(precision) << std::setw(width) << std::fixed << this->electron_charge << std::endl;
    std::cout << std::right << std::setw(block_size) << "band_energy" << " = "
              << std::left << std::setprecision(precision) << std::setw(width) << std::fixed << this->band_energy << std::endl;
    std::cout << std::right << std::setw(block_size) << "entropy_energy" << " = "
              << std::left << std::setprecision(precision) << std::setw(width) << std::fixed << this->entropy_energy << std::endl;
    this->domain_vertices.show();
    this->domain_vertices_with_band.show();
    return;
}

template<typename T>
void Density_solver<T>::density_dump(const std::string& dir_name) const {
    bool if_has_directory_root = false;
    std::filesystem::path dir_root(dir_name);
    if (this->domain_vertices.get_comm_rank() == 0) {
        if (!std::filesystem::exists(dir_root)) {
            if (!std::filesystem::create_directories(dir_root)) {
                std::cerr << "Failed to create directory: " << dir_name << std::endl;
            } else {
                if_has_directory_root = true;
            }
        } else {
            if_has_directory_root = true;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &if_has_directory_root, 1, MPI_C_BOOL, MPI_LOR, this->domain_vertices.comm);
    if (!if_has_directory_root) return;
    uint comm_i = this->domain_vertices.get_active_comm_i();
    uint comm_j = this->domain_vertices.get_active_comm_j();
    uint comm_k = this->domain_vertices.get_active_comm_k();
    uint comm_b = this->domain_vertices.get_active_comm_b();
    char subdir_k[20];
    char subdir_j[20];
    char subdir_i[20];
    std::snprintf(subdir_k, sizeof(subdir_k), "comm_k_%d", comm_k);
    std::snprintf(subdir_j, sizeof(subdir_j), "comm_j_%d", comm_j);
    std::snprintf(subdir_i, sizeof(subdir_i), "comm_i_%d", comm_i);
    std::filesystem::path dir_k = dir_root / subdir_k;
    std::filesystem::path dir_j = dir_k / subdir_j;
    std::filesystem::path dir_i = dir_j / subdir_i;

    bool if_has_directory_k = false;
    if (comm_b == 0 && comm_i == 0 && comm_j == 0) {
        if (!std::filesystem::exists(dir_k)) {
            if (!std::filesystem::create_directories(dir_k)) {
                std::cerr << "Failed to create directory: " << dir_k << std::endl;
            } else {
                if_has_directory_k = true;
            }
        } else {
            if_has_directory_k = true;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &if_has_directory_k, 1, MPI_C_BOOL, MPI_LOR, this->domain_vertices.get_domain_3d_comm());
    if (!if_has_directory_k) return;

    bool if_has_directory_j = false;
    if (comm_b == 0 && comm_i == 0) {
        if (!std::filesystem::exists(dir_j)) {
            if (!std::filesystem::create_directories(dir_j)) {
                std::cerr << "Failed to create directory: " << dir_j << std::endl;
            } else {
                if_has_directory_j = true;
            }
        } else {
            if_has_directory_j = true;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &if_has_directory_j, 1, MPI_C_BOOL, MPI_LOR, this->domain_vertices.get_domain_3d_comm());
    if (!if_has_directory_j) return;

    bool if_has_directory_i = false;
    if (comm_b == 0) {
        if (!std::filesystem::exists(dir_i)) {
            if (!std::filesystem::create_directories(dir_i)) {
                std::cerr << "Failed to create directory: " << dir_i << std::endl;
            } else {
                if_has_directory_i = true;
            }
        } else {
            if_has_directory_i = true;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &if_has_directory_i, 1, MPI_C_BOOL, MPI_LOR, this->domain_vertices.get_domain_3d_comm());
    if (!if_has_directory_i) return;

    std::string dens_filename = dir_i / "density.bin";
    std::ofstream ofs(dens_filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "open file failed\n";
    }
    ofs.write(
        reinterpret_cast<const char*>(this->electron_densities[0].data),
        this->electron_densities[0].length * sizeof(T)
    );
    ofs.close();
    MPI_Barrier(this->domain_vertices.get_domain_3d_comm());
    return;
}

template<typename T>
void Density_solver<T>::density_load(const std::string& dir_name){
    std::filesystem::path dir_root(dir_name);
    uint comm_i = this->domain_vertices.get_active_comm_i();
    uint comm_j = this->domain_vertices.get_active_comm_j();
    uint comm_k = this->domain_vertices.get_active_comm_k();
    uint comm_b = this->domain_vertices.get_active_comm_b();
    char subdir_k[20];
    char subdir_j[20];
    char subdir_i[20];
    std::snprintf(subdir_k, sizeof(subdir_k), "comm_k_%d", comm_k);
    std::snprintf(subdir_j, sizeof(subdir_j), "comm_j_%d", comm_j);
    std::snprintf(subdir_i, sizeof(subdir_i), "comm_i_%d", comm_i);
    std::filesystem::path dir_k = dir_root / subdir_k;
    std::filesystem::path dir_j = dir_k / subdir_j;
    std::filesystem::path dir_i = dir_j / subdir_i;

    bool if_has_directory_i = false;
    if (comm_b == 0) {
        if (!std::filesystem::exists(dir_i)) {
            if_has_directory_i = false;
        } else {
            if_has_directory_i = true;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &if_has_directory_i, 1, MPI_C_BOOL, MPI_LOR, this->domain_vertices.get_domain_3d_comm());
    assert(if_has_directory_i);
    if (!if_has_directory_i) return;

    std::string dens_filename = dir_i / "density.bin";
    std::ifstream ifs(dens_filename, std::ios::binary);
    if (!ifs) {
        std::cerr << "Error: cannot open file " << dens_filename << std::endl;
    }
    ifs.read(reinterpret_cast<char*>(this->electron_densities[0].data),
                this->electron_densities[0].length * sizeof(T));
    ifs.close();
    MPI_Barrier(this->domain_vertices.get_domain_3d_comm());
    return;
}

template class Density_solver<float>;
template class Density_solver<double>;
