#include "density_matrix_solver.h"
#include <filesystem>

#ifdef ENABLE_DENSITY_MATRIX_TIMER
#pragma message("Building with ENABLE_DENSITY_MATRIX_TIMER.")
Density_matrix_solver_timer::Density_matrix_solver_timer() {}
Density_matrix_solver_timer::~Density_matrix_solver_timer() {}
void Density_matrix_solver_timer::reset() {
    this->density_matrix_solver.reset();
    this->density_matrix_solver_distribute_veff.reset();
    this->density_matrix_solver_kernel.reset();
    return;
}
void Density_matrix_solver_timer::show(std::ostream& output) const {
    output << std::left << std::setw(20) << "density_matrix_solver"                      << ": " << this->density_matrix_solver.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "  density_matrix_solver_distribute_veff"    << ": " << this->density_matrix_solver_distribute_veff.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "  density_matrix_solver_kernel"             << ": " << this->density_matrix_solver_kernel.time_cost_millisecond() << " [ms]" << std::endl;
    return;
}
#endif //ENABLE_DENSITY_MATRIX_TIMER

template<typename T>
Density_matrix_solver<T>::Density_matrix_solver(const Density_matrix_solver_control& density_matrix_solver_control,
                                                const Mesh_control& mesh_control,
                                                const Geometry& geometry,
                                                const Stencil<T>& stencil,
                                                const Domain_parallel_vertices_4D& domain_vertices)
                                            :   density_matrix_solver_control(density_matrix_solver_control),
                                                mesh_control(mesh_control),
                                                geometry(geometry),
                                                stencil(stencil),
                                                domain_vertices(domain_vertices),
                                                exarr_mpi_package(this->domain_vertices),
                                                xlsdft(this->density_matrix_solver_control.xlsdft_control,
                                                this->mesh_control,
                                                this->geometry,
                                                this->stencil,
                                                this->domain_vertices) {}

template<typename T>
Density_matrix_solver<T>::~Density_matrix_solver() {}

template<typename T>
T Density_matrix_solver<T>::evaluate_chemical_potential(const Smearing& smearing, const T electron_charge,
                                                        const T smearing_coef, const T trial_chemical_potential) const {
    if (this->density_matrix_solver_control.method == 0) { //xlsdft
        // return this->xlsdft.evaluate_chemical_potential(smearing, electron_charge, smearing_coef, trial_chemical_potential);
        return this->xlsdft.evaluate_chemical_potential2(smearing, electron_charge, smearing_coef, trial_chemical_potential);
        // return this->xlsdft.evaluate_chemical_potential2_with_fg_electron(smearing, electron_charge, smearing_coef, trial_chemical_potential);
    } else {
        assert(this->density_matrix_solver_control.method == 0);
        return (T)0.0;
    }
}

template<typename T>
T Density_matrix_solver<T>::evaluate_chemical_potential_mp(
                const Smearing& smearing, const T electron_charge,
                const T smearing_coef, const T trial_chemical_potential,
                Memory_pool<T, Fast_memory>& pool_fast,
                Memory_pool<T, Capacity_memory>& pool_cap) const {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    if (this->density_matrix_solver_control.method == 0) { //xlsdft
        // return this->xlsdft.evaluate_chemical_potential(smearing, electron_charge, smearing_coef, trial_chemical_potential);
        // return this->xlsdft.evaluate_chemical_potential2(smearing, electron_charge, smearing_coef, trial_chemical_potential);
        // return this->xlsdft.evaluate_chemical_potential_mp(smearing, electron_charge, smearing_coef, trial_chemical_potential, pool_fast, pool_cap);
        // return this->xlsdft.evaluate_chemical_potential2_mp(smearing, electron_charge, smearing_coef, trial_chemical_potential, pool_fast, pool_cap);
        // return this->xlsdft.evaluate_chemical_potential3_mp(smearing, electron_charge, smearing_coef, trial_chemical_potential, pool_fast, pool_cap);
        // return this->xlsdft.evaluate_chemical_potential2_with_fg_electron(smearing, electron_charge, smearing_coef, trial_chemical_potential);
        // return this->xlsdft.evaluate_chemical_potential_with_fg_electron_mp(smearing, electron_charge, smearing_coef, trial_chemical_potential, pool_fast, pool_cap);
        // return this->xlsdft.evaluate_chemical_potential2_with_fg_electron_mp(smearing, electron_charge, smearing_coef, trial_chemical_potential, pool_fast, pool_cap);
        if (std::getenv("CAL_LIGEPS") != nullptr) {
            return this->xlsdft.evaluate_chemical_potential3_with_fg_electron_mp(smearing, electron_charge, smearing_coef, trial_chemical_potential, pool_fast, pool_cap);
        } else {
            return this->xlsdft.evaluate_chemical_potential3_mp(smearing, electron_charge, smearing_coef, trial_chemical_potential, pool_fast, pool_cap);
        }
    } else {
        assert(this->density_matrix_solver_control.method == 0);
        return (T)0.0;
    }
}

template<typename T>
T Density_matrix_solver<T>::evaluate_band_energy(const Smearing& smearing, const T& chemical_potential, const T& smearing_coef) const {
    if (this->density_matrix_solver_control.method == 0) { //xlsdft
        return this->xlsdft.evaluate_band_energy(smearing, chemical_potential, smearing_coef);
    } else {
        assert(this->density_matrix_solver_control.method == 0);
        return (T)0.0;
    }
}

template<typename T>
T Density_matrix_solver<T>::evaluate_entropy_energy(const Smearing& smearing, const T& chemical_potential, const T& smearing_coef) const {
    if (this->density_matrix_solver_control.method == 0) { //xlsdft
        return this->xlsdft.evaluate_entropy_energy(smearing, chemical_potential, smearing_coef);
    } else {
        assert(this->density_matrix_solver_control.method == 0);
        return (T)0.0;
    }
}

template<typename T>
void Density_matrix_solver<T>::update_electron_density(Array_3D<T>& electron_density, const Smearing& smearing,
                                                       const T chemical_potential, const T smearing_coef, const T dV,
                                                       const Domain_3D_to_4D_mpi_package& domain_band_mpi_package) {
    if (this->density_matrix_solver_control.method == 0) {
        Vertices_4D local_vertices = this->domain_vertices.get_4D_local_vertices();
        if (domain_band_mpi_package.need_comm) {
            Array_3D<T> dia_density_matrix = this->xlsdft.cal_electron_charge_density(smearing,
                                                        smearing_coef/dV, chemical_potential);
            // Array_3D<T> dia_density_matrix = this->xlsdft.cal_electron_charge_density_with_fg_electron(smearing,
            //                                             smearing_coef/dV, chemical_potential);
            domain_band_mpi_package.recv_data(electron_density, dia_density_matrix);
        } else {
            electron_density = this->xlsdft.cal_electron_charge_density(smearing,
                                                        smearing_coef/dV, chemical_potential);
        }
    } else {
        assert(this->density_matrix_solver_control.method == 0);
    }
    return;
}

template<typename T>
void Density_matrix_solver<T>::update_electron_density_mp(Array_3D<T>& electron_density, const Smearing& smearing,
                                                        const T chemical_potential, const T smearing_coef, const T dV,
                                                        const Domain_3D_to_4D_mpi_package& domain_band_mpi_package,
                                                        Memory_pool<T, Fast_memory>& pool_fast,
                                                        Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    if (this->density_matrix_solver_control.method == 0) {
        Vertices_4D local_vertices = this->domain_vertices.get_4D_local_vertices();
        if (domain_band_mpi_package.need_comm) {
            Array_3D<T> dia_density_matrix = std::getenv("CAL_LIGEPS") != nullptr
                                            ? this->xlsdft.cal_electron_charge_density_with_fg_electron(smearing,
                                                                        smearing_coef/dV, chemical_potential)
                                            : this->xlsdft.cal_electron_charge_density(smearing,
                                                                        smearing_coef/dV, chemical_potential);
            // Array_3D<T> dia_density_matrix = this->xlsdft.cal_electron_charge_density(smearing,
            //                                             smearing_coef/dV, chemical_potential);
            // Array_3D<T> dia_density_matrix = this->xlsdft.cal_electron_charge_density_with_fg_electron(smearing,
            //                                             smearing_coef/dV, chemical_potential);
            domain_band_mpi_package.recv_data(electron_density, dia_density_matrix);
        } else {
            // electron_density = this->xlsdft.cal_electron_charge_density(smearing,
            //                                             smearing_coef/dV, chemical_potential);
            if (std::getenv("CAL_LIGEPS") != nullptr) {
                this->xlsdft.cal_electron_charge_density_with_fg_electron_mp(electron_density.data, smearing,
                                                            smearing_coef/dV, chemical_potential,
                                                            pool_fast, pool_cap);
            } else {
                this->xlsdft.cal_electron_charge_density_mp(electron_density.data, smearing,
                                                            smearing_coef/dV, chemical_potential,
                                                            pool_fast, pool_cap);
            }
        }
    } else {
        assert(this->density_matrix_solver_control.method == 0);
    }
    return;
}

template<typename T>
void Density_matrix_solver<T>::cal_nonlocal_forces(Array_2D<T>& nonlocal_forces, const T& chemical_potential, const Smearing& smearing,
                                                   const Spin& spin, const std::vector<Psp8_file>& psp8_files) const {
    if (this->density_matrix_solver_control.method == 0) { //xlsdft
        this->xlsdft.cal_nonlocal_forces(nonlocal_forces, chemical_potential, smearing, spin, psp8_files);
    } else {
        assert(this->density_matrix_solver_control.method == 0);
    }
    return;
}

template<typename T>
void Density_matrix_solver<T>::print_eigens(const T& chemical_potential, const Smearing& smearing,
                                            const Spin& spin, const std::string& fname) const {
    if (this->density_matrix_solver_control.method == 0) {
        // this->xlsdft.print_eigens(chemical_potential, smearing, spin, fname);
        bool if_has_directory_root = false;
        std::filesystem::path dir_root(fname);
        if (this->domain_vertices.get_comm_rank() == 0) {
            if (!std::filesystem::exists(dir_root)) {
                if (!std::filesystem::create_directories(dir_root)) {
                    std::cerr << "Failed to create directory: " << fname << std::endl;
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

        this->xlsdft.print_eigens_divided(chemical_potential, smearing, spin, dir_i);
    } else {
        assert(this->density_matrix_solver_control.method == 0);
    }
    return;
}

template<typename T>
void Density_matrix_solver<T>::print_pdos(const T& chemical_potential, const Spin& spin, const std::string& dir_name) const {
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

    if (this->density_matrix_solver_control.method == 0) {
        this->xlsdft.print_pdos(chemical_potential, spin, dir_i);
    } else {
        // assert(this->density_matrix_solver_control.method == 0);
    }
    return;
}

template<typename T>
double Density_matrix_solver<T>::evalutate_flops() {
    double flops = 0.0;
    if (this->density_matrix_solver_control.method == 0) {  // xlsdft
        flops += this->xlsdft.evalutate_flops();
    } else {
        assert(this->density_matrix_solver_control.method == 0);
    }
    return flops;
}

template<typename T>
void Density_matrix_solver<T>::run(const Array_3D<T>& effective_potentail_loc) {
    this->run_mp(effective_potentail_loc.data);
    // #ifdef ENABLE_DENSITY_MATRIX_TIMER
    //     this->density_matrix_solver_timer.reset();
    //     this->density_matrix_solver_timer.density_matrix_solver.start();
    // #endif //ENABLE_DENSITY_MATRIX_TIMER
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // if (this->density_matrix_solver_control.method == 0) {  // xlsdft
    //     Array_3D<T> ex_effective_potentail_loc(this->ex_vertices, 0);
    //     if (this->domain_vertices.is_active) {
    //         #ifdef ENABLE_DENSITY_MATRIX_TIMER
    //             this->density_matrix_solver_timer.density_matrix_solver_distribute_veff.start();
    //         #endif //ENABLE_DENSITY_MATRIX_TIMER
    //         if (this->domain_vertices.get_active_comm_b() == 0) {
    //             this->exarr_mpi_package.fill_domain_par_ex_arr(effective_potentail_loc, ex_effective_potentail_loc);
    //         }
    //         if (this->domain_vertices.get_active_comm_nb() > 1) {
    //             MPI_Bcast(ex_effective_potentail_loc.data, ex_effective_potentail_loc.length,
    //                       Linalg::get_mpi_datatype<T>(), 0,
    //                       this->domain_vertices.band_comm);
    //         }
    //         #ifdef ENABLE_DENSITY_MATRIX_TIMER
    //             this->density_matrix_solver_timer.density_matrix_solver_distribute_veff.stop();
    //             this->density_matrix_solver_timer.density_matrix_solver_kernel.start();
    //         #endif //ENABLE_DENSITY_MATRIX_TIMER
    //         bool print_flag = this->domain_vertices.get_domain_3d_comm_rank() == 0 ? true : false;
    //         this->xlsdft.run(ex_effective_potentail_loc, print_flag);
    //         #ifdef ENABLE_DENSITY_MATRIX_TIMER
    //             this->density_matrix_solver_timer.density_matrix_solver_kernel.stop();
    //         #endif //ENABLE_DENSITY_MATRIX_TIMER
    //     }
    // } else {
    //     assert(this->density_matrix_solver_control.method == 0);
    // }
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // if (this->domain_vertices.get_comm_rank() == 0) {
    //     std::cout << "The density_matrix_solver run took " << Tools::time_cost(begin, end) << "." << std::endl;
    // }
    // #ifdef ENABLE_DENSITY_MATRIX_TIMER
    //     this->density_matrix_solver_timer.density_matrix_solver.stop();
    //     if (this->domain_vertices.get_comm_rank() == 0) this->density_matrix_solver_timer.show();
    // #endif //ENABLE_DENSITY_MATRIX_TIMER
    return;
}

template<typename T>
void Density_matrix_solver<T>::run_mp(T const* const effective_potentail_loc) {
    constexpr size_t GB = 1024 * 1024 * 1024 / sizeof(T);
    Memory_pool<T, Fast_memory> pool_fast(3.5 * GB);
    Memory_pool<T, Capacity_memory> pool_cap(3.5 * GB);
    this->run_mp(effective_potentail_loc, pool_fast, pool_cap);
    // #ifdef ENABLE_DENSITY_MATRIX_TIMER
    //     this->density_matrix_solver_timer.reset();
    //     this->density_matrix_solver_timer.density_matrix_solver.start();
    // #endif //ENABLE_DENSITY_MATRIX_TIMER
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // if (this->density_matrix_solver_control.method == 0) {  // xlsdft
    //     Array_3D<T> ex_effective_potentail_loc(this->ex_vertices, 0);
    //     if (this->domain_vertices.is_active) {
    //         #ifdef ENABLE_DENSITY_MATRIX_TIMER
    //             this->density_matrix_solver_timer.density_matrix_solver_distribute_veff.start();
    //         #endif //ENABLE_DENSITY_MATRIX_TIMER
    //         if (this->domain_vertices.get_active_comm_b() == 0) {
    //             // this->exarr_mpi_package.fill_domain_par_ex_arr(effective_potentail_loc, ex_effective_potentail_loc);
    //             this->exarr_mpi_package.fill_domain_par_ex_arr(
    //                 effective_potentail_loc, this->domain_vertices.get_3D_local_vertices(),
    //                 ex_effective_potentail_loc.data, this->ex_vertices);
    //         }
    //         if (this->domain_vertices.get_active_comm_nb() > 1) {
    //             MPI_Bcast(ex_effective_potentail_loc.data, ex_effective_potentail_loc.length,
    //                       Linalg::get_mpi_datatype<T>(), 0,
    //                       this->domain_vertices.band_comm);
    //         }
    //         #ifdef ENABLE_DENSITY_MATRIX_TIMER
    //             this->density_matrix_solver_timer.density_matrix_solver_distribute_veff.stop();
    //             this->density_matrix_solver_timer.density_matrix_solver_kernel.start();
    //         #endif //ENABLE_DENSITY_MATRIX_TIMER
    //         bool print_flag = this->domain_vertices.get_domain_3d_comm_rank() == 0 ? true : false;
    //         this->xlsdft.run(ex_effective_potentail_loc, print_flag);
    //         #ifdef ENABLE_DENSITY_MATRIX_TIMER
    //             this->density_matrix_solver_timer.density_matrix_solver_kernel.stop();
    //         #endif //ENABLE_DENSITY_MATRIX_TIMER
    //     }
    // } else {
    //     assert(this->density_matrix_solver_control.method == 0);
    // }
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // if (this->domain_vertices.get_comm_rank() == 0) {
    //     std::cout << "The density_matrix_solver run took " << Tools::time_cost(begin, end) << "." << std::endl;
    // }
    // #ifdef ENABLE_DENSITY_MATRIX_TIMER
    //     this->density_matrix_solver_timer.density_matrix_solver.stop();
    //     if (this->domain_vertices.get_comm_rank() == 0) this->density_matrix_solver_timer.show();
    // #endif //ENABLE_DENSITY_MATRIX_TIMER
    return;
}

template<typename T>
void Density_matrix_solver<T>::run_mp(T const* const effective_potentail_loc,
                                    Memory_pool<T, Fast_memory>& pool_fast,
                                    Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    #ifdef ENABLE_DENSITY_MATRIX_TIMER
        this->density_matrix_solver_timer.reset();
        this->density_matrix_solver_timer.density_matrix_solver.start();
    #endif //ENABLE_DENSITY_MATRIX_TIMER
    if (this->density_matrix_solver_control.method == 0) {  // xlsdft
        // Array_3D<T> ex_effective_potentail_loc(this->ex_vertices, 0);
        const uint ex_effective_potentail_loc_length = this->ex_vertices.get_size();
        T* ex_effective_potentail_loc = pool_cap.allocate(ex_effective_potentail_loc_length);
        if (this->domain_vertices.is_active) {
            #ifdef ENABLE_DENSITY_MATRIX_TIMER
                this->density_matrix_solver_timer.density_matrix_solver_distribute_veff.start();
            #endif //ENABLE_DENSITY_MATRIX_TIMER
            if (this->domain_vertices.get_active_comm_b() == 0) {
                // this->exarr_mpi_package.fill_domain_par_ex_arr(effective_potentail_loc, ex_effective_potentail_loc);
                this->exarr_mpi_package.fill_domain_par_ex_arr(
                    effective_potentail_loc, this->domain_vertices.get_3D_local_vertices(),
                    ex_effective_potentail_loc, this->ex_vertices);
            }
            if (this->domain_vertices.get_active_comm_nb() > 1) {
                MPI_Bcast(ex_effective_potentail_loc, ex_effective_potentail_loc_length,
                          Linalg::get_mpi_datatype<T>(), 0,
                          this->domain_vertices.band_comm);
            }
            #ifdef ENABLE_DENSITY_MATRIX_TIMER
                this->density_matrix_solver_timer.density_matrix_solver_distribute_veff.stop();
                this->density_matrix_solver_timer.density_matrix_solver_kernel.start();
            #endif //ENABLE_DENSITY_MATRIX_TIMER
            bool print_flag = this->domain_vertices.get_domain_3d_comm_rank() == 0 ? true : false;
            this->xlsdft.run_mp(ex_effective_potentail_loc, this->ex_vertices, print_flag,
                                pool_fast, pool_cap);
            #ifdef ENABLE_DENSITY_MATRIX_TIMER
                this->density_matrix_solver_timer.density_matrix_solver_kernel.stop();
            #endif //ENABLE_DENSITY_MATRIX_TIMER
        }
    } else {
        assert(this->density_matrix_solver_control.method == 0);
    }
    #ifdef ENABLE_DENSITY_MATRIX_TIMER
        this->density_matrix_solver_timer.density_matrix_solver.stop();
        if (this->domain_vertices.get_comm_rank() == 0) this->density_matrix_solver_timer.show();
        // #if defined(ENABLE_TIMER)
        //     this->print_timer_statistics(this->domain_vertices.get_comm_rank() == 0, std::cout);
        // #endif //ENABLE_TIMER
    #endif //ENABLE_DENSITY_MATRIX_TIMER
    return;
}

#if defined(ENABLE_TIMER)
template<typename T>
void Density_matrix_solver<T>::print_timer_statistics(const bool if_print, std::ostream& output) const {
#if defined(LOW_MEMORY)
    constexpr int nitems = 11 + 3 + 7 + 3;
#else
    constexpr int nitems = 11 + 2 + 6 + 3;
#endif

    const MPI_Comm domain_comm =
        this->domain_vertices.get_domain_3d_comm();

    if (this->domain_vertices.get_active_comm_b() == 0 &&
        this->domain_vertices.is_active) {

        std::vector<double> t_locals(nitems, 0.0);
        std::vector<double> max_values(nitems);
        std::vector<double> min_values(nitems);
        std::vector<double> mean_values(nitems);
        std::vector<double> stddev_values(nitems);
        std::vector<double> reciprocal_mean_values(nitems);

        const uint nelement = this->xlsdft.local_element_num;

        for (uint ielement = 0; ielement < nelement; ielement++) {
            uint count = 0;

            // CheFSI
            t_locals[count++] +=
                this->xlsdft.eigen_solvers[ielement].chefsi.chefsi_timer.chefsi.time_cost_millisecond();

            t_locals[count++] +=
                this->xlsdft.eigen_solvers[ielement].chefsi.chefsi_timer.lanczos.time_cost_millisecond();

            t_locals[count++] +=
                this->xlsdft.eigen_solvers[ielement].chefsi.chefsi_timer.filter.time_cost_millisecond();

            t_locals[count++] +=
                this->xlsdft.eigen_solvers[ielement].chefsi.chefsi_timer.filter_copy.time_cost_millisecond();

            t_locals[count++] +=
                this->xlsdft.eigen_solvers[ielement].chefsi.chefsi_timer.filter_product.time_cost_millisecond();

            t_locals[count++] +=
                this->xlsdft.eigen_solvers[ielement].chefsi.chefsi_timer.filter_lap.time_cost_millisecond();

            t_locals[count++] +=
                this->xlsdft.eigen_solvers[ielement].chefsi.chefsi_timer.filter_nloc.time_cost_millisecond();

            t_locals[count++] +=
                this->xlsdft.eigen_solvers[ielement].chefsi.chefsi_timer.H_psi.time_cost_millisecond();

            t_locals[count++] +=
                this->xlsdft.eigen_solvers[ielement].chefsi.chefsi_timer.projection.time_cost_millisecond();

            t_locals[count++] +=
                this->xlsdft.eigen_solvers[ielement].chefsi.chefsi_timer.diagonalization.time_cost_millisecond();

            t_locals[count++] +=
                this->xlsdft.eigen_solvers[ielement].chefsi.chefsi_timer.rotation.time_cost_millisecond();

            // Eigen solver
            t_locals[count++] +=
                this->xlsdft.eigen_solvers[ielement].eigen_solver_timer.eigen_solver.time_cost_millisecond();

#if defined(LOW_MEMORY)
            t_locals[count++] +=
                this->xlsdft.eigen_solvers[ielement].eigen_solver_timer.low_memory_chi.time_cost_millisecond();
#endif

            t_locals[count++] +=
                this->xlsdft.eigen_solvers[ielement].eigen_solver_timer.eigen_solver_kernel.time_cost_millisecond();

#if defined(LOW_MEMORY)
            assert(count == 11 + 3);
#else
            assert(count == 11 + 2);
#endif
        }

#if defined(LOW_MEMORY)
        int count = 11 + 3;
#else
        int count = 11 + 2;
#endif

        // XLSDFT
        t_locals[count++] +=
            this->xlsdft.xlsdft_timer.xlsdft.time_cost_millisecond();

        t_locals[count++] +=
            this->xlsdft.xlsdft_timer.xlsdft_copy_veff.time_cost_millisecond();

#if defined(LOW_MEMORY)
        t_locals[count++] +=
            this->xlsdft.xlsdft_timer.xlsdft_low_memory_psi.time_cost_millisecond();
#endif

        t_locals[count++] +=
            this->xlsdft.xlsdft_timer.xlsdft_eigen_solver.time_cost_millisecond();

        t_locals[count++] +=
            this->xlsdft.xlsdft_timer.xlsdft_barrier1.time_cost_millisecond();

        t_locals[count++] +=
            this->xlsdft.xlsdft_timer.xlsdft_barrier2.time_cost_millisecond();

        t_locals[count++] +=
            this->xlsdft.xlsdft_timer.xlsdft_barrier3.time_cost_millisecond();

        assert(count == nitems - 3);

        t_locals[count++] +=
            this->density_matrix_solver_timer.density_matrix_solver.time_cost_millisecond();

        t_locals[count++] +=
            this->density_matrix_solver_timer.density_matrix_solver_distribute_veff.time_cost_millisecond();

        t_locals[count++] +=
            this->density_matrix_solver_timer.density_matrix_solver_kernel.time_cost_millisecond();

        assert(count == nitems);

        /*
         * This must be called by every rank in domain_comm.
         * Do NOT put it inside "if (if_print)".
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
         * Only the selected root rank prints.
         */
        if (if_print) {

            const char* timer_names[nitems] = {
                // CheFSI
                "CheFSI",
                "  Lanczos",
                "  Filter",
                "    Filter copy",
                "    Filter product",
                "    Filter lap",
                "    Filter nloc",
                "  H_psi",
                "  Projection",
                "  Diagonalization",
                "  Rotation",

                // Eigen solver
                "Eigen solver",
#if defined(LOW_MEMORY)
                "  Low memory chi",
#endif
                "  Eigen solver kernel",

                // XLSDFT
                "XLSDFT",
                "  Copy Veff",
#if defined(LOW_MEMORY)
                "  Low memory psi",
#endif
                "  Eigen solver",
                "  Barrier 1",
                "  Barrier 2",
                "  Barrier 3",

                // Density matrix solver
                "Density matrix solver",
                "  Distribute Veff",
                "  Density matrix kernel"
            };

            constexpr int name_width = 25;
            constexpr int value_width = 15;

            output
                << '\n'
                << "========================== Timer Statistics =========================="
                << "====================================================\n";

            output
                << std::left
                << std::setw(name_width) << "Timer"
                << std::right
                << std::setw(value_width) << "Root(ms)"
                << std::setw(value_width) << "Mean(ms)"
                << std::setw(value_width) << "Max(ms)"
                << std::setw(value_width) << "Min(ms)"
                << std::setw(value_width) << "Std(ms)"
                << std::setw(value_width) << "Mean(1/t)"
                << '\n';

            output
                << std::string(
                       name_width + value_width * 6,
                       '-')
                << '\n';

            output << std::scientific << std::setprecision(6);

            for (int i = 0; i < nitems; i++) {
                output
                    << std::left
                    << std::setw(name_width) << timer_names[i]
                    << std::right
                    << std::setw(value_width) << t_locals[i]
                    << std::setw(value_width) << mean_values[i]
                    << std::setw(value_width) << max_values[i]
                    << std::setw(value_width) << min_values[i]
                    << std::setw(value_width) << stddev_values[i]
                    << std::setw(value_width) << reciprocal_mean_values[i]
                    << '\n';
            }

            output
                << std::string(
                       name_width + value_width * 6,
                       '-')
                << '\n'
                << "Mean(1/t) unit: 1/ms\n"
                << "======================================================================"
                << "====================================================\n";
        }
    }
}
#endif //ENABLE_TIMER

template<typename T>
void Density_matrix_solver<T>::init(const std::vector<Psp8_file>& psp8_files) {
    // method control
    if (this->density_matrix_solver_control.method == 0) { //xlsdft
        // exarr_mpi_package
        int nodeses[3] = {(int)(std::round(this->density_matrix_solver_control.xlsdft_control.buffers[0]
                        / this->mesh_control.delta_x + 1e-12)),
                          (int)(std::round(this->density_matrix_solver_control.xlsdft_control.buffers[1]
                        / this->mesh_control.delta_y + 1e-12)),
                          (int)(std::round(this->density_matrix_solver_control.xlsdft_control.buffers[2]
                        / this->mesh_control.delta_z + 1e-12))};
        this->exarr_mpi_package.init_full(mesh_control.is_periodic, nodeses);
        // ex_vertices
        if (this->domain_vertices.is_active) {
            Vertices_3D vertices = this->domain_vertices.get_3D_local_vertices();
            this->ex_vertices.set_vertices(vertices.is - nodeses[0], vertices.get_ie() + nodeses[0],
                                           vertices.js - nodeses[1], vertices.get_je() + nodeses[1],
                                           vertices.ks - nodeses[2], vertices.get_ke() + nodeses[2]);
        }
        // xlsdft
        this->xlsdft.init(psp8_files);
    } else {
        assert(!"ONLY XLSDFT SUPPORTED!");
    }
    return;
}

template<typename T>
template<typename T2>
void Density_matrix_solver<T>::init(const Density_matrix_solver<T2>& other) {
    this->exarr_mpi_package.init(other.exarr_mpi_package);
    this->ex_vertices.set_vertices(other.ex_vertices);
    this->xlsdft.init(other.xlsdft);
    return;
}
template void Density_matrix_solver<float>::init(const Density_matrix_solver<float>& chefsi);
template void Density_matrix_solver<double>::init(const Density_matrix_solver<double>& chefsi);
template void Density_matrix_solver<float>::init(const Density_matrix_solver<double>& chefsi);
template void Density_matrix_solver<double>::init(const Density_matrix_solver<float>& chefsi);

template<typename T>
void Density_matrix_solver<T>::destructor() {
    this->exarr_mpi_package.destructor();
    // this->ex_vertices.destructor();
    this->xlsdft.destructor();
    return;
}

template<typename T>
void Density_matrix_solver<T>::show() const {
    this->exarr_mpi_package.show();
    this->ex_vertices.show();
    this->xlsdft.show();
    return;
}

template class Density_matrix_solver<float>;
template class Density_matrix_solver<double>;
