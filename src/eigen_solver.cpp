#include "eigen_solver.h"

#ifdef ENABLE_EIGEN_SOLVER_TIMER
#pragma message("Building with ENABLE_EIGEN_SOLVER_TIMER.")
Eigen_solver_timer::Eigen_solver_timer() {}
Eigen_solver_timer::~Eigen_solver_timer() {}
void Eigen_solver_timer::reset() {
    this->eigen_solver.reset();
    #if (defined(LOW_MEMORY))
    this->low_memory_chi.reset();
    #endif
    this->eigen_solver_kernel.reset();
    return;
}
void Eigen_solver_timer::show(std::ostream& output) const {
    output << std::left << std::setw(20) << "eigen_solver"           << ": " << this->eigen_solver.time_cost_millisecond() << " [ms]" << std::endl;
    #if (defined(LOW_MEMORY))
    output << std::left << std::setw(20) << "  low_memory_chi"       << ": " << this->low_memory_chi.time_cost_millisecond() << " [ms]" << std::endl;
    #endif
    output << std::left << std::setw(20) << "  eigen_solver_kernel"  << ": " << this->eigen_solver_kernel.time_cost_millisecond() << " [ms]" << std::endl;
    return;
}
#endif //ENABLE_EIGEN_SOLVER_TIMER

template<typename T>
Eigen_solver<T>::Eigen_solver(const Eigen_solver_control& eigen_solver_control,
                              const Mesh_control& mesh_control,
                              const Geometry& geometry,
                              const Stencil<T>& stencil,
                              const std::vector<Psp8_file>& psp8_files,
                              const Domain_parallel_vertices_4D& domain_vertices) 
                            : eigen_solver_control(eigen_solver_control),
                              mesh_control(mesh_control),
                              geometry(geometry),
                              stencil(stencil),
                              psp8_files(psp8_files),
                              domain_vertices(domain_vertices),
                              effective_potential_nloc(this->geometry,
                                                       this->mesh_control,
                                                       this->psp8_files,
                                                       this->domain_vertices),
                              exarr_mpi_package(this->domain_vertices),
                              intra_band_exarr_mpi_package(this->intra_band_domain_vertices),
                              chefsi(eigen_solver_control.chefsi_control,
                                     this->mesh_control,
                                     this->stencil,
                                     this->domain_vertices,
                                     this->intra_band_exarr_mpi_package) {}

template<typename T>
Eigen_solver<T>::~Eigen_solver() {}

template<typename T>
void Eigen_solver<T>::cal_int_density_per_band(const Vertices_3D& region, std::vector<T>& eigen_value_per_band,
                                               std::vector<T>& int_density_per_band) const {
    if (this->eigen_solver_control.method == 0) { //chefsi
        this->chefsi.cal_int_density_per_band(region, this->eigen_values, this->eigen_vectors,
                                              eigen_value_per_band, int_density_per_band);
        return;
    } else {
        assert(this->eigen_solver_control.method == 0);
        return;
    }
}

template<typename T>
Array_3D<T> Eigen_solver<T>::cal_electron_charge_density(const Smearing& smearing, const T& smearing_coef,
                                                         const T& chemical_potential, const Vertices_3D& region) const {
    if (this->eigen_solver_control.method == 0) { //chefsi
        return this->chefsi.cal_electron_charge_density(smearing, smearing_coef, chemical_potential, region,
                                                        this->eigen_values, this->eigen_vectors);
    } else {
        assert(this->eigen_solver_control.method == 0);
        Array_3D<T> electron_charge_density;
        return electron_charge_density;
    }
}

template<typename T>
T Eigen_solver<T>::cal_electron_charge(const Smearing& smearing, const T& smearing_coef,
                                       const T& chemical_potential, const Vertices_3D& region) const {
    if (this->eigen_solver_control.method == 0) { //chefsi
        return this->chefsi.cal_electron_charge(smearing, smearing_coef, chemical_potential, region,
                                                this->eigen_values, this->eigen_vectors);
    } else {
        assert(this->eigen_solver_control.method == 0);
        return (T)0.0;
    }
}

template<typename T>
T Eigen_solver<T>::evaluate_chemical_potential(const Smearing& smearing, const T& electron_charge,
                                               const T& smearing_coef) const {
    T tol = T(1e-12);
    if (unlikely(Linalg::get_type_id<T>() == 1)) tol = T(1e-4);
    return Eigen_solver_method::evaluate_chemical_potential<T>(this->eigen_values.data,
                                                            this->domain_vertices.shared_vertices.nb,
                                                            this->eigen_values[0] - T(1.0),
                                                            this->eigen_values[this->domain_vertices.shared_vertices.nb - 1] + T(1.0),
                                                            smearing,
                                                            electron_charge,
                                                            smearing_coef,
                                                            100,
                                                            tol);
}

template<typename T>
T Eigen_solver<T>::evaluate_band_energy(const Smearing& smearing, const T& chemical_potential, const T& smearing_coef) const {
    if (this->eigen_solver_control.method == 0) {
        uint nstates = this->domain_vertices.shared_vertices.nb;
        Array_0D<T> occ(nstates);
        Smearing_method::smear(this->eigen_values.data, occ.data, chemical_potential, smearing, nstates);
        occ *= smearing_coef;
        Array_0D<T> band_energy_arr(nstates);
        Linalg::hadamard_product_general(band_energy_arr.data, this->eigen_values.data, occ.data, nstates);
        T band_energy = Linalg::vector_sum(band_energy_arr.data, nstates);
        return band_energy;
    } else {
        assert(this->eigen_solver_control.method == 0);
        return (T)0.0;
    }
}

template<typename T>
T Eigen_solver<T>::evaluate_band_energy(const Smearing& smearing, const T& chemical_potential,
                                        const T& smearing_coef, const Vertices_3D& region) const {
    if (this->eigen_solver_control.method == 0) { //chefsi
        if (this->eigen_vectors.data == nullptr) return (T)0.0; // in case eigen solver fails to produce eigen vectors, e.g., due to too LWO MEMORY
        return this->chefsi.evaluate_band_energy(smearing, chemical_potential, smearing_coef, region,
                                                 this->eigen_values, this->eigen_vectors);
    } else {
        assert(this->eigen_solver_control.method == 0);
        return (T)0.0;
    }
}

template<typename T>
T Eigen_solver<T>::evaluate_entropy_energy(const Smearing& smearing, const T& chemical_potential, const T& smearing_coef) const {
    if (this->eigen_solver_control.method == 0) {
        uint nstates = this->domain_vertices.shared_vertices.nb;
        Array_0D<T> occ(nstates);
        Smearing_method::smear(this->eigen_values.data, occ.data, chemical_potential, smearing, nstates);
        Array_0D<T> entropy_energy_arr(nstates);
        Smearing_method::generate_entropy_energy_arr(occ.data, entropy_energy_arr.data, smearing, smearing_coef, nstates);
        T entropy_energy = Linalg::vector_sum(entropy_energy_arr.data, nstates);
        return entropy_energy;
    } else {
        assert(this->eigen_solver_control.method == 0);
        return (T)0.0;
    }
}

template<typename T>
T Eigen_solver<T>::evaluate_entropy_energy(const Smearing& smearing, const T& chemical_potential,
                                           const T& smearing_coef, const Vertices_3D& region) const {
    if (this->eigen_solver_control.method == 0) { //chefsi
        if (this->eigen_vectors.data == nullptr) return (T)0.0; // in case eigen solver fails to produce eigen vectors, e.g., due to too LWO MEMORY
        return this->chefsi.evaluate_entropy_energy(smearing, chemical_potential, smearing_coef, region,
                                                    this->eigen_values, this->eigen_vectors);
    } else {
        assert(this->eigen_solver_control.method == 0);
        return (T)0.0;
    }
}

template<typename T>
void Eigen_solver<T>::update_electron_density(Array_3D<T>& electron_density, const Smearing& smearing,
                                              const T& chemical_potential, const T& smearing_coef, const T& dV,
                                              const Domain_3D_to_4D_mpi_package& domain_band_mpi_package) {
    if (this->eigen_solver_control.method == 0) {
        Vertices_4D local_vertices = this->domain_vertices.get_4D_local_vertices();
        uint m = local_vertices.Vertices_3D::get_size();
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        Array_4D<T> eigen_vectors_square(local_vertices);
        Linalg::vector_norm_square(eigen_vectors_square.data, this->eigen_vectors.data, eigen_vectors_square.length);
        Array_0D<T> occ(local_vertices.nb);
        Smearing_method::smear(this->eigen_values.data + local_vertices.bs, occ.data, chemical_potential, smearing, local_vertices.nb);
        Array_0D<T> factor = occ * (smearing_coef/dV);
        Array_3D<T> band_gathered_electron_density(local_vertices);
        if (m > 0) {
            if (this->domain_vertices.get_active_comm_nb() > 0) {
                Array_3D<T> band_separated_electron_density(local_vertices);
                Linalg::matrix_vector_product(eigen_vectors_square.data, 1, factor.data, band_separated_electron_density.data,
                                              m, local_vertices.nb);
                MPI_Reduce(band_separated_electron_density.data, band_gathered_electron_density.data,
                        m, mpi_datatype, MPI_SUM, 0, this->domain_vertices.band_comm);
            } else {
                Linalg::matrix_vector_product(eigen_vectors_square.data, 1, factor.data, band_gathered_electron_density.data,
                                              m, local_vertices.nb);
            }
        }
        domain_band_mpi_package.recv_data(electron_density, band_gathered_electron_density);
    } else {
        assert(this->eigen_solver_control.method == 0);
    }
    return;
}

template<typename T>
void Eigen_solver<T>::cal_nonlocal_forces(Array_2D<T>& forces, const T& chemical_potential, const Smearing& smearing, const Spin& spin) const {
    #ifdef LOW_MEMORY
    Effective_potential_nloc<T> effective_potential_nloc_temp = this->effective_potential_nloc;
    effective_potential_nloc_temp.init();
    this->cal_nonlocal_forces(forces, chemical_potential, smearing, spin, effective_potential_nloc_temp);
    #else
    this->cal_nonlocal_forces(forces, chemical_potential, smearing, spin, this->effective_potential_nloc);
    #endif
    return;
}

template<typename T>
void Eigen_solver<T>::cal_nonlocal_forces(Array_2D<T>& nonlocal_forces, const T& chemical_potential, const Smearing& smearing,
                                          const Spin& spin, const Effective_potential_nloc<T>& effective_potential_nloc) const {
    if (this->eigen_solver_control.method == 0) { //chefsi
        this->chefsi.cal_nonlocal_forces(nonlocal_forces, chemical_potential, smearing, spin, effective_potential_nloc,
                                         this->eigen_values, this->eigen_vectors);
    } else {
        assert(this->eigen_solver_control.method == 0);
    }
    return;
}

#if (defined(LOW_MEMORY))
template<typename T>
template<typename T2>
void Eigen_solver<T>::cal_nonlocal_forces(Array_2D<T>& nonlocal_forces, const T& chemical_potential, const Smearing& smearing,
                                          const Spin& spin, const Effective_potential_nloc<T>& effective_potential_nloc,
                                          const Array_4D<T2> eigen_vectors_other_type) const {
    Array_4D<T> eigen_vectors_temp(eigen_vectors_other_type.as_type(eigen_vectors.data));
    this->eigen_vectors.fill_overlap(eigen_vectors_temp);
    if (this->eigen_solver_control.method == 0) { //chefsi
        this->chefsi.cal_nonlocal_forces(nonlocal_forces, chemical_potential, smearing, spin, effective_potential_nloc,
                                         this->eigen_values, eigen_vectors_temp);
    } else {
        assert(this->eigen_solver_control.method == 0);
    }
    return;
}
template void Eigen_solver<double>::cal_nonlocal_forces(Array_2D<double>& nonlocal_forces, const double& chemical_potential, const Smearing& smearing,
                                                        const Spin& spin, const Effective_potential_nloc<double>& effective_potential_nloc,
                                                        const Array_4D<float> eigen_vectors_other_type) const;
template void Eigen_solver<float>::cal_nonlocal_forces(Array_2D<float>& nonlocal_forces, const float& chemical_potential, const Smearing& smearing,
                                                       const Spin& spin, const Effective_potential_nloc<float>& effective_potential_nloc,
                                                       const Array_4D<float> eigen_vectors_other_type) const;
#if defined(FP16_FLAG)
template void Eigen_solver<double>::cal_nonlocal_forces(Array_2D<double>& nonlocal_forces, const double& chemical_potential, const Smearing& smearing,
                                                        const Spin& spin, const Effective_potential_nloc<double>& effective_potential_nloc,
                                                        const Array_4D<__fp16> eigen_vectors_other_type) const;
template void Eigen_solver<float>::cal_nonlocal_forces(Array_2D<float>& nonlocal_forces, const float& chemical_potential, const Smearing& smearing,
                                                        const Spin& spin, const Effective_potential_nloc<float>& effective_potential_nloc,
                                                        const Array_4D<__fp16> eigen_vectors_other_type) const;
#endif
#endif

template<typename T>
void Eigen_solver<T>::print_eigens(const T& chemical_potential, const Smearing& smearing,
                                   const Spin& spin, const std::string& fname) const {
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::ofstream outfile(fname);
        this->print_eigens(chemical_potential, smearing, spin, outfile);
        outfile.close();
    }
    return;
}

template<typename T>
void Eigen_solver<T>::print_eigens(const T& chemical_potential, const Smearing& smearing,
                                   const Spin& spin, std::ostream &file) const {
    const uint nspin = spin.generate_nspin();
    const uint nstates = this->domain_vertices.shared_vertices.nb;
    if (nspin == 1) {
        file << "Final eigenvalues (Ha) and occupation numbers, with chemical potential "
                    << std::fixed << std::setprecision(12) << chemical_potential << std::endl;
        file << std::endl;
        file << std::fixed << std::setprecision(12) << "kred #" << 1 << " = ("
             << 0.0 << ","
             << 0.0 << ","
             << 0.0 << ")"
             << std::endl;
        file << "weight = " << 1.0
             << std::endl;
        file << std::setw(10) << "n"
             << std::setw(20) << "eigval"
             << std::setw(20) << "occ" << std::endl;
        Array_0D<T> occs(nstates);
        Smearing_method::smear(this->eigen_values.data, occs.data, chemical_potential,
                            smearing, nstates);
        Linalg::scalar_product_general(occs.data, T(2.0), nstates);
        for (uint ib = 0; ib < nstates; ib++) {
            file << std::setw(10) << ib + 1
                 << std::setw(20) << std::setprecision(12) << this->eigen_values[ib]
                 << std::setw(20) << std::setprecision(12) << occs[ib] << std::endl;
        }
    } else if (nspin == 2) {
        assert(nspin == 1);
    }
    return;
}

template<typename T>
void Eigen_solver<T>::print_eigens(const T& chemical_potential, const Smearing& smearing,
                                   const Spin& spin, const Vertices_3D& region, const std::string& fname) const {
    Array_0D<T> fracs(this->domain_vertices.shared_vertices.nb, 0);
    this->get_region_fracs(region, fracs.data);
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::ofstream outfile(fname);
        this->print_eigens(chemical_potential, smearing, spin, fracs.data, outfile);
        outfile.close();
    }
    return;
}

template<typename T>
void Eigen_solver<T>::print_eigens(const T& chemical_potential, const Smearing& smearing,
                                   const Spin& spin, T const* const fracs, std::ostream &file) const {
    const uint nspin = spin.generate_nspin();
    const uint nstates = this->domain_vertices.shared_vertices.nb;
    if (nspin == 1) {
        file << "Final eigenvalues (Ha) and occupation numbers, with chemical potential "
                    << std::fixed << std::setprecision(12) << chemical_potential << std::endl;
        file << std::endl;
        file << std::fixed << std::setprecision(12) << "kred #" << 1 << " = ("
             << 0.0 << ","
             << 0.0 << ","
             << 0.0 << ")"
             << std::endl;
        file << "weight = " << 1.0
             << std::endl;
        file << std::setw(10) << "n"
             << std::setw(20) << "eigval"
             << std::setw(20) << "frac"
             << std::setw(20) << "occ" << std::endl;
        Array_0D<T> occs(nstates);
        Smearing_method::smear(this->eigen_values.data, occs.data, chemical_potential,
                            smearing, nstates);
        Linalg::scalar_product_general(occs.data, T(2.0), nstates);
        for (uint ib = 0; ib < nstates; ib++) {
            file << std::setw(10) << ib + 1
                 << std::setw(20) << std::setprecision(12) << this->eigen_values[ib]
                 << std::setw(20) << std::setprecision(12) << fracs[ib]
                 << std::setw(20) << std::setprecision(12) << occs[ib]
                 << std::endl;
        }
    } else if (nspin == 2) {
        assert(nspin == 1);
    }
    return;
}

template<typename T>
void Eigen_solver<T>::print_pdos(const T& chemical_potential, const Spin& spin, const std::string& fname) const {
    const std::vector<Array_2D<T>> pdos_atoms = this->cal_pdos();
    const uint nspin = spin.generate_nspin();
    const uint natom = this->geometry.atoms.size();
    assert(nspin == 1);
    if (this->domain_vertices.get_comm_rank()==0) {
        if (nspin == 1) {
            std::ofstream outfile(fname);
            outfile << "Final PDOS (Ha) with index (ispin, n, l, m), with chemical potential "
                    << std::fixed << std::setprecision(12) << chemical_potential
                    << std::right << std::endl;
            for (uint iatom = 0; iatom < natom; iatom++) {
                const Atom& atom = this->geometry.atoms[iatom];
                const Array_2D<T>& pdos_atom = pdos_atoms[iatom];
                Atom_method::print_pdos<T>(outfile, atom, nspin, pdos_atom, this->eigen_values,
                                        this->psp8_files[atom.type], chemical_potential, false);
            }
            outfile.close();
        }
    }
    return;
}

template<typename T>
void Eigen_solver<T>::print_psi_sparc(const Spin& spin, const std::string& fname) const {
    const uint nspin = spin.generate_nspin();
    const uint nstates = this->eigen_solver_control.nstates;
    assert(nspin == 1);
    if (this->domain_vertices.get_comm_size() != 1) {
        std::cout << "WARNING: ONLY np = 1 is supported in print_psi!" << std::endl;
    }
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::ofstream output(fname, std::ios::binary);
        if (!output.is_open()) {
            std::cerr << "\nCannot open file \"" << fname << "\"\n";
            std::exit(EXIT_FAILURE);
        }
        int one = 1;
        output.write(reinterpret_cast<const char*>(&(this->mesh_control.nx)), sizeof(int));
        output.write(reinterpret_cast<const char*>(&(this->mesh_control.ny)), sizeof(int));
        output.write(reinterpret_cast<const char*>(&(this->mesh_control.nz)), sizeof(int));
        output.write(reinterpret_cast<const char*>(&(this->mesh_control.nd)), sizeof(int));
        output.write(reinterpret_cast<const char*>(&(this->mesh_control.delta_x)), sizeof(double));
        output.write(reinterpret_cast<const char*>(&(this->mesh_control.delta_y)), sizeof(double));
        output.write(reinterpret_cast<const char*>(&(this->mesh_control.delta_z)), sizeof(double));
        output.write(reinterpret_cast<const char*>(&(this->mesh_control.delta_V)), sizeof(double));
        output.write(reinterpret_cast<const char*>(&(one)), sizeof(int)); //Nspinor_eig
        output.write(reinterpret_cast<const char*>(&(one)), sizeof(int)); //isGammaPoint
        output.write(reinterpret_cast<const char*>(&(nspin)), sizeof(int)); //nspin
        output.write(reinterpret_cast<const char*>(&(one)), sizeof(int)); //nkpt
        output.write(reinterpret_cast<const char*>(&(nstates)), sizeof(int)); //nstates
        // printf("DEBUG:: nx,ny,nz,nd,dx,dy,dz,dv,Nspinor_eig,isGammaPoint,nspin,nkpt,nband = [%d,%d,%d,%d,%f,%f,%f,%f,%d,%d,%d,%d,%d]\n",
        //         this->mesh_control.nx, this->mesh_control.ny, this->mesh_control.nz, this->mesh_control.nd,
        //         this->mesh_control.delta_x, this->mesh_control.delta_y, this->mesh_control.delta_z, this->mesh_control.delta_V,
        //         one, one, nspin, one, nstates);

        Array_4D<T> psi(this->domain_vertices.shared_vertices);
        Linalg::scalar_product_general(psi.data, this->eigen_vectors.data,
                                T(1.0/std::sqrt(this->mesh_control.delta_V)), psi.length);

        uint ispin = 0;
        uint kpt_index = 0;
        uint Nd_ex = this->domain_vertices.shared_vertices.Vertices_3D::get_size();
        double kpt_vec[3] = {0.0,0.0,0.0};
        for (uint istate = 0; istate < nstates; istate++) {
            output.write(reinterpret_cast<const char*>(&ispin), sizeof(int));
            output.write(reinterpret_cast<const char*>(&kpt_index), sizeof(int));
            output.write(reinterpret_cast<const char*>(kpt_vec), 3 * sizeof(double));
            output.write(reinterpret_cast<const char*>(&istate), sizeof(int));
            output.write(reinterpret_cast<const char*>(psi.data + Nd_ex*istate), Nd_ex * sizeof(T));
            // printf("DEBUG:: spin_index, kpt_index, kpt_vec, band_index = [%d,%d,%f,%f,%f,%d]\n",
            //         ispin, kpt_index, kpt_vec[0], kpt_vec[1], kpt_vec[2],istate);
        }

        output.close();
    }

}

template<typename T>
std::vector<Array_2D<T>> Eigen_solver<T>::cal_pdos() const {
    return this->cal_pdos(this->geometry.atoms, this->eigen_vectors);
}

template<typename T>
std::vector<Array_2D<T>> Eigen_solver<T>::cal_pdos(const std::vector<Atom>& atoms, const Array_4D<T>& eigen_vectors) const {
    assert(this->geometry.cell_type <= 2);
    // if (!this->domain_vertices.is_active) return std::vector<Array_2D<T>>(0);

    std::vector<Array_2D<T>> pdos_atoms;
    const uint natom = atoms.size();
    pdos_atoms.reserve(natom);

    const Vertices_3D local_vertices_3d = this->domain_vertices.get_3D_local_vertices();
    const uint& nd = local_vertices_3d.get_size();
    const Vertices_4D& shared_vertices_4d = this->domain_vertices.get_4D_shared_vertices();
    Array_4D<T> atomic_orbitals_Ynlm;
    uint n_atomic_orbital_Ynlm = 0;
    T atomic_orbital_Ynlm_cutoff = T(0);

    uint atom_type = UINT_MAX;
    for (uint iatom = 0; iatom < natom; iatom++) {
        const Atom& atom = atoms[iatom];
        if (!this->psp8_files[atom.type].if_has_upf_file) {
            pdos_atoms.emplace_back(Array_2D<T>(Vertices_2D(0,0),0));
            continue;
        }
        const Upf_file& upf_file = this->psp8_files[atom.type].upf_file;
        if (atom.type != atom_type) {
            //update atomic_orbitals_Ynlm
            n_atomic_orbital_Ynlm = 0;
            atomic_orbital_Ynlm_cutoff = T(0);
            for (int i_chi = 0; i_chi < upf_file.n_chi; i_chi++) {
                n_atomic_orbital_Ynlm += 2 * upf_file.atomic_orbitals[i_chi].l + 1;
                for (int i_grid = 0; i_grid < upf_file.n_grid; i_grid++) {
                    if (upf_file.atomic_orbitals[i_chi].chi[i_grid] > 10-12) {
                        atomic_orbital_Ynlm_cutoff = std::max(atomic_orbital_Ynlm_cutoff, T(upf_file.r_grid[i_grid]));
                    }
                }
            }
            atomic_orbital_Ynlm_cutoff = std::min(atomic_orbital_Ynlm_cutoff, T(6.0));
            atomic_orbitals_Ynlm.reconstructor(Vertices_4D(local_vertices_3d, n_atomic_orbital_Ynlm));
            atom_type = atom.type;
        }

        pdos_atoms.emplace_back(Array_2D<T>(Vertices_2D(n_atomic_orbital_Ynlm, this->domain_vertices.shared_vertices.nb),0));
        Array_2D<T>& pdos_atom = pdos_atoms.back();

        if (this->domain_vertices.is_active) {
            Vertices_3D atom_vertices = atom.generate_rc_vertices(
                                        this->geometry.cell_type, this->mesh_control.delta_x,
                                        this->mesh_control.delta_y, this->mesh_control.delta_z,
                                        atomic_orbital_Ynlm_cutoff, atomic_orbital_Ynlm_cutoff,
                                        atomic_orbital_Ynlm_cutoff);
            int i_left = this->mesh_control.is_periodic[0]
                    ? Tools::group_of(atom_vertices.get_is() - shared_vertices_4d.get_is(), shared_vertices_4d.get_ni())
                    : 0;
            int i_right = this->mesh_control.is_periodic[0]
                    ? Tools::group_of(atom_vertices.get_ie() - shared_vertices_4d.get_is(), shared_vertices_4d.get_ni())
                    : 0;
            int j_left = this->mesh_control.is_periodic[1]
                    ? Tools::group_of(atom_vertices.get_js() - shared_vertices_4d.get_js(), shared_vertices_4d.get_nj())
                    : 0;
            int j_right = this->mesh_control.is_periodic[1]
                    ? Tools::group_of(atom_vertices.get_je() - shared_vertices_4d.get_js(), shared_vertices_4d.get_nj())
                    : 0;
            int k_left = this->mesh_control.is_periodic[2]
                    ? Tools::group_of(atom_vertices.get_ks() - shared_vertices_4d.get_ks(), shared_vertices_4d.get_nk())
                    : 0;
            int k_right = this->mesh_control.is_periodic[2]
                    ? Tools::group_of(atom_vertices.get_ke() - shared_vertices_4d.get_ks(), shared_vertices_4d.get_nk())
                    : 0;
            Linalg::set_value_general(atomic_orbitals_Ynlm.data, T(0), atomic_orbitals_Ynlm.length);
            // Array_0D<T> square_sum1(n_atomic_orbital_Ynlm, 0);
            for (int i_vertice = i_left; i_vertice <= i_right; i_vertice++) {
                for (int j_vertice = j_left; j_vertice <= j_right; j_vertice++) {
                    for (int k_vertice = k_left; k_vertice <= k_right; k_vertice++) {
                        const Vertices_3D vertices_ijk = 
                            atom_vertices.get_overlap_vertices(local_vertices_3d,
                                                                i_vertice * int(shared_vertices_4d.get_ni()),
                                                                j_vertice * int(shared_vertices_4d.get_nj()),
                                                                k_vertice * int(shared_vertices_4d.get_nk()));
                        if (vertices_ijk.get_size() == 0) continue;
                        Array_4D<T> atomic_orbitals_nlm_ijk =
                                            atom.generate_atomic_orbitals_nlm_array<T>(
                                            this->geometry.cell_type, n_atomic_orbital_Ynlm, vertices_ijk,
                                            upf_file, atomic_orbital_Ynlm_cutoff, this->mesh_control.delta_x,
                                            this->mesh_control.delta_y, this->mesh_control.delta_z);
                        const uint nd_ijk = vertices_ijk.get_size();
                        for (uint i_atomic_orbital_Ynlm = 0; i_atomic_orbital_Ynlm < n_atomic_orbital_Ynlm; i_atomic_orbital_Ynlm++) {
                            // square_sum1[i_atomic_orbital_Ynlm] += Linalg::vector_norm_square_sum(
                            //                             atomic_orbitals_nlm_ijk.data + nd_ijk * i_atomic_orbital_Ynlm, nd_ijk);
                            Vertices_3D vertices_000 = vertices_ijk.get_shifed_vertices(
                                                        -i_vertice * int(shared_vertices_4d.get_ni()),
                                                        -j_vertice * int(shared_vertices_4d.get_nj()),
                                                        -k_vertice * int(shared_vertices_4d.get_nk()));
                            Vertices_method::accumulate_overlap(
                                atomic_orbitals_nlm_ijk.data + nd_ijk * i_atomic_orbital_Ynlm, vertices_000,
                                atomic_orbitals_Ynlm.data + nd * i_atomic_orbital_Ynlm, local_vertices_3d, vertices_000);
                        }
                    }
                }
            }
            Array_0D<T> square_sum2(n_atomic_orbital_Ynlm, 0);
            for (uint i_atomic_orbital_Ynlm = 0; i_atomic_orbital_Ynlm < n_atomic_orbital_Ynlm; i_atomic_orbital_Ynlm++) {
                square_sum2[i_atomic_orbital_Ynlm] += Linalg::vector_norm_square_sum(
                                            atomic_orbitals_Ynlm.data + nd * i_atomic_orbital_Ynlm, nd);
            }
            // Linalg::scalar_product_general(square_sum1.data, T(this->mesh_control.delta_V), n_atomic_orbital_Ynlm);
            // Linalg::scalar_product_general(square_sum2.data, T(this->mesh_control.delta_V), n_atomic_orbital_Ynlm);
            // MPI_Allreduce(MPI_IN_PLACE, square_sum1.data, n_atomic_orbital_Ynlm, Linalg::get_mpi_datatype(square_sum.data),
            //                 MPI_SUM, this->domain_vertices.domain_3d_comm);
            MPI_Allreduce(MPI_IN_PLACE, square_sum2.data, n_atomic_orbital_Ynlm, Linalg::get_mpi_datatype<T>(),
                            MPI_SUM, this->domain_vertices.domain_3d_comm);
            for (uint i_atomic_orbital_Ynlm = 0; i_atomic_orbital_Ynlm < n_atomic_orbital_Ynlm; i_atomic_orbital_Ynlm++) {
                // T nornalize_factor = 1.0/std::sqrt(square_sum1[i_atomic_orbital_Ynlm]);
                T nornalize_factor = 1.0/std::sqrt(square_sum2[i_atomic_orbital_Ynlm]);
                Linalg::scalar_product_general(atomic_orbitals_Ynlm.data + nd * i_atomic_orbital_Ynlm, nornalize_factor, nd);
            }
        
            // Linalg::matrix_product(this->eigen_vectors.data, 0, atomic_orbitals_Ynlm.data, 1,
            //                     pdos_atom.data + n_atomic_orbital_Ynlm * this->domain_vertices.local_vertices.bs, 0,
            //                     this->domain_vertices.local_vertices.nb, n_atomic_orbital_Ynlm, nd);
            Linalg::matrix_product(atomic_orbitals_Ynlm.data, 0, eigen_vectors.data, 1,
                                pdos_atom.data + n_atomic_orbital_Ynlm * this->domain_vertices.local_vertices.bs, 1,
                                n_atomic_orbital_Ynlm, this->domain_vertices.local_vertices.nb, nd);
        }

        MPI_Allreduce(MPI_IN_PLACE, pdos_atom.data, n_atomic_orbital_Ynlm * this->domain_vertices.shared_vertices.nb,
                      Linalg::get_mpi_datatype<T>(), MPI_SUM, this->domain_vertices.comm);
        // Linalg::scalar_product_general(pdos_atom.data, T(std::sqrt(this->mesh_control.delta_V)), n_atomic_orbital_Ynlm * this->domain_vertices.shared_vertices.nb);
        Linalg::vector_norm_square(pdos_atom.data, n_atomic_orbital_Ynlm * this->domain_vertices.shared_vertices.nb);
    }
    return pdos_atoms;
}

template<typename T>
void Eigen_solver<T>::get_region_fracs(const Vertices_3D& region, T* const fracs) const {
    if (this->domain_vertices.is_active) {
        const Vertices_4D& local_vertices = this->domain_vertices.local_vertices;
        const uint local_nstates = local_vertices.nb;
        Vertices_3D overlapped_region = region.get_overlap_vertices(
                                this->domain_vertices.local_vertices);
        Array_4D<T> sub_eigen_vector = this->eigen_vectors.sub_arr(Vertices_4D(overlapped_region,
                                        local_vertices.bs, local_vertices.get_be()));
        uint Nd = region.get_size();
        for (uint i = 0; i < local_nstates; i++) {
            fracs[local_vertices.bs + i] = Linalg::vector_norm_square_sum(sub_eigen_vector.data + i * Nd, Nd);
        }
        MPI_Allreduce(MPI_IN_PLACE, fracs, this->domain_vertices.shared_vertices.nb, Linalg::get_mpi_datatype<T>(), MPI_SUM, this->domain_vertices.domain_3d_comm);
        MPI_Allreduce(MPI_IN_PLACE, fracs, this->domain_vertices.shared_vertices.nb, Linalg::get_mpi_datatype<T>(), MPI_SUM, this->domain_vertices.band_comm);
    }
    return;
}

template<typename T>
void Eigen_solver<T>::get_eigens(T* const eigens) const {
    if (this->eigen_solver_control.method == 0) { //chefsi
        Linalg::set_value_general(eigens, this->eigen_values.data, this->eigen_solver_control.nstates);
    } else {
        assert(this->eigen_solver_control.method == 0);
    }
    return;
}

template<typename T>
void Eigen_solver<T>::run(const Array_3D<T>& effective_potentail_loc, const bool print_flag) {
    this->run_mp(effective_potentail_loc.data, print_flag);
    // #ifdef USE_OPENMP

    // #ifdef ENABLE_EIGEN_SOLVER_TIMER
    //     #pragma omp master
    //     {
    //         this->eigen_solver_timer.reset();
    //         this->eigen_solver_timer.eigen_solver.start();
    //     }
    // #endif //ENABLE_EIGEN_SOLVER_TIMER

    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // #ifdef LOW_MEMORY
    // #pragma omp single
    // {
    //     #ifdef ENABLE_EIGEN_SOLVER_TIMER
    //         this->eigen_solver_timer.low_memory_chi.start();
    //         this->effective_potential_nloc.init(print_flag);
    //         this->eigen_solver_timer.low_memory_chi.stop();
    //     #else
    //         this->effective_potential_nloc.init(print_flag);
    //     #endif //ENABLE_EIGEN_SOLVER_TIMER
    // }
    // #endif

    // #ifdef ENABLE_EIGEN_SOLVER_TIMER
    //     #pragma omp master
    //     this->eigen_solver_timer.eigen_solver_kernel.start();
    // #endif //ENABLE_EIGEN_SOLVER_TIMER
    // if (this->eigen_solver_control.method == 0) {
    //     this->chefsi.run(this->eigen_vectors, this->eigen_values, effective_potentail_loc,
    //                      this->effective_potential_nloc, print_flag);
    // } else {
    //     assert(this->eigen_solver_control.method == 0);
    // }
    // #ifdef ENABLE_EIGEN_SOLVER_TIMER
    //     #pragma omp master
    //     this->eigen_solver_timer.eigen_solver_kernel.stop();
    // #endif //ENABLE_EIGEN_SOLVER_TIMER
    // #pragma omp barrier
    // #pragma omp single nowait
    // {
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
    //     std::cout << "The Eigen_solver run took " << Tools::time_cost(begin, end) << "." << std::endl;
    // }
    // }

    // #ifdef LOW_MEMORY
    // #pragma omp single
    // {
    //     #ifdef ENABLE_EIGEN_SOLVER_TIMER
    //         this->eigen_solver_timer.low_memory_chi.start();
    //         this->effective_potential_nloc.destructor();
    //         this->eigen_solver_timer.low_memory_chi.stop();
    //     #else
    //         this->effective_potential_nloc.destructor();
    //     #endif //ENABLE_EIGEN_SOLVER_TIMER
    // }
    // #endif

    // #pragma omp master
    // if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
    //     this->print_runtime_result();
    // }

    // #ifdef ENABLE_EIGEN_SOLVER_TIMER
    //     #pragma omp master
    //     {
    //         this->eigen_solver_timer.eigen_solver.stop();
    //         if (this->domain_vertices.get_comm_rank() == 0 && print_flag) this->eigen_solver_timer.show();
    //     }
    // #endif //ENABLE_EIGEN_SOLVER_TIMER

    // #else //USE_OPENMP

    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // #ifdef LOW_MEMORY
    // this->effective_potential_nloc.init(print_flag);
    // #endif
    // if (this->eigen_solver_control.method == 0) {
    //     this->chefsi.run(this->eigen_vectors, this->eigen_values, effective_potentail_loc,
    //                      this->effective_potential_nloc, print_flag);
    // } else {
    //     assert(this->eigen_solver_control.method == 0);
    // }
    // if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
    //     this->print_runtime_result();
    // }
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
    //     std::cout << "The Eigen_solver run took " << Tools::time_cost(begin, end) << "." << std::endl;
    // }
    // #ifdef LOW_MEMORY
    // this->effective_potential_nloc.destructor();
    // #endif

    // #endif //USE_OPENMP
    // return;
}

template<typename T>
void Eigen_solver<T>::run_mp(T const* const effective_potentail_loc, const bool print_flag) {
    constexpr size_t GB = 1024 * 1024 * 1024 / sizeof(T);
    Memory_pool<T, Fast_memory> pool_fast(3 * GB);
    Memory_pool<T, Capacity_memory> pool_cap(3 * GB);
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    const uint local_vertices_size = this->domain_vertices.get_4D_local_vertices().get_size();
    T*& eigen_vectors_in = this->eigen_vectors.data;
    T* eigen_vectors_out = pool_fast.allocate(local_vertices_size);
    this->run_mp(eigen_vectors_in, eigen_vectors_out, effective_potentail_loc,
                    print_flag, pool_fast, pool_cap);
    Linalg::set_value_general(this->eigen_vectors.data, eigen_vectors_out, local_vertices_size);
    return;
}

template<typename T>
void Eigen_solver<T>::run_mp(T*& eigen_vectors_in, T*& eigen_vectors_out, T const* const effective_potentail_loc,
            const bool print_flag, Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    #ifdef ENABLE_EIGEN_SOLVER_TIMER
        this->eigen_solver_timer.reset();
        this->eigen_solver_timer.eigen_solver.start();
    #endif //ENABLE_EIGEN_SOLVER_TIMER

    #ifdef LOW_MEMORY
        #ifdef ENABLE_EIGEN_SOLVER_TIMER
            this->eigen_solver_timer.low_memory_chi.start();
            // this->effective_potential_nloc.init(print_flag);
            this->effective_potential_nloc.init_mp(print_flag, pool_fast, pool_cap);
            this->eigen_solver_timer.low_memory_chi.stop();
        #else
            // this->effective_potential_nloc.init(print_flag);
            this->effective_potential_nloc.init_mp(print_flag, pool_fast, pool_cap);
        #endif //ENABLE_EIGEN_SOLVER_TIMER
    #endif

    #ifdef ENABLE_EIGEN_SOLVER_TIMER
        this->eigen_solver_timer.eigen_solver_kernel.start();
    #endif //ENABLE_EIGEN_SOLVER_TIMER
    if (this->eigen_solver_control.method == 0) {
        this->chefsi.run_mp(eigen_vectors_in, eigen_vectors_out, this->eigen_values.data, effective_potentail_loc,
                         this->effective_potential_nloc, print_flag, pool_fast, pool_cap);
    } else {
        assert(this->eigen_solver_control.method == 0);
    }
    #ifdef ENABLE_EIGEN_SOLVER_TIMER
        this->eigen_solver_timer.eigen_solver_kernel.stop();
    #endif //ENABLE_EIGEN_SOLVER_TIMER

    #ifdef LOW_MEMORY
        #ifdef ENABLE_EIGEN_SOLVER_TIMER
            this->eigen_solver_timer.low_memory_chi.start();
            // this->effective_potential_nloc.destructor();
            this->effective_potential_nloc.destructor_mp();
            this->eigen_solver_timer.low_memory_chi.stop();
        #else
            // this->effective_potential_nloc.destructor();
            this->effective_potential_nloc.destructor_mp();
        #endif //ENABLE_EIGEN_SOLVER_TIMER
    #endif

    if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
        this->print_runtime_result();
    }

    #ifdef ENABLE_EIGEN_SOLVER_TIMER
        this->eigen_solver_timer.eigen_solver.stop();
        if (this->domain_vertices.get_comm_rank() == 0 && print_flag) this->eigen_solver_timer.show();
    #endif //ENABLE_EIGEN_SOLVER_TIMER
}

template<typename T>
void Eigen_solver<T>::print_runtime_result() {
    const uint nstate = this->eigen_values.length;
    const uint print_num = std::min(5u, nstate/2);
    std::cout << YELLOW << "Eigen_solver :: " << RESET << std::endl;
    std::cout << YELLOW << "First " << print_num << " eigenvalues: [";
    for (uint i = 0; i < print_num - 1; i ++) {
        std::cout << std::setprecision(6) << std::fixed << this->eigen_values[i] << ", ";
    }
    std::cout << std::setprecision(6) << std::fixed << this->eigen_values[print_num - 1] << "]." << YELLOW << std::endl;
    std::cout << YELLOW << "Last " << print_num << " eigenvalues: [";
    for (uint i = print_num; i > 1; i--) {
        std::cout << std::setprecision(6) << std::fixed << this->eigen_values[nstate - i] << ", ";
    }
    std::cout << std::setprecision(6) << std::fixed << this->eigen_values[nstate - 1] << "]." << RESET << std::endl;
}

template<typename T>
double Eigen_solver<T>::evalutate_flops() {
    if (this->eigen_solver_control.method == 0) {
        return this->chefsi.evalutate_flops();
    } else {
        assert(this->eigen_solver_control.method == 0);
    }
    return 0.0;
}

template<typename T>
void Eigen_solver<T>::init() {
    // if (this->domain_vertices.get_4D_shared_vertices().get_size() == 0) return;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_4D_shared_vertices().get_size() == 0) {
        // init eigen_values and eigen_vectors
        this->eigen_values.reconstructor(this->domain_vertices.get_4D_shared_vertices().nb, 0);
        this->eigen_vectors.reconstructor(this->domain_vertices.get_4D_local_vertices());
        // exarr_mpi_package
        this->exarr_mpi_package.comm = this->exarr_mpi_package.domain_vertices.get_domain_4d_comm();
        // chefsi
        bool const is_periodic[4] = {this->mesh_control.is_periodic[0], this->mesh_control.is_periodic[1],
                                     this->mesh_control.is_periodic[2], false};
        this->chefsi.init(is_periodic, this->eigen_solver_control.is_rand_fixed);
        // intra_band_domain_vertices
        this->intra_band_domain_vertices.deepcopy_mpi_comm(this->domain_vertices.comm);
        this->intra_band_domain_vertices.set_is_active(false);
        this->intra_band_domain_vertices.set_local_vertices(Vertices_4D(0,0,0,0));
        this->intra_band_exarr_mpi_package.comm = this->intra_band_domain_vertices.get_domain_4d_comm();
        #ifndef LOW_MEMORY
        this->effective_potential_nloc.init();
        #endif
    } else if (this->eigen_solver_control.method == 0) {   //Chefsi
        // init eigen_values and eigen_vectors
        this->eigen_values.reconstructor(this->domain_vertices.get_4D_shared_vertices().nb, 0);
        this->eigen_vectors.reconstructor(this->domain_vertices.get_4D_local_vertices());
        // check the nstates
        if ((this->domain_vertices.get_comm_rank() == 0)
         && (this->domain_vertices.get_4D_shared_vertices().nb * 2
          > this->domain_vertices.get_4D_shared_vertices().Vertices_3D::get_size())) {
            std::cout << "WARNING:: nstates is " << this->domain_vertices.get_4D_shared_vertices().nb
                      << ", is kind of too big when the grids number is "
                      << this->domain_vertices.get_4D_shared_vertices().Vertices_3D::get_size()
                      << "."
                      << std::endl;
        }
        if (this->eigen_solver_control.is_rand_fixed) {
            Parallel_vertices::domain_vertices_rand<T>(this->eigen_vectors, this->domain_vertices, -0.5, 0.5);
        } else {
            this->eigen_vectors.seededrand(this->domain_vertices.get_comm_rank() * 100 + 1);
        }
        // exarr_mpi_package
        bool const is_periodic[4] = {this->mesh_control.is_periodic[0], this->mesh_control.is_periodic[1],
                                     this->mesh_control.is_periodic[2], false};
        int const FDn[4] = {this->stencil.FDn, this->stencil.FDn, this->stencil.FDn, 0};
        this->exarr_mpi_package.init(is_periodic, FDn);
        // chefsi
        this->chefsi.init(is_periodic, this->eigen_solver_control.is_rand_fixed);
        // intra_band_domain_vertices
        // Domain_parallel_vertices_4D intra_band_domain_vertices;
        if (this->domain_vertices.is_active) {
            this->intra_band_domain_vertices.set_chunk_size_i(this->domain_vertices.get_chunk_size_i());
            this->intra_band_domain_vertices.set_chunk_size_j(this->domain_vertices.get_chunk_size_j());
            this->intra_band_domain_vertices.set_chunk_size_k(this->domain_vertices.get_chunk_size_k());
            this->intra_band_domain_vertices.set_chunk_size_b(this->domain_vertices.get_4D_local_vertices().get_nb());
            Vertices_4D intra_band_shared_vertices(this->domain_vertices.get_3D_shared_vertices(),
                                                   this->domain_vertices.get_4D_local_vertices().get_bs(),
                                                   this->domain_vertices.get_4D_local_vertices().get_be());
            this->intra_band_domain_vertices.init(intra_band_shared_vertices, this->domain_vertices.domain_3d_comm);
            this->intra_band_exarr_mpi_package.init(is_periodic, FDn);
        } else {
            this->intra_band_domain_vertices.init(this->domain_vertices.get_4D_shared_vertices(), this->domain_vertices.domain_3d_comm);
            this->intra_band_domain_vertices.set_is_active(false);
            this->intra_band_domain_vertices.set_local_vertices(Vertices_4D(0,0,0,0));
            this->intra_band_exarr_mpi_package.comm = this->intra_band_domain_vertices.get_domain_4d_comm();
        }
        #ifndef LOW_MEMORY
        this->effective_potential_nloc.init();
        #endif
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
    // if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The eigen_solver init took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

template<typename T>
template<typename T2>
void Eigen_solver<T>::init(const Eigen_solver<T2>& eigen_solver) {
    this->eigen_values.deepcopy(std::move(eigen_solver.eigen_values.as_type(this->eigen_values.data)));
    this->eigen_vectors.deepcopy(std::move(eigen_solver.eigen_vectors.as_type(this->eigen_vectors.data)));
    #ifndef LOW_MEMORY
    this->effective_potential_nloc.init(eigen_solver.effective_potential_nloc);
    #endif
    this->exarr_mpi_package.init(eigen_solver.exarr_mpi_package);
    this->intra_band_domain_vertices.init(eigen_solver.intra_band_domain_vertices);
    this->intra_band_exarr_mpi_package.init(eigen_solver.intra_band_exarr_mpi_package);
    this->chefsi.init(eigen_solver.chefsi);
    return;
}
template void Eigen_solver<float>::init(const Eigen_solver<float>& eigen_solver);
template void Eigen_solver<double>::init(const Eigen_solver<double>& eigen_solver);
template void Eigen_solver<float>::init(const Eigen_solver<double>& eigen_solver);
template void Eigen_solver<double>::init(const Eigen_solver<float>& eigen_solver);

template<typename T>
void Eigen_solver<T>::destructor() {
    this->eigen_values.destructor();
    this->eigen_vectors.destructor();
    #ifndef LOW_MEMORY
    this->effective_potential_nloc.destructor();
    #endif
    this->exarr_mpi_package.destructor();
    this->intra_band_domain_vertices.destructor();
    this->intra_band_exarr_mpi_package.destructor();
    this->chefsi.destructor();
    return;
}

template<typename T>
void Eigen_solver<T>::show() const {
    this->eigen_solver_control.show();
    this->domain_vertices.show();
    this->eigen_values.show();
    this->eigen_vectors.show();
    this->effective_potential_nloc.show();
    this->exarr_mpi_package.show();
    this->intra_band_exarr_mpi_package.show();
    return;
}

template class Eigen_solver<float>;
template class Eigen_solver<double>;

namespace Eigen_solver_method {
    template<typename T>
    T evaluate_chemical_potential(T const * const eigen_values, uint const length, const T lower_bound, const T upper_bound,
                                const Smearing& smearing, const T electron_charge, const T smearing_coef, const uint max_iter, const T tol) {
        T a = lower_bound;
        T b = upper_bound;
        T* occ = new T[length];
        Smearing_method::smear(eigen_values, occ, a, smearing, length);
        T fa = Linalg::vector_sum(occ, length) * smearing_coef - electron_charge;
        Smearing_method::smear(eigen_values, occ, b, smearing, length);
        T fb = Linalg::vector_sum(occ, length) * smearing_coef - electron_charge;
        T c = T(0);

        uint ext_loop_count = 0;
        while (fa * fb > 0.0 && ext_loop_count++ < 10) {
            T w = b - a;
            a -= w / 2.0;
            b += w / 2.0;
            c = b;
            Smearing_method::smear(eigen_values, occ, a, smearing, length);
            fa = Linalg::vector_sum(occ, length) * smearing_coef - electron_charge;
            Smearing_method::smear(eigen_values, occ, b, smearing, length);
            fb = Linalg::vector_sum(occ, length) * smearing_coef - electron_charge;
        }

        if (fa * fb > T(0.0)) {
            assert(0 && "Cannot find the chemical potential in the given range!");
        }

        T fc = fb;
        T e = T(0);
        T d = T(0);
        T tol1 = T(0);
        T xm = T(0);
        T s = T(0);
        T p = T(0);
        T q = T(0);
        T r = T(0);
        T tol1q = T(0);
        T min1 = T(0);
        T min2 = T(0);
        T eq = T(0);
        #define EPSILON 1e-16
        #define SIGN(a, b) ((b) > T(0.0) ? std::fabs((a)) : -std::fabs((a)))
        for (uint iter = 1; iter <= max_iter; iter++) {
            if ((fb > T(0.0) && fc > T(0.0)) || (fb < T(0.0) && fc < T(0.0))) {
                c = a;
                fc = fa;
                e = d = b - a;
            }
            if (std::fabs(fc) < std::fabs(fb)) {
                a = b;
                b = c;
                c = a;
                fa = fb;
                fb = fc;
                fc = fa;
            }
            tol1 = 2.0 * T(EPSILON) * std::fabs(b) + 0.5 * tol;
            xm = 0.5 * (c - b);
            if (std::fabs(xm) <= tol1 || std::fabs(fb) < T(EPSILON)) {
                delete[] occ;
                return b;
            }
            if (std::fabs(e) >= tol1 && std::fabs(fa) > std::fabs(fb)) {
                // attempt inverse quadratic interpolation
                s = fb / fa;
                if (a == c) {
                    p = 2.0 * xm * s;
                    q = 1.0 - s;
                } else {
                    q = fa / fc;
                    r = fb / fc;
                    p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                    q = (q - 1.0) * (r - 1.0) * (s - 1.0);
                }
                if (p > 0.0) {
                    // check whether in bounds
                    q = -q;
                }
                p = fabs(p);
                tol1q = tol1 * q;
                min1 = 3.0 * xm * q - fabs(tol1q);
                eq = e * q;
                min2 = fabs(eq);
                if (2.0 * p < (min1 < min2 ? min1 : min2)) {
                    // accept interpolation
                    e = d;
                    d = p / q;
                } else {
                    // Bounds decreasing too slowly, use bisection
                    d = xm;
                    e = d;
                }
            } else {
                d = xm;
                e = d;
            }
            // move last best guess to a
            a = b;
            fa = fb;

            if (fabs(d) > tol1) {
                // evaluate new trial root
                b += d;
            } else {
                b += SIGN(tol1, xm);
            }
            Smearing_method::smear(eigen_values, occ, b, smearing, length);
            fb = Linalg::vector_sum(occ, length) * smearing_coef - electron_charge;
        }
        #undef EPSILON
        #undef SIGN

        delete[] occ;
        return T(0.0);
    }
    template float evaluate_chemical_potential<float>(float const * const eigen_values, uint const length, const float lower_bound, const float upper_bound,
                                const Smearing& smearing, const float electron_charge, const float smearing_coef, const uint max_iter, const float tol);
    template double evaluate_chemical_potential<double>(double const * const eigen_values, uint const length, const double lower_bound, const double upper_bound,
                                const Smearing& smearing, const double electron_charge, const double smearing_coef, const uint max_iter, const double tol);
}
