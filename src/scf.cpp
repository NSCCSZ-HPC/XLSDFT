#include "scf.h"

Scf::Scf(const Preparation& preparation,
         const MPI_Comm& comm)
       : comm(comm),
         scf_control(preparation.control.scf_control),
         misc_control(preparation.control.misc_control),
         mesh_control(preparation.control.mesh_control),
         psp8_files(preparation.psp8_files),
         density_solver_control(preparation.control.density_solver_control),
         spin_control(preparation.control.spin_control),
         stencil_control(preparation.control.stencil_control),
         geometry(preparation.geometry),
         global_domain_mpi_package(this->global_domain_vertices, this->domain_vertices),
         exarr_mpi_package(this->domain_vertices),
         spin(preparation.control.spin_control),
         seffective_potential_loc_solver(preparation.control.poisson_solver_control,
                                         preparation.control.exchange_correlation_solver_control,
                                         preparation.geometry,
                                         this->spin,
                                         this->sstencil,
                                         preparation.control.mesh_control,
                                         this->psp8_files,
                                         this->domain_vertices,
                                         this->exarr_mpi_package),
         deffective_potential_loc_solver(preparation.control.poisson_solver_control,
                                         preparation.control.exchange_correlation_solver_control,
                                         preparation.geometry,
                                         this->spin,
                                         this->dstencil,
                                         preparation.control.mesh_control,
                                         this->psp8_files,
                                         this->domain_vertices,
                                         this->exarr_mpi_package),
         sdensity_solver(preparation.control.density_solver_control,
                         preparation.control.mesh_control,
                         preparation.geometry,
                         this->spin,
                         this->sstencil,
                         this->psp8_files,
                         this->domain_vertices),
         ddensity_solver(preparation.control.density_solver_control,
                         preparation.control.mesh_control,
                         preparation.geometry,
                         this->spin,
                         this->dstencil,
                         this->psp8_files,
                         this->domain_vertices),
         smixing(preparation.control.scf_control.mixing_control,
                 this->spin, this->sstencil, this->exarr_mpi_package,
                 this->scf_control.mixing_variable),
         dmixing(preparation.control.scf_control.mixing_control,
                 this->spin, this->dstencil, this->exarr_mpi_package,
                 this->scf_control.mixing_variable) {}

Scf::Scf(const Control& control, const Geometry& geometry,
        const std::vector<Psp8_file>& psp8_files, const MPI_Comm& comm)
       : comm(comm),
         scf_control(control.scf_control),
         misc_control(control.misc_control),
         mesh_control(control.mesh_control),
         psp8_files(psp8_files),
         density_solver_control(control.density_solver_control),
         spin_control(control.spin_control),
         stencil_control(control.stencil_control),
         geometry(geometry),
         global_domain_mpi_package(this->global_domain_vertices, this->domain_vertices),
         exarr_mpi_package(this->domain_vertices),
         spin(control.spin_control),
         seffective_potential_loc_solver(control.poisson_solver_control,
                                         control.exchange_correlation_solver_control,
                                         geometry,
                                         this->spin,
                                         this->sstencil,
                                         control.mesh_control,
                                         this->psp8_files,
                                         this->domain_vertices,
                                         this->exarr_mpi_package),
         deffective_potential_loc_solver(control.poisson_solver_control,
                                         control.exchange_correlation_solver_control,
                                         geometry,
                                         this->spin,
                                         this->dstencil,
                                         control.mesh_control,
                                         this->psp8_files,
                                         this->domain_vertices,
                                         this->exarr_mpi_package),
         sdensity_solver(control.density_solver_control,
                         control.mesh_control,
                         geometry,
                         this->spin,
                         this->sstencil,
                         this->psp8_files,
                         this->domain_vertices),
         ddensity_solver(control.density_solver_control,
                         control.mesh_control,
                         geometry,
                         this->spin,
                         this->dstencil,
                         this->psp8_files,
                         this->domain_vertices),
         smixing(control.scf_control.mixing_control,
                 this->spin, this->sstencil, this->exarr_mpi_package,
                 this->scf_control.mixing_variable),
         dmixing(control.scf_control.mixing_control,
                 this->spin, this->dstencil, this->exarr_mpi_package,
                 this->scf_control.mixing_variable) {}

Scf::~Scf() {}

void Scf::scale_electron_density() {
    // if (this->scf_control.max_single_precision_iter > 0) {
    //     float ratio = - this->seffective_potential_loc_solver.pseudo_charge_solver.pseudo_charge
    //                   / this->sdensity_solver.electron_charge;
    //     this->sdensity_solver.electron_density *= ratio;
    //     this->sdensity_solver.electron_charge *= ratio;
    //     if (this->domain_vertices.get_comm_rank() == 0) std::cout << "scale electron density with the ratio = " << std::setprecision(13) << ratio << std::endl;
    // } else {
        double ratio = - this->deffective_potential_loc_solver.pseudo_charge_solver.pseudo_charge
                       / this->ddensity_solver.electron_charge;
        this->ddensity_solver.electron_density_init *= ratio;
        this->ddensity_solver.electron_charge *= ratio;
        if (this->spin.get_spin_type() == 0) {
            this->ddensity_solver.electron_densities[0] = this->ddensity_solver.electron_density_init;
        } else if (this->spin.get_spin_type() == 1) {
            const uint length = this->ddensity_solver.electron_density_init.length;
            Linalg::set_value_general(this->ddensity_solver.electron_densities[0].data,
                            this->ddensity_solver.electron_density_init.data, length);
            Linalg::hadamard_plus_general(this->ddensity_solver.electron_densities[0].data,
                            this->ddensity_solver.magnetization_z.data, length);
            Linalg::scalar_product_general(this->ddensity_solver.electron_densities[0].data,
                            0.5, length);
            Linalg::hadamard_minus_general(this->ddensity_solver.electron_densities[1].data,
                            this->ddensity_solver.electron_densities[0].data,
                            this->ddensity_solver.magnetization_z.data, length);
        } else {
            assert(this->spin.get_spin_type() == 0 || this->spin.get_spin_type() == 1);
        }
        if (this->domain_vertices.get_comm_rank() == 0) std::cout << "scale electron density with the ratio = " << std::setprecision(13) << ratio << std::endl;
    // }
    return;
}

void Scf::electron_density_pretreatment() {
    if (!this->double_precision_flag) {
        assert(false && "electron_density_pretreatment is only for double precision.");
    } else {
        int rank = this->domain_vertices.get_comm_rank();
        int comm_i = (int)this->domain_vertices.get_active_comm_i();
        int comm_j = (int)this->domain_vertices.get_active_comm_j();
        int comm_k = (int)this->domain_vertices.get_active_comm_k();
        int color_i = this->domain_vertices.is_active ? comm_i : INT_MAX;
        int color_jk = this->domain_vertices.is_active ? comm_j + comm_k * (int)this->domain_vertices.get_active_comm_nj() : INT_MAX;
        MPI_Comm i_comm;
        MPI_Comm jk_comm;
        MPI_Comm_split(this->domain_vertices.comm, color_i, rank, &i_comm);
        MPI_Comm_split(this->domain_vertices.comm, color_jk, rank, &jk_comm);
        uint nspin = this->spin.generate_nspin();
        if (this->domain_vertices.is_active) {
            this->deffective_potential_loc_solver.cal_effective_potential_loc(
                    this->ddensity_solver.electron_densities, this->ddensity_solver.electron_density_core);
            const Vertices_3D local_vertices = this->domain_vertices.get_3D_local_vertices();
            const uint64_t ni = local_vertices.ni;
            const uint64_t nj = local_vertices.nj;
            const uint64_t nk = local_vertices.nk;
            const Array_3D<double>& electrostatic_potential = this->deffective_potential_loc_solver.poisson_solver.electrostatic_potential;
            Array_0D<double> effective_potential_dir_i(ni, 0);
            // Array_0D<double> F_effective_potential_dir_i(ni);
            // Array_0D<double> G_effective_potential_dir_i(ni);
            for (uint64_t k = 0; k < nk; k++) {
                for (uint64_t j = 0; j < nj; j++) {
                    Linalg::hadamard_plus_general(effective_potential_dir_i.data,
                                    electrostatic_potential.data + (k * nj + j) * ni, ni);
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, effective_potential_dir_i.data, ni, MPI_DOUBLE, MPI_SUM, i_comm);
            double spin_coef = nspin == 2 ? 0.5 : 1.0;
            Linalg::scalar_product_general(effective_potential_dir_i.data,
                                            spin_coef * 0.25 * M_1_PI
                                            / double(this->domain_vertices.get_3D_shared_vertices().nj
                                                   * this->domain_vertices.get_3D_shared_vertices().nk), ni);
            assert(this->geometry.cell_type <= 2);
            int rank_jk;
            int size_jk;
            MPI_Comm_rank(jk_comm, &rank_jk);
            MPI_Comm_size(jk_comm, &size_jk);
            std::vector<int> counts(size_jk, (int)this->domain_vertices.chunk_size_i);
            counts.back() = (int)this->domain_vertices.get_last_block_size_i();
            std::vector<int> displs(size_jk);
            displs[0] = 0;
            for (int i = 1; i < size_jk; i++) {
                displs[i] = displs[i-1] + counts[i-1];
            }
            int64_t global_ni = (int)this->domain_vertices.get_3D_shared_vertices().ni;
            Array_0D<double> effective_potential_dir_i_global(global_ni);
            Array_0D<double> D2_effective_potential_dir_i_global(global_ni);
            MPI_Allgatherv(effective_potential_dir_i.data, (int)ni, MPI_DOUBLE,
                           effective_potential_dir_i_global.data, counts.data(), displs.data(), MPI_DOUBLE, jk_comm);
            Tools::second_deriv_fd<double>(D2_effective_potential_dir_i_global.data, effective_potential_dir_i_global.data, this->dstencil.get_D2_coeffs_x(), this->dstencil.FDn, global_ni, this->mesh_control.is_periodic[0]);
            double sum_G = Linalg::vector_sum(D2_effective_potential_dir_i_global.data, global_ni);
            Array_0D<double> density_shift(ni);
            Linalg::scalar_minus_general(density_shift.data, D2_effective_potential_dir_i_global.data+displs[rank_jk], sum_G / (double)global_ni, ni);
            for (uint ispin = 0; ispin < nspin; ispin++) {
                for (uint64_t k = 0; k < nk; k++) {
                    for (uint64_t j = 0; j < nj; j++) {
                        Linalg::hadamard_plus_general(
                            this->ddensity_solver.electron_densities[ispin].data + (k * nj + j) * ni,
                                        density_shift.data, ni);
                    }
                }
            }
        }
        MPI_Comm_free(&i_comm);
        MPI_Comm_free(&jk_comm);
    }
    return;
}

void Scf::set_electron_density(const Array_3D<double>& electron_density) {
    if (!this->double_precision_flag) {
        assert(false && "set_electron_density is only for double precision.");
    } else {
        Linalg::set_value_general(this->ddensity_solver.electron_densities[0].data,
            electron_density.data, this->ddensity_solver.electron_densities[0].length);
        Linalg::set_value_general(this->dmixing.x_km1.data,
            this->ddensity_solver.electron_densities[0].data, this->dmixing.x_km1.length);
    }
}

//TODO dmixing.x_km1 remove
double& Scf::evaluate_error(const uint& iter) {
    const uint Nd = this->domain_vertices.get_3D_local_vertices().get_size();
    uint spin_type = this->spin.get_spin_type();
    assert(this->scf_control.mixing_variable == 0 || this->scf_control.mixing_variable == 1);
    if (!this->double_precision_flag) {
        if (spin_type == 0) {
            Array_0D<float>& old = this->smixing.x_km1;
            Array_0D<float>& input = this->scf_control.mixing_variable == 0
                                    ? this->sdensity_solver.electron_densities[0]
                                    : this->seffective_potential_loc_solver.effective_potentail_locs[0];
            Array_0D<float> differences(input - old);
            float diff_2norm = Linalg::vector_2norm(differences.data, Nd, this->domain_vertices.comm);
            float two_norm = Linalg::vector_2norm(input.data, Nd, this->domain_vertices.comm);
            this->error = (double) diff_2norm / two_norm;
        } else if (spin_type == 1) {
            Array_0D<float> old(Nd * 2);
            Array_0D<float> input(Nd * 2);
            if (this->scf_control.mixing_variable == 0) {
                Linalg::set_value_general(old.data, this->sdensity_solver.electron_densities_in[0].data, Nd);
                Linalg::set_value_general(old.data + Nd, this->sdensity_solver.electron_densities_in[1].data, Nd);
                Linalg::set_value_general(input.data, this->sdensity_solver.electron_densities[0].data, Nd);
                Linalg::set_value_general(input.data + Nd, this->sdensity_solver.electron_densities[1].data, Nd);
                Linalg::set_value_general(this->sdensity_solver.electron_densities_in[0].data, this->sdensity_solver.electron_densities[0].data, Nd);
                Linalg::set_value_general(this->sdensity_solver.electron_densities_in[1].data, this->sdensity_solver.electron_densities[1].data, Nd);
            } else {
                Linalg::set_value_general(old.data, this->smixing.x_km1.data, Nd * 2);
                Linalg::set_value_general(input.data, this->seffective_potential_loc_solver.effective_potentail_locs[0].data, Nd);
                Linalg::set_value_general(input.data + Nd, this->seffective_potential_loc_solver.effective_potentail_locs[1].data, Nd);
            }
            Array_0D<float> differences(input - old);
            float diff_2norm = Linalg::vector_2norm(differences.data, Nd * 2, this->domain_vertices.comm);
            float two_norm = Linalg::vector_2norm(input.data, Nd * 2, this->domain_vertices.comm);
            this->error = (double) diff_2norm / two_norm;
        } else {
            assert(spin_type == 0 || spin_type == 1);
        }
    }  else {
        if (spin_type == 0) {
            Array_0D<double>& old = this->dmixing.x_km1;
            Array_0D<double>& input = this->scf_control.mixing_variable == 0
                                    ? this->ddensity_solver.electron_densities[0]
                                    : this->deffective_potential_loc_solver.effective_potentail_locs[0];
            Array_0D<double> differences(input - old);
            double diff_2norm = Linalg::vector_2norm(differences.data, Nd, this->domain_vertices.comm);
            double two_norm = Linalg::vector_2norm(input.data, Nd, this->domain_vertices.comm);
            this->error = (double) diff_2norm / two_norm;
        } else if (spin_type == 1) {
            Array_0D<double> old(Nd * 2);
            Array_0D<double> input(Nd * 2);
            if (this->scf_control.mixing_variable == 0) {
                Linalg::set_value_general(old.data, this->ddensity_solver.electron_densities_in[0].data, Nd);
                Linalg::set_value_general(old.data + Nd, this->ddensity_solver.electron_densities_in[1].data, Nd);
                Linalg::set_value_general(input.data, this->ddensity_solver.electron_densities[0].data, Nd);
                Linalg::set_value_general(input.data + Nd, this->ddensity_solver.electron_densities[1].data, Nd);
                Linalg::set_value_general(this->ddensity_solver.electron_densities_in[0].data, this->ddensity_solver.electron_densities[0].data, Nd);
                Linalg::set_value_general(this->ddensity_solver.electron_densities_in[1].data, this->ddensity_solver.electron_densities[1].data, Nd);
            } else {
                Linalg::set_value_general(old.data, this->dmixing.x_km1.data, Nd * 2);
                Linalg::set_value_general(input.data, this->deffective_potential_loc_solver.effective_potentail_locs[0].data, Nd);
                Linalg::set_value_general(input.data + Nd, this->deffective_potential_loc_solver.effective_potentail_locs[1].data, Nd);
            }
            Array_0D<double> differences(input - old);
            double diff_2norm = Linalg::vector_2norm(differences.data, Nd * 2, this->domain_vertices.comm);
            double two_norm = Linalg::vector_2norm(input.data, Nd * 2, this->domain_vertices.comm);
            this->error = (double) diff_2norm / two_norm;
        } else {
            assert(spin_type == 0 || spin_type == 1);
        }
    }
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "iteration: " << iter+1
                  << ", scf error = " << std::scientific << std::setw(6) << std::setprecision(3) << this->error
                  << "." << std::endl; 
    }
    return this->error;
}

double& Scf::evaluate_ion_elecst_energy() {
    uint length = this->domain_vertices.get_3D_local_vertices().get_size();
    if (!this->double_precision_flag) {
        Array_0D<float> temp(length);
        Linalg::hadamard_product_general(temp.data,
                                         this->seffective_potential_loc_solver.pseudo_charge_solver.pseudo_charge_density.data,
                                         this->seffective_potential_loc_solver.poisson_solver.electrostatic_potential.data,
                                         length,
                                         (float)(0.5 * this->mesh_control.delta_V));
        this->ion_elecst_energy = (double) Linalg::vector_sum(temp.data, length, this->domain_vertices.comm);
    } else {
        Array_0D<double> temp(length);
        Linalg::hadamard_product_general(temp.data,
                                         this->deffective_potential_loc_solver.pseudo_charge_solver.pseudo_charge_density.data,
                                         this->deffective_potential_loc_solver.poisson_solver.electrostatic_potential.data,
                                         length,
                                         (double)(0.5 * this->mesh_control.delta_V));
        this->ion_elecst_energy = (double) Linalg::vector_sum(temp.data, length, this->domain_vertices.comm);
    }
    return this->ion_elecst_energy;
}

//TODO dmixing.x_km1 remove
double& Scf::evaluate_electron_elecst_energy() {
    const uint length = this->domain_vertices.get_3D_local_vertices().get_size();
    const uint spin_type = this->spin.get_spin_type();
    if (!this->double_precision_flag) {
        Array_0D<float> temp(length);
        if (this->scf_control.mixing_variable == 0) {
            if (spin_type == 0) {
                Linalg::hadamard_product_general(temp.data,
                                         this->smixing.x_km1.data,
                                         this->seffective_potential_loc_solver.poisson_solver.electrostatic_potential.data,
                                         length);
            } else if (spin_type == 1) {
                Linalg::hadamard_product_general(temp.data,
                                         this->sdensity_solver.electron_densities_in[0].data,
                                         this->seffective_potential_loc_solver.poisson_solver.electrostatic_potential.data,
                                         length);
                Linalg::accumulate_hadamard_product_general(temp.data,
                                         this->sdensity_solver.electron_densities_in[1].data,
                                         this->seffective_potential_loc_solver.poisson_solver.electrostatic_potential.data,
                                         length);
            } else {
                assert(spin_type == 0 || spin_type == 1);
            }
        } else if (this->scf_control.mixing_variable == 1) {
            if (spin_type == 0) {
                Linalg::hadamard_product_general(temp.data,
                                             this->sdensity_solver.electron_densities[0].data,
                                             this->seffective_potential_loc_solver.poisson_solver.electrostatic_potential.data,
                                             length);
            } else if (spin_type == 1) {
                Linalg::hadamard_product_general(temp.data,
                                         this->sdensity_solver.electron_densities[0].data,
                                         this->seffective_potential_loc_solver.poisson_solver.electrostatic_potential.data,
                                         length);
                Linalg::accumulate_hadamard_product_general(temp.data,
                                         this->sdensity_solver.electron_densities[1].data,
                                         this->seffective_potential_loc_solver.poisson_solver.electrostatic_potential.data,
                                         length);
            } else {
                assert(spin_type == 0 || spin_type == 1);
            }
        } else {
            assert(this->scf_control.mixing_variable == 0);
        }
        this->electron_elecst_energy = (double) Linalg::vector_sum(temp.data, length, this->domain_vertices.comm) * 0.5 * this->mesh_control.delta_V;
    } else {
        Array_0D<double> temp(length);
        if (this->scf_control.mixing_variable == 0) {
            if (spin_type == 0) {
                Linalg::hadamard_product_general(temp.data,
                                         this->dmixing.x_km1.data,
                                         this->deffective_potential_loc_solver.poisson_solver.electrostatic_potential.data,
                                         length);
            } else if (spin_type == 1) {
                Linalg::hadamard_product_general(temp.data,
                                         this->ddensity_solver.electron_densities_in[0].data,
                                         this->deffective_potential_loc_solver.poisson_solver.electrostatic_potential.data,
                                         length);
                Linalg::accumulate_hadamard_product_general(temp.data,
                                         this->ddensity_solver.electron_densities_in[1].data,
                                         this->deffective_potential_loc_solver.poisson_solver.electrostatic_potential.data,
                                         length);
            } else {
                assert(spin_type == 0 || spin_type == 1);
            }
        } else if (this->scf_control.mixing_variable == 1) {
            if (spin_type == 0) {
                Linalg::hadamard_product_general(temp.data,
                                             this->ddensity_solver.electron_densities[0].data,
                                             this->deffective_potential_loc_solver.poisson_solver.electrostatic_potential.data,
                                             length);
            } else if (spin_type == 1) {
                Linalg::hadamard_product_general(temp.data,
                                         this->ddensity_solver.electron_densities[0].data,
                                         this->deffective_potential_loc_solver.poisson_solver.electrostatic_potential.data,
                                         length);
                Linalg::accumulate_hadamard_product_general(temp.data,
                                         this->ddensity_solver.electron_densities[1].data,
                                         this->deffective_potential_loc_solver.poisson_solver.electrostatic_potential.data,
                                         length);
            } else {
                assert(spin_type == 0 || spin_type == 1);
            }
        } else {
            assert(this->scf_control.mixing_variable == 0);
        }
        this->electron_elecst_energy = (double) Linalg::vector_sum(temp.data, length, this->domain_vertices.comm) * 0.5 * this->mesh_control.delta_V;
    }
    return this->electron_elecst_energy;
}

double& Scf::evaluate_electron_exchange_correlation_energy() {
    const uint length = this->domain_vertices.get_3D_local_vertices().get_size();
    const uint spin_type = this->spin.get_spin_type();
    if (!this->double_precision_flag) {
        Array_0D<float> temp(length);
        if (this->scf_control.mixing_variable == 0) {
            if (spin_type == 0) {
                Linalg::hadamard_product_general(temp.data,
                                            this->smixing.x_km1.data,
                                            this->seffective_potential_loc_solver.exchange_correlation_solver.exchange_correlation_potentials[0].data,
                                            length);
            } else if (spin_type == 1) {
                Linalg::hadamard_product_general(temp.data,
                                            this->sdensity_solver.electron_densities_in[0].data,
                                            this->seffective_potential_loc_solver.exchange_correlation_solver.exchange_correlation_potentials[0].data,
                                            length);
                Linalg::accumulate_hadamard_product_general(temp.data,
                                            this->sdensity_solver.electron_densities_in[1].data,
                                            this->seffective_potential_loc_solver.exchange_correlation_solver.exchange_correlation_potentials[1].data,
                                            length);
            } else {
                assert(spin_type == 0 || spin_type == 1);
            }
        } else if (this->scf_control.mixing_variable == 1) {
            if (spin_type == 0) {
                Linalg::hadamard_product_general(temp.data,
                                                this->sdensity_solver.electron_densities[0].data,
                                                this->seffective_potential_loc_solver.exchange_correlation_solver.exchange_correlation_potentials[0].data,
                                                length);
            } else if (spin_type == 1) {
                Linalg::hadamard_product_general(temp.data,
                                                this->sdensity_solver.electron_densities[0].data,
                                                this->seffective_potential_loc_solver.exchange_correlation_solver.exchange_correlation_potentials[0].data,
                                                length);
                Linalg::accumulate_hadamard_product_general(temp.data,
                                                this->sdensity_solver.electron_densities[1].data,
                                                this->seffective_potential_loc_solver.exchange_correlation_solver.exchange_correlation_potentials[1].data,
                                                length);
            } else {
                assert(spin_type == 0 || spin_type == 1);
            }
        } else {
            assert(this->scf_control.mixing_variable == 0);
        }
        this->electron_exchange_correlation_energy = (double) Linalg::vector_sum(temp.data, length, this->domain_vertices.comm) * this->mesh_control.delta_V;
    } else {
        Array_0D<double> temp(length);
        if (this->scf_control.mixing_variable == 0) {
            if (spin_type == 0) {
                Linalg::hadamard_product_general(temp.data,
                                            this->dmixing.x_km1.data,
                                            this->deffective_potential_loc_solver.exchange_correlation_solver.exchange_correlation_potentials[0].data,
                                            length);
            } else if (spin_type == 1) {
                Linalg::hadamard_product_general(temp.data,
                                            this->ddensity_solver.electron_densities_in[0].data,
                                            this->deffective_potential_loc_solver.exchange_correlation_solver.exchange_correlation_potentials[0].data,
                                            length);
                Linalg::accumulate_hadamard_product_general(temp.data,
                                            this->ddensity_solver.electron_densities_in[1].data,
                                            this->deffective_potential_loc_solver.exchange_correlation_solver.exchange_correlation_potentials[1].data,
                                            length);
            } else {
                assert(spin_type == 0 || spin_type == 1);
            }
        } else if (this->scf_control.mixing_variable == 1) {
            if (spin_type == 0) {
                Linalg::hadamard_product_general(temp.data,
                                                this->ddensity_solver.electron_densities[0].data,
                                                this->deffective_potential_loc_solver.exchange_correlation_solver.exchange_correlation_potentials[0].data,
                                                length);
            } else if (spin_type == 1) {
                Linalg::hadamard_product_general(temp.data,
                                                this->ddensity_solver.electron_densities[0].data,
                                                this->deffective_potential_loc_solver.exchange_correlation_solver.exchange_correlation_potentials[0].data,
                                                length);
                Linalg::accumulate_hadamard_product_general(temp.data,
                                                this->ddensity_solver.electron_densities[1].data,
                                                this->deffective_potential_loc_solver.exchange_correlation_solver.exchange_correlation_potentials[1].data,
                                                length);
            } else {
                assert(spin_type == 0 || spin_type == 1);
            }
        } else {
            assert(this->scf_control.mixing_variable == 0);
        }
        this->electron_exchange_correlation_energy = (double) Linalg::vector_sum(temp.data, length, this->domain_vertices.comm) * this->mesh_control.delta_V;
    }
    return this->electron_exchange_correlation_energy;
}

double Scf::evaluate_exchange_correlation_energy() {
    const uint spin_type = this->spin.get_spin_type();
    assert(this->scf_control.mixing_variable == 0 || this->scf_control.mixing_variable == 1);
    if (!this->double_precision_flag) {
        if (spin_type == 0) {
            if (this->scf_control.mixing_variable == 0) {
                std::vector<Array_3D<float>> temp(1, this->smixing.x_km1);
                return (double)this->seffective_potential_loc_solver.evaluate_exchange_correlation_energy(temp, this->sdensity_solver.electron_density_core);
            } else if (this->scf_control.mixing_variable == 1) {
                return (double)this->seffective_potential_loc_solver.evaluate_exchange_correlation_energy(this->sdensity_solver.electron_densities, this->sdensity_solver.electron_density_core);
            }
        } else if (spin_type == 1) {
            if (this->scf_control.mixing_variable == 0) {
                return (double)this->seffective_potential_loc_solver.evaluate_exchange_correlation_energy(this->sdensity_solver.electron_densities_in, this->sdensity_solver.electron_density_core);
            } else if (this->scf_control.mixing_variable == 1) {
                return (double)this->seffective_potential_loc_solver.evaluate_exchange_correlation_energy(this->sdensity_solver.electron_densities, this->sdensity_solver.electron_density_core);
            }
        } else {
            assert(spin_type == 0 || spin_type == 1);
        }
    } else {
        if (spin_type == 0) {
            if (this->scf_control.mixing_variable == 0) {
                std::vector<Array_3D<double>> temp(1, this->dmixing.x_km1);
                return (double)this->deffective_potential_loc_solver.evaluate_exchange_correlation_energy(temp, this->ddensity_solver.electron_density_core);
            } else if (this->scf_control.mixing_variable == 1) {
                return (double)this->deffective_potential_loc_solver.evaluate_exchange_correlation_energy(this->ddensity_solver.electron_densities, this->ddensity_solver.electron_density_core);
            }
        } else if (spin_type == 1) {
            if (this->scf_control.mixing_variable == 0) {
                return (double)this->deffective_potential_loc_solver.evaluate_exchange_correlation_energy(this->ddensity_solver.electron_densities_in, this->ddensity_solver.electron_density_core);
            } else if (this->scf_control.mixing_variable == 1) {
                return (double)this->deffective_potential_loc_solver.evaluate_exchange_correlation_energy(this->ddensity_solver.electron_densities, this->ddensity_solver.electron_density_core);
            }
        } else {
            assert(spin_type == 0 || spin_type == 1);
        }
    }
    return 0.0;
}

double& Scf::evaluate_free_energy() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    this->free_energy = this->deffective_potential_loc_solver.pseudo_charge_solver.self_and_correction_energy;
    this->free_energy += this->evaluate_ion_elecst_energy();
    this->free_energy -= this->evaluate_electron_elecst_energy();
    this->free_energy -= this->evaluate_electron_exchange_correlation_energy();
    this->free_energy += this->evaluate_exchange_correlation_energy();
    if (!this->double_precision_flag) {
        this->free_energy += (double) this->sdensity_solver.evaluate_band_energy();
        this->free_energy += (double) this->sdensity_solver.evaluate_entropy_energy();
        
    } else {
        this->free_energy += (double) this->ddensity_solver.evaluate_band_energy();
        this->free_energy += (double) this->ddensity_solver.evaluate_entropy_energy();
    }
    if (this->domain_vertices.get_comm_rank() == 0) {
        printf("Etot    = %18.12f\n", this->free_energy);
        if (this->double_precision_flag) {
            printf("Eband   = %18.12f\n", this->ddensity_solver.band_energy);
        } else {
            printf("Eband   = %18.12f\n", this->sdensity_solver.band_energy);
        }
        printf("E1      = %18.12f\n", this->ion_elecst_energy);
        printf("E2      = %18.12f\n", this->electron_elecst_energy);
        printf("E3      = %18.12f\n", this->electron_exchange_correlation_energy);
        if (this->double_precision_flag) {
            printf("Exc     = %18.12f\n", this->deffective_potential_loc_solver.exchange_correlation_solver.exchange_correlation_energy);
        } else {
            printf("Exc     = %18.12f\n", this->seffective_potential_loc_solver.exchange_correlation_solver.exchange_correlation_energy);
        }
        printf("Esc     = %18.12f\n", this->deffective_potential_loc_solver.pseudo_charge_solver.self_and_correction_energy);
        if (this->double_precision_flag) {
            printf("Entropy = %18.12f\n", this->ddensity_solver.entropy_energy);
        } else {
            printf("Entropy = %18.12f\n", this->sdensity_solver.entropy_energy);
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The evaluate_free_energy took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return this->free_energy;
}

void Scf::density_mixing(const uint iter) {
    const uint spin_type = this->spin.get_spin_type();
    assert(spin_type == 0 || spin_type == 1);
    if (!this->double_precision_flag) {
        if (spin_type == 0) {
            this->smixing.run(this->sdensity_solver.electron_densities[0], iter, this->domain_vertices.comm);
        } else {
            Vertices_3D vertices_k_boost = this->domain_vertices.get_3D_local_vertices();
            uint Nd = vertices_k_boost.get_size();
            // vertices_k_boost.set_nk(vertices_k_boost.get_nk() * 2);
            vertices_k_boost.nk *= 2;
            Array_3D<float> smixing_temp(vertices_k_boost);
            Linalg::set_value_general(this->sdensity_solver.electron_densities_in[0].data, this->sdensity_solver.electron_densities[0].data, Nd);
            Linalg::set_value_general(this->sdensity_solver.electron_densities_in[1].data, this->sdensity_solver.electron_densities[1].data, Nd);
            Linalg::hadamard_plus_general(smixing_temp.data, this->sdensity_solver.electron_densities[0].data, this->sdensity_solver.electron_densities[1].data, Nd);
            Linalg::hadamard_minus_general(smixing_temp.data + Nd, this->sdensity_solver.electron_densities[0].data, this->sdensity_solver.electron_densities[1].data, Nd);
            this->smixing.run(smixing_temp, iter, this->domain_vertices.comm);
            Linalg::hadamard_plus_general(this->sdensity_solver.electron_densities[0].data, smixing_temp.data, smixing_temp.data + Nd, Nd);
            Linalg::scalar_product_general(this->sdensity_solver.electron_densities[0].data, 0.5, Nd);
            Linalg::hadamard_minus_general(this->sdensity_solver.electron_densities[1].data, this->sdensity_solver.electron_densities[0].data, smixing_temp.data + Nd, Nd);
        }
    } else {
        if (spin_type == 0) {
            this->dmixing.run(this->ddensity_solver.electron_densities[0], iter, this->domain_vertices.comm);
        } else {
            Vertices_3D vertices_k_boost = this->domain_vertices.get_3D_local_vertices();
            uint Nd = vertices_k_boost.get_size();
            // vertices_k_boost.set_nk(vertices_k_boost.get_nk() * 2);
            vertices_k_boost.nk *= 2;
            Array_3D<double> dmixing_temp(vertices_k_boost);
            Linalg::hadamard_plus_general(dmixing_temp.data, this->ddensity_solver.electron_densities[0].data, this->ddensity_solver.electron_densities[1].data, Nd);
            Linalg::hadamard_minus_general(dmixing_temp.data + Nd, this->ddensity_solver.electron_densities[0].data, this->ddensity_solver.electron_densities[1].data, Nd);
            this->dmixing.run(dmixing_temp, iter, this->domain_vertices.comm);
            Linalg::hadamard_plus_general(this->ddensity_solver.electron_densities[0].data, dmixing_temp.data, dmixing_temp.data + Nd, Nd);
            Linalg::scalar_product_general(this->ddensity_solver.electron_densities[0].data, 0.5, Nd);
            Linalg::hadamard_minus_general(this->ddensity_solver.electron_densities[1].data, this->ddensity_solver.electron_densities[0].data, dmixing_temp.data + Nd, Nd);
            Linalg::set_value_general(this->ddensity_solver.electron_densities_in[0].data, this->ddensity_solver.electron_densities[0].data, Nd);
            Linalg::set_value_general(this->ddensity_solver.electron_densities_in[1].data, this->ddensity_solver.electron_densities[1].data, Nd);
        }
    }
    return;
}

void Scf::density_mixing_mp(const uint iter,
                        Memory_pool<double, Fast_memory>& pool_fast,
                        Memory_pool<double, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<double, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<double, Capacity_memory>> scope_cap(pool_cap);
    const uint spin_type = this->spin.get_spin_type();
    assert(spin_type == 0 || spin_type == 1);
    if (!this->double_precision_flag) {
        if (spin_type == 0) {
            this->smixing.run(this->sdensity_solver.electron_densities[0], iter, this->domain_vertices.comm);
        } else {
            Vertices_3D vertices_k_boost = this->domain_vertices.get_3D_local_vertices();
            uint Nd = vertices_k_boost.get_size();
            // vertices_k_boost.set_nk(vertices_k_boost.get_nk() * 2);
            vertices_k_boost.nk *= 2;
            Array_3D<float> smixing_temp(vertices_k_boost);
            Linalg::set_value_general(this->sdensity_solver.electron_densities_in[0].data, this->sdensity_solver.electron_densities[0].data, Nd);
            Linalg::set_value_general(this->sdensity_solver.electron_densities_in[1].data, this->sdensity_solver.electron_densities[1].data, Nd);
            Linalg::hadamard_plus_general(smixing_temp.data, this->sdensity_solver.electron_densities[0].data, this->sdensity_solver.electron_densities[1].data, Nd);
            Linalg::hadamard_minus_general(smixing_temp.data + Nd, this->sdensity_solver.electron_densities[0].data, this->sdensity_solver.electron_densities[1].data, Nd);
            this->smixing.run(smixing_temp, iter, this->domain_vertices.comm);
            Linalg::hadamard_plus_general(this->sdensity_solver.electron_densities[0].data, smixing_temp.data, smixing_temp.data + Nd, Nd);
            Linalg::scalar_product_general(this->sdensity_solver.electron_densities[0].data, 0.5, Nd);
            Linalg::hadamard_minus_general(this->sdensity_solver.electron_densities[1].data, this->sdensity_solver.electron_densities[0].data, smixing_temp.data + Nd, Nd);
        }
    } else {
        if (spin_type == 0) {
            this->dmixing.run_mp(this->ddensity_solver.electron_densities[0].data, iter, this->domain_vertices.comm,
                                     pool_fast, pool_cap);
        } else {
            Vertices_3D vertices_k_boost = this->domain_vertices.get_3D_local_vertices();
            uint Nd = vertices_k_boost.get_size();
            // vertices_k_boost.set_nk(vertices_k_boost.get_nk() * 2);
            vertices_k_boost.nk *= 2;
            Array_3D<double> dmixing_temp(vertices_k_boost);
            Linalg::hadamard_plus_general(dmixing_temp.data, this->ddensity_solver.electron_densities[0].data, this->ddensity_solver.electron_densities[1].data, Nd);
            Linalg::hadamard_minus_general(dmixing_temp.data + Nd, this->ddensity_solver.electron_densities[0].data, this->ddensity_solver.electron_densities[1].data, Nd);
            this->dmixing.run_mp(dmixing_temp.data, iter, this->domain_vertices.comm, pool_fast, pool_cap);
            Linalg::hadamard_plus_general(this->ddensity_solver.electron_densities[0].data, dmixing_temp.data, dmixing_temp.data + Nd, Nd);
            Linalg::scalar_product_general(this->ddensity_solver.electron_densities[0].data, 0.5, Nd);
            Linalg::hadamard_minus_general(this->ddensity_solver.electron_densities[1].data, this->ddensity_solver.electron_densities[0].data, dmixing_temp.data + Nd, Nd);
            Linalg::set_value_general(this->ddensity_solver.electron_densities_in[0].data, this->ddensity_solver.electron_densities[0].data, Nd);
            Linalg::set_value_general(this->ddensity_solver.electron_densities_in[1].data, this->ddensity_solver.electron_densities[1].data, Nd);
        }
    }
    return;
}

void Scf::potential_mixing(const uint& iter) {
    const uint spin_type = this->spin.get_spin_type();
    assert(spin_type == 0 || spin_type == 1);
    if (!this->double_precision_flag) {
        if (spin_type == 0) {
            this->smixing.run(this->seffective_potential_loc_solver.effective_potentail_locs[0], iter, this->domain_vertices.comm);
        } else {
            Vertices_3D vertices_k_boost = this->domain_vertices.get_3D_local_vertices();
            uint Nd = vertices_k_boost.get_size();
            // vertices_k_boost.set_nk(vertices_k_boost.get_nk() * 2);
            vertices_k_boost.nk *= 2;
            Array_3D<float> smixing_temp(vertices_k_boost);
            Linalg::set_value_general(smixing_temp.data, this->seffective_potential_loc_solver.effective_potentail_locs[0].data, Nd);
            Linalg::set_value_general(smixing_temp.data + Nd, this->seffective_potential_loc_solver.effective_potentail_locs[1].data, Nd);
        }
    } else {
        if (spin_type == 0) {
            this->dmixing.run(this->deffective_potential_loc_solver.effective_potentail_locs[0], iter, this->domain_vertices.comm);
        } else {
            Vertices_3D vertices_k_boost = this->domain_vertices.get_3D_local_vertices();
            uint Nd = vertices_k_boost.get_size();
            // vertices_k_boost.set_nk(vertices_k_boost.get_nk() * 2);
            vertices_k_boost.nk *= 2;
            Array_3D<double> dmixing_temp(vertices_k_boost);
            Linalg::set_value_general(dmixing_temp.data, this->deffective_potential_loc_solver.effective_potentail_locs[0].data, Nd);
            Linalg::set_value_general(dmixing_temp.data + Nd, this->deffective_potential_loc_solver.effective_potentail_locs[1].data, Nd);
        }
    }
    return;
}

void Scf::precision_conversion() {
    assert(0 && "precision_conversion() is disabled temporarily");
    // std::cout << "precision_conversion" << std::endl;
    // this->ddensity_solver.electron_density.deepcopy(
    //     this->sdensity_solver.electron_density.as_type(
    //         this->ddensity_solver.electron_density.data));
    // this->ddensity_solver.chemical_potential = (double)this->sdensity_solver.chemical_potential;
    // this->ddensity_solver.generate_electron_charge();
    // this->deffective_potential_loc_solver.poisson_solver.electrostatic_potential.deepcopy(
    //     this->seffective_potential_loc_solver.poisson_solver.electrostatic_potential.as_type(
    //         this->deffective_potential_loc_solver.poisson_solver.electrostatic_potential.data));
    // this->scale_electron_density();
    // this->dmixing.init(this->smixing);
    // if (this->scf_control.mixing_variable == 0) this->dmixing.set_x_km1(this->ddensity_solver.electron_density);
    // if (this->density_solver_control.method == 0) {
    //     if (this->density_solver_control.eigen_solver_control.method == 0) {
    //         this->ddensity_solver.eigen_solver.chefsi.lanczos.eig_min = this->sdensity_solver.eigen_solver.chefsi.lanczos.eig_min;
    //         this->ddensity_solver.eigen_solver.chefsi.lanczos.eig_max = this->sdensity_solver.eigen_solver.chefsi.lanczos.eig_max;
    //         this->ddensity_solver.eigen_solver.chefsi.lanczos.lambda_cutoff = this->sdensity_solver.eigen_solver.chefsi.lanczos.lambda_cutoff;
    //         this->ddensity_solver.eigen_solver.eigen_vectors.deepcopy(
    //             this->sdensity_solver.eigen_solver.eigen_vectors.as_type(
    //                 this->ddensity_solver.eigen_solver.eigen_vectors.data));
    //         this->ddensity_solver.eigen_solver.eigen_values.deepcopy(
    //             this->sdensity_solver.eigen_solver.eigen_values.as_type(
    //                 this->ddensity_solver.eigen_solver.eigen_values.data));
    //     }
    // }
    // // destruct singgle precision
    // this->sstencil.destructor();
    // this->seffective_potential_loc_solver.destructor();
    // this->sdensity_solver.destructor();
    // this->smixing.destructor();
    return;
}

void Scf::cal_forces() {
    if (!this->double_precision_flag) {
        this->seffective_potential_loc_solver.cal_effective_potential_loc(
            this->sdensity_solver.electron_densities, this->sdensity_solver.electron_density_core);
        Force_solver<float> force_solver(this->domain_vertices, this->exarr_mpi_package,
                                         this->mesh_control, this->geometry,
                                         this->sstencil, this->sdensity_solver,
                                         this->psp8_files);
        force_solver.init();
        force_solver.cal_local_forces(this->seffective_potential_loc_solver.poisson_solver.electrostatic_potential,
                                      this->seffective_potential_loc_solver.pseudo_charge_solver.pseudo_charge_density,
                                      this->seffective_potential_loc_solver.pseudo_charge_solver.pseudo_charge_density_ref,
                                      this->seffective_potential_loc_solver.pseudo_charge_solver.pseudo_charge_density_potiential_correction);
        force_solver.cal_xc_forces(this->seffective_potential_loc_solver.exchange_correlation_solver.exchange_correlation_potentials);
        force_solver.cal_nonlocal_forces();
        force_solver.balance_forces();
        if (this->domain_vertices.get_comm_rank() == 0) force_solver.show();
    } else {
        this->deffective_potential_loc_solver.cal_effective_potential_loc(
            this->ddensity_solver.electron_densities, this->ddensity_solver.electron_density_core);
        Force_solver<double> force_solver(this->domain_vertices, this->exarr_mpi_package,
                                          this->mesh_control, this->geometry,
                                          this->dstencil, this->ddensity_solver,
                                          this->psp8_files);
        force_solver.init();
        force_solver.cal_local_forces(this->deffective_potential_loc_solver.poisson_solver.electrostatic_potential,
                                      this->deffective_potential_loc_solver.pseudo_charge_solver.pseudo_charge_density,
                                      this->deffective_potential_loc_solver.pseudo_charge_solver.pseudo_charge_density_ref,
                                      this->deffective_potential_loc_solver.pseudo_charge_solver.pseudo_charge_density_potiential_correction);
        force_solver.cal_xc_forces(this->deffective_potential_loc_solver.exchange_correlation_solver.exchange_correlation_potentials);
        force_solver.cal_nonlocal_forces();
        force_solver.balance_forces();
        if (this->domain_vertices.get_comm_rank() == 0) force_solver.show();
    }
    return;
}

void Scf::init() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // basic
    // domain_vertices
    Vertices_3D shared_vertices(this->mesh_control.nx, this->mesh_control.ny, this->mesh_control.nz);
    this->global_domain_vertices.init(shared_vertices, this->comm, 1);
    // if (poisson_solver_control.comm_ni != 0) this->domain_vertices.set_comm_ni(poisson_solver_control.comm_ni, mesh_control.nx);
    // // if (poisson_solver_control.comm_nj != 0) this->domain_vertices.set_comm_nj(poisson_solver_control.comm_nj, mesh_control.ny);
    // // if (poisson_solver_control.comm_nk != 0) this->domain_vertices.set_comm_nk(poisson_solver_control.comm_nk, mesh_control.nz);
    // // this->domain_vertices.init(shared_vertices, comm, poisson_solver_control.comm_np_max);
    if (this->density_solver_control.method == 1) {
        // this->domain_vertices.set_shared_vertices(shared_vertices);
        this->domain_vertices.set_comm_ni(this->density_solver_control.density_matrix_solver_control.xlsdft_control.element_comm_numbers[0], shared_vertices.ni);
        this->domain_vertices.set_comm_nj(this->density_solver_control.density_matrix_solver_control.xlsdft_control.element_comm_numbers[1], shared_vertices.nj);
        this->domain_vertices.set_comm_nk(this->density_solver_control.density_matrix_solver_control.xlsdft_control.element_comm_numbers[2], shared_vertices.nk);

        // this->domain_vertices.set_comm_ni(16*18, shared_vertices.ni);
        // this->domain_vertices.set_comm_nj(1, shared_vertices.nj);
        // this->domain_vertices.set_comm_nk(1, shared_vertices.nk);
    }
    this->domain_vertices.init(shared_vertices, this->comm);
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "scf.domain_vertices.show()" << std::endl;
        this->domain_vertices.show();
    }

    // mpi package init
    int FDn[3] = {this->stencil_control.order/2, this->stencil_control.order/2, this->stencil_control.order/2};
    this->exarr_mpi_package.init(mesh_control.is_periodic, FDn);
    this->global_domain_mpi_package.init();

    // stencil, seffective_potential_loc_solver
    this->double_precision_flag = this->scf_control.max_single_precision_iter > 0 ? false : true;
    if (!this->double_precision_flag) {
        // this->sstencil.init(this->geometry.cell_type, this->stencil_control.order,
        //                     this->mesh_control.delta_x, this->mesh_control.delta_y,
        //                     this->mesh_control.delta_z);
        // this->seffective_potential_loc_solver.init();
        // this->sdensity_solver.init();
        // this->scale_electron_density();

        this->dstencil.init(this->geometry.cell_type, this->stencil_control.order,
                            this->mesh_control.delta_x, this->mesh_control.delta_y,
                            this->mesh_control.delta_z);
        this->sstencil.init(this->dstencil);
        this->deffective_potential_loc_solver.init();
        this->seffective_potential_loc_solver.init(this->deffective_potential_loc_solver);
        this->ddensity_solver.init();
        this->scale_electron_density();
        this->sdensity_solver.init(this->ddensity_solver);
        const uint spin_type = this->spin.get_spin_type();
        if (spin_type == 0) {
            if (this->scf_control.mixing_variable == 0) {
                this->smixing.init(this->sdensity_solver.electron_densities[0], "SCF");
            } else if (this->scf_control.mixing_variable == 1) {
                this->smixing.init(this->seffective_potential_loc_solver.effective_potentail_locs[0], "SCF");
            } else {
                assert(this->scf_control.mixing_variable == 0 || this->scf_control.mixing_variable == 1);
            }
        } else if (spin_type == 1) {
            Vertices_3D vertices_k_boost = this->domain_vertices.get_3D_local_vertices();
            uint Nd = vertices_k_boost.get_size();
            // vertices_k_boost.set_nk(vertices_k_boost.get_nk() * 2);
            vertices_k_boost.nk *= 2;
            Array_3D<float> smixing_init(vertices_k_boost);
            if (this->scf_control.mixing_variable == 0) {
                Linalg::set_value_general(smixing_init.data, this->sdensity_solver.electron_density_init.data, Nd);
                Linalg::set_value_general(smixing_init.data + Nd, this->sdensity_solver.magnetization_z.data, Nd);
                Linalg::set_value_general(this->sdensity_solver.electron_densities_in[0].data, this->sdensity_solver.electron_densities[0].data, Nd);
                Linalg::set_value_general(this->sdensity_solver.electron_densities_in[1].data, this->sdensity_solver.electron_densities[1].data, Nd);
            } else if (this->scf_control.mixing_variable == 1) {
                Linalg::set_value_general(smixing_init.data, this->seffective_potential_loc_solver.effective_potentail_locs[0].data, Nd);
                Linalg::set_value_general(smixing_init.data + Nd, this->seffective_potential_loc_solver.effective_potentail_locs[1].data, Nd);
            } else {
                assert(this->scf_control.mixing_variable == 0 || this->scf_control.mixing_variable == 1);
            }
            this->smixing.init(smixing_init, "SCF");
            // assert(spin_type == 0);
        } else {
            assert(spin_type == 0 || spin_type == 1);
        }
    } else {
        this->dstencil.init(this->geometry.cell_type, this->stencil_control.order,
                            this->mesh_control.delta_x, this->mesh_control.delta_y,
                            this->mesh_control.delta_z);
        this->deffective_potential_loc_solver.init();
        this->ddensity_solver.init();
        this->scale_electron_density();
        if (std::getenv("CAL_LIGEPS") != nullptr) {
            constexpr bool if_electron_density_pretreatment = true;
            if (if_electron_density_pretreatment) {
                this->electron_density_pretreatment();
            }
        }
        const uint spin_type = this->spin.get_spin_type();
        if (spin_type == 0) {
            if (this->scf_control.mixing_variable == 0) {
                this->dmixing.init(this->ddensity_solver.electron_densities[0], "SCF");
            } else if (this->scf_control.mixing_variable == 1) {
                this->dmixing.init(this->deffective_potential_loc_solver.effective_potentail_locs[0], "SCF");
            } else {
                assert(this->scf_control.mixing_variable == 0 || this->scf_control.mixing_variable == 1);
            }
        } else if (spin_type == 1) {
            Vertices_3D vertices_k_boost = this->domain_vertices.get_3D_local_vertices();
            uint Nd = vertices_k_boost.get_size();
            // vertices_k_boost.set_nk(vertices_k_boost.get_nk() * 2);
            vertices_k_boost.nk *= 2;
            Array_3D<double> dmixing_init(vertices_k_boost);
            if (this->scf_control.mixing_variable == 0) {
                Linalg::set_value_general(dmixing_init.data, this->ddensity_solver.electron_density_init.data, Nd);
                Linalg::set_value_general(dmixing_init.data + Nd, this->ddensity_solver.magnetization_z.data, Nd);
                Linalg::set_value_general(this->ddensity_solver.electron_densities_in[0].data, this->ddensity_solver.electron_densities[0].data, Nd);
                Linalg::set_value_general(this->ddensity_solver.electron_densities_in[1].data, this->ddensity_solver.electron_densities[1].data, Nd);
            } else if (this->scf_control.mixing_variable == 1) {
                Linalg::set_value_general(dmixing_init.data, this->deffective_potential_loc_solver.effective_potentail_locs[0].data, Nd);
                Linalg::set_value_general(dmixing_init.data + Nd, this->deffective_potential_loc_solver.effective_potentail_locs[1].data, Nd);
            } else {
                assert(this->scf_control.mixing_variable == 0 || this->scf_control.mixing_variable == 1);
            }
            this->dmixing.init(dmixing_init, "SCF");
            // assert(spin_type == 0);
        } else {
            assert(spin_type == 0 || spin_type == 1);
        }
        if (this->domain_vertices.get_comm_rank() == 0) {
            std::cout << GREEN << "The scf calculate " << this->ddensity_solver.evalutate_flops()
                      << " flops in 1 SCF step (when rho_trigger = 1)." << RESET << std::endl;
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The scf init took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

void Scf::run() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    uint iter = 0;

    if (this->misc_control.if_read_density) {
        this->ddensity_solver.density_load();
        Linalg::set_value_general(this->dmixing.x_km1.data,
            this->ddensity_solver.electron_densities[0].data, this->dmixing.x_km1.length);
    }
    
    while (iter < this->scf_control.max_iter) {
        if (this->domain_vertices.get_comm_rank() == 0) std::cout << "\nSCF loop " << iter + 1 << " start."<< std::endl;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        if (!this->double_precision_flag) {
            this->seffective_potential_loc_solver.cal_effective_potential_loc(
                this->sdensity_solver.electron_densities, this->sdensity_solver.electron_density_core);
            if (this->scf_control.mixing_variable == 1) {
                this->evaluate_free_energy();
                if (this->evaluate_error(iter) < this->scf_control.tolerance) break;
                this->potential_mixing(iter);
            }
            this->sdensity_solver.run(this->seffective_potential_loc_solver.effective_potentail_locs);
            if (this->scf_control.mixing_variable == 0) {
                this->evaluate_free_energy();
                if (this->evaluate_error(iter) < this->scf_control.tolerance) {
                    iter++;
                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    if (this->domain_vertices.get_comm_rank() == 0)
                        std::cout << "SCF loop " << iter << " took "
                                                 << Tools::time_cost(begin, end) << "."<< std::endl;
                    break;
                };
                this->density_mixing(iter);
            }
            if ((this->error < this->scf_control.single_precision_tolerance) ||
                (iter + 1 == this->scf_control.max_single_precision_iter)) {
                double_precision_flag = true;
                this->precision_conversion();
            }
        } else {
            this->deffective_potential_loc_solver.cal_effective_potential_loc(
                this->ddensity_solver.electron_densities, this->ddensity_solver.electron_density_core);
            if (this->scf_control.mixing_variable == 1) {
                this->evaluate_free_energy();
                if (this->evaluate_error(iter) < this->scf_control.tolerance) break;
                this->potential_mixing(iter);
            }
            this->ddensity_solver.run(this->deffective_potential_loc_solver.effective_potentail_locs);
            if (this->scf_control.mixing_variable == 0) {
                this->evaluate_free_energy();
                if (this->evaluate_error(iter) < this->scf_control.tolerance) {
                    iter++;
                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    if (this->domain_vertices.get_comm_rank() == 0)
                        std::cout << "SCF loop " << iter << " took "
                                                 << Tools::time_cost(begin, end) << "."<< std::endl;
                    break;
                }
                this->density_mixing(iter);
            }
        }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (this->domain_vertices.get_comm_rank() == 0) std::cout << "SCF loop " << iter + 1 << " took "
                                                                  << Tools::time_cost(begin, end) << "."<< std::endl;
        iter++;
        if (this->misc_control.if_print_density && iter%10 == 0) {
            this->ddensity_solver.density_dump();
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) std::cout << "\nSCF total "
                                                              << iter
                                                              << " loops took "
                                                              << Tools::time_cost(begin, end) << "."
                                                              << std::endl << std::endl;
    if (this->misc_control.if_print_density) {
        this->ddensity_solver.density_dump();
    }
    if (this->misc_control.if_print_eigen) {
        if (this->double_precision_flag) {
            this->ddensity_solver.print_eigens();
            // this->ddensity_solver.eigen_solver.print_psi_sparc(this->spin);
        } else {
            this->sdensity_solver.print_eigens();
        }
    }
    if (this->misc_control.if_print_pdos) {
        if (this->double_precision_flag) {
            this->ddensity_solver.print_pdos();
        } else {
            this->sdensity_solver.print_pdos();
        }
    }
    if (this->misc_control.if_print_forces) this->cal_forces();
    return;
}

void Scf::run_mp() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    uint iter = 0;

    if (this->misc_control.if_read_density) {
        this->ddensity_solver.density_load();
        Linalg::set_value_general(this->dmixing.x_km1.data,
            this->ddensity_solver.electron_densities[0].data, this->dmixing.x_km1.length);
    }
    
    constexpr size_t GB = 1024 * 1024 * 1024 / sizeof(double);
    Memory_pool<double, Fast_memory> pool_fast(3.5 * GB);
    Memory_pool<double, Capacity_memory> pool_cap(3.5 * GB);
    while (iter < this->scf_control.max_iter) {
        if (this->domain_vertices.get_comm_rank() == 0) std::cout << "\nSCF loop " << iter + 1 << " start."<< std::endl;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        if (!this->double_precision_flag) {
            this->seffective_potential_loc_solver.cal_effective_potential_loc(
                this->sdensity_solver.electron_densities, this->sdensity_solver.electron_density_core);
            if (this->scf_control.mixing_variable == 1) {
                this->evaluate_free_energy();
                if (this->evaluate_error(iter) < this->scf_control.tolerance) break;
                this->potential_mixing(iter);
            }
            this->sdensity_solver.run(this->seffective_potential_loc_solver.effective_potentail_locs);
            if (this->scf_control.mixing_variable == 0) {
                this->evaluate_free_energy();
                if (this->evaluate_error(iter) < this->scf_control.tolerance) {
                    iter++;
                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    if (this->domain_vertices.get_comm_rank() == 0)
                        std::cout << "SCF loop " << iter << " took "
                                                 << Tools::time_cost(begin, end) << "."<< std::endl;
                    break;
                };
                this->density_mixing(iter);
            }
            if ((this->error < this->scf_control.single_precision_tolerance) ||
                (iter + 1 == this->scf_control.max_single_precision_iter)) {
                double_precision_flag = true;
                this->precision_conversion();
            }
        } else {
            Memory_pool_scope<Memory_pool<double, Fast_memory>> scope_fast(pool_fast);
            Memory_pool_scope<Memory_pool<double, Capacity_memory>> scope_cap(pool_cap);
            this->deffective_potential_loc_solver.cal_effective_potential_loc(
                this->ddensity_solver.electron_densities, this->ddensity_solver.electron_density_core);
            if (this->scf_control.mixing_variable == 1) {
                this->evaluate_free_energy();
                if (this->evaluate_error(iter) < this->scf_control.tolerance) break;
                this->potential_mixing(iter);
            }
            double const* effective_potentail_locs_data[2];
            for (uint ispin = 0; ispin < this->spin.generate_nspin(); ispin++) {
                effective_potentail_locs_data[ispin] = this->deffective_potential_loc_solver.effective_potentail_locs[ispin].data;
            }
            this->ddensity_solver.run_mp(effective_potentail_locs_data, pool_fast, pool_cap);
            if (this->scf_control.mixing_variable == 0) {
                this->evaluate_free_energy();
                if (this->evaluate_error(iter) < this->scf_control.tolerance) {
                    iter++;
                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    if (this->domain_vertices.get_comm_rank() == 0)
                        std::cout << "SCF loop " << iter << " took "
                                                 << Tools::time_cost(begin, end) << "."<< std::endl;
                    break;
                }
                this->density_mixing_mp(iter, pool_fast, pool_cap);
            }
        }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (this->domain_vertices.get_comm_rank() == 0) std::cout << "SCF loop " << iter + 1 << " took "
                                                                  << Tools::time_cost(begin, end) << "."<< std::endl;
        iter++;
        if (this->misc_control.if_print_density && iter%10 == 0) {
            this->ddensity_solver.density_dump();
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) std::cout << "\nSCF total "
                                                              << iter
                                                              << " loops took "
                                                              << Tools::time_cost(begin, end) << "."
                                                              << std::endl << std::endl;
    if (this->misc_control.if_print_density) {
        this->ddensity_solver.density_dump();
    }
    if (this->misc_control.if_print_eigen) {
        if (this->double_precision_flag) {
            this->ddensity_solver.print_eigens();
            // this->ddensity_solver.eigen_solver.print_psi_sparc(this->spin);
        } else {
            this->sdensity_solver.print_eigens();
        }
    }
    if (this->misc_control.if_print_pdos) {
        if (this->double_precision_flag) {
            this->ddensity_solver.print_pdos();
        } else {
            this->sdensity_solver.print_pdos();
        }
    }
    if (this->misc_control.if_print_forces) this->cal_forces();
    return;
}

void Scf::run_mp(Memory_pool<double, Fast_memory>& pool_fast, Memory_pool<double, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<double, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<double, Capacity_memory>> scope_cap(pool_cap);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    uint iter = 0;

    if (this->misc_control.if_read_density) {
        this->ddensity_solver.density_load();
        Linalg::set_value_general(this->dmixing.x_km1.data,
            this->ddensity_solver.electron_densities[0].data, this->dmixing.x_km1.length);
    }
 
    while (iter < this->scf_control.max_iter) {
        if (this->domain_vertices.get_comm_rank() == 0) std::cout << "\nSCF loop " << iter + 1 << " start."<< std::endl;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        if (!this->double_precision_flag) {
            this->seffective_potential_loc_solver.cal_effective_potential_loc(
                this->sdensity_solver.electron_densities, this->sdensity_solver.electron_density_core);
            if (this->scf_control.mixing_variable == 1) {
                this->evaluate_free_energy();
                if (this->evaluate_error(iter) < this->scf_control.tolerance) break;
                this->potential_mixing(iter);
            }
            this->sdensity_solver.run(this->seffective_potential_loc_solver.effective_potentail_locs);
            if (this->scf_control.mixing_variable == 0) {
                this->evaluate_free_energy();
                if (this->evaluate_error(iter) < this->scf_control.tolerance) {
                    iter++;
                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    if (this->domain_vertices.get_comm_rank() == 0)
                        std::cout << "SCF loop " << iter << " took "
                                                 << Tools::time_cost(begin, end) << "."<< std::endl;
                    break;
                };
                this->density_mixing_mp(iter, pool_fast, pool_cap);
            }
            if ((this->error < this->scf_control.single_precision_tolerance) ||
                (iter + 1 == this->scf_control.max_single_precision_iter)) {
                double_precision_flag = true;
                this->precision_conversion();
            }
        } else {
            Memory_pool_scope<Memory_pool<double, Fast_memory>> scope_fast2(pool_fast);
            Memory_pool_scope<Memory_pool<double, Capacity_memory>> scope_cap2(pool_cap);
            this->deffective_potential_loc_solver.cal_effective_potential_loc_mp(
                this->ddensity_solver.electron_densities,
                this->ddensity_solver.electron_density_core,
                pool_fast, pool_cap
            );
            if (this->scf_control.mixing_variable == 1) {
                this->evaluate_free_energy();
                if (this->evaluate_error(iter) < this->scf_control.tolerance) break;
                this->potential_mixing(iter);
            }
            double const* effective_potentail_locs_data[2];
            for (uint ispin = 0; ispin < this->spin.generate_nspin(); ispin++) {
                effective_potentail_locs_data[ispin] = this->deffective_potential_loc_solver.effective_potentail_locs[ispin].data;
            }
            this->ddensity_solver.run_mp(effective_potentail_locs_data, pool_fast, pool_cap);
            if (this->scf_control.mixing_variable == 0) {
                // this->evaluate_free_energy();
                if (this->evaluate_error(iter) < this->scf_control.tolerance) {
                    iter++;
                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    if (this->domain_vertices.get_comm_rank() == 0)
                        std::cout << "SCF loop " << iter << " took "
                                                 << Tools::time_cost(begin, end) << "."<< std::endl;
                    break;
                }
                this->density_mixing_mp(iter, pool_fast, pool_cap);
            }
        }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (this->domain_vertices.get_comm_rank() == 0) std::cout << "SCF loop " << iter + 1 << " took "
                                                                  << Tools::time_cost(begin, end) << "."<< std::endl;
        iter++;

        #if defined(ENABLE_TIMER)
        this->ddensity_solver.print_timer_statistics(this->domain_vertices.get_comm_rank() == 0, std::cout);
        Xlsdft_performance xlsdft_performance;
        xlsdft_performance.update(this->ddensity_solver.density_matrix_solver.xlsdft.eigen_solvers,
                                    this->ddensity_solver.density_matrix_solver.xlsdft.xlsdft_timer);
        xlsdft_performance.reduce(this->domain_vertices.comm, 0);
        xlsdft_performance.show(this->domain_vertices.get_comm_rank() == 0, std::cout);
        #endif

        if (this->misc_control.if_print_density && iter%10 == 0) {
            this->ddensity_solver.density_dump();
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) std::cout << "\nSCF total "
                                                              << iter
                                                              << " loops took "
                                                              << Tools::time_cost(begin, end) << "."
                                                              << std::endl << std::endl;
    this->evaluate_free_energy();
    
    if (this->misc_control.if_print_density) {
        this->ddensity_solver.density_dump();
    }
    if (this->misc_control.if_print_eigen) {
        if (this->double_precision_flag) {
            this->ddensity_solver.print_eigens();
            // this->ddensity_solver.eigen_solver.print_psi_sparc(this->spin);
        } else {
            this->sdensity_solver.print_eigens();
        }
    }
    if (this->misc_control.if_print_pdos) {
        if (this->double_precision_flag) {
            this->ddensity_solver.print_pdos();
        } else {
            this->sdensity_solver.print_pdos();
        }
    }
    if (this->misc_control.if_print_forces) this->cal_forces();
    return;
}

void Scf::destructor() {
    this->global_domain_vertices.destructor();
    this->domain_vertices.destructor();
    this->exarr_mpi_package.destructor();
    this->global_domain_mpi_package.destructor();
    if (!this->double_precision_flag) {
        this->dstencil.destructor();
        this->sstencil.destructor();
        this->deffective_potential_loc_solver.destructor();
        this->seffective_potential_loc_solver.destructor();
        this->ddensity_solver.destructor();
        this->sdensity_solver.destructor();
        this->smixing.destructor();
    } else {
        this->dstencil.destructor();
        this->deffective_potential_loc_solver.destructor();
        this->ddensity_solver.destructor();
        this->dmixing.destructor();
    }
    return;
}

void Scf::show() const {
    int block_size = 20;
    int precision = 13;
    int width = 18;
    std::cout << std::right << std::setw(block_size) << "ion_elecst_energy" << " = "
              << std::left << std::setprecision(precision) << std::setw(width) << std::fixed << this->ion_elecst_energy << std::endl;
    std::cout << std::right << std::setw(block_size) << "electron_elecst_energy" << " = "
              << std::left << std::setprecision(precision) << std::setw(width) << std::fixed << this->electron_elecst_energy << std::endl;
    std::cout << std::right << std::setw(block_size) << "electron_exchange_correlation_energy" << " = "
              << std::left << std::setprecision(precision) << std::setw(width) << std::fixed << this->electron_exchange_correlation_energy << std::endl;
    std::cout << std::right << std::setw(block_size) << "free_energy" << " = "
              << std::left << std::setprecision(precision) << std::setw(width) << std::fixed << this->free_energy << std::endl;
    return;
}
