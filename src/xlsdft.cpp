#include "xlsdft.h"
#include "chefsi_layout.h"
#include "nloc_fixture_dump.hpp"
#include "xlsdft_nchi_diag.hpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <type_traits>
#include <set>
#include <tuple>

#ifdef ENABLE_XLSDFT_TIMER
#pragma message("Building with ENABLE_XLSDFT_TIMER.")
Xlsdft_timer::Xlsdft_timer() {}
Xlsdft_timer::~Xlsdft_timer() {}
void Xlsdft_timer::reset() {
    this->xlsdft.reset();
    this->xlsdft_copy_veff.reset();
    #if (defined(LOW_MEMORY))
    this->xlsdft_low_memory_psi.reset();
    #endif
    this->xlsdft_eigen_solver.reset();
    this->xlsdft_barrier1.reset();
    this->xlsdft_barrier2.reset();
    this->xlsdft_barrier3.reset();
    return;
}
void Xlsdft_timer::show(std::ostream& output) const {
    output << std::left << std::setw(20) << "xlsdft"                     << ": " << this->xlsdft.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "  xlsdft_copy_veff"         << ": " << this->xlsdft_copy_veff.time_cost_millisecond() << " [ms]" << std::endl;
    #if (defined(LOW_MEMORY))
    output << std::left << std::setw(20) << "  xlsdft_low_memory_psi"    << ": " << this->xlsdft_low_memory_psi.time_cost_millisecond() << " [ms]" << std::endl;
    #endif
    output << std::left << std::setw(20) << "  xlsdft_eigen_solver"      << ": " << this->xlsdft_eigen_solver.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "  xlsdft_barrier1"      << ": " << this->xlsdft_barrier1.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "  xlsdft_barrier2"      << ": " << this->xlsdft_barrier2.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "  xlsdft_barrier3"      << ": " << this->xlsdft_barrier3.time_cost_millisecond() << " [ms]" << std::endl;
    return;
}
#endif //ENABLE_XLSDFT_TIMER

template<typename T>
Xlsdft<T>::Xlsdft(const Xlsdft_control& xlsdft_control,
                  const Mesh_control& mesh_control,
                  const Geometry& geometry,
                  const Stencil<T>& stencil,
                  const Domain_parallel_vertices_4D& domain_vertices)
                : xlsdft_control(xlsdft_control),
                  mesh_control(mesh_control),
                  geometry(geometry),
                  stencil(stencil),
                  domain_vertices(domain_vertices) {}

template<typename T>
Xlsdft<T>::~Xlsdft(){}

template<typename T>
void Xlsdft<T>::cal_int_density_per_band(std::vector<T>& eigen_value_per_band,
                                         std::vector<T>& int_density_per_band) const {
    uint nstates_total = 0;
    for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
        nstates_total += eigen_solvers[ielement].eigen_solver_control.nstates;
    }
    eigen_value_per_band.resize(0);
    int_density_per_band.resize(0);
    eigen_value_per_band.reserve(nstates_total);
    int_density_per_band.reserve(nstates_total);

    for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
        if (this->domain_verticeses[ielement].shared_vertices.get_size() == 0) continue;
        const Vertices_3D& element_vertices = this->element_verticeses[ielement];
        Vertices_3D region = element_vertices.get_overlap_vertices(
                                this->domain_verticeses[ielement].get_3D_local_vertices());
        std::vector<T> eigen_value_per_band_temp;
        std::vector<T> int_density_per_band_temp;
        this->eigen_solvers[ielement].cal_int_density_per_band(region, eigen_value_per_band_temp,
                                                               int_density_per_band_temp);
        eigen_value_per_band.insert(eigen_value_per_band.end(), eigen_value_per_band_temp.begin(),
                                    eigen_value_per_band_temp.end());
        int_density_per_band.insert(int_density_per_band.end(), int_density_per_band_temp.begin(),
                                    int_density_per_band_temp.end());
    }
    return;
}

template<typename T>
void Xlsdft<T>::cal_int_density_per_band2(std::vector<T>& eigen_value_per_band,
                                         std::vector<T>& int_density_per_band) const {
    uint nstates_total = 0;
    for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
        nstates_total += eigen_solvers[ielement].eigen_solver_control.nstates;
    }
    eigen_value_per_band.resize(0);
    int_density_per_band.resize(0);
    eigen_value_per_band.reserve(nstates_total);
    int_density_per_band.reserve(nstates_total);

    for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
        if (this->domain_verticeses[ielement].shared_vertices.get_size() == 0) continue;
        const Vertices_3D& element_vertices = this->element_verticeses[ielement];
        Vertices_3D region = element_vertices.get_overlap_vertices(
                                this->domain_verticeses[ielement].get_3D_local_vertices());
        std::vector<T> eigen_value_per_band_temp;
        std::vector<T> int_density_per_band_temp;
        this->eigen_solvers[ielement].cal_int_density_per_band(region, eigen_value_per_band_temp,
                                                               int_density_per_band_temp);
        if (this->eigen_solvers[ielement].domain_vertices.is_active) {
            const Domain_parallel_vertices_4D& domain_vertices = this->eigen_solvers[ielement].domain_vertices;
            const uint local_nb = domain_vertices.local_vertices.get_nb();
            const uint nstates = domain_vertices.shared_vertices.get_nb();
            const MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
            if (domain_vertices.get_active_comm_i() == 0
             && domain_vertices.get_active_comm_j() == 0
             && domain_vertices.get_active_comm_k() == 0) {
                std::vector<T> int_density_per_band_temp2(local_nb, 0);
                MPI_Reduce(int_density_per_band_temp.data(), int_density_per_band_temp2.data(), local_nb,
                           mpi_datatype, MPI_SUM, 0,
                           domain_vertices.domain_3d_comm);
                int_density_per_band_temp.resize(nstates);
                int* recvcounts = new int [domain_vertices.get_active_comm_nb()]();
                int* displs = new int [domain_vertices.get_active_comm_nb() + 1]();
                displs[0] = 0;
                for (uint ib = 0; ib < domain_vertices.get_active_comm_nb(); ib++) {
                    recvcounts[ib] = ib + 1 < domain_vertices.get_active_comm_nb()
                                   ? domain_vertices.chunk_size_b
                                   : domain_vertices.get_last_block_size_b();
                    displs[ib + 1] = displs[ib] + recvcounts[ib];
                }
                MPI_Gatherv(int_density_per_band_temp2.data(), local_nb, mpi_datatype, int_density_per_band_temp.data(),
                            recvcounts, displs, mpi_datatype, 0, domain_vertices.band_comm);
                std::vector<T> eigen_value_per_band_temp2(nstates,0);
                eigen_value_per_band_temp2.swap(eigen_value_per_band_temp);
                MPI_Gatherv(eigen_value_per_band_temp2.data(), local_nb, mpi_datatype, eigen_value_per_band_temp.data(),
                            recvcounts, displs, mpi_datatype, 0, domain_vertices.band_comm);
                delete [] recvcounts;
                delete [] displs;
            } else {
                MPI_Reduce(int_density_per_band_temp.data(), int_density_per_band_temp.data(), local_nb,
                           mpi_datatype, MPI_SUM, 0,
                           domain_vertices.domain_3d_comm);
            }
        }
        eigen_value_per_band.insert(eigen_value_per_band.end(), eigen_value_per_band_temp.begin(),
                                    eigen_value_per_band_temp.end());
        int_density_per_band.insert(int_density_per_band.end(), int_density_per_band_temp.begin(),
                                    int_density_per_band_temp.end());
    }
    return;
}

template<typename T>
T Xlsdft<T>::get_max_eigen_value() const {
    T max_eigen_value = -(T)1e10;
    for (typename std::vector<Eigen_solver<T>>::const_iterator it = this->eigen_solvers.cbegin();
            it != this->eigen_solvers.cend(); ++it) {
        if (it->domain_vertices.shared_vertices.get_size() == 0) continue;
        if (it->eigen_values[it->eigen_values.length - 1] > max_eigen_value) {
            max_eigen_value = it->eigen_values[it->eigen_values.length - 1];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &max_eigen_value, 1, Linalg::get_mpi_datatype<T>(),
                  MPI_MAX, this->domain_vertices.comm);
    return max_eigen_value;
}

template<typename T>
T Xlsdft<T>::get_min_eigen_value() const {
    T min_eigen_value = (T)1e10;
    for (typename std::vector<Eigen_solver<T>>::const_iterator it = this->eigen_solvers.cbegin();
            it != this->eigen_solvers.cend(); ++it) {
        if (it->domain_vertices.shared_vertices.get_size() == 0) continue;
        if (it->eigen_values[0] < min_eigen_value) {
            min_eigen_value = it->eigen_values[0];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &min_eigen_value, 1, Linalg::get_mpi_datatype<T>(),
                  MPI_MIN, this->domain_vertices.comm);
    return min_eigen_value;
}

template<typename T>
void Xlsdft<T>::get_eigen_value_range(T* const range) const {
    range[0] = T(1e10);
    range[1] = T(-1e10);
    for (typename std::vector<Eigen_solver<T>>::const_iterator it = this->eigen_solvers.cbegin();
            it != this->eigen_solvers.cend(); ++it) {
        if (it->domain_vertices.shared_vertices.get_size() == 0) continue;
        range[0] = std::min(it->eigen_values[0], range[0]);
        range[1] = std::max(it->eigen_values[it->eigen_values.length - 1], range[1]);
    }
    range[0] = -range[0];
    MPI_Allreduce(MPI_IN_PLACE, range, 2, Linalg::get_mpi_datatype<T>(),
                  MPI_MAX, this->domain_vertices.comm);
    range[0] = -range[0];
    return;
}

template<typename T>
Array_3D<T> Xlsdft<T>::cal_electron_charge_density(const Smearing& smearing, const T smearing_coef,
                                            const T chemical_potential) const {
    const Vertices_3D& local_vertices = this->domain_vertices.get_3D_local_vertices();
    MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
    Array_3D<T> electron_density(local_vertices, 0);
    for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
        if (!this->domain_verticeses[ielement].is_active || this->domain_verticeses[ielement].shared_vertices.get_size() == 0) continue;
        const Vertices_3D& element_vertices = this->element_verticeses[ielement];
        Array_3D<T> electron_charge_density_temp(element_vertices, 0);
        Vertices_3D region = element_vertices.get_overlap_vertices(
                                this->domain_verticeses[ielement].local_vertices);
        if (this->domain_verticeses[ielement].get_domain_3d_comm_size() == 1) {
            electron_charge_density_temp
                    = this->eigen_solvers[ielement].
                                cal_electron_charge_density(smearing,
                                    smearing_coef, chemical_potential, region);
        } else {
            Array_3D<T> electron_charge_density_temp_local
                    = this->eigen_solvers[ielement].
                                cal_electron_charge_density(smearing,
                                    smearing_coef, chemical_potential, region);
            electron_charge_density_temp.be_filled_overlap(electron_charge_density_temp_local);
            if (this->domain_verticeses[ielement].get_active_comm_b() == 0) {
                if (this->domain_verticeses[ielement].get_domain_3d_comm_rank() == 0) {
                    MPI_Reduce(MPI_IN_PLACE, electron_charge_density_temp.data,
                           electron_charge_density_temp.length, mpi_datatype, MPI_SUM, 0,
                           this->domain_verticeses[ielement].get_domain_3d_comm());
                } else {
                    MPI_Reduce(electron_charge_density_temp.data, electron_charge_density_temp.data,
                           electron_charge_density_temp.length, mpi_datatype, MPI_SUM, 0,
                           this->domain_verticeses[ielement].get_domain_3d_comm());
                }
            }
        }
        electron_density.be_filled_overlap(electron_charge_density_temp);
    }
    return electron_density;
}

template<typename T>
void Xlsdft<T>::cal_electron_charge_density_mp(T* const electron_density, const Smearing& smearing,
                                            const T smearing_coef, const T chemical_potential,
                                            Memory_pool<T, Fast_memory>& pool_fast,
                                            Memory_pool<T, Capacity_memory>& pool_cap) const {
    const Vertices_3D& local_vertices = this->domain_vertices.get_3D_local_vertices();
    // MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
    // Array_3D<T> electron_density(local_vertices, 0);
    for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
        if (!this->domain_verticeses[ielement].is_active || this->domain_verticeses[ielement].shared_vertices.get_size() == 0) continue;
        assert(this->domain_verticeses[ielement].comm == MPI_COMM_SELF);
        Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
        Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
        // const Vertices_3D& element_vertices = this->element_verticeses[ielement];
        const Vertices_4D psi_vertices = this->eigen_solvers[ielement].eigen_vectors.get_vertices();
        const Vertices_3D region = psi_vertices.Vertices_3D::get_vertices();
        const uint nstates = psi_vertices.nb;
        const uint region_size = region.get_size();
        T* element_density = pool_fast.allocate(region_size);
        #pragma omp parallel
        Linalg::set_value_general(element_density, T(0), region_size);
        T* occ = pool_fast.allocate(nstates);
        #pragma omp parallel
        Smearing_method::smear(this->eigen_solvers[ielement].eigen_values.data, occ, chemical_potential, smearing, nstates);
        for (uint istate = 0; istate < nstates; istate++) {
            #pragma omp parallel
            Linalg::accumulate_vector_norm_square(element_density,
                        this->eigen_solvers[ielement].eigen_vectors.data + istate * region_size,
                        region_size, occ[istate] * smearing_coef);
        }
        // electron_density.be_filled_overlap(electron_charge_density_temp);
        #pragma omp parallel
        Vertices_method::fill_region(element_density, region,
                                    electron_density, local_vertices,
                                    region);
    }
    return;
}

template<typename T>
Array_3D<T> Xlsdft<T>::cal_electron_charge_density_with_fg_electron(const Smearing& smearing, const T& smearing_coef,
                                            const T& chemical_potential) const {
    const Vertices_3D& local_vertices = this->domain_vertices.get_3D_local_vertices();
    MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
    Array_3D<T> electron_density(local_vertices, 0);

    T upper_limit = this->get_max_eigen_value() + T(1.0);
    const T beta = T(smearing.smearing_control.beta);
    for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
        if (!this->domain_verticeses[ielement].is_active || this->domain_verticeses[ielement].shared_vertices.get_size() == 0) continue;

        T lambda_max = this->eigen_solvers[ielement].eigen_values[this->eigen_solvers[ielement].eigen_values.length - 1];
        T volume = this->shared_verticeses[ielement].get_size() * this->mesh_control.delta_V;
        T coef = T(1) / T(this->shared_verticeses[ielement].get_size());
        T U0 = this->U0s[ielement];
        T shift = Xlsdft_method::fg_electron_number_for_ext_fpmd(lambda_max, U0, upper_limit, chemical_potential, volume, beta, coef);
        shift *= smearing_coef;

        const Vertices_3D& element_vertices = this->element_verticeses[ielement];
        Array_3D<T> electron_charge_density_temp(element_vertices, 0);
        Vertices_3D region = element_vertices.get_overlap_vertices(
                                this->domain_verticeses[ielement].local_vertices);
        if (this->domain_verticeses[ielement].get_domain_3d_comm_size() == 1) {
            electron_charge_density_temp
                    = this->eigen_solvers[ielement].
                                cal_electron_charge_density(smearing,
                                    smearing_coef, chemical_potential, region);
        } else {
            Array_3D<T> electron_charge_density_temp_local
                    = this->eigen_solvers[ielement].
                                cal_electron_charge_density(smearing,
                                    smearing_coef, chemical_potential, region);
            electron_charge_density_temp.be_filled_overlap(electron_charge_density_temp_local);
            if (this->domain_verticeses[ielement].get_active_comm_b() == 0) {
                if (this->domain_verticeses[ielement].get_domain_3d_comm_rank() == 0) {
                    MPI_Reduce(MPI_IN_PLACE, electron_charge_density_temp.data,
                           electron_charge_density_temp.length, mpi_datatype, MPI_SUM, 0,
                           this->domain_verticeses[ielement].get_domain_3d_comm());
                } else {
                    MPI_Reduce(electron_charge_density_temp.data, electron_charge_density_temp.data,
                           electron_charge_density_temp.length, mpi_datatype, MPI_SUM, 0,
                           this->domain_verticeses[ielement].get_domain_3d_comm());
                }
            }
        }
        Linalg::scalar_plus_general(electron_charge_density_temp.data, shift, electron_charge_density_temp.length);
        electron_density.be_filled_overlap(electron_charge_density_temp);
    }
    return electron_density;
}

template<typename T>
void Xlsdft<T>::cal_electron_charge_density_with_fg_electron_mp(T* const electron_density, const Smearing& smearing,
                                                            const T smearing_coef, const T chemical_potential,
                                                            Memory_pool<T, Fast_memory>& pool_fast,
                                                            Memory_pool<T, Capacity_memory>& pool_cap) const {
    const Vertices_3D& local_vertices = this->domain_vertices.get_3D_local_vertices();
    // MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
    // Array_3D<T> electron_density(local_vertices, 0);
    T upper_limit = this->get_max_eigen_value() + T(1.0);
    const T beta = T(smearing.smearing_control.beta);
    for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
        if (!this->domain_verticeses[ielement].is_active || this->domain_verticeses[ielement].shared_vertices.get_size() == 0) continue;
        assert(this->domain_verticeses[ielement].comm == MPI_COMM_SELF);
        Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
        Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
        // const Vertices_3D& element_vertices = this->element_verticeses[ielement];
        const Vertices_4D psi_vertices = this->eigen_solvers[ielement].eigen_vectors.get_vertices();
        const Vertices_3D region = psi_vertices.Vertices_3D::get_vertices();
        const uint nstates = psi_vertices.nb;
        const uint region_size = region.get_size();
        T U0 = this->U0s[ielement];
        T lambda_max = this->eigen_solvers[ielement].eigen_values[this->eigen_solvers[ielement].eigen_values.length - 1];
        T volume = this->shared_verticeses[ielement].get_size() * this->mesh_control.delta_V;
        T coef = T(1) / T(this->shared_verticeses[ielement].get_size());
        T shift = Xlsdft_method::fg_electron_number_for_ext_fpmd(lambda_max, U0, upper_limit, chemical_potential, volume, beta, coef);
        shift *= smearing_coef;
        T* element_density = pool_fast.allocate(region_size);
        #pragma omp parallel
        Linalg::set_value_general(element_density, T(0), region_size);
        T* occ = pool_fast.allocate(nstates);
        #pragma omp parallel
        Smearing_method::smear(this->eigen_solvers[ielement].eigen_values.data, occ, chemical_potential, smearing, nstates);
        for (uint istate = 0; istate < nstates; istate++) {
            #pragma omp parallel
            Linalg::accumulate_vector_norm_square(element_density,
                        this->eigen_solvers[ielement].eigen_vectors.data + istate * region_size,
                        region_size, occ[istate] * smearing_coef);
        }
        #pragma omp parallel
        Linalg::scalar_plus_general(element_density, shift, region_size);
        // electron_density.be_filled_overlap(electron_charge_density_temp);
        #pragma omp parallel
        Vertices_method::fill_region(element_density, region,
                                    electron_density, local_vertices,
                                    region);
    }
    return;
}

template<typename T>
T Xlsdft<T>::cal_electron_charge(const Smearing& smearing, const T& smearing_coef,
                                 const T& chemical_potential) const {
    T electron_charge = (T)0.0;
    for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
        if (this->domain_verticeses[ielement].shared_vertices.get_size() == 0) continue;
        const Vertices_3D& element_vertices = this->element_verticeses[ielement];
        Vertices_3D region = element_vertices.get_overlap_vertices(
                                this->domain_verticeses[ielement].local_vertices);
        electron_charge += this->eigen_solvers[ielement].cal_electron_charge(smearing,
                                    smearing_coef, chemical_potential, region);
    }
    return electron_charge;
}

template<typename T>
T Xlsdft<T>::evaluate_chemical_potential(const Smearing& smearing, const T electron_charge,
                                         const T smearing_coef, const T trial_chemical_potential) const {
    // chemical_potential control
    T tol = 1e-10;
    if (Linalg::get_type_id<T>() == 1) tol = 1e-4;
    uint max_it = 100;

    std::vector<T> eigen_value_per_band;
    std::vector<T> int_density_per_band;
    this->cal_int_density_per_band(eigen_value_per_band, int_density_per_band);
    uint nstates = eigen_value_per_band.size();
    Array_0D<T> occ(nstates);
    T lower_bound = this->get_min_eigen_value() - (T)1.0;
    T upper_bound = this->get_max_eigen_value() + (T)1.0;
    T n = -(T)1.0;
    uint count = 0;
    T chemical_potential = trial_chemical_potential;
    if (unlikely(std::fabs(trial_chemical_potential) == (T)0.0)) {
        chemical_potential = (T)0.5 * (lower_bound + upper_bound);
    }
    while (true) {
        // n = this->cal_electron_charge(smearing, smearing_coef, chemical_potential);
        Smearing_method::smear(eigen_value_per_band.data(), occ.data, chemical_potential, smearing, nstates);
        n = smearing_coef * Linalg::vector_dot_product(occ.data, int_density_per_band.data(),
                                                       nstates, this->domain_vertices.comm);
        if (++count == max_it) break;
        if (unlikely(fabs(electron_charge - n) < tol)) break;
        if (n > electron_charge) {
            upper_bound = chemical_potential;
        } else {
            lower_bound = chemical_potential;
        }
        chemical_potential = (T)0.5 * (lower_bound + upper_bound);
    }
    if (count == max_it && this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "WARNING:: XLSDFT CANNOT FIND chemical_potential AFTER " << count << " iteration." << std::endl;
        std::cout << "WARNING:: WITH chemical_potential = " << chemical_potential
                  << ", n = " << n
                  << ", electron_charge = " << electron_charge
                  << "." << std::endl;
    }
    return chemical_potential;
}

template<typename T>
T Xlsdft<T>::evaluate_chemical_potential2(const Smearing& smearing, const T electron_charge,
                                          const T smearing_coef, const T trial_chemical_potential) const {
    T tol = T(1e-12);
    if (Linalg::get_type_id<T>() == 1) tol = T(1e-4);
    const uint max_it = 100;
    std::vector<T> eigen_value_per_band;
    std::vector<T> int_density_per_band;
    this->cal_int_density_per_band2(eigen_value_per_band, int_density_per_band);
    bool is_very_first = unlikely(std::fabs(trial_chemical_potential) == (T)0.0);
    T chemical_potential = T(0.0);
    T lower_bound = is_very_first ? this->get_min_eigen_value() - T(1.0) : trial_chemical_potential - T(1.0);
    T upper_bound = is_very_first ? this->get_max_eigen_value() + T(1.0) : trial_chemical_potential + T(1.0);
    if (likely(this->domain_vertices.get_active_comm_b() == 0 && this->domain_vertices.is_active)) {
        chemical_potential = Xlsdft_method::evaluate_chemical_potential2<T>(
                                eigen_value_per_band.data(),
                                int_density_per_band.data(),
                                eigen_value_per_band.size(),
                                this->domain_vertices.get_domain_3d_comm(),
                                lower_bound,
                                upper_bound,
                                smearing,
                                electron_charge,
                                smearing_coef,
                                max_it,
                                tol);
    } else if (!this->domain_vertices.is_active) {
        chemical_potential = T(9999.9);
    }
    MPI_Bcast(&chemical_potential, 1, Linalg::get_mpi_datatype<T>(),
              0, this->domain_vertices.band_comm);
    return chemical_potential;
}

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
double vector_norm_square_sum_serial(
    const double* const ptr,
    const uint64_t length)
{
#ifdef __ARM_FEATURE_SVE
    svfloat64_t vSum = svdup_f64(0.0);
    const uint64_t svcnt = 8;
    for (uint64_t i = 0; i < length; i += svcnt) {
        const svbool_t pg = svwhilelt_b64(i, length);
        const svfloat64_t v = svld1(pg, ptr + i);
        vSum = svmla_m(pg, vSum, v, v);
    }
    return svaddv_f64(svptrue_b64(), vSum);
#else
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (uint64_t i = 0; i < length; ++i) {
        sum += ptr[i] * ptr[i];
    }
    return sum;
#endif
}

float vector_norm_square_sum_serial(
    const float* const ptr,
    const uint64_t length)
{
#ifdef __ARM_FEATURE_SVE
    svfloat32_t vSum = svdup_f32(0.0f);
    const uint64_t svcnt = 16;
    for (uint64_t i = 0; i < length; i += svcnt) {
        const svbool_t pg = svwhilelt_b32(i, length);
        const svfloat32_t v = svld1(pg, ptr + i);
        vSum = svmla_m(pg, vSum, v, v);
    }
    return svaddv_f32(svptrue_b32(), vSum);
#else
    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (uint64_t i = 0; i < length; ++i) {
        sum += ptr[i] * ptr[i];
    }
    return sum;
#endif
}

template<typename T>
T Xlsdft<T>::evaluate_chemical_potential_mp(const Smearing& smearing, const T electron_charge,
                                            const T smearing_coef, const T trial_chemical_potential,
                                            Memory_pool<T, Fast_memory>& pool_fast,
                                            Memory_pool<T, Capacity_memory>& pool_cap) const {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    uint nstates_total = 0;
    for (uint i_element = 0; i_element < this->local_element_num; i_element++) {
        const uint nstates_i = this->eigen_solver_controls[i_element].nstates;
        nstates_total += nstates_i;
    }
    T* eigen_values_total = pool_fast.allocate(nstates_total);
    T* band_fractions_total = pool_fast.allocate(nstates_total);
    uint offset = 0;
    for (uint i_element = 0; i_element < this->local_element_num; i_element++) {
        const uint nstates_i = this->eigen_solver_controls[i_element].nstates;
        const Vertices_4D region = this->eigen_solvers[i_element].eigen_vectors.get_vertices();
        const uint nd_inner = region.Vertices_3D::get_size();
        T const* eigen_vectors_data = this->eigen_solvers[i_element].eigen_vectors.data;
        #pragma omp parallel
        #pragma omp for schedule(static, 64/sizeof(T))
        for (uint istate = 0; istate < nstates_i; istate++) {
            T band_fraction;
            #pragma omp parallel if(0)
            band_fraction = vector_norm_square_sum_serial(eigen_vectors_data + istate * nd_inner, nd_inner);
            band_fractions_total[offset + istate] = band_fraction;
        }
        #pragma omp parallel
        Linalg::set_value_general(eigen_values_total + offset, this->eigen_solvers[i_element].eigen_values.data, nstates_i);
        offset += nstates_i;
    }

    const MPI_Comm domain_comm = this->domain_vertices.get_domain_3d_comm();
    T* occ = pool_fast.allocate(nstates_total);
    std::function<T(const T)> net_charge =
    [
        eigen_values_total,
        band_fractions_total,
        smearing_coef,
        occ,
        smearing,
        nstates_total,
        electron_charge,
        domain_comm
    ] (const T chemical_potential) {
        #pragma omp parallel
        Smearing_method::smear(eigen_values_total, occ, chemical_potential, smearing, nstates_total);
        T total_electron = T(0);
        #pragma omp parallel
        total_electron = Linalg::vector_dot_product(occ, band_fractions_total, nstates_total);
        T net_charge = total_electron * smearing_coef;
        MPI_Allreduce(MPI_IN_PLACE, &net_charge, 1, Linalg::get_mpi_datatype<T>(), MPI_SUM, domain_comm);
        net_charge -= electron_charge;
        return net_charge;
    };
    constexpr T tol = std::is_same_v<T, double> ? T(1e-12) : T(1e-4);
    const uint max_it = 100;
    bool is_very_first = unlikely(std::fabs(trial_chemical_potential) == (T)0.0);
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast2(pool_fast);
    T* range = pool_fast.allocate(2);
    this->get_eigen_value_range(range);
    T lower_bound;
    T upper_bound;
    if (is_very_first) {
        lower_bound = range[0];
        upper_bound = range[1];
    } else {
        lower_bound = std::min(range[0], trial_chemical_potential - T(1.0));
        upper_bound = std::max(range[1], trial_chemical_potential + T(1.0));
    }
    T chemical_potential = T(0.0);
    if (likely(this->domain_vertices.get_active_comm_b() == 0 && this->domain_vertices.is_active)) {
        chemical_potential = Tools::BrentsFun(net_charge, lower_bound, upper_bound, max_it, tol);
    } else if (!this->domain_vertices.is_active) {
        chemical_potential = T(9999.9);
    }
    MPI_Bcast(&chemical_potential, 1, Linalg::get_mpi_datatype<T>(),
              0, this->domain_vertices.band_comm);
    return chemical_potential;
}

template<typename T>
T Xlsdft<T>::evaluate_chemical_potential2_mp(const Smearing& smearing, const T electron_charge,
                                            const T smearing_coef, const T trial_chemical_potential,
                                            Memory_pool<T, Fast_memory>& pool_fast,
                                            Memory_pool<T, Capacity_memory>& pool_cap) const {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    uint nstates_total = 0;
    for (uint i_element = 0; i_element < this->local_element_num; i_element++) {
        const uint nstates_i = this->eigen_solver_controls[i_element].nstates;
        nstates_total += nstates_i;
    }
    T* eigen_values_total = pool_fast.allocate(nstates_total);
    T* band_fractions_total = pool_fast.allocate(nstates_total);
    uint offset = 0;
    for (uint i_element = 0; i_element < this->local_element_num; i_element++) {
        const uint nstates_i = this->eigen_solver_controls[i_element].nstates;
        const Vertices_4D region = this->eigen_solvers[i_element].eigen_vectors.get_vertices();
        const uint nd_inner = region.Vertices_3D::get_size();
        T const* eigen_vectors_data = this->eigen_solvers[i_element].eigen_vectors.data;
        #pragma omp parallel
        #pragma omp for schedule(static, 64/sizeof(T))
        for (uint istate = 0; istate < nstates_i; istate++) {
            T band_fraction;
            #pragma omp parallel if(0)
            band_fraction = vector_norm_square_sum_serial(eigen_vectors_data + istate * nd_inner, nd_inner);
            band_fractions_total[offset + istate] = band_fraction;
        }
        #pragma omp parallel
        Linalg::set_value_general(eigen_values_total + offset, this->eigen_solvers[i_element].eigen_values.data, nstates_i);
        offset += nstates_i;
    }

    const MPI_Comm domain_comm = this->domain_vertices.get_domain_3d_comm();
    T* occ = pool_fast.allocate(nstates_total);
    std::function<void (T const* const in, T *const out, uint const n)> net_charge =
    [
        eigen_values_total,
        band_fractions_total,
        smearing_coef,
        occ,
        smearing,
        nstates_total,
        electron_charge,
        domain_comm
    ] (T const* const chemical_potentials, T *const net_charges, uint const n) {
        for (uint i = 0; i < n; i++) {
            #pragma omp parallel
            Smearing_method::smear(eigen_values_total, occ, chemical_potentials[i], smearing, nstates_total);
            T total_electron = T(0);
            #pragma omp parallel
            total_electron = Linalg::vector_dot_product(occ, band_fractions_total, nstates_total);
            net_charges[i] = total_electron * smearing_coef;
        }
        MPI_Allreduce(MPI_IN_PLACE, net_charges, n, Linalg::get_mpi_datatype<T>(), MPI_SUM, domain_comm);
        Linalg::scalar_minus_general(net_charges, electron_charge, n);
        return;
    };
    constexpr T tol = std::is_same_v<T, double> ? T(1e-12) : T(1e-4);
    const uint max_it = 100;
    bool is_very_first = unlikely(std::fabs(trial_chemical_potential) == (T)0.0);
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast2(pool_fast);
    T* range = pool_fast.allocate(2);
    this->get_eigen_value_range(range);
    T lower_bound;
    T upper_bound;
    if (is_very_first) {
        lower_bound = range[0];
        upper_bound = range[1];
    } else {
        lower_bound = std::min(range[0], trial_chemical_potential - T(1.0));
        upper_bound = std::max(range[1], trial_chemical_potential + T(1.0));
    }
    uint const n = 8;
    T chemical_potential = T(0.0);
    if (likely(this->domain_vertices.get_active_comm_b() == 0 && this->domain_vertices.is_active)) {
        chemical_potential = Tools::BatchedBrent(net_charge, lower_bound, upper_bound, max_it, tol, n);
    } else if (!this->domain_vertices.is_active) {
        chemical_potential = T(9999.9);
    }
    MPI_Bcast(&chemical_potential, 1, Linalg::get_mpi_datatype<T>(),
              0, this->domain_vertices.band_comm);
    return chemical_potential;
}

template<typename T>
T Xlsdft<T>::evaluate_chemical_potential3_mp(const Smearing& smearing, const T electron_charge,
                                            const T smearing_coef, const T trial_chemical_potential,
                                            Memory_pool<T, Fast_memory>& pool_fast,
                                            Memory_pool<T, Capacity_memory>& pool_cap) const {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    
    uint nstates_total = 0;
    for (uint i_element = 0; i_element < this->local_element_num; i_element++) {
        nstates_total += this->eigen_solver_controls[i_element].nstates;
    }
    
    T* eigen_values_total = pool_fast.allocate(nstates_total);
    T* band_fractions_total = pool_fast.allocate(nstates_total);
    uint offset = 0;
    
    for (uint i_element = 0; i_element < this->local_element_num; i_element++) {
        const uint nstates_i = this->eigen_solver_controls[i_element].nstates;
        const Vertices_4D region = this->eigen_solvers[i_element].eigen_vectors.get_vertices();
        const uint nd_inner = region.Vertices_3D::get_size();
        T const* eigen_vectors_data = this->eigen_solvers[i_element].eigen_vectors.data;
        
        #pragma omp parallel
        #pragma omp for schedule(static, 64/sizeof(T))
        for (uint istate = 0; istate < nstates_i; istate++) {
            T band_fraction;
            #pragma omp parallel if(0)
            band_fraction = vector_norm_square_sum_serial(eigen_vectors_data + istate * nd_inner, nd_inner);
            band_fractions_total[offset + istate] = band_fraction;
        }
        #pragma omp parallel
        Linalg::set_value_general(eigen_values_total + offset, this->eigen_solvers[i_element].eigen_values.data, nstates_i);
        offset += nstates_i;
    }

    const MPI_Comm domain_comm = this->domain_vertices.get_domain_3d_comm();
    const uint fallback_n = 8;
    const uint num_bins = 2048;

    // 内存池统一接管巨型数组与通信缓存，防止堆内存碎片
    T* occ = pool_fast.allocate(nstates_total);
    T* docc = pool_fast.allocate(nstates_total);
    T* mpi_buf_deriv = pool_fast.allocate(2 * fallback_n); // 用于导数版的 2*n 通信缓冲

    const T beta = (T)smearing.smearing_control.beta;

    // =========================================================================
    // 接口 1: 全局能谱直方图生成器 (1次 Allreduce)
    // =========================================================================
    std::function<void(const T, const T, const uint, T* const)> build_global_hist =
    [eigen_values_total, band_fractions_total, smearing_coef, nstates_total, domain_comm] 
    (const T L, const T R, const uint nbins, T* const out_hist) {
        T bin_width = (R - L) / T(nbins);
        
        // 初始化目标空间为 0
        for(uint i = 0; i < nbins; i++) out_hist[i] = T(0.0);

        #pragma omp parallel
        {
            // 每个线程维护私有的局域直方图，避免原子锁争用
            std::vector<T> thread_hist(nbins, T(0.0));
            #pragma omp for schedule(static)
            for (uint j = 0; j < nstates_total; j++) {
                T val = eigen_values_total[j];
                if (val >= L && val <= R) {
                    int bin_idx = std::min((int)nbins - 1, std::max(0, (int)((val - L) / bin_width)));
                    thread_hist[bin_idx] += band_fractions_total[j] * smearing_coef;
                }
            }
            #pragma omp critical
            {
                for (uint b = 0; b < nbins; b++) out_hist[b] += thread_hist[b];
            }
        }
        
        // 使用 MPI_IN_PLACE 直接在 out_hist 上完成规约
        MPI_Allreduce(MPI_IN_PLACE, out_hist, nbins, Linalg::get_mpi_datatype<T>(), MPI_SUM, domain_comm);
    };

    // =========================================================================
    // 接口 2: 带导数的批量评估器 (复用你们的 Linalg::scalar_minus_general)
    // =========================================================================
    std::function<void(const T* const, T* const, T* const, const uint)> net_charge_with_deriv =
    [eigen_values_total, band_fractions_total, smearing_coef, occ, docc, mpi_buf_deriv, 
     smearing, nstates_total, electron_charge, domain_comm] 
    (const T* const chem_pots, T* const out_f, T* const out_df, const uint n) {
        for (uint i = 0; i < n; i++) {
            #pragma omp parallel
            Smearing_method::smear_with_deriv(eigen_values_total, occ, docc, chem_pots[i], smearing, nstates_total);
            
            T total_e = T(0);
            #pragma omp parallel
            total_e = Linalg::vector_dot_product(occ, band_fractions_total, nstates_total);
            
            T total_de = T(0);
            #pragma omp parallel
            total_de = Linalg::vector_dot_product(docc, band_fractions_total, nstates_total);
            
            mpi_buf_deriv[i]     = total_e * smearing_coef;
            mpi_buf_deriv[n + i] = total_de * smearing_coef;
        }
        
        MPI_Allreduce(MPI_IN_PLACE, mpi_buf_deriv, 2 * n, Linalg::get_mpi_datatype<T>(), MPI_SUM, domain_comm);
        
        for (uint i = 0; i < n; i++) {
            out_f[i]  = mpi_buf_deriv[i];
            out_df[i] = mpi_buf_deriv[n + i];
        }
        
        // 调用底层 Linalg 进行精准的减法运算
        Linalg::scalar_minus_general(out_f, electron_charge, n);
    };

    // =========================================================================
    // 接口 3: 纯函数批量评估器 (Fallback 收尾专用)
    // =========================================================================
    std::function<void (T const* const, T *const, uint const)> net_charge_pure =
    [eigen_values_total, band_fractions_total, smearing_coef, occ, smearing, nstates_total, electron_charge, domain_comm] 
    (T const* const chemical_potentials, T *const net_charges, uint const n) {
        for (uint i = 0; i < n; i++) {
            #pragma omp parallel
            Smearing_method::smear(eigen_values_total, occ, chemical_potentials[i], smearing, nstates_total);
            
            T total_electron = T(0);
            #pragma omp parallel
            total_electron = Linalg::vector_dot_product(occ, band_fractions_total, nstates_total);
            net_charges[i] = total_electron * smearing_coef;
        }
        MPI_Allreduce(MPI_IN_PLACE, net_charges, n, Linalg::get_mpi_datatype<T>(), MPI_SUM, domain_comm);
        Linalg::scalar_minus_general(net_charges, electron_charge, n);
    };

    // =========================================================================
    // 接口 4: 本地零通信单点求解模型 (支持 FD 和 Gaussian 自动切换)
    // =========================================================================
    const int s_method = smearing.smearing_control.method;
    std::function<T(const T)> local_smear = [beta, s_method](const T delta_E) {
        if (s_method == 0) return T(1.0) / (T(1.0) + std::exp(beta * delta_E));
        else               return T(0.5) * (T(1.0) - erf(beta * delta_E));
    };

    // =========================================================================
    // 初始化物理参数并进入算法
    // =========================================================================
    constexpr T tol = std::is_same_v<T, double> ? T(1e-12) : T(1e-4);
    const uint max_it = 100;
    bool is_very_first = unlikely(std::fabs(trial_chemical_potential) == (T)0.0);
    
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast2(pool_fast);
    T* range = pool_fast.allocate(2);
    this->get_eigen_value_range(range);
    T lower_bound;
    T upper_bound;
    if (is_very_first) {
        lower_bound = range[0];
        upper_bound = range[1];
    } else {
        lower_bound = std::min(range[0], trial_chemical_potential - T(1.0));
        upper_bound = std::max(range[1], trial_chemical_potential + T(1.0));
    }
    
    T chemical_potential = T(0.0);
    if (likely(this->domain_vertices.get_active_comm_b() == 0 && this->domain_vertices.is_active)) {
        // 调用终极版 SpectralHistogramRoot 算法
        chemical_potential = Tools::SpectralHistogramRoot<T>(
            build_global_hist,
            net_charge_with_deriv,
            net_charge_pure,
            local_smear,
            lower_bound, upper_bound,
            max_it, tol, electron_charge,
            num_bins, fallback_n
        );
    } else if (!this->domain_vertices.is_active) {
        chemical_potential = T(9999.9);
    }
    
    MPI_Bcast(&chemical_potential, 1, Linalg::get_mpi_datatype<T>(),
              0, this->domain_vertices.band_comm);
    
    return chemical_potential;
}

template<typename T>
T Xlsdft<T>::evaluate_chemical_potential2_with_fg_electron(const Smearing& smearing, const T electron_charge,
                                          const T smearing_coef, const T trial_chemical_potential) const {
    (void) trial_chemical_potential;
    T tol = T(1e-12);
    if (Linalg::get_type_id<T>() == 1) tol = T(1e-4);
    const uint max_it = 100;
    std::vector<T> eigen_value_per_band;
    std::vector<T> int_density_per_band;
    this->cal_int_density_per_band2(eigen_value_per_band, int_density_per_band);
    T chemical_potential = T(0.0);
    T lower_bound = this->get_min_eigen_value() - T(1.0);
    T upper_bound = this->get_max_eigen_value() + T(1.0);
    uint nelement = this->local_element_num;
    std::vector<T> lambda_maxs(nelement, T(0.0));
    const std::vector<T>& U0s = this->U0s;
    std::vector<T> volumes(nelement, T(0.0));
    std::vector<T> coefs(nelement, T(0.0));
    for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
        lambda_maxs[ielement] = this->eigen_solvers[ielement].eigen_values[this->eigen_solvers[ielement].eigen_values.length - 1];
        volumes[ielement] = this->shared_verticeses[ielement].get_size() * this->mesh_control.delta_V;
        coefs[ielement] = T(this->element_verticeses[ielement].get_size()) / T(this->shared_verticeses[ielement].get_size());
    }
    if (likely(this->domain_vertices.get_active_comm_b() == 0 && this->domain_vertices.is_active)) {
        chemical_potential = Xlsdft_method::evaluate_chemical_potential2_with_fg_electron<T>(
                                eigen_value_per_band.data(),
                                int_density_per_band.data(),
                                eigen_value_per_band.size(),
                                this->domain_vertices.get_domain_3d_comm(),
                                lower_bound,
                                upper_bound,
                                smearing,
                                electron_charge,
                                smearing_coef,
                                max_it,
                                tol,
                                nelement,
                                lambda_maxs.data(),
                                U0s.data(),
                                volumes.data(),
                                coefs.data());
    } else if (!this->domain_vertices.is_active) {
        chemical_potential = T(9999.9);
    }
    MPI_Bcast(&chemical_potential, 1, Linalg::get_mpi_datatype<T>(),
              0, this->domain_vertices.band_comm);
    return chemical_potential;
}

template<typename T>
T Xlsdft<T>::evaluate_chemical_potential_with_fg_electron_mp(const Smearing& smearing, const T electron_charge,
                                                            const T smearing_coef, const T trial_chemical_potential,
                                                            Memory_pool<T, Fast_memory>& pool_fast,
                                                            Memory_pool<T, Capacity_memory>& pool_cap) const {
    (void) trial_chemical_potential;
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    uint nstates_total = 0;
    for (uint i_element = 0; i_element < this->local_element_num; i_element++) {
        const uint nstates_i = this->eigen_solver_controls[i_element].nstates;
        nstates_total += nstates_i;
    }
    T* eigen_values_total = pool_fast.allocate(nstates_total);
    T* band_fractions_total = pool_fast.allocate(nstates_total);
    T* U0s = pool_fast.allocate(this->local_element_num);
    T* volumes = pool_fast.allocate(this->local_element_num);
    T* coefs = pool_fast.allocate(this->local_element_num);
    T* lambda_maxs = pool_fast.allocate(this->local_element_num);
    Linalg::set_value_general(U0s, this->U0s.data(), this->local_element_num);
    const uint nelement = this->local_element_num;
    uint offset = 0;
    for (uint i_element = 0; i_element < this->local_element_num; i_element++) {
        const uint nstates_i = this->eigen_solver_controls[i_element].nstates;
        const Vertices_4D region = this->eigen_solvers[i_element].eigen_vectors.get_vertices();
        const uint nd_inner = region.Vertices_3D::get_size();
        volumes[i_element] = this->shared_verticeses[i_element].get_size() * this->mesh_control.delta_V;
        coefs[i_element] = T(this->element_verticeses[i_element].get_size()) / T(this->shared_verticeses[i_element].get_size());
        lambda_maxs[i_element] = this->eigen_solvers[i_element].eigen_values.data[this->eigen_solvers[i_element].eigen_values.length-1];
        T const* eigen_vectors_data = this->eigen_solvers[i_element].eigen_vectors.data;
        #pragma omp parallel
        #pragma omp for schedule(static, 64/sizeof(T))
        for (uint istate = 0; istate < nstates_i; istate++) {
            T band_fraction;
            #pragma omp parallel if(0)
            band_fraction = vector_norm_square_sum_serial(eigen_vectors_data + istate * nd_inner, nd_inner);
            band_fractions_total[offset + istate] = band_fraction;
        }
        #pragma omp parallel
        Linalg::set_value_general(eigen_values_total + offset, this->eigen_solvers[i_element].eigen_values.data, nstates_i);
        offset += nstates_i;
    }

    T* range = pool_fast.allocate(2);
    this->get_eigen_value_range(range);
    T lower_bound = range[0];
    T upper_bound = range[1];

    const MPI_Comm domain_comm = this->domain_vertices.get_domain_3d_comm();
    T* occ = pool_fast.allocate(nstates_total);
    std::function<T(const T)> net_charge =
    [
        eigen_values_total,
        band_fractions_total,
        smearing_coef,
        occ,
        smearing,
        nstates_total,
        electron_charge,
        domain_comm,
        nelement,
        lambda_maxs,
        U0s,
        upper_bound,
        volumes,
        coefs
    ] (const T chemical_potential) {
        #pragma omp parallel
        Smearing_method::smear(eigen_values_total, occ, chemical_potential, smearing, nstates_total);
        T total_electron = T(0);
        #pragma omp parallel
        total_electron = Linalg::vector_dot_product(occ, band_fractions_total, nstates_total);
        T fg_electron = Xlsdft_method::xlsdft_fg_electron_number_for_ext_fpmd<T>(
            nelement, lambda_maxs, U0s, upper_bound, chemical_potential,
            volumes, smearing.smearing_control.beta, coefs);
        T net_charge = (total_electron + fg_electron) * smearing_coef;
        MPI_Allreduce(MPI_IN_PLACE, &net_charge, 1, Linalg::get_mpi_datatype<T>(), MPI_SUM, domain_comm);
        net_charge -= electron_charge;
        return net_charge;
    };

    constexpr T tol = std::is_same_v<T, double> ? T(1e-12) : T(1e-4);
    const uint max_it = 100;
    T chemical_potential = T(0.0);
    if (likely(this->domain_vertices.get_active_comm_b() == 0 && this->domain_vertices.is_active)) {
        chemical_potential = Tools::BrentsFun(net_charge, lower_bound, upper_bound, max_it, tol);
    } else if (!this->domain_vertices.is_active) {
        chemical_potential = T(9999.9);
    }
    MPI_Bcast(&chemical_potential, 1, Linalg::get_mpi_datatype<T>(),
              0, this->domain_vertices.band_comm);
    return chemical_potential;
}

template<typename T>
T Xlsdft<T>::evaluate_chemical_potential2_with_fg_electron_mp(const Smearing& smearing, const T electron_charge,
                                                            const T smearing_coef, const T trial_chemical_potential,
                                                            Memory_pool<T, Fast_memory>& pool_fast,
                                                            Memory_pool<T, Capacity_memory>& pool_cap) const {
    (void) trial_chemical_potential;
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    uint nstates_total = 0;
    for (uint i_element = 0; i_element < this->local_element_num; i_element++) {
        const uint nstates_i = this->eigen_solver_controls[i_element].nstates;
        nstates_total += nstates_i;
    }
    T* eigen_values_total = pool_fast.allocate(nstates_total);
    T* band_fractions_total = pool_fast.allocate(nstates_total);
    T* U0s = pool_fast.allocate(this->local_element_num);
    T* volumes = pool_fast.allocate(this->local_element_num);
    T* coefs = pool_fast.allocate(this->local_element_num);
    T* lambda_maxs = pool_fast.allocate(this->local_element_num);
    Linalg::set_value_general(U0s, this->U0s.data(), this->local_element_num);
    const uint nelement = this->local_element_num;
    uint offset = 0;
    for (uint i_element = 0; i_element < this->local_element_num; i_element++) {
        const uint nstates_i = this->eigen_solver_controls[i_element].nstates;
        const Vertices_4D region = this->eigen_solvers[i_element].eigen_vectors.get_vertices();
        const uint nd_inner = region.Vertices_3D::get_size();
        volumes[i_element] = this->shared_verticeses[i_element].get_size() * this->mesh_control.delta_V;
        coefs[i_element] = T(this->element_verticeses[i_element].get_size()) / T(this->shared_verticeses[i_element].get_size());
        lambda_maxs[i_element] = this->eigen_solvers[i_element].eigen_values.data[this->eigen_solvers[i_element].eigen_values.length-1];
        T const* eigen_vectors_data = this->eigen_solvers[i_element].eigen_vectors.data;
        #pragma omp parallel
        #pragma omp for schedule(static, 64/sizeof(T))
        for (uint istate = 0; istate < nstates_i; istate++) {
            T band_fraction;
            #pragma omp parallel if(0)
            band_fraction = vector_norm_square_sum_serial(eigen_vectors_data + istate * nd_inner, nd_inner);
            band_fractions_total[offset + istate] = band_fraction;
        }
        #pragma omp parallel
        Linalg::set_value_general(eigen_values_total + offset, this->eigen_solvers[i_element].eigen_values.data, nstates_i);
        offset += nstates_i;
    }

    T* range = pool_fast.allocate(2);
    this->get_eigen_value_range(range);
    T lower_bound = range[0];
    T upper_bound = range[1];

    const MPI_Comm domain_comm = this->domain_vertices.get_domain_3d_comm();
    T* occ = pool_fast.allocate(nstates_total);
    std::function<void (T const* const in, T *const out, uint const n)> net_charge =
    [
        eigen_values_total,
        band_fractions_total,
        smearing_coef,
        occ,
        smearing,
        nstates_total,
        electron_charge,
        domain_comm,
        nelement,
        lambda_maxs,
        U0s,
        upper_bound,
        volumes,
        coefs
    ] (T const* const chemical_potentials, T *const net_charges, uint const n) {
        for (uint i = 0; i < n; i++) {
            #pragma omp parallel
            Smearing_method::smear(eigen_values_total, occ, chemical_potentials[i], smearing, nstates_total);
            T total_electron = T(0);
            #pragma omp parallel
            total_electron = Linalg::vector_dot_product(occ, band_fractions_total, nstates_total);
            T fg_electron = Xlsdft_method::xlsdft_fg_electron_number_for_ext_fpmd<T>(
                    nelement, lambda_maxs, U0s, upper_bound, chemical_potentials[i],
                    volumes, smearing.smearing_control.beta, coefs);
            net_charges[i] = (total_electron + fg_electron) * smearing_coef;
        }
        MPI_Allreduce(MPI_IN_PLACE, net_charges, n, Linalg::get_mpi_datatype<T>(), MPI_SUM, domain_comm);
        Linalg::scalar_minus_general(net_charges, electron_charge, n);
        return;
    };

    constexpr T tol = std::is_same_v<T, double> ? T(1e-12) : T(1e-4);
    const uint max_it = 100;
    uint const n = 8;
    T chemical_potential = T(0.0);
    if (likely(this->domain_vertices.get_active_comm_b() == 0 && this->domain_vertices.is_active)) {
        chemical_potential = Tools::BatchedBrent(net_charge, lower_bound, upper_bound, max_it, tol, n);
    } else if (!this->domain_vertices.is_active) {
        chemical_potential = T(9999.9);
    }
    MPI_Bcast(&chemical_potential, 1, Linalg::get_mpi_datatype<T>(),
              0, this->domain_vertices.band_comm);
    return chemical_potential;
}

template<typename T>
T Xlsdft<T>::evaluate_chemical_potential3_with_fg_electron_mp(const Smearing& smearing,
                                                            const T electron_charge,
                                                            const T smearing_coef,
                                                            const T trial_chemical_potential,
                                                            Memory_pool<T, Fast_memory>& pool_fast,
                                                            Memory_pool<T, Capacity_memory>& pool_cap) const {
    (void) trial_chemical_potential;
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);

    uint nstates_total = 0;
    for (uint i_element = 0; i_element < this->local_element_num; i_element++) {
        nstates_total += this->eigen_solver_controls[i_element].nstates;
    }

    T* eigen_values_total = pool_fast.allocate(nstates_total);
    T* band_fractions_total = pool_fast.allocate(nstates_total);
    T* U0s = pool_fast.allocate(this->local_element_num);
    T* volumes = pool_fast.allocate(this->local_element_num);
    T* coefs = pool_fast.allocate(this->local_element_num);
    T* lambda_maxs = pool_fast.allocate(this->local_element_num);
    Linalg::set_value_general(U0s, this->U0s.data(), this->local_element_num);

    const uint nelement = this->local_element_num;
    uint offset = 0;
    for (uint i_element = 0; i_element < nelement; i_element++) {
        const uint nstates_i = this->eigen_solver_controls[i_element].nstates;
        const Vertices_4D region = this->eigen_solvers[i_element].eigen_vectors.get_vertices();
        const uint nd_inner = region.Vertices_3D::get_size();
        volumes[i_element] = this->shared_verticeses[i_element].get_size() * this->mesh_control.delta_V;
        coefs[i_element] = T(this->element_verticeses[i_element].get_size()) /
                           T(this->shared_verticeses[i_element].get_size());
        lambda_maxs[i_element] = this->eigen_solvers[i_element].eigen_values.data[
                                    this->eigen_solvers[i_element].eigen_values.length - 1];
        T const* eigen_vectors_data = this->eigen_solvers[i_element].eigen_vectors.data;

        #pragma omp parallel
        #pragma omp for schedule(static, 64/sizeof(T))
        for (uint istate = 0; istate < nstates_i; istate++) {
            T band_fraction;
            #pragma omp parallel if(0)
            band_fraction = vector_norm_square_sum_serial(
                                eigen_vectors_data + istate * nd_inner, nd_inner);
            band_fractions_total[offset + istate] = band_fraction;
        }
        #pragma omp parallel
        Linalg::set_value_general(eigen_values_total + offset,
                                  this->eigen_solvers[i_element].eigen_values.data,
                                  nstates_i);
        offset += nstates_i;
    }

    T* range = pool_fast.allocate(2);
    this->get_eigen_value_range(range);
    const T lower_bound = range[0];
    const T upper_bound = range[1];
    // Keep this identical to evaluate_chemical_potential2_with_fg_electron_mp.
    const T fg_upper_limit = upper_bound;

    const MPI_Comm domain_comm = this->domain_vertices.get_domain_3d_comm();
    const uint fallback_n = 8;
    const uint num_bins = 2048;
    const T beta = T(smearing.smearing_control.beta);
    constexpr T fg_step = T(0.01);
    constexpr T sqrt_2_pi_square =
        T(1.41421356237309504880168872420969807856967187537694807317667973799) /
        T(9.869604401089358618834490999876151135313699407240790626413533);

    T* occ = pool_fast.allocate(nstates_total);
    T* docc = pool_fast.allocate(nstates_total);
    T* mpi_buf_deriv = pool_fast.allocate(2 * fallback_n);
    // The first half is the discrete histogram and the second half is the FG histogram.
    T* hist_reduce_buf = pool_fast.allocate(2 * num_bins);
    T* fg_hist = hist_reduce_buf + num_bins;

    std::function<void(const T, const T, const uint, T* const)> build_global_hist =
    [eigen_values_total, band_fractions_total, smearing_coef, nstates_total,
     domain_comm, nelement, lambda_maxs, U0s, fg_upper_limit, volumes, coefs,
     hist_reduce_buf]
    (const T L, const T R, const uint nbins, T* const out_hist) {
        assert(nbins > 0);
        assert(R > L);
        const T bin_width = (R - L) / T(nbins);
        for (uint i = 0; i < 2 * nbins; i++) hist_reduce_buf[i] = T(0.0);

        #pragma omp parallel
        {
            std::vector<T> thread_hist(2 * nbins, T(0.0));

            #pragma omp for schedule(static)
            for (uint j = 0; j < nstates_total; j++) {
                const T value = eigen_values_total[j];
                if (value >= L && value <= R) {
                    const int bin = std::min((int)nbins - 1,
                                             std::max(0, (int)((value - L) / bin_width)));
                    thread_hist[bin] += band_fractions_total[j] * smearing_coef;
                }
            }

            #pragma omp for schedule(dynamic)
            for (uint i_element = 0; i_element < nelement; i_element++) {
                if (lambda_maxs[i_element] >= fg_upper_limit) continue;
                const int n_fg_states = std::ceil(
                    (fg_upper_limit - lambda_maxs[i_element]) / fg_step);
                const T element_prefactor = smearing_coef * fg_step * sqrt_2_pi_square *
                                            volumes[i_element] * coefs[i_element];
                for (int i = 0; i < n_fg_states; i++) {
                    const T lambda = lambda_maxs[i_element] + (T(i) + T(0.5)) * fg_step;
                    const T kinetic_energy = lambda - U0s[i_element];
                    if (kinetic_energy <= T(0.0) || lambda < L ||
                        lambda > R + fg_step / T(2.0)) continue;
                    // ceil() can place the final midpoint at most step/2 above R.
                    const int bin = std::min((int)nbins - 1,
                                             std::max(0, (int)((lambda - L) / bin_width)));
                    thread_hist[nbins + bin] += element_prefactor * std::sqrt(kinetic_energy);
                }
            }

            #pragma omp critical
            {
                for (uint b = 0; b < 2 * nbins; b++) {
                    hist_reduce_buf[b] += thread_hist[b];
                }
            }
        }

        // Both histograms share one collective operation.
        MPI_Allreduce(MPI_IN_PLACE, hist_reduce_buf, 2 * nbins,
                      Linalg::get_mpi_datatype<T>(), MPI_SUM, domain_comm);
        for (uint b = 0; b < nbins; b++) out_hist[b] = hist_reduce_buf[b];
    };

    std::function<void(const T* const, T* const, T* const, const uint)>
    net_charge_with_deriv =
    [eigen_values_total, band_fractions_total, smearing_coef, occ, docc,
     mpi_buf_deriv, smearing, nstates_total, electron_charge, domain_comm,
     nelement, lambda_maxs, U0s, fg_upper_limit, volumes, beta, coefs]
    (const T* const chem_pots, T* const out_f, T* const out_df, const uint n) {
        assert(n <= fallback_n);
        for (uint i = 0; i < n; i++) {
            #pragma omp parallel
            Smearing_method::smear_with_deriv(eigen_values_total, occ, docc,
                                               chem_pots[i], smearing, nstates_total);
            T total_electron = T(0.0);
            #pragma omp parallel
            total_electron = Linalg::vector_dot_product(
                                occ, band_fractions_total, nstates_total);
            T total_derivative = T(0.0);
            #pragma omp parallel
            total_derivative = Linalg::vector_dot_product(
                                docc, band_fractions_total, nstates_total);

            T fg_electron = T(0.0);
            T fg_derivative = T(0.0);
            for (uint i_element = 0; i_element < nelement; i_element++) {
                if (lambda_maxs[i_element] >= fg_upper_limit) continue;
                const int n_fg_states = std::ceil(
                    (fg_upper_limit - lambda_maxs[i_element]) / fg_step);
                T element_electron = T(0.0);
                T element_derivative = T(0.0);
                #pragma omp parallel for reduction(+:element_electron,element_derivative)
                for (int j = 0; j < n_fg_states; j++) {
                    const T lambda = lambda_maxs[i_element] + (T(j) + T(0.5)) * fg_step;
                    const T kinetic_energy = lambda - U0s[i_element];
                    const T x = beta * (lambda - chem_pots[i]);
                    if (likely(x < T(1e2) && kinetic_energy > T(0.0))) {
                        const T occupation = T(1.0) / (T(1.0) + std::exp(x));
                        const T dos = std::sqrt(kinetic_energy);
                        element_electron += dos * occupation;
                        element_derivative += dos * beta * occupation *
                                              (T(1.0) - occupation);
                    }
                }
                const T element_prefactor = fg_step * sqrt_2_pi_square *
                                            volumes[i_element] * coefs[i_element];
                fg_electron += element_electron * element_prefactor;
                fg_derivative += element_derivative * element_prefactor;
            }

            mpi_buf_deriv[i] = (total_electron + fg_electron) * smearing_coef;
            mpi_buf_deriv[n + i] =
                (total_derivative + fg_derivative) * smearing_coef;
        }

        MPI_Allreduce(MPI_IN_PLACE, mpi_buf_deriv, 2 * n,
                      Linalg::get_mpi_datatype<T>(), MPI_SUM, domain_comm);
        for (uint i = 0; i < n; i++) {
            out_f[i] = mpi_buf_deriv[i];
            out_df[i] = mpi_buf_deriv[n + i];
        }
        Linalg::scalar_minus_general(out_f, electron_charge, n);
    };

    std::function<void(const T* const, T* const, const uint)> net_charge_pure =
    [eigen_values_total, band_fractions_total, smearing_coef, occ, smearing,
     nstates_total, electron_charge, domain_comm, nelement, lambda_maxs, U0s,
     fg_upper_limit, volumes, beta, coefs]
    (const T* const chemical_potentials, T* const net_charges, const uint n) {
        for (uint i = 0; i < n; i++) {
            #pragma omp parallel
            Smearing_method::smear(eigen_values_total, occ, chemical_potentials[i],
                                   smearing, nstates_total);
            T total_electron = T(0.0);
            #pragma omp parallel
            total_electron = Linalg::vector_dot_product(
                                occ, band_fractions_total, nstates_total);
            const T fg_electron =
                Xlsdft_method::xlsdft_fg_electron_number_for_ext_fpmd<T>(
                    nelement, lambda_maxs, U0s, fg_upper_limit,
                    chemical_potentials[i], volumes, beta, coefs);
            net_charges[i] = (total_electron + fg_electron) * smearing_coef;
        }
        MPI_Allreduce(MPI_IN_PLACE, net_charges, n,
                      Linalg::get_mpi_datatype<T>(), MPI_SUM, domain_comm);
        Linalg::scalar_minus_general(net_charges, electron_charge, n);
    };

    const int smearing_method = smearing.smearing_control.method;
    std::function<T(const T)> local_smear = [beta, smearing_method](const T delta_E) {
        if (smearing_method == 0) {
            return T(1.0) / (T(1.0) + std::exp(beta * delta_E));
        }
        return T(0.5) * (T(1.0) - erf(beta * delta_E));
    };

    std::function<T(const T)> local_fg_model =
    [fg_hist, lower_bound, upper_bound, beta](const T chemical_potential) {
        const T bin_width = (upper_bound - lower_bound) / T(num_bins);
        T fg_electron = T(0.0);
        for (uint b = 0; b < num_bins; b++) {
            if (fg_hist[b] <= T(0.0)) continue;
            const T energy = lower_bound + (T(b) + T(0.5)) * bin_width;
            const T x = beta * (energy - chemical_potential);
            T occupation;
            if (x >= T(1e2)) occupation = T(0.0);
            else if (x <= T(-1e2)) occupation = T(1.0);
            else occupation = T(1.0) / (T(1.0) + std::exp(x));
            fg_electron += fg_hist[b] * occupation;
        }
        return fg_electron;
    };

    constexpr T tol = std::is_same_v<T, double> ? T(1e-12) : T(1e-4);
    const uint max_it = 100;
    T chemical_potential = T(0.0);
    if (likely(this->domain_vertices.get_active_comm_b() == 0 &&
               this->domain_vertices.is_active)) {
        chemical_potential = Tools::SpectralHistogramRoot<T>(
            build_global_hist,
            net_charge_with_deriv,
            net_charge_pure,
            local_smear,
            lower_bound, upper_bound,
            max_it, tol, electron_charge,
            num_bins, fallback_n,
            local_fg_model);
    } else if (!this->domain_vertices.is_active) {
        chemical_potential = T(9999.9);
    }

    MPI_Bcast(&chemical_potential, 1, Linalg::get_mpi_datatype<T>(),
              0, this->domain_vertices.band_comm);
    return chemical_potential;
}

template<typename T>
T Xlsdft<T>::evaluate_band_energy(const Smearing& smearing, const T& chemical_potential, const T& smearing_coef) const {
    T band_energy = (T)0.0;
    for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
        if (this->domain_verticeses[ielement].shared_vertices.get_size() == 0) continue;
        const Vertices_3D& element_vertices = this->element_verticeses[ielement];
        Vertices_3D region = element_vertices.get_overlap_vertices(
                                this->domain_verticeses[ielement].local_vertices);
        band_energy += this->eigen_solvers[ielement].evaluate_band_energy(smearing,
                                    chemical_potential, smearing_coef, region);
    }
    // allreduce band energy from the rank 0 of element comms
    if (likely(this->domain_vertices.get_active_comm_b() == 0 && this->domain_vertices.is_active)) {
        MPI_Allreduce(MPI_IN_PLACE, &band_energy, 1, Linalg::get_mpi_datatype<T>(),
                      MPI_SUM, this->domain_vertices.get_domain_3d_comm());
    }
    return band_energy;
}

template<typename T>
T Xlsdft<T>::evaluate_entropy_energy(const Smearing& smearing, const T& chemical_potential, const T& smearing_coef) const {
    T entropy_energy = (T)0.0;
    for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
        if (this->domain_verticeses[ielement].shared_vertices.get_size() == 0) continue;
        const Vertices_3D& element_vertices = this->element_verticeses[ielement];
        Vertices_3D region = element_vertices.get_overlap_vertices(
                                this->domain_verticeses[ielement].local_vertices);
        entropy_energy += this->eigen_solvers[ielement].evaluate_entropy_energy(smearing,
                                    chemical_potential, smearing_coef, region);
    }
    // allreduce band energy from the rank 0 of element comms
    if (likely(this->domain_vertices.get_active_comm_b() == 0 && this->domain_vertices.is_active)) {
        MPI_Allreduce(MPI_IN_PLACE, &entropy_energy, 1, Linalg::get_mpi_datatype<T>(),
                      MPI_SUM, this->domain_vertices.get_domain_3d_comm());
    }
    return entropy_energy;
}

template<typename T>
const Vertices_3D Xlsdft<T>::get_atoms_region(const Vertices_3D& element_vertices) const {
    Vertices_3D atoms_region = element_vertices;
    const Vertices_3D& unitcell_3D_shared_vertices = this->domain_vertices.get_3D_shared_vertices();
    const Mesh_control& unitcell_mesh_control = this->mesh_control;
    if ((unitcell_mesh_control.is_periodic[0]
     && atoms_region.get_ie() <= unitcell_3D_shared_vertices.get_ie())
     || (!unitcell_mesh_control.is_periodic[0]
     && atoms_region.get_ie() < unitcell_3D_shared_vertices.get_ie())) {
        atoms_region.ni += 1;
    }
    if ((unitcell_mesh_control.is_periodic[1]
     && atoms_region.get_je() <= unitcell_3D_shared_vertices.get_je())
     || (!unitcell_mesh_control.is_periodic[1]
     && atoms_region.get_je() < unitcell_3D_shared_vertices.get_je())) {
        atoms_region.nj += 1;
    }
    if ((unitcell_mesh_control.is_periodic[2]
     && atoms_region.get_ke() <= unitcell_3D_shared_vertices.get_ke())
     || (!unitcell_mesh_control.is_periodic[2]
     && atoms_region.get_ke() < unitcell_3D_shared_vertices.get_ke())) {
        atoms_region.nk += 1;
    }
    return atoms_region;
}

template<typename T>
void Xlsdft<T>::cal_nonlocal_forces(Array_2D<T>& nonlocal_forces, const T& chemical_potential, const Smearing& smearing,
                                    const Spin& spin, const std::vector<Psp8_file>& psp8_files) const {
    for (uint i_element = 0; i_element < this->element_verticeses.size(); i_element++) {
        const Vertices_3D& element_vertices = this->element_verticeses[i_element];
        const Vertices_3D atomic_vertices = this->get_atoms_region(element_vertices);
        if (element_vertices.get_size() == 0) continue;
        Geometry geometry_temp(this->ex_geometries[i_element]);
        std::vector<Atom>().swap(geometry_temp.atoms);
        for (std::vector<Atom>::const_iterator it_atom = this->geometry.atoms.begin();
                                        it_atom != this->geometry.atoms.end(); ++it_atom) {
            it_atom->generate_valid_images(geometry_temp.atoms, this->geometry.cell_type,
                                            this->domain_vertices.get_3D_shared_vertices(),
                                            atomic_vertices, this->mesh_control.is_periodic,
                                            this->mesh_control.delta_x, this->mesh_control.delta_y,
                                            this->mesh_control.delta_z,
                                            this->geometry.a, this->geometry.b, this->geometry.c,
                                            0.0, 0.0, 0.0);
        }
        Array_0D<int> atom_indexes_temp(geometry_temp.atoms.size());
        for (uint iatom = 0; iatom < geometry_temp.atoms.size(); iatom++) {
            atom_indexes_temp[iatom] = geometry_temp.atoms[iatom].index;
            geometry_temp.atoms[iatom].index = iatom;
        }
        Domain_parallel_vertices_3D domain_vertices_temp;
        domain_vertices_temp.set_shared_vertices(this->shared_verticeses[i_element]);
        domain_vertices_temp.set_chunk_size_i(this->domain_verticeses[i_element].chunk_size_i);
        domain_vertices_temp.set_chunk_size_j(this->domain_verticeses[i_element].chunk_size_j);
        domain_vertices_temp.set_chunk_size_k(this->domain_verticeses[i_element].chunk_size_k);
        domain_vertices_temp.init(this->shared_verticeses[i_element], this->domain_verticeses[i_element].comm);
        // domain_vertices_temp.set_local_vertices(this->domain_verticeses[i_element].get_3D_local_vertices());
        Effective_potential_nloc<T> effective_potential_nloc_temp(geometry_temp, this->mesh_controls[i_element], psp8_files, domain_vertices_temp);
        effective_potential_nloc_temp.init(false);
        for (uint iatom = 0; iatom < geometry_temp.atoms.size(); iatom++) {
            effective_potential_nloc_temp.nloc_projectors[iatom].is_real = true;
        }
        Array_2D<T> nonlocal_forces_temp(Vertices_2D(3, geometry_temp.atoms.size()), 0);
        #if (defined(LOW_MEMORY))
        this->eigen_solvers[i_element].cal_nonlocal_forces(nonlocal_forces_temp, chemical_potential, smearing,
                                                       spin, effective_potential_nloc_temp,
                                                       this->eigen_vectorses_lp[i_element]);
        #else
        this->eigen_solvers[i_element].cal_nonlocal_forces(nonlocal_forces_temp, chemical_potential, smearing,
                                                       spin, effective_potential_nloc_temp);
        #endif
        for (uint iatom = 0; iatom < geometry_temp.atoms.size(); iatom++) {
            nonlocal_forces[atom_indexes_temp[iatom] * 3    ] = nonlocal_forces_temp[iatom * 3    ];
            nonlocal_forces[atom_indexes_temp[iatom] * 3 + 1] = nonlocal_forces_temp[iatom * 3 + 1];
            nonlocal_forces[atom_indexes_temp[iatom] * 3 + 2] = nonlocal_forces_temp[iatom * 3 + 2];
        }
    }
    // allreduce band energy from the rank 0 of element comms
    if (likely(this->domain_vertices.get_active_comm_b() == 0 && this->domain_vertices.is_active)) {
        MPI_Allreduce(MPI_IN_PLACE, nonlocal_forces.data, nonlocal_forces.length, Linalg::get_mpi_datatype<T>(),
                      MPI_SUM, this->domain_vertices.get_domain_3d_comm());
    }
    return;
}

template<typename T>
void Xlsdft<T>::print_eigens(const T& chemical_potential, const Smearing& smearing,
                                   const Spin& spin, const std::string& fname) const {
    const uint nspin = spin.generate_nspin();
    if (nspin != 1) {
        assert(false);
    }
    uint nstates = 0;
    for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
        if (this->domain_verticeses[ielement].shared_vertices.get_size() == 0) continue;
        nstates += this->eigen_solvers[ielement].eigen_solver_control.nstates;
    }
    Array_0D<T> lambdas(nstates, 0);
    Array_0D<T> fracs(nstates, 0);

    uint state_offset = 0;
    for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
        if (this->domain_verticeses[ielement].shared_vertices.get_size() == 0) continue;
        const Vertices_3D& element_vertices = this->element_verticeses[ielement];
        Vertices_3D region = element_vertices.get_overlap_vertices(
                                this->domain_verticeses[ielement].local_vertices);
        this->eigen_solvers[ielement].get_region_fracs(region, fracs.data + state_offset);
        this->eigen_solvers[ielement].get_eigens(lambdas.data + state_offset);
        state_offset += this->eigen_solvers[ielement].eigen_solver_control.nstates;
    }
    if (this->domain_vertices.is_active) {
        // MPI_Allreduce(MPI_IN_PLACE, fracs.data, nstates, Linalg::get_mpi_datatype<T>(),
        //               MPI_SUM, this->domain_vertices.get_band_comm());
        if (likely(this->domain_vertices.get_active_comm_b() == 0)) {
            int ncomm = this->domain_vertices.get_domain_3d_comm_size();
            Array_0D<int> globel_nstates(ncomm, 0);
            MPI_Gather(&nstates, 1, MPI_UNSIGNED,
                       globel_nstates.data, 1, MPI_UNSIGNED, 0,
                       this->domain_vertices.get_domain_3d_comm());
            int total_nstates = 0;
            Array_0D<int> nstates_offset(ncomm + 1, 0);
            for (int i = 0; i < ncomm; i++) {
                total_nstates += globel_nstates[i];
                nstates_offset[i + 1] = total_nstates;
            }

            Array_0D<T> globel_lambdas(total_nstates, 0);
            Array_0D<T> globel_fracs(total_nstates, 0);
            Array_0D<T> globel_occs(total_nstates, 0);
            MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
            MPI_Gatherv(lambdas.data, nstates, mpi_datatype,
                        globel_lambdas.data, globel_nstates.data, nstates_offset.data,
                        mpi_datatype, 0, this->domain_vertices.get_domain_3d_comm());
            MPI_Gatherv(fracs.data, nstates, mpi_datatype,
                        globel_fracs.data, globel_nstates.data, nstates_offset.data,
                        mpi_datatype, 0, this->domain_vertices.get_domain_3d_comm());
            if (this->domain_vertices.get_comm_rank()==0) {
                Smearing_method::smear(globel_lambdas.data, globel_occs.data, chemical_potential,
                                        smearing, total_nstates);
                Linalg::scalar_product_general(globel_occs.data, T(2.0), total_nstates);
                std::ofstream outfile(fname);
                outfile << "Final eigenvalues (Ha) and occupation numbers, with chemical potential "
                            << std::fixed << std::setprecision(12) << chemical_potential << std::endl;
                outfile << std::endl;
                outfile << std::fixed << std::setprecision(12) << "kred #" << 1 << " = ("
                        << 0.0 << ","
                        << 0.0 << ","
                        << 0.0 << ")"
                        << std::endl;
                outfile << "weight = " << 1.0
                        << std::endl;
                outfile << std::setw(10) << "n"
                        << std::setw(20) << "eigval"
                        << std::setw(20) << "frac"
                        << std::setw(20) << "occ" << std::endl;
                for (int ib = 0; ib < total_nstates; ib++) {
                    outfile << std::setw(10) << ib + 1
                            << std::setw(20) << std::setprecision(12) << globel_lambdas[ib]
                            << std::setw(20) << std::setprecision(12) << globel_fracs[ib]
                            << std::setw(20) << std::setprecision(12) << globel_occs[ib] << std::endl;
                }
                outfile.close();
            }
        }
    }
    return;
}

template<typename T>
void Xlsdft<T>::print_eigens_divided(const T& chemical_potential, const Smearing& smearing,
                                   const Spin& spin, const std::string& dir_name) const {
    const uint nspin = spin.generate_nspin();
    if (nspin != 1) {
        assert(false);
    }
    const uint comm_i = this->domain_vertices.get_active_comm_i();
    const uint comm_j = this->domain_vertices.get_active_comm_j();
    const uint comm_k = this->domain_vertices.get_active_comm_k();
    for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
        if (this->domain_verticeses[ielement].shared_vertices.get_size() == 0) continue;
        std::string filename = "EIGENS_" + std::to_string(comm_i)
                                   + "_" + std::to_string(comm_j)
                                   + "_" + std::to_string(comm_k)
                                   + "_" + std::to_string(ielement);
        std::filesystem::path full_path = std::filesystem::path(dir_name) / filename;
        this->eigen_solvers[ielement].print_eigens(chemical_potential, smearing, spin,
            this->element_verticeses[ielement], full_path);
    }
    return;
}

template<typename T>
void Xlsdft<T>::print_pdos(const T& chemical_potential, const Spin& spin, const std::string& dir_name) const {
    const uint nspin = spin.generate_nspin();
    assert(nspin == 1);
    for (uint i_element = 0; i_element < this->element_verticeses.size(); i_element++) {
        const Vertices_3D& element_vertices = this->element_verticeses[i_element];
        const Vertices_3D atomic_vertices = this->get_atoms_region(element_vertices);
        if (element_vertices.get_size() == 0) continue;
        std::vector<Atom> atoms;
        for (std::vector<Atom>::const_iterator it_atom = this->geometry.atoms.begin();
                                        it_atom != this->geometry.atoms.end(); ++it_atom) {
            it_atom->generate_valid_images(atoms, this->geometry.cell_type,
                                            this->domain_vertices.get_3D_shared_vertices(),
                                            atomic_vertices, this->mesh_control.is_periodic,
                                            this->mesh_control.delta_x, this->mesh_control.delta_y,
                                            this->mesh_control.delta_z,
                                            this->geometry.a, this->geometry.b, this->geometry.c,
                                            0.0, 0.0, 0.0);
        }
        #if (defined(LOW_MEMORY))
        Array_4D<T> eigen_vectors(this->eigen_vectorses_lp[i_element].as_type(this->eigen_solvers[i_element].eigen_vectors.data));
        const std::vector<Array_2D<T>> pdos_atoms = this->eigen_solvers[i_element].cal_pdos(atoms, eigen_vectors);
        #else
        const std::vector<Array_2D<T>> pdos_atoms = this->eigen_solvers[i_element].cal_pdos(atoms, this->eigen_solvers[i_element].eigen_vectors);
        #endif
        if (this->eigen_solvers[i_element].domain_vertices.get_comm_rank() == 0) {
            for (uint i_atom = 0; i_atom < atoms.size(); i_atom++) {
                const Atom& atom = atoms[i_atom];
                const Psp8_file& psp8_file = this->eigen_solvers[i_element].psp8_files[atom.type];
                if (!psp8_file.if_has_upf_file) continue;
                std::string filename = "atom_" + std::to_string(atom.index + 1);
                std::filesystem::path full_path = std::filesystem::path(dir_name) / filename;
                std::ofstream fout(full_path);
                if (!fout.is_open()) {
                    std::cerr << "Error: Cannot create file " << full_path << std::endl;
                } else {
                    Atom_method::print_pdos<T>(fout, atom, nspin, pdos_atoms[i_atom],
                                                this->eigen_solvers[i_element].eigen_values,
                                                psp8_file, chemical_potential, true);
                    fout.close();
                }
            }
        }
    }
    return;
}

// template<typename T>
// void Xlsdft<T>::print_timer_statistics(const bool if_print, std::ostream& output) const {
//     (void) if_print;
//     #if (defined(LOW_MEMORY))
//     constexpr int nitems = 11 + 3 + 7;
//     #else
//     constexpr int nitems = 11 + 2 + 6;
//     #endif
//     const MPI_Comm domain_comm = this->domain_vertices.get_domain_3d_comm();
//     if (this->domain_vertices.get_active_comm_b() == 0 && this->domain_vertices.is_active) {
//         std::vector<double> t_locals(nitems, 0.0);
//         std::vector<double> max_values(nitems);
//         std::vector<double> min_values(nitems);
//         std::vector<double> mean_values(nitems);
//         std::vector<double> stddev_values(nitems);
//         std::vector<double> reciprocal_mean_values(nitems);
//         const uint nelement = this->local_element_num;
//         for (uint ielement = 0; ielement < nelement; ielement++) {
//             uint count = 0;
//             //chefsi
//             t_locals[count++] += this->eigen_solvers[ielement].chefsi.chefsi_timer.chefsi.time_cost_millisecond();
//             t_locals[count++] += this->eigen_solvers[ielement].chefsi.chefsi_timer.lanczos.time_cost_millisecond();
//             t_locals[count++] += this->eigen_solvers[ielement].chefsi.chefsi_timer.filter.time_cost_millisecond();
//             t_locals[count++] += this->eigen_solvers[ielement].chefsi.chefsi_timer.filter_copy.time_cost_millisecond();
//             t_locals[count++] += this->eigen_solvers[ielement].chefsi.chefsi_timer.filter_product.time_cost_millisecond();
//             t_locals[count++] += this->eigen_solvers[ielement].chefsi.chefsi_timer.filter_lap.time_cost_millisecond();
//             t_locals[count++] += this->eigen_solvers[ielement].chefsi.chefsi_timer.filter_nloc.time_cost_millisecond();
//             t_locals[count++] += this->eigen_solvers[ielement].chefsi.chefsi_timer.H_psi.time_cost_millisecond();
//             t_locals[count++] += this->eigen_solvers[ielement].chefsi.chefsi_timer.projection.time_cost_millisecond();
//             t_locals[count++] += this->eigen_solvers[ielement].chefsi.chefsi_timer.diagonalization.time_cost_millisecond();
//             t_locals[count++] += this->eigen_solvers[ielement].chefsi.chefsi_timer.rotation.time_cost_millisecond();
//             //eigen_solver
//             t_locals[count++] += this->eigen_solvers[ielement].eigen_solver_timer.eigen_solver.time_cost_millisecond();
//             #if (defined(LOW_MEMORY))
//             t_locals[count++] += this->eigen_solvers[ielement].eigen_solver_timer.low_memory_chi.time_cost_millisecond();
//             #endif
//             t_locals[count++] += this->eigen_solvers[ielement].eigen_solver_timer.eigen_solver_kernel.time_cost_millisecond();
//         }
//         #if (defined(LOW_MEMORY))
//         int count = 11 + 3;
//         #else
//         int count = 11 + 2;
//         #endif
//         //xlsdft
//         t_locals[count++] += this->xlsdft_timer.xlsdft.time_cost_millisecond();
//         t_locals[count++] += this->xlsdft_timer.xlsdft_copy_veff.time_cost_millisecond();
//         #if (defined(LOW_MEMORY))
//         t_locals[count++] += this->xlsdft_timer.xlsdft_low_memory_psi.time_cost_millisecond();
//         #endif
//         t_locals[count++] += this->xlsdft_timer.xlsdft_eigen_solver.time_cost_millisecond();
//         t_locals[count++] += this->xlsdft_timer.xlsdft_barrier1.time_cost_millisecond();
//         t_locals[count++] += this->xlsdft_timer.xlsdft_barrier2.time_cost_millisecond();
//         t_locals[count++] += this->xlsdft_timer.xlsdft_barrier3.time_cost_millisecond();

//         void calc_mpi_statistics(double const* const local_values,
//                             const int nitems,
//                             const MPI_Comm comm,
//                             double const* max_values,
//                             double const* min_values,
//                             double const* mean_values,
//                             double const* stddev_values,
//                             double const* reciprocal_mean_values);
//         Tools::calc_mpi_statistics(t_locals.data(), nitems, domain_comm, max_values.data(),
//                                     min_values.data(), mean_values.data(), stddev_values.data(),
//                                     reciprocal_mean_values.data());
//     }
//     return;
// }

#if defined(ENABLE_TIMER)
template<typename T>
void Xlsdft<T>::print_timer_statistics(const bool if_print, std::ostream& output) const {
#if defined(LOW_MEMORY)
    constexpr int nitems = 11 + 3 + 7;
#else
    constexpr int nitems = 11 + 2 + 6;
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

        const uint nelement = this->local_element_num;

        for (uint ielement = 0; ielement < nelement; ielement++) {
            uint count = 0;

            // CheFSI
            t_locals[count++] +=
                this->eigen_solvers[ielement].chefsi.chefsi_timer.chefsi.time_cost_millisecond();

            t_locals[count++] +=
                this->eigen_solvers[ielement].chefsi.chefsi_timer.lanczos.time_cost_millisecond();

            t_locals[count++] +=
                this->eigen_solvers[ielement].chefsi.chefsi_timer.filter.time_cost_millisecond();

            t_locals[count++] +=
                this->eigen_solvers[ielement].chefsi.chefsi_timer.filter_copy.time_cost_millisecond();

            t_locals[count++] +=
                this->eigen_solvers[ielement].chefsi.chefsi_timer.filter_product.time_cost_millisecond();

            t_locals[count++] +=
                this->eigen_solvers[ielement].chefsi.chefsi_timer.filter_lap.time_cost_millisecond();

            t_locals[count++] +=
                this->eigen_solvers[ielement].chefsi.chefsi_timer.filter_nloc.time_cost_millisecond();

            t_locals[count++] +=
                this->eigen_solvers[ielement].chefsi.chefsi_timer.H_psi.time_cost_millisecond();

            t_locals[count++] +=
                this->eigen_solvers[ielement].chefsi.chefsi_timer.projection_time_cost_millisecond();

            t_locals[count++] +=
                this->eigen_solvers[ielement].chefsi.chefsi_timer.diagonalization.time_cost_millisecond();

            t_locals[count++] +=
                this->eigen_solvers[ielement].chefsi.chefsi_timer.rotation.time_cost_millisecond();

            // Eigen solver
            t_locals[count++] +=
                this->eigen_solvers[ielement].eigen_solver_timer.eigen_solver.time_cost_millisecond();

#if defined(LOW_MEMORY)
            t_locals[count++] +=
                this->eigen_solvers[ielement].eigen_solver_timer.low_memory_chi.time_cost_millisecond();
#endif

            t_locals[count++] +=
                this->eigen_solvers[ielement].eigen_solver_timer.eigen_solver_kernel.time_cost_millisecond();

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
            this->xlsdft_timer.xlsdft.time_cost_millisecond();

        t_locals[count++] +=
            this->xlsdft_timer.xlsdft_copy_veff.time_cost_millisecond();

#if defined(LOW_MEMORY)
        t_locals[count++] +=
            this->xlsdft_timer.xlsdft_low_memory_psi.time_cost_millisecond();
#endif

        t_locals[count++] +=
            this->xlsdft_timer.xlsdft_eigen_solver.time_cost_millisecond();

        t_locals[count++] +=
            this->xlsdft_timer.xlsdft_barrier1.time_cost_millisecond();

        t_locals[count++] +=
            this->xlsdft_timer.xlsdft_barrier2.time_cost_millisecond();

        t_locals[count++] +=
            this->xlsdft_timer.xlsdft_barrier3.time_cost_millisecond();

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
                "  Barrier 3"
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
#endif

template<typename T>
double Xlsdft<T>::evalutate_flops() {
    double flops = 0.0;
    for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
        flops += this->eigen_solvers[ielement].evalutate_flops();
    }
    flops *= double(this->domain_vertices.get_domain_3d_comm_size());
    return flops;
}

template<typename T>
void Xlsdft<T>::run(const Array_3D<T>& ex_effective_potentail_loc, const bool& print_flag) {
    this->run_mp(ex_effective_potentail_loc.data, ex_effective_potentail_loc.get_vertices(), print_flag);
    // #ifdef ENABLE_XLSDFT_TIMER
    //     this->xlsdft_timer.xlsdft_barrier1.start();
    // #endif //ENABLE_XLSDFT_TIMER
    // MPI_Barrier(this->domain_vertices.get_mpi_comm());
    // #ifdef ENABLE_XLSDFT_TIMER
    //     this->xlsdft_timer.xlsdft_barrier1.stop();
    // #endif //ENABLE_XLSDFT_TIMER
    // #ifdef ENABLE_XLSDFT_TIMER
    //     this->xlsdft_timer.reset();
    //     this->xlsdft_timer.xlsdft.start();
    // #endif //ENABLE_XLSDFT_TIMER
    // bool print_flag_local = this->domain_vertices.get_active_comm_b() == 0 ? true : false;
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
    //     if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
    //         std::cout << "The index " << ielement << " element of its element comm start the eigen_solver." << std::endl;
    //     }
    //     #ifdef ENABLE_XLSDFT_TIMER
    //         this->xlsdft_timer.xlsdft_copy_veff.start();
    //     #endif //ENABLE_XLSDFT_TIMER
    //     const Vertices_3D& local_vertices_3d = this->eigen_solvers[ielement].domain_vertices.get_3D_local_vertices();
    //     Array_3D<T> effective_potentail_loc(local_vertices_3d);
    //     ex_effective_potentail_loc.fill_overlap(effective_potentail_loc);
    //     #ifdef ENABLE_XLSDFT_TIMER
    //         this->xlsdft_timer.xlsdft_copy_veff.stop();
    //     #endif //ENABLE_XLSDFT_TIMER
    //     // copy data from fp16
    //     #if (defined(LOW_MEMORY))
    //     #ifdef ENABLE_XLSDFT_TIMER
    //         this->xlsdft_timer.xlsdft_low_memory_psi.start();
    //     #endif //ENABLE_XLSDFT_TIMER
    //     Array_4D<T> eigen_vectors_rv;
    //     #ifdef USE_HBM
    //         eigen_vectors_rv.reconstructor_hbm(this->eigen_vectorses_lp[ielement].get_vertices());
    //     #else
    //         eigen_vectors_rv.reconstructor(this->eigen_vectorses_lp[ielement].get_vertices());
    //     #endif
    //     #pragma omp parallel
    //     Linalg::convert_type(eigen_vectors_rv.data, this->eigen_vectorses_lp[ielement].data, eigen_vectors_rv.length);
    //     this->eigen_solvers[ielement].eigen_vectors.deepcopy(std::move(eigen_vectors_rv));
    //     // this->eigen_solvers[ielement].eigen_vectors.deepcopy(
    //     //         std::move(this->eigen_vectorses_lp[ielement].as_type(this->eigen_solvers[ielement].eigen_vectors.data)));
    //     this->eigen_vectorses_lp[ielement].destructor();
    //     #ifdef ENABLE_XLSDFT_TIMER
    //         this->xlsdft_timer.xlsdft_low_memory_psi.stop();
    //     #endif //ENABLE_XLSDFT_TIMER
    //     #endif
    //     #ifdef ENABLE_XLSDFT_TIMER
    //         this->xlsdft_timer.xlsdft_eigen_solver.start();
    //     #endif //ENABLE_XLSDFT_TIMER
    //     //run eigen_solver
    //     T vloc_sum = Linalg::vector_sum(effective_potentail_loc.data, effective_potentail_loc.length, MPI_COMM_NULL);
    //     this->U0s[ielement] = vloc_sum/T(effective_potentail_loc.length);
    //     this->eigen_solvers[ielement].run(effective_potentail_loc, print_flag && print_flag_local);
    //     #ifdef ENABLE_XLSDFT_TIMER
    //         this->xlsdft_timer.xlsdft_eigen_solver.stop();
    //     #endif //ENABLE_XLSDFT_TIMER
    //     // store data to fp16
    //     #if (defined(LOW_MEMORY))
    //     #ifdef ENABLE_XLSDFT_TIMER
    //         this->xlsdft_timer.xlsdft_low_memory_psi.start();
    //     #endif //ENABLE_XLSDFT_TIMER
    //     const Vertices_4D& eigen_region = this->eigen_solvers[ielement].domain_vertices.local_vertices;
    //     const Vertices_3D& element_vertices = this->element_verticeses[ielement];
    //     Vertices_4D region(element_vertices.get_overlap_vertices(eigen_region),
    //                             eigen_region.bs,
    //                             eigen_region.get_be());
    //     Array_4D<T> region_eigen_vectors_temp(region);
    //     this->eigen_vectorses_lp[ielement].reconstructor(this->eigen_solvers[ielement].eigen_vectors.get_vertices());
    //     #pragma omp parallel
    //     {
    //         this->eigen_solvers[ielement].eigen_vectors.sub_arr(region_eigen_vectors_temp);
    //         Linalg::convert_type(this->eigen_vectorses_lp[ielement].data, this->eigen_solvers[ielement].eigen_vectors.data, this->eigen_vectorses_lp[ielement].length);
    //     }
    //     // region_eigen_vectors_temp.deepcopy(
    //             // std::move(this->eigen_solvers[ielement].eigen_vectors.sub_arr(region).as_type(region_eigen_vectors_temp.data)));
    //     // this->eigen_vectorses_lp[ielement].deepcopy(
    //     //         std::move(this->eigen_solvers[ielement].eigen_vectors.as_type(this->eigen_vectorses_lp[ielement].data)));
    //     // this->eigen_solvers[ielement].eigen_vectors.destructor();
    //     #ifdef USE_HBM
    //         this->eigen_solvers[ielement].eigen_vectors.destructor_hbm();
    //     #endif
    //     this->eigen_solvers[ielement].eigen_vectors.deepcopy(std::move(region_eigen_vectors_temp));
    //     #ifdef ENABLE_XLSDFT_TIMER
    //         this->xlsdft_timer.xlsdft_low_memory_psi.stop();
    //     #endif //ENABLE_XLSDFT_TIMER
    //     #endif
    // }
    // #ifdef ENABLE_XLSDFT_TIMER
    //     this->xlsdft_timer.xlsdft_barrier2.start();
    // #endif //ENABLE_XLSDFT_TIMER
    // MPI_Barrier(this->domain_vertices.get_mpi_comm());
    // #ifdef ENABLE_XLSDFT_TIMER
    //     this->xlsdft_timer.xlsdft_barrier2.stop();
    // #endif //ENABLE_XLSDFT_TIMER
    // #ifdef ENABLE_XLSDFT_TIMER
    //     this->xlsdft_timer.xlsdft_barrier3.start();
    // #endif //ENABLE_XLSDFT_TIMER
    // MPI_Barrier(this->domain_vertices.get_mpi_comm());
    // #ifdef ENABLE_XLSDFT_TIMER
    //     this->xlsdft_timer.xlsdft_barrier3.stop();
    // #endif //ENABLE_XLSDFT_TIMER
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
    //     std::cout << "The Xlsdft run took " << Tools::time_cost(begin, end) << "." << std::endl;
    // }
    // #ifdef ENABLE_XLSDFT_TIMER
    //     this->xlsdft_timer.xlsdft.stop();
    //     if (this->domain_vertices.get_comm_rank() == 0 && print_flag) this->xlsdft_timer.show();
    // #endif //ENABLE_XLSDFT_TIMER
    return;
}

template<typename T>
void Xlsdft<T>::run_mp(T const* const ex_effective_potentail_loc, const Vertices_3D& ex_effective_potentail_vertices, const bool print_flag) {
    constexpr size_t GB = 1024 * 1024 * 1024 / sizeof(T);
    Memory_pool<T, Fast_memory> pool_fast(3.75 * GB);
    Memory_pool<T, Capacity_memory> pool_cap(3.75 * GB);
    this->run_mp(ex_effective_potentail_loc, ex_effective_potentail_vertices, print_flag,
                    pool_fast, pool_cap);
    // #ifdef ENABLE_XLSDFT_TIMER
    //     this->xlsdft_timer.xlsdft_barrier1.start();
    // #endif //ENABLE_XLSDFT_TIMER
    // MPI_Barrier(this->domain_vertices.get_mpi_comm());
    // #ifdef ENABLE_XLSDFT_TIMER
    //     this->xlsdft_timer.xlsdft_barrier1.stop();
    // #endif //ENABLE_XLSDFT_TIMER
    // #ifdef ENABLE_XLSDFT_TIMER
    //     this->xlsdft_timer.reset();
    //     this->xlsdft_timer.xlsdft.start();
    // #endif //ENABLE_XLSDFT_TIMER
    // bool print_flag_local = this->domain_vertices.get_active_comm_b() == 0 ? true : false;
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
    //     if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
    //         std::cout << "The index " << ielement << " element of its element comm start the eigen_solver." << std::endl;
    //     }
    //     #ifdef ENABLE_XLSDFT_TIMER
    //         this->xlsdft_timer.xlsdft_copy_veff.start();
    //     #endif //ENABLE_XLSDFT_TIMER
    //     const Vertices_3D& local_vertices_3d = this->eigen_solvers[ielement].domain_vertices.get_3D_local_vertices();
    //     Array_3D<T> effective_potentail_loc(local_vertices_3d);
    //     // ex_effective_potentail_loc.fill_overlap(effective_potentail_loc);
    //     Vertices_method::fill_region(ex_effective_potentail_loc, ex_effective_potentail_vertices,
    //                              effective_potentail_loc.data, local_vertices_3d, local_vertices_3d);
    //     #ifdef ENABLE_XLSDFT_TIMER
    //         this->xlsdft_timer.xlsdft_copy_veff.stop();
    //     #endif //ENABLE_XLSDFT_TIMER
    //     // copy data from fp16
    //     #if (defined(LOW_MEMORY))
    //     #ifdef ENABLE_XLSDFT_TIMER
    //         this->xlsdft_timer.xlsdft_low_memory_psi.start();
    //     #endif //ENABLE_XLSDFT_TIMER
    //     Array_4D<T> eigen_vectors_rv;
    //     #ifdef USE_HBM
    //         eigen_vectors_rv.reconstructor_hbm(this->eigen_vectorses_lp[ielement].get_vertices());
    //     #else
    //         eigen_vectors_rv.reconstructor(this->eigen_vectorses_lp[ielement].get_vertices());
    //     #endif
    //     #pragma omp parallel
    //     Linalg::convert_type(eigen_vectors_rv.data, this->eigen_vectorses_lp[ielement].data, eigen_vectors_rv.length);
    //     this->eigen_solvers[ielement].eigen_vectors.deepcopy(std::move(eigen_vectors_rv));
    //     // this->eigen_solvers[ielement].eigen_vectors.deepcopy(
    //     //         std::move(this->eigen_vectorses_lp[ielement].as_type(this->eigen_solvers[ielement].eigen_vectors.data)));
    //     this->eigen_vectorses_lp[ielement].destructor();
    //     #ifdef ENABLE_XLSDFT_TIMER
    //         this->xlsdft_timer.xlsdft_low_memory_psi.stop();
    //     #endif //ENABLE_XLSDFT_TIMER
    //     #endif
    //     #ifdef ENABLE_XLSDFT_TIMER
    //         this->xlsdft_timer.xlsdft_eigen_solver.start();
    //     #endif //ENABLE_XLSDFT_TIMER
    //     //run eigen_solver
    //     T vloc_sum = Linalg::vector_sum(effective_potentail_loc.data, effective_potentail_loc.length, MPI_COMM_NULL);
    //     this->U0s[ielement] = vloc_sum/T(effective_potentail_loc.length);
    //     this->eigen_solvers[ielement].run(effective_potentail_loc, print_flag && print_flag_local);
    //     #ifdef ENABLE_XLSDFT_TIMER
    //         this->xlsdft_timer.xlsdft_eigen_solver.stop();
    //     #endif //ENABLE_XLSDFT_TIMER
    //     // store data to fp16
    //     #if (defined(LOW_MEMORY))
    //     #ifdef ENABLE_XLSDFT_TIMER
    //         this->xlsdft_timer.xlsdft_low_memory_psi.start();
    //     #endif //ENABLE_XLSDFT_TIMER
    //     const Vertices_4D& eigen_region = this->eigen_solvers[ielement].domain_vertices.local_vertices;
    //     const Vertices_3D& element_vertices = this->element_verticeses[ielement];
    //     Vertices_4D region(element_vertices.get_overlap_vertices(eigen_region),
    //                             eigen_region.bs,
    //                             eigen_region.get_be());
    //     Array_4D<T> region_eigen_vectors_temp(region);
    //     this->eigen_vectorses_lp[ielement].reconstructor(this->eigen_solvers[ielement].eigen_vectors.get_vertices());
    //     #pragma omp parallel
    //     {
    //         this->eigen_solvers[ielement].eigen_vectors.sub_arr(region_eigen_vectors_temp);
    //         Linalg::convert_type(this->eigen_vectorses_lp[ielement].data, this->eigen_solvers[ielement].eigen_vectors.data, this->eigen_vectorses_lp[ielement].length);
    //     }
    //     // region_eigen_vectors_temp.deepcopy(
    //             // std::move(this->eigen_solvers[ielement].eigen_vectors.sub_arr(region).as_type(region_eigen_vectors_temp.data)));
    //     // this->eigen_vectorses_lp[ielement].deepcopy(
    //     //         std::move(this->eigen_solvers[ielement].eigen_vectors.as_type(this->eigen_vectorses_lp[ielement].data)));
    //     // this->eigen_solvers[ielement].eigen_vectors.destructor();
    //     #ifdef USE_HBM
    //         this->eigen_solvers[ielement].eigen_vectors.destructor_hbm();
    //     #endif
    //     this->eigen_solvers[ielement].eigen_vectors.deepcopy(std::move(region_eigen_vectors_temp));
    //     #ifdef ENABLE_XLSDFT_TIMER
    //         this->xlsdft_timer.xlsdft_low_memory_psi.stop();
    //     #endif //ENABLE_XLSDFT_TIMER
    //     #endif
    // }
    // #ifdef ENABLE_XLSDFT_TIMER
    //     this->xlsdft_timer.xlsdft_barrier2.start();
    // #endif //ENABLE_XLSDFT_TIMER
    // MPI_Barrier(this->domain_vertices.get_mpi_comm());
    // #ifdef ENABLE_XLSDFT_TIMER
    //     this->xlsdft_timer.xlsdft_barrier2.stop();
    // #endif //ENABLE_XLSDFT_TIMER
    // #ifdef ENABLE_XLSDFT_TIMER
    //     this->xlsdft_timer.xlsdft_barrier3.start();
    // #endif //ENABLE_XLSDFT_TIMER
    // MPI_Barrier(this->domain_vertices.get_mpi_comm());
    // #ifdef ENABLE_XLSDFT_TIMER
    //     this->xlsdft_timer.xlsdft_barrier3.stop();
    // #endif //ENABLE_XLSDFT_TIMER
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
    //     std::cout << "The Xlsdft run took " << Tools::time_cost(begin, end) << "." << std::endl;
    // }
    // #ifdef ENABLE_XLSDFT_TIMER
    //     this->xlsdft_timer.xlsdft.stop();
    //     if (this->domain_vertices.get_comm_rank() == 0 && print_flag) this->xlsdft_timer.show();
    // #endif //ENABLE_XLSDFT_TIMER
    return;
}

template<typename T>
void Xlsdft<T>::reserve_retained_packed_pool(Memory_pool<T, Fast_memory>& pool_fast) {
    if constexpr (std::is_same_v<T, double>) {
        if (this->retained_pool_reserved_ || this->local_element_num == 0) return;
        size_t max_wf_elems = 0;
        uint representative = 0;
        std::set<std::tuple<size_t, size_t, size_t, size_t>> shapes;
        for (uint e = 0; e < this->local_element_num; ++e) {
            const auto& v = this->eigen_solvers[e].domain_vertices.get_3D_local_vertices();
            const size_t nb = this->eigen_solvers[e].domain_vertices.get_4D_local_vertices().nb;
            shapes.emplace(v.ni, v.nj, v.nk, nb);
            const size_t n = chefsi_layout::wf_stride(v.get_size(), nb);
            if (n > max_wf_elems) { max_wf_elems = n; representative = e; }
        }
        const char* mode = std::getenv("XLSDFT_TRIPLE_LOTTERY");
        assert(mode == nullptr || std::strcmp(mode, "0") == 0 ||
               std::strcmp(mode, "1") == 0 || std::strcmp(mode, "fixed") == 0);
        const bool enable = mode == nullptr || std::strcmp(mode, "0") != 0;
        if (enable) {
            this->triple_panels_.allocate(pool_fast, max_wf_elems);
            pool_fast.set_persistent_floor(pool_fast.mark());
            const auto& solver = this->eigen_solvers[representative];
            triple_head::tune(this->triple_panels_, solver.domain_vertices.get_3D_local_vertices(),
                              solver.domain_vertices.get_4D_local_vertices().nb, solver.stencil,
                              pool_fast, this->domain_vertices.get_comm_rank(),
                              mode == nullptr || std::strcmp(mode, "fixed") != 0, shapes.size());
            this->retained_packed_psi_ = this->triple_panels_.head[0];
            this->wf_scratch_panel_ = this->triple_panels_.head[1];
        } else {
            this->retained_packed_psi_ = pool_fast.allocate(max_wf_elems);
            this->wf_scratch_panel_ = pool_fast.allocate(max_wf_elems);
            pool_fast.set_persistent_floor(pool_fast.mark());
        }
        this->packed_live_ = this->retained_packed_psi_;
        this->packed_alt_ = this->wf_scratch_panel_;
        this->retained_pool_reserved_ = true;
        assert(pool_fast.mark() >= pool_fast.persistent_floor());
        assert(pool_fast.persistent_floor() > 0);
    }
}

template<typename T>
void Xlsdft<T>::run_mp(T const* const ex_effective_potentail_loc,
                      const Vertices_3D& ex_effective_potentail_vertices,
                      const bool print_flag,
                      Memory_pool<T, Fast_memory>& pool_fast,
                      Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    #ifdef ENABLE_XLSDFT_TIMER
        this->xlsdft_timer.xlsdft_barrier1.start();
    #endif //ENABLE_XLSDFT_TIMER
    MPI_Barrier(this->domain_vertices.get_mpi_comm());
    #ifdef ENABLE_XLSDFT_TIMER
        this->xlsdft_timer.xlsdft_barrier1.stop();
    #endif //ENABLE_XLSDFT_TIMER
    #ifdef ENABLE_XLSDFT_TIMER
        this->xlsdft_timer.reset();
        this->xlsdft_timer.xlsdft.start();
    #endif //ENABLE_XLSDFT_TIMER
    bool print_flag_local = this->domain_vertices.get_active_comm_b() == 0 ? true : false;
    const bool nchi_diag = xlsdft_nchi_diag_enabled();
    size_t rank_nchi_sum = 0;
    size_t rank_projector_bytes_sum = 0;
    size_t rank_projector_bytes_max = 0;
    double rank_chi_build_ms_sum = 0.0;
    double rank_filter_nloc_ms_sum = 0.0;
    double rank_chefsi_ms_sum = 0.0;
    for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
        Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast2(pool_fast);
        Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap2(pool_cap);
        if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
            std::cout << "The index " << ielement << " element of its element comm start the eigen_solver." << std::endl;
        }
        
        const Vertices_3D& local_vertices_3d = this->eigen_solvers[ielement].domain_vertices.get_3D_local_vertices();
        const uint K = local_vertices_3d.get_size();
        const uint nb = this->eigen_solvers[ielement].domain_vertices.get_4D_local_vertices().nb;
        bool run_legacy_path = true;
        if constexpr (std::is_same_v<T, double>) {
            if (std::getenv("CHEFSI_USE_OPT") != nullptr) {
            run_legacy_path = false;
            assert(this->retained_pool_reserved_);
            assert(pool_fast.persistent_floor() > 0);
            assert(pool_fast.mark() >= pool_fast.persistent_floor());

            #if (defined(LOW_MEMORY))
            const Vertices_4D vertices_eigen_vectors = this->eigen_vectorses_lp[ielement].get_vertices();
            #else
            const Vertices_4D vertices_eigen_vectors = this->domain_verticeses[ielement].get_4D_local_vertices();
            #endif
            const uint eigen_vector_length = vertices_eigen_vectors.get_size();
            const uint nat_elems = K * nb;
            assert(nat_elems <= eigen_vector_length);

            double* packed_in = this->packed_live_;
            double* packed_out = this->packed_alt_;
            // Reuse packed_alt_ as natural scratch: tile input (LOW_MEMORY) and untile
            // output after swap (wf_stride >= K*nb).

            const bool use_retained_direct =
                this->retained_packed_initialized_ && ielement == 0 &&
                this->local_element_num == 1;
            if (!use_retained_direct) {
                #if (defined(LOW_MEMORY))
                #ifdef ENABLE_XLSDFT_TIMER
                    this->xlsdft_timer.xlsdft_low_memory_psi.start();
                #endif //ENABLE_XLSDFT_TIMER
                #pragma omp parallel
                Linalg::convert_type(packed_out,
                                     this->eigen_vectorses_lp[ielement].data, nat_elems);
                #ifdef ENABLE_XLSDFT_TIMER
                    this->xlsdft_timer.xlsdft_low_memory_psi.stop();
                #endif //ENABLE_XLSDFT_TIMER
                chefsi_layout::tile_16(nb, K, packed_in, packed_out);
                #else
                chefsi_layout::tile_16(nb, K, packed_in,
                                       this->eigen_solvers[ielement].eigen_vectors.data);
                #endif
                this->retained_packed_initialized_ = true;
            }

            #ifdef ENABLE_XLSDFT_TIMER
                this->xlsdft_timer.xlsdft_copy_veff.start();
            #endif //ENABLE_XLSDFT_TIMER
            const uint effective_potentail_loc_length = K;
            double* effective_potentail_loc =
                reinterpret_cast<double*>(pool_fast.allocate(effective_potentail_loc_length));
            Vertices_method::fill_region(ex_effective_potentail_loc, ex_effective_potentail_vertices,
                                         effective_potentail_loc, local_vertices_3d, local_vertices_3d);
            double vloc_sum = Linalg::vector_sum(effective_potentail_loc, effective_potentail_loc_length,
                                                 MPI_COMM_NULL);
            this->U0s[ielement] = T(vloc_sum / double(effective_potentail_loc_length));
            #ifdef ENABLE_XLSDFT_TIMER
                this->xlsdft_timer.xlsdft_copy_veff.stop();
            #endif //ENABLE_XLSDFT_TIMER

            #ifdef ENABLE_XLSDFT_TIMER
                this->xlsdft_timer.xlsdft_eigen_solver.start();
            #endif //ENABLE_XLSDFT_TIMER
            nloc_fixture_dump::current_element() = static_cast<int>(ielement);
            if (this->triple_panels_.enabled) {
                auto& panels = this->triple_panels_;
                panels.check_guards();
                this->eigen_solvers[ielement].chefsi.opt_third_panel = panels.unused(packed_in, packed_out);
                this->eigen_solvers[ielement].chefsi.opt_third_panel_elems = panels.elems;
                if (!panels.usage_reported) {
                    if (triple_head::verbose_enabled()) {
                        std::fprintf(stderr, "TRIPLE_BUFFER_USE rank=%d input=%d output=%d third=%d offsets=%zu,%zu,%zu distinct=1\n",
                                     this->domain_vertices.get_comm_rank(), panels.index(packed_in), panels.index(packed_out),
                                     panels.index(this->eigen_solvers[ielement].chefsi.opt_third_panel),
                                     panels.offset[0], panels.offset[1], panels.offset[2]);
                    }
                    panels.usage_reported = true;
                }
            }
            this->eigen_solvers[ielement].run_mp_opt(packed_in, packed_out,
                                                     effective_potentail_loc,
                                                     print_flag && print_flag_local,
                                                     pool_fast, pool_cap);
            if (nchi_diag) {
                const Xlsdft_element_workload_diag& diag =
                    this->eigen_solvers[ielement].element_workload_diag;
                if (diag.valid) {
                    rank_nchi_sum += diag.nchi;
                    rank_projector_bytes_sum += diag.projector_bytes;
                    if (diag.projector_bytes > rank_projector_bytes_max) {
                        rank_projector_bytes_max = diag.projector_bytes;
                    }
                    rank_chi_build_ms_sum += diag.chi_build_ms;
                    rank_filter_nloc_ms_sum += diag.filter_nloc_ms;
                    rank_chefsi_ms_sum += diag.chefsi_ms;
                    const int rank = this->domain_vertices.get_comm_rank();
                    std::fprintf(stderr,
                                 "NCHI rank=%d elem=%u nchi=%zu nproj=%zu bytes=%zu "
                                 "chi_ms=%.3f filter_nloc_ms=%.3f chefsi_ms=%.3f eigen_ms=%.3f\n",
                                 rank, ielement, diag.nchi, diag.n_projectors,
                                 diag.projector_bytes, diag.chi_build_ms,
                                 diag.filter_nloc_ms, diag.chefsi_ms, diag.eigen_ms);
                }
            }
            #ifdef ENABLE_XLSDFT_TIMER
                this->xlsdft_timer.xlsdft_eigen_solver.stop();
            #endif //ENABLE_XLSDFT_TIMER

            // CheFSI returns packed_in -> SR tile_16 output (psi_buf after filter
            // swaps). packed_out may point at freed pool scratch (psi); use the
            // retained panel that is not the tile_16 output for natural untile.
            this->packed_live_ = packed_in;
            if (this->triple_panels_.enabled) {
                this->triple_panels_.check_guards();
                this->packed_alt_ = this->triple_panels_.alternate(packed_in);
            } else {
                this->packed_alt_ =
                    (packed_in == this->retained_packed_psi_) ? this->wf_scratch_panel_
                                                             : this->retained_packed_psi_;
            }
            double* const tile16_out = this->packed_live_;
            double* const natural_scratch = this->packed_alt_;

            #if (defined(LOW_MEMORY))
            #ifdef ENABLE_XLSDFT_TIMER
                this->xlsdft_timer.xlsdft_low_memory_psi.start();
            #endif //ENABLE_XLSDFT_TIMER
            const Vertices_4D& eigen_region = this->eigen_solvers[ielement].domain_vertices.local_vertices;
            const Vertices_3D& element_vertices = this->element_verticeses[ielement];
            Vertices_4D region(element_vertices.get_overlap_vertices(eigen_region),
                               eigen_region.bs, eigen_region.get_be());
            chefsi_layout::untile_16_production(nb, K, natural_scratch, tile16_out);
            #pragma omp parallel
            Vertices_method::fill_region(natural_scratch, vertices_eigen_vectors,
                                         this->eigen_solvers[ielement].eigen_vectors.data,
                                         region, region);
            #pragma omp parallel
            Linalg::convert_type(this->eigen_vectorses_lp[ielement].data, natural_scratch,
                                 eigen_vector_length);
            #ifdef ENABLE_XLSDFT_TIMER
                this->xlsdft_timer.xlsdft_low_memory_psi.stop();
            #endif //ENABLE_XLSDFT_TIMER
            #else
            chefsi_layout::untile_16_production(nb, K, natural_scratch, tile16_out);
            #pragma omp parallel
            Linalg::set_value_general(this->eigen_solvers[ielement].eigen_vectors.data,
                                      natural_scratch, nat_elems);
            #endif
            }
        }
        if (run_legacy_path) {
        // copy data from eigenvector
        #if (defined(LOW_MEMORY))
            #ifdef ENABLE_XLSDFT_TIMER
                this->xlsdft_timer.xlsdft_low_memory_psi.start();
            #endif //ENABLE_XLSDFT_TIMER
            const Vertices_4D vertices_eigen_vectors = this->eigen_vectorses_lp[ielement].get_vertices();
            const uint eigen_vector_length = vertices_eigen_vectors.get_size();
            T* eigen_vectors_in = pool_fast.allocate(eigen_vector_length);
            T* eigen_vectors_out = pool_fast.allocate(eigen_vector_length);
            #pragma omp parallel
            Linalg::convert_type(eigen_vectors_in, this->eigen_vectorses_lp[ielement].data, eigen_vector_length);
            #ifdef ENABLE_XLSDFT_TIMER
                this->xlsdft_timer.xlsdft_low_memory_psi.stop();
            #endif //ENABLE_XLSDFT_TIMER
        #else
            const Vertices_4D vertices_eigen_vectors = this->domain_verticeses[ielement].get_4D_local_vertices();
            const uint eigen_vector_length = vertices_eigen_vectors.get_size();
            T* eigen_vectors_in = pool_fast.allocate(eigen_vector_length);
            T* eigen_vectors_out = pool_fast.allocate(eigen_vector_length);
            #pragma omp parallel
            Linalg::set_value_general(eigen_vectors_in, this->eigen_solvers[ielement].eigen_vectors.data, eigen_vector_length);
        #endif

        #ifdef ENABLE_XLSDFT_TIMER
            this->xlsdft_timer.xlsdft_eigen_solver.start();
        #endif //ENABLE_XLSDFT_TIMER

        #ifdef ENABLE_XLSDFT_TIMER
            this->xlsdft_timer.xlsdft_copy_veff.start();
        #endif //ENABLE_XLSDFT_TIMER
        const uint effective_potentail_loc_length = local_vertices_3d.get_size();
        T* effective_potentail_loc = pool_fast.allocate(effective_potentail_loc_length);
        Vertices_method::fill_region(ex_effective_potentail_loc, ex_effective_potentail_vertices,
                                 effective_potentail_loc, local_vertices_3d, local_vertices_3d);
        T vloc_sum = Linalg::vector_sum(effective_potentail_loc, effective_potentail_loc_length, MPI_COMM_NULL);
        this->U0s[ielement] = vloc_sum/T(effective_potentail_loc_length);
        #ifdef ENABLE_XLSDFT_TIMER
            this->xlsdft_timer.xlsdft_copy_veff.stop();
        #endif //ENABLE_XLSDFT_TIMER

            this->eigen_solvers[ielement].run_mp(eigen_vectors_in, eigen_vectors_out,
                                                 effective_potentail_loc,
                                                 print_flag && print_flag_local, pool_fast,
                                                 pool_cap);
        #ifdef ENABLE_XLSDFT_TIMER
            this->xlsdft_timer.xlsdft_eigen_solver.stop();
        #endif //ENABLE_XLSDFT_TIMER
        #if !(defined(LOW_MEMORY))
            #pragma omp parallel
            Linalg::set_value_general(this->eigen_solvers[ielement].eigen_vectors.data,
                                      eigen_vectors_out, eigen_vector_length);
        #endif
        #if (defined(LOW_MEMORY))
            #ifdef ENABLE_XLSDFT_TIMER
                this->xlsdft_timer.xlsdft_low_memory_psi.start();
            #endif //ENABLE_XLSDFT_TIMER
            const Vertices_4D& eigen_region = this->eigen_solvers[ielement].domain_vertices.local_vertices;
            const Vertices_3D& element_vertices = this->element_verticeses[ielement];
            Vertices_4D region(element_vertices.get_overlap_vertices(eigen_region),
                                    eigen_region.bs, eigen_region.get_be());
            #pragma omp parallel
            Vertices_method::fill_region(eigen_vectors_out, vertices_eigen_vectors, 
                                this->eigen_solvers[ielement].eigen_vectors.data, region, region);
            #pragma omp parallel
            Linalg::convert_type(this->eigen_vectorses_lp[ielement].data, eigen_vectors_out, eigen_vector_length);
            #ifdef ENABLE_XLSDFT_TIMER
                this->xlsdft_timer.xlsdft_low_memory_psi.stop();
            #endif //ENABLE_XLSDFT_TIMER
        #endif
        }  // run_legacy_path
    }
    if (nchi_diag) {
        const int rank = this->domain_vertices.get_comm_rank();
        MPI_Comm comm = this->domain_vertices.get_mpi_comm();
        std::fprintf(stderr,
                     "NCHI_TOTAL rank=%d elements=%u nchi=%zu bytes_sum=%zu bytes_max=%zu "
                     "chi_ms=%.3f filter_nloc_ms=%.3f chefsi_ms=%.3f\n",
                     rank, this->local_element_num, rank_nchi_sum,
                     rank_projector_bytes_sum, rank_projector_bytes_max,
                     rank_chi_build_ms_sum, rank_filter_nloc_ms_sum, rank_chefsi_ms_sum);

        const unsigned long long local_nchi = rank_nchi_sum;
        unsigned long long max_nchi = 0;
        unsigned long long min_nchi = 0;
        MPI_Allreduce(&local_nchi, &max_nchi, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
        MPI_Allreduce(&local_nchi, &min_nchi, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm);

        const double local_filter_nloc_ms = rank_filter_nloc_ms_sum;
        double max_filter_nloc_ms = 0.0;
        double min_filter_nloc_ms = 0.0;
        MPI_Allreduce(&local_filter_nloc_ms, &max_filter_nloc_ms, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(&local_filter_nloc_ms, &min_filter_nloc_ms, 1, MPI_DOUBLE, MPI_MIN, comm);

        if (rank == 0) {
            const double nchi_spread_pct =
                min_nchi > 0
                    ? 100.0 * (static_cast<double>(max_nchi) / static_cast<double>(min_nchi) - 1.0)
                    : 0.0;
            const double filter_nloc_spread_pct =
                min_filter_nloc_ms > 0.0
                    ? 100.0 * (max_filter_nloc_ms / min_filter_nloc_ms - 1.0)
                    : 0.0;
            std::fprintf(stderr,
                         "NCHI_CROSS_RANK nchi_min=%llu nchi_max=%llu spread=%.1f%% "
                         "filter_nloc_ms_min=%.1f filter_nloc_ms_max=%.1f spread=%.1f%%\n",
                         min_nchi, max_nchi, nchi_spread_pct,
                         min_filter_nloc_ms, max_filter_nloc_ms, filter_nloc_spread_pct);
        }

        unsigned long long max_bytes_sum = 0;
        const unsigned long long local_bytes_sum = rank_projector_bytes_sum;
        MPI_Allreduce(&local_bytes_sum, &max_bytes_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
        if (rank == 0) {
            std::fprintf(stderr,
                         "NCHI_CACHE_EST max_rank_bytes_sum=%llu (per-rank if all elements cached)\n",
                         max_bytes_sum);
        }
    }
    #ifdef ENABLE_XLSDFT_TIMER
        this->xlsdft_timer.xlsdft_barrier2.start();
    #endif //ENABLE_XLSDFT_TIMER
    MPI_Barrier(this->domain_vertices.get_mpi_comm());
    #ifdef ENABLE_XLSDFT_TIMER
        this->xlsdft_timer.xlsdft_barrier2.stop();
    #endif //ENABLE_XLSDFT_TIMER
    #ifdef ENABLE_XLSDFT_TIMER
        this->xlsdft_timer.xlsdft_barrier3.start();
    #endif //ENABLE_XLSDFT_TIMER
    MPI_Barrier(this->domain_vertices.get_mpi_comm());
    #ifdef ENABLE_XLSDFT_TIMER
        this->xlsdft_timer.xlsdft_barrier3.stop();
    #endif //ENABLE_XLSDFT_TIMER
    #ifdef ENABLE_XLSDFT_TIMER
        this->xlsdft_timer.xlsdft.stop();
    // #if defined(ENABLE_TIMER)
    //     this->print_timer_statistics(print_flag, std::cout);
    // #endif //ENABLE_TIMER
        if (this->domain_vertices.get_comm_rank() == 0 && print_flag) this->xlsdft_timer.show();
        // Xlsdft_performance xlsdft_performance;
        // xlsdft_performance.update(this->eigen_solvers, this->xlsdft_timer);
        // xlsdft_performance.reduce(this->domain_vertices.comm, 0);
        // xlsdft_performance.show(print_flag, std::cout);
    #endif //ENABLE_XLSDFT_TIMER
    return;
}

template<typename T>
void Xlsdft<T>::init(const std::vector<Psp8_file>& psp8_files) {
    assert(this->geometry.cell_type <= 2);
    const Vertices_3D& shared_vertices_3d = this->domain_vertices.get_3D_shared_vertices();
    bool print_flag = this->domain_vertices.get_comm_rank() == 0;
    // [ni, nj, nk] of elements in the element comm
    uint local_element_ni = this->xlsdft_control.element_numbers[0]/this->xlsdft_control.element_comm_numbers[0];
    uint local_element_nj = this->xlsdft_control.element_numbers[1]/this->xlsdft_control.element_comm_numbers[1];
    uint local_element_nk = this->xlsdft_control.element_numbers[2]/this->xlsdft_control.element_comm_numbers[2];
    // the size of the element comm
    this->local_element_num = local_element_ni * local_element_nj * local_element_nk;

    std::chrono::steady_clock::time_point begin_group_geo = std::chrono::steady_clock::now();
    // the element comm vertices 
    Vertices_3D element_comm_vertices = this->domain_vertices.get_3D_local_vertices();
    int64_t ex_nnodes[3] = {
        int64_t(std::round(this->xlsdft_control.buffers[0]/this->mesh_control.delta_x + 1e-12) + 1),
        int64_t(std::round(this->xlsdft_control.buffers[1]/this->mesh_control.delta_y + 1e-12) + 1),
        int64_t(std::round(this->xlsdft_control.buffers[2]/this->mesh_control.delta_z + 1e-12) + 1)
    };
    Vertices_3D ex_element_comm_vertices = element_comm_vertices.generate_ex_vertices<int64_t>(ex_nnodes);
    bool if_one_more_grid[3] = {ex_element_comm_vertices.get_ie() == shared_vertices_3d.get_ie()
                                    && !this->mesh_control.is_periodic[0] ? false : true,
                                    ex_element_comm_vertices.get_je() == shared_vertices_3d.get_je()
                                    && !this->mesh_control.is_periodic[1] ? false : true,
                                    ex_element_comm_vertices.get_ke() == shared_vertices_3d.get_ke()
                                    && !this->mesh_control.is_periodic[2] ? false : true};
    Geometry geo_element_group = this->geometry.generate_extended_geometry(this->domain_vertices.get_3D_shared_vertices(),
                                                            ex_element_comm_vertices, this->mesh_control.delta_x, 
                                                            this->mesh_control.delta_y, this->mesh_control.delta_z,
                                                            this->mesh_control.is_periodic, if_one_more_grid, psp8_files);
    geo_element_group.origin[0] = this->geometry.origin[0];
    geo_element_group.origin[1] = this->geometry.origin[1];
    geo_element_group.origin[2] = this->geometry.origin[2];
    geo_element_group.a = this->geometry.a;
    geo_element_group.b = this->geometry.b;
    geo_element_group.c = this->geometry.c;
    geo_element_group.sync_from_angles();
    std::vector<Atom> atoms_no_image;
    uint64_t atom_index = ULONG_MAX;
    for (uint64_t i_atom = 0; i_atom < uint64_t(geo_element_group.natom); i_atom++) {
        Atom& i_atom_e = geo_element_group.atoms[i_atom];
        uint64_t i_atom_index = uint64_t(i_atom_e.index);
        if (i_atom_index != atom_index) {
            atom_index = i_atom_index;
            atoms_no_image.emplace_back(i_atom_e);
            // atoms_no_image.emplace_back(i_atom_e.x, i_atom_e.y, i_atom_e.z, i_atom_e.type, i_atom_e.index);
            // atoms_no_image.back().is_spin = i_atom_e.is_spin;
            // atoms_no_image.back().atom_spin[0] = i_atom_e.atom_spin[0];
            // atoms_no_image.back().atom_spin[1] = i_atom_e.atom_spin[1];
            // atoms_no_image.back().atom_spin[2] = i_atom_e.atom_spin[2];
        }
    }
    geo_element_group.atoms.swap(atoms_no_image);
    geo_element_group.natom = geo_element_group.atoms.size();
    // if (print_flag) geo_element_group.show();
    std::chrono::steady_clock::time_point end_group_geo = std::chrono::steady_clock::now();
    if (print_flag) {
        std::cout << "The XLSDFT group geometry generate took " << Tools::time_cost(begin_group_geo, end_group_geo) << "." << std::endl;
    }

    // verticeses of the element in the element comm
    this->element_verticeses = element_comm_vertices.split(local_element_ni, local_element_nj,
                                                                       local_element_nk);
    // shared verticeses of the element in the element comm
    int64_t nnodes[3];
    nnodes[0] = std::round(this->xlsdft_control.buffers[0]/this->mesh_control.delta_x + 1e-12);
    nnodes[1] = std::round(this->xlsdft_control.buffers[1]/this->mesh_control.delta_y + 1e-12);
    nnodes[2] = std::round(this->xlsdft_control.buffers[2]/this->mesh_control.delta_z + 1e-12);
    this->element_indexes.reserve(local_element_num);
    this->shared_verticeses.reserve(local_element_num);
    this->mesh_controls.reserve(local_element_num);
    this->ex_geometries.reserve(local_element_num);
    this->eigen_solver_controls.resize(local_element_num);
    this->domain_verticeses.resize(local_element_num);
    this->eigen_solvers.reserve(local_element_num);
    this->U0s.resize(local_element_num);
    #if (defined(LOW_MEMORY))
    this->eigen_vectorses_lp.resize(local_element_num);
    #endif
    uint count = 0;
    for (std::vector<Vertices_3D>::iterator it = this->element_verticeses.begin();
                                    it != this->element_verticeses.end(); ++it) {
        std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
        if (print_flag) {
            std::cout << "The index " << count << " element of XLSDFT is initializing." << count << std::endl;
        }
        // element_indexes
        this->element_indexes.emplace_back(count);
        // element shared verticeses
        this->shared_verticeses.emplace_back(it->generate_ex_vertices<int64_t>(nnodes));
        // element mesh controls
        this->mesh_controls.emplace_back(this->mesh_control);
        this->mesh_controls[count].nx = this->shared_verticeses[count].ni;
        this->mesh_controls[count].ny = this->shared_verticeses[count].nj;
        this->mesh_controls[count].nz = this->shared_verticeses[count].nk;
        this->mesh_controls[count].nd = this->shared_verticeses[count].get_size();
        this->mesh_controls[count].is_periodic[0] = this->xlsdft_control.is_periodic[0];
        this->mesh_controls[count].is_periodic[1] = this->xlsdft_control.is_periodic[1];
        this->mesh_controls[count].is_periodic[2] = this->xlsdft_control.is_periodic[2];
        // ex element geometries
        // element_shared_vertices_temp is to make sure that all the real atoms are included in the geometries
        Vertices_3D element_shared_vertices_temp = this->shared_verticeses[count];
        bool if_one_more_grid[3] = {element_shared_vertices_temp.get_ie() == shared_vertices_3d.get_ie()
                                    && !this->mesh_control.is_periodic[0] ? false : true,
                                    element_shared_vertices_temp.get_je() == shared_vertices_3d.get_je()
                                    && !this->mesh_control.is_periodic[1] ? false : true,
                                    element_shared_vertices_temp.get_ke() == shared_vertices_3d.get_ke()
                                    && !this->mesh_control.is_periodic[2] ? false : true};
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        this->ex_geometries.emplace_back(
                  geo_element_group.generate_extended_geometry(this->domain_vertices.get_3D_shared_vertices(),
                                                            element_shared_vertices_temp, this->mesh_control.delta_x, 
                                                            this->mesh_control.delta_y, this->mesh_control.delta_z,
                                                            this->mesh_control.is_periodic, if_one_more_grid,
                                                            psp8_files));
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (print_flag) {
            std::cout << "The XLSDFT element geometry " << count << " init took " << Tools::time_cost(begin, end) << "." << std::endl;
        }
        // Eigen solver controls
        uint natom = ex_geometries[count].natom;
        double nelectron = 1.0;
        for (uint iatom = 0; iatom < natom; iatom++) {
            this->ex_geometries[count].atoms[iatom].index = iatom;
            uint atom_type = this->ex_geometries[count].atoms[iatom].type;
            nelectron += (psp8_files[atom_type].zion) > 1.0
                       ? (psp8_files[atom_type].zion)
                       : 1.0;
        }
        const uint nzions = std::floor(nelectron / 2.0);
        this->eigen_solver_controls[count].set_method(0);
        this->eigen_solver_controls[count].set_nstates(this->xlsdft_control.element_nstates > 0
                                                        ? this->xlsdft_control.element_nstates
                                                        : nzions > 1
                                                        ? uint((double)nzions * this->xlsdft_control.basis_per_atom)
                                                        : 1);
        this->eigen_solver_controls[count].set_is_rand_fixed(this->xlsdft_control.is_rand_fixed);
        this->eigen_solver_controls[count].chefsi_control.init(this->xlsdft_control.chefsi_control);
        // domain_verticeses
        Vertices_4D shared_vertices_4d(this->shared_verticeses[count], this->eigen_solver_controls[count].nstates);
        this->domain_verticeses[count].init(shared_vertices_4d, this->domain_vertices.band_comm);
        // eigen_solvers
        this->eigen_solvers.emplace_back(this->eigen_solver_controls[count], this->mesh_controls[count],
                                         this->ex_geometries[count], this->stencil,
                                         psp8_files, this->domain_verticeses[count]);
        this->eigen_solvers[count].init();

        #if (defined(LOW_MEMORY))
        this->eigen_vectorses_lp[count].deepcopy(
                std::move(this->eigen_solvers[count].eigen_vectors.as_type(this->eigen_vectorses_lp[count].data)));
        this->eigen_solvers[count].eigen_vectors.destructor();
        // _mp
        const Vertices_4D& eigen_region = this->eigen_solvers[count].domain_vertices.local_vertices;
        const Vertices_3D& element_vertices = this->element_verticeses[count];
        const Vertices_4D region(element_vertices.get_overlap_vertices(eigen_region),
                                eigen_region.bs, eigen_region.get_be());
        this->eigen_solvers[count].eigen_vectors.constructor(region);
        #endif
        std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
        if (print_flag) {
            std::cout << "The XLSDFT element " << count << " init took " << Tools::time_cost(begin1, end1) << "." << std::endl;
        }
        count++;
    }
    return;
}

template<typename T>
template<typename T2>
void Xlsdft<T>::init(const Xlsdft<T2>& xlsdft) {
    this->local_element_num = xlsdft.local_element_num;
    this->element_indexes = xlsdft.element_indexes;
    this->element_verticeses = xlsdft.element_verticeses;
    this->ex_geometries = xlsdft.ex_geometries;
    this->shared_verticeses = xlsdft.shared_verticeses;
    this->eigen_solver_controls = xlsdft.eigen_solver_controls;
    this->mesh_controls = xlsdft.mesh_controls;
    this->domain_verticeses.resize(xlsdft.domain_verticeses.size());
    for (uint ielement = 0; ielement < this->local_element_num; ielement++) {
        this->domain_verticeses[ielement].init(xlsdft.domain_verticeses[ielement]);
    }
    this->eigen_solvers.reserve(xlsdft.eigen_solvers.size());
    for (typename std::vector<Eigen_solver<T2>>::const_iterator it = xlsdft.eigen_solvers.cbegin();
        it != xlsdft.eigen_solvers.cend(); ++it) {
        this->eigen_solvers.emplace_back(it->eigen_solver_control, it->mesh_control,
                                         it->geometry, this->stencil,
                                         it->psp8_files, it->domain_vertices);
        (this->eigen_solvers.end() - 1)->init(*it);
    }
    #if (defined(LOW_MEMORY))
    this->eigen_vectorses_lp.reserve(xlsdft.eigen_vectorses_lp.size());
    #if defined(FP16_FLAG)
    for (typename std::vector<Array_4D<__fp16>>::const_iterator it = xlsdft.eigen_vectorses_lp.cbegin();
    #else
    for (typename std::vector<Array_4D<float>>::const_iterator it = xlsdft.eigen_vectorses_lp.cbegin();
    #endif
        it != xlsdft.eigen_vectorses_lp.cend(); ++it) {
        this->eigen_vectorses_lp.emplace_back(*it);
    }
    #endif
    return;
}
template void Xlsdft<float>::init(const Xlsdft<float>& eigen_solver);
template void Xlsdft<double>::init(const Xlsdft<double>& eigen_solver);
template void Xlsdft<float>::init(const Xlsdft<double>& eigen_solver);
template void Xlsdft<double>::init(const Xlsdft<float>& eigen_solver);

template<typename T>
void Xlsdft<T>::destructor() {
    std::vector<uint>().swap(this->element_indexes);
    std::vector<Vertices_3D>().swap(this->element_verticeses);
    std::vector<Geometry>().swap(this->ex_geometries);
    std::vector<Vertices_3D>().swap(this->shared_verticeses);
    std::vector<Eigen_solver_control>().swap(this->eigen_solver_controls);
    std::vector<Mesh_control>().swap(this->mesh_controls);
    std::vector<Domain_parallel_vertices_4D>().swap(this->domain_verticeses);
    std::vector<Eigen_solver<T>>().swap(this->eigen_solvers);
    #if defined(LOW_MEMORY)
    #if defined(FP16_FLAG)
    std::vector<Array_4D<__fp16>>
    #else
    std::vector<Array_4D<float>>
    #endif
    ().swap(this->eigen_vectorses_lp);
    #endif
    return;
}

template<typename T>
void Xlsdft<T>::show() const {
    this->xlsdft_control.show();
    std::cout << "local_element_num = " << this->local_element_num << std::endl;
    for (uint i = 0; i < this->local_element_num; i++) {
        this->element_verticeses[i].show();
    }
    return;
}

template class Xlsdft<float>;
template class Xlsdft<double>;

namespace Xlsdft_method {

    template<typename T>
    T fg_electron_number_for_ext_fpmd(const T lambda_max, const T U0, const T upper_limit,
                                      const T chemical_potential, const T volume, const T beta,
                                      const T coef) {
        if (lambda_max >= upper_limit) {
            return T(0);
        }
        constexpr T sqrt_2 = 1.41421356237309504880168872420969807856967187537694807317667973799;
        constexpr T pi_square = 9.869604401089358618834490999876151135313699407240790626413533;
        constexpr T sqrt_2_pi_square = sqrt_2 / pi_square;
        constexpr T step = T(0.01);
        const int N_state = std::ceil((upper_limit - lambda_max) / step);
        T integral = T(0);
        // T f = T(1.0) / (T(1.0) + std::exp(beta * (lambda - chemical_potential)));
        #pragma omp parallel for reduction(+:integral)
        for (int i = 0; i < N_state; i++) {
            T lambda = lambda_max + (i + 0.5) * step;
            T x1 = lambda - U0;
            T x2 = beta * (lambda - chemical_potential);
            if (likely(x2 < 1e2 && x1 > T(0))) {
                T fx = std::sqrt(x1) / (T(1.0) + std::exp(x2));
                integral += fx;
            }
        }
        integral *= step * sqrt_2_pi_square * volume * coef;
        return integral;
    }

    template<typename T>
    T xlsdft_fg_electron_number_for_ext_fpmd(const uint nelement, const T* lambda_maxs, const T* U0s, const T upper_limit,
                                      const T chemical_potential, const T* volumes, const T beta,
                                      const T* coefs) {
        T fg_electron_number = T(0);
        for (uint i = 0; i < nelement; i++) {
            T fg_electron_element = fg_electron_number_for_ext_fpmd(lambda_maxs[i], U0s[i], upper_limit, chemical_potential, volumes[i], beta, coefs[i]);
            fg_electron_number += fg_electron_element;
        }
        return fg_electron_number;
    }

    template<typename T>
    T evaluate_chemical_potential2_with_fg_electron(T const * const eigen_values, T const* const fracs, const uint length, const MPI_Comm domain_3d_comm, const T lower_bound, const T upper_bound,
                                const Smearing& smearing, const T electron_charge, const T smearing_coef, const uint max_iter, const T tol,
                                const uint nelement, const T* lambda_maxs, const T* U0s, const T* volumes, const T* coefs) {
        T a = lower_bound;
        T b = upper_bound;
        T* occ = new T[length];
        const MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        const T beta = T(smearing.smearing_control.beta);

        T fg_electron_a = xlsdft_fg_electron_number_for_ext_fpmd(nelement, lambda_maxs, U0s, upper_bound, a, volumes, beta, coefs);
        Smearing_method::smear(eigen_values, occ, a, smearing, length);
        T fa = Linalg::vector_dot_product(occ, fracs, length, MPI_COMM_NULL);
        fa = (fa + fg_electron_a) * smearing_coef;
        MPI_Allreduce(MPI_IN_PLACE, &fa, 1, mpi_datatype, MPI_SUM, domain_3d_comm);
        fa -= electron_charge;


        T fg_electron_b = xlsdft_fg_electron_number_for_ext_fpmd(nelement, lambda_maxs, U0s, upper_bound, b, volumes, beta, coefs);
        Smearing_method::smear(eigen_values, occ, b, smearing, length);
        T fb = Linalg::vector_dot_product(occ, fracs, length, MPI_COMM_NULL);
        fb = (fb + fg_electron_b) * smearing_coef;
        MPI_Allreduce(MPI_IN_PLACE, &fb, 1, mpi_datatype, MPI_SUM, domain_3d_comm);
        fb -= electron_charge;

        T c = T(0);

        uint ext_loop_count = 0;
        while (fa * fb > 0.0 && ext_loop_count++ < 10) {
            T w = b - a;
            a -= w / 2.0;
            b += w / 2.0;
            c = b;

            fg_electron_a = xlsdft_fg_electron_number_for_ext_fpmd(nelement, lambda_maxs, U0s, upper_bound, a, volumes, beta, coefs);
            Smearing_method::smear(eigen_values, occ, a, smearing, length);
            fa = Linalg::vector_dot_product(occ, fracs, length, MPI_COMM_NULL);
            fa = (fa + fg_electron_a) * smearing_coef;
            MPI_Allreduce(MPI_IN_PLACE, &fa, 1, mpi_datatype, MPI_SUM, domain_3d_comm);
            fa -= electron_charge;

            fg_electron_b = xlsdft_fg_electron_number_for_ext_fpmd(nelement, lambda_maxs, U0s, upper_bound, b, volumes, beta, coefs);
            Smearing_method::smear(eigen_values, occ, b, smearing, length);
            fb = Linalg::vector_dot_product(occ, fracs, length, MPI_COMM_NULL);
            fb = (fb + fg_electron_b) * smearing_coef;
            MPI_Allreduce(MPI_IN_PLACE, &fb, 1, mpi_datatype, MPI_SUM, domain_3d_comm);
            fb -= electron_charge;
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

            fg_electron_b = xlsdft_fg_electron_number_for_ext_fpmd(nelement, lambda_maxs, U0s, upper_bound, b, volumes, beta, coefs);
            Smearing_method::smear(eigen_values, occ, b, smearing, length);
            fb = Linalg::vector_dot_product(occ, fracs, length, MPI_COMM_NULL);
            fb = (fb + fg_electron_b) * smearing_coef;
            MPI_Allreduce(MPI_IN_PLACE, &fb, 1, mpi_datatype, MPI_SUM, domain_3d_comm);
            fb -= electron_charge;
        }
        #undef EPSILON
        #undef SIGN

        delete[] occ;
        return T(0.0);
    }
    template float evaluate_chemical_potential2_with_fg_electron(float const * const eigen_values, float const* const fracs, const uint length, const MPI_Comm domain_3d_comm, const float lower_bound, const float upper_bound,
                                const Smearing& smearing, const float electron_charge, const float smearing_coef, const uint max_iter, const float tol,
                                const uint nelement, const float* lambda_maxs, const float* U0s, const float* volumes, const float* coefs);
    template double evaluate_chemical_potential2_with_fg_electron(double const * const eigen_values, double const* const fracs, const uint length, const MPI_Comm domain_3d_comm, const double lower_bound, const double upper_bound,
                                const Smearing& smearing, const double electron_charge, const double smearing_coef, const uint max_iter, const double tol,
                                const uint nelement, const double* lambda_maxs, const double* U0s, const double* volumes, const double* coefs);

    template<typename T>
    T evaluate_chemical_potential2(T const * const eigen_values, T const* const fracs, const uint length, const MPI_Comm domain_3d_comm, const T lower_bound, const T upper_bound,
                                const Smearing& smearing, const T electron_charge, const T smearing_coef, const uint max_iter, const T tol) {
        T a = lower_bound;
        T b = upper_bound;
        T* occ = new T[length];
        Smearing_method::smear(eigen_values, occ, a, smearing, length);
        T fa = Linalg::vector_dot_product(occ, fracs, length, domain_3d_comm) * smearing_coef - electron_charge;
        Smearing_method::smear(eigen_values, occ, b, smearing, length);
        T fb = Linalg::vector_dot_product(occ, fracs, length, domain_3d_comm) * smearing_coef - electron_charge;
        T c = T(0);

        uint ext_loop_count = 0;
        while (fa * fb > 0.0 && ext_loop_count++ < 10) {
            T w = b - a;
            a -= w / 2.0;
            b += w / 2.0;
            c = b;
            Smearing_method::smear(eigen_values, occ, a, smearing, length);
            fa = Linalg::vector_dot_product(occ, fracs, length, domain_3d_comm) * smearing_coef - electron_charge;
            Smearing_method::smear(eigen_values, occ, b, smearing, length);
            fb = Linalg::vector_dot_product(occ, fracs, length, domain_3d_comm) * smearing_coef - electron_charge;
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
            fb = Linalg::vector_dot_product(occ, fracs, length, domain_3d_comm) * smearing_coef - electron_charge;
        }
        #undef EPSILON
        #undef SIGN

        delete[] occ;
        return T(0.0);
    }
    template float evaluate_chemical_potential2(float const * const eigen_values, float const* const fracs, const uint length, const MPI_Comm domain_3d_comm, const float lower_bound, const float upper_bound,
                                const Smearing& smearing, const float electron_charge, const float smearing_coef, const uint max_iter, const float tol);
    template double evaluate_chemical_potential2(double const * const eigen_values, double const* const fracs, const uint length, const MPI_Comm domain_3d_comm, const double lower_bound, const double upper_bound,
                                const Smearing& smearing, const double electron_charge, const double smearing_coef, const uint max_iter, const double tol);
}
