#include "effective_potential_loc_solver.h"

template<typename T>
Effective_potential_loc_solver<T>::Effective_potential_loc_solver(const Poisson_solver_control& poisson_solver_control,
                                                                  const Exchange_correlation_solver_control& exchange_correlation_solver_control,
                                                                  const Geometry& geometry,
                                                                  const Spin& spin,
                                                                  const Stencil<T>& stencil,
                                                                  const Mesh_control& mesh_control,
                                                                  const std::vector<Psp8_file>& psp8_files,
                                                                  const Domain_parallel_vertices_3D& domain_vertices,
                                                                  const Exarr_3D_mpi_package& exarr_mpi_package)
    : poisson_solver_control(poisson_solver_control),
      exchange_correlation_solver_control(exchange_correlation_solver_control),
      geometry(geometry),
      spin(spin),
      stencil(stencil),
      mesh_control(mesh_control),
      psp8_files(psp8_files),
      domain_vertices(domain_vertices),
      exarr_mpi_package(exarr_mpi_package),
      pseudo_charge_solver(geometry,
                           mesh_control,
                           psp8_files,
                           stencil,
                           this->domain_vertices),
      poisson_solver(poisson_solver_control,
                     stencil,
                     mesh_control,
                     this->domain_vertices,
                     this->exarr_mpi_package),
      exchange_correlation_solver(exchange_correlation_solver_control,
                                  this->spin,
                                  this->stencil,
                                  this->exarr_mpi_package) {}

template<typename T>
Effective_potential_loc_solver<T>::~Effective_potential_loc_solver() {}

template<typename T>
std::vector<Array_3D<T>>& Effective_potential_loc_solver<T>::cal_effective_potential_loc(
                                                            const std::vector<Array_3D<T>>& electron_densities,
                                                            const Array_3D<T>& electron_density_core) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->spin.get_spin_type() == 0 || this->spin.get_spin_type() == 1) {
        const uint nspin = this->spin.generate_nspin();
        this->poisson_solver.cal_electrostatic_potential(electron_densities, this->pseudo_charge_solver.pseudo_charge_density);
        this->exchange_correlation_solver.cal_exchange_correlation_potential(electron_densities, electron_density_core);
        for (uint ispin = 0; ispin < nspin; ispin++) {
            this->effective_potentail_locs[ispin] = this->poisson_solver.electrostatic_potential
                                                + this->exchange_correlation_solver.exchange_correlation_potentials[ispin];
        }
    } else {
        assert(this->spin.get_spin_type() == 0 || this->spin.get_spin_type() == 1);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The cal_effective_potential_loc took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return this->effective_potentail_locs;
}

template<typename T>
std::vector<Array_3D<T>>& Effective_potential_loc_solver<T>::cal_effective_potential_loc_mp(
                            const std::vector<Array_3D<T>>& electron_densities,
                            const Array_3D<T>& electron_density_core,
                            Memory_pool<T, Fast_memory>& pool_fast,
                            Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->spin.get_spin_type() == 0 || this->spin.get_spin_type() == 1) {
        const uint nspin = this->spin.generate_nspin();
        T const* electron_densities_data[2];
        for (uint ispin = 0; ispin < nspin; ispin++) {
            electron_densities_data[ispin] = electron_densities[ispin].data;
        }
        this->poisson_solver.cal_electrostatic_potential_mp(
            electron_densities_data,
            this->pseudo_charge_solver.pseudo_charge_density.data,
            pool_fast, pool_cap);
        bool const if_add_core = electron_density_core.length == 0 ? false : true;
        this->exchange_correlation_solver.cal_exchange_correlation_potential_mp(
            electron_densities_data, electron_density_core.data, if_add_core,
            pool_fast, pool_cap);
        for (uint ispin = 0; ispin < nspin; ispin++) {
            this->effective_potentail_locs[ispin] = this->poisson_solver.electrostatic_potential
                                                + this->exchange_correlation_solver.exchange_correlation_potentials[ispin];
        }
    } else {
        assert(this->spin.get_spin_type() == 0 || this->spin.get_spin_type() == 1);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The cal_effective_potential_loc took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return this->effective_potentail_locs;
}

template<typename T>
T& Effective_potential_loc_solver<T>::evaluate_exchange_correlation_energy(const std::vector<Array_3D<T>>& electron_densities, const Array_3D<T>& electron_density_core) {
    return this->exchange_correlation_solver.evaluate_exchange_correlation_energy(electron_densities, electron_density_core, (T)this->mesh_control.delta_V, this->domain_vertices.comm);
}

template<typename T>
void Effective_potential_loc_solver<T>::init() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    Vertices_3D vertices = domain_vertices.get_3D_local_vertices();
    // effective_potentail_loc
    const uint nspin = this->spin.generate_nspin();
    this->effective_potentail_locs.reserve(nspin);
    for (uint ispin = 0; ispin < this->spin.generate_nspin(); ispin++) {
        this->effective_potentail_locs.emplace_back(vertices);
    }
    this->pseudo_charge_solver.init();
    this->poisson_solver.init();
    this->exchange_correlation_solver.init(vertices);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The effective_potential_loc_solver init took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

template<typename T>
template<typename T2> 
void Effective_potential_loc_solver<T>::init(const Effective_potential_loc_solver<T2>& effective_potential_loc_solver) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    std::vector<Array_3D<T>>().swap(this->effective_potentail_locs);
    this->effective_potentail_locs.reserve(effective_potential_loc_solver.effective_potentail_locs.size());
    T temp = T(0);
    for (uint ispin = 0; ispin < this->spin.generate_nspin(); ispin++) {
        this->effective_potentail_locs.emplace_back(
            effective_potential_loc_solver.effective_potentail_locs[ispin].as_type(&temp));
    }
    this->pseudo_charge_solver.init(effective_potential_loc_solver.pseudo_charge_solver);
    this->poisson_solver.init(effective_potential_loc_solver.poisson_solver);
    this->exchange_correlation_solver.init(effective_potential_loc_solver.exchange_correlation_solver);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The effective_potential_loc_solver init from other took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}
template void Effective_potential_loc_solver<float>::init(const Effective_potential_loc_solver<float>& effective_potential_loc_solver);
template void Effective_potential_loc_solver<double>::init(const Effective_potential_loc_solver<double>& effective_potential_loc_solver);
template void Effective_potential_loc_solver<float>::init(const Effective_potential_loc_solver<double>& effective_potential_loc_solver);
template void Effective_potential_loc_solver<double>::init(const Effective_potential_loc_solver<float>& effective_potential_loc_solver);

template<typename T>
void Effective_potential_loc_solver<T>::destructor() {
    std::vector<Array_3D<T>>().swap(this->effective_potentail_locs);
    this->pseudo_charge_solver.destructor();
    this->poisson_solver.destructor();
    this->exchange_correlation_solver.destructor();
    return;
}

template<typename T>
void Effective_potential_loc_solver<T>::show() const {
    // this->effective_potentail_loc.show();
    this->poisson_solver.show();
    this->exchange_correlation_solver.show();
    return;
}

template class Effective_potential_loc_solver<float>;
template class Effective_potential_loc_solver<double>;
