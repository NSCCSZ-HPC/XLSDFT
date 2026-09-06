#include "exchange_correlation_solver.h"

template<typename T>
Exchange_correlation_solver<T>::Exchange_correlation_solver(
    const Exchange_correlation_solver_control& exchange_correlation_solver_control,
    const Spin& spin,
    const Stencil<T>& stencil,
    const Exarr_3D_mpi_package& exarr_mpi_package)
    : exchange_correlation_solver_control(exchange_correlation_solver_control),
      spin(spin),
      stencil(stencil),
      exarr_mpi_package(exarr_mpi_package) {}

template<typename T>
Exchange_correlation_solver<T>::~Exchange_correlation_solver() {}

// template<typename T>
// std::vector<Array_3D<T>> Exchange_correlation_solver<T>::pretreat_electron_density(const std::vector<Array_3D<T>>& electron_densities) {
//     const uint nspin = this->spin.generate_nspin();
//     const uint Nd = electron_densities[0].length;
//     std::vector<Array_3D<T>> treated_electron_density(electron_densities);
//     for (uint ispin = 0; ispin < nspin; ispin++) {
//         for (uint i = 0; i < Nd; i++) {
//             if (treated_electron_density[ispin].data[i] < this->exchange_correlation_solver_control.xc_rhotol) {
//                 treated_electron_density[ispin].data[i] = this->exchange_correlation_solver_control.xc_rhotol;
//             }
//         }
//     }
//     return treated_electron_density;
// }

template<typename T>
Array_3D<T> Exchange_correlation_solver<T>::pretreat_electron_density(const std::vector<Array_3D<T>>& electron_densities,
                                                                        const Array_3D<T>& electron_density_core) {
    // const uint nspin = this->spin.generate_nspin();
    const bool if_add_core = electron_density_core.length == 0 ? false : true;
    const uint spin_type = this->spin.get_spin_type();
    const Vertices_3D vertices = electron_densities[0].get_vertices();
    const uint Nd = vertices.get_size();
    Array_3D<T> treated_electron_densities;
    if (spin_type == 0) {
        treated_electron_densities.reconstructor(vertices);
        if (if_add_core) {
            Linalg::hadamard_plus_general(treated_electron_densities.data, electron_densities[0].data,
                                        electron_density_core.data, Nd);
        } else {
            Linalg::set_value_general(treated_electron_densities.data, electron_densities[0].data, Nd);
        }
        for (uint i = 0; i < Nd; i++) {
            if (treated_electron_densities.data[i] < this->exchange_correlation_solver_control.xc_rhotol) {
                treated_electron_densities.data[i] = this->exchange_correlation_solver_control.xc_rhotol;
            }
        }
    } else if (spin_type == 1) {
        const uint ncol = 3;
        Vertices_3D vertices_k_boost(vertices);
        // vertices_k_boost.set_nk(vertices_k_boost.get_nk() * ncol);
        vertices_k_boost.nk *= ncol;
        treated_electron_densities.reconstructor(vertices_k_boost);
        if (if_add_core) {
            for (uint i = 0; i < Nd; i++) {
                T temp = T(0.5) * electron_density_core.data[i];
                T temp1 = electron_densities[0].data[i] + temp;
                T temp2 = electron_densities[1].data[i] + temp;
                if (temp1 < this->exchange_correlation_solver_control.xc_rhotol) {
                    treated_electron_densities.data[i + Nd] = this->exchange_correlation_solver_control.xc_rhotol;
                } else {
                    treated_electron_densities.data[i + Nd] = temp1;
                }
                if (temp2 < this->exchange_correlation_solver_control.xc_rhotol) {
                    treated_electron_densities.data[i + Nd * 2] = this->exchange_correlation_solver_control.xc_rhotol;
                } else {
                    treated_electron_densities.data[i + Nd * 2] = temp2;
                }
                treated_electron_densities.data[i] = treated_electron_densities.data[i + Nd] + treated_electron_densities.data[i + Nd * 2];
            }
        } else {
            for (uint i = 0; i < Nd; i++) {
                if (electron_densities[0].data[i] < this->exchange_correlation_solver_control.xc_rhotol) {
                    treated_electron_densities.data[i + Nd] = this->exchange_correlation_solver_control.xc_rhotol;
                } else {
                    treated_electron_densities.data[i + Nd] = electron_densities[0].data[i];
                }
                if (electron_densities[1].data[i] < this->exchange_correlation_solver_control.xc_rhotol) {
                    treated_electron_densities.data[i + Nd * 2] = this->exchange_correlation_solver_control.xc_rhotol;
                } else {
                    treated_electron_densities.data[i + Nd * 2] = electron_densities[1].data[i];
                }
                treated_electron_densities.data[i] = treated_electron_densities.data[i + Nd] + treated_electron_densities.data[i + Nd * 2];
            }
        }
    } else {
        assert(spin_type == 0 || spin_type == 1);
    }
    return treated_electron_densities;
}

template<typename T>
void Exchange_correlation_solver<T>::pretreat_electron_density(
                                    T* const treated_electron_densities,
                                    T const* const* const electron_densities,
                                    T const* const electron_density_core,
                                    bool const if_add_core) {
    // const uint nspin = this->spin.generate_nspin();
    // const bool if_add_core = electron_density_core.length == 0 ? false : true;
    const uint spin_type = this->spin.get_spin_type();
    const Vertices_3D vertices = this->exarr_mpi_package.domain_vertices.get_3D_local_vertices();
    const uint Nd = vertices.get_size();
    // Array_3D<T> treated_electron_densities;
    if (spin_type == 0) {
        // treated_electron_densities.reconstructor(vertices);
        if (if_add_core) {
            // Linalg::hadamard_plus_general(treated_electron_densities.data, electron_densities[0].data,
            //                             electron_density_core.data, Nd);
            Linalg::hadamard_plus_general(treated_electron_densities, electron_densities[0],
                                        electron_density_core, Nd);
        } else {
            // Linalg::set_value_general(treated_electron_densities.data, electron_densities[0].data, Nd);
            Linalg::set_value_general(treated_electron_densities, electron_densities[0], Nd);
        }
        for (uint i = 0; i < Nd; i++) {
            if (treated_electron_densities[i] < this->exchange_correlation_solver_control.xc_rhotol) {
                treated_electron_densities[i] = this->exchange_correlation_solver_control.xc_rhotol;
            }
        }
    } else if (spin_type == 1) {
        // const uint ncol = 3;
        // Vertices_3D vertices_k_boost(vertices);
        // // vertices_k_boost.set_nk(vertices_k_boost.get_nk() * ncol);
        // vertices_k_boost.nk *= ncol;
        // treated_electron_densities.reconstructor(vertices_k_boost);
        if (if_add_core) {
            for (uint i = 0; i < Nd; i++) {
                T temp = T(0.5) * electron_density_core[i];
                T temp1 = electron_densities[0][i] + temp;
                T temp2 = electron_densities[1][i] + temp;
                if (temp1 < this->exchange_correlation_solver_control.xc_rhotol) {
                    treated_electron_densities[i + Nd] = this->exchange_correlation_solver_control.xc_rhotol;
                } else {
                    treated_electron_densities[i + Nd] = temp1;
                }
                if (temp2 < this->exchange_correlation_solver_control.xc_rhotol) {
                    treated_electron_densities[i + Nd * 2] = this->exchange_correlation_solver_control.xc_rhotol;
                } else {
                    treated_electron_densities[i + Nd * 2] = temp2;
                }
                treated_electron_densities[i] = treated_electron_densities[i + Nd] + treated_electron_densities[i + Nd * 2];
            }
        } else {
            for (uint i = 0; i < Nd; i++) {
                if (electron_densities[0][i] < this->exchange_correlation_solver_control.xc_rhotol) {
                    treated_electron_densities[i + Nd] = this->exchange_correlation_solver_control.xc_rhotol;
                } else {
                    treated_electron_densities[i + Nd] = electron_densities[0][i];
                }
                if (electron_densities[1][i] < this->exchange_correlation_solver_control.xc_rhotol) {
                    treated_electron_densities[i + Nd * 2] = this->exchange_correlation_solver_control.xc_rhotol;
                } else {
                    treated_electron_densities[i + Nd * 2] = electron_densities[1][i];
                }
                treated_electron_densities[i] = treated_electron_densities[i + Nd] + treated_electron_densities[i + Nd * 2];
            }
        }
    } else {
        assert(spin_type == 0 || spin_type == 1);
    }
    return;
}

template<typename T>
void Exchange_correlation_solver<T>::cal_sigma(Array_3D<T>& sigma, const Array_3D<T>& Drho_x,
                                               const Array_3D<T>& Drho_y, const Array_3D<T>& Drho_z) {
    this->cal_sigma(sigma.data, sigma.length, 1, Drho_x.data, Drho_y.data, Drho_z.data);
    return;
}

template<typename T>
void Exchange_correlation_solver<T>::cal_sigma(T* __restrict__ sigma, const uint nd, const uint ncol,
    T const* const __restrict__ Drho_x, T const* const __restrict__ Drho_y, T const* const __restrict__ Drho_z) {
    assert(this->stencil.cell_type <= 2);
    for (uint i = 0; i < nd * ncol; i++) {
        sigma[i] = Drho_x[i] * Drho_x[i] +
                   Drho_y[i] * Drho_y[i] +
                   Drho_z[i] * Drho_z[i];
        // sigma_data[i] = std::max(sigma_data[i], 0.0); xc_sigmatol
    }
    return;
}

/**
 * @brief   slater exchange
 * 
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/exchangeCorrelation.c#L459
 */
template<typename T>
Array_3D<T> Exchange_correlation_solver<T>::slater_exchange_potential(const Array_3D<T>& electron_density) {
    Array_3D<T> exchange_potential(electron_density.get_vertices());
    Linalg::vector_cbrt(exchange_potential.data, electron_density.data, exchange_potential.length);
    // exchange parameter
    T C3 = 0.9847450218426965; // (3/pi)^(1/3)
    exchange_potential *= - C3;
    return exchange_potential;
}

/**
 * @brief   slater exchange
 * 
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/exchangeCorrelation.c#L459
 */
template<typename T>
Array_3D<T> Exchange_correlation_solver<T>::slater_exchange_energy_density(const Array_3D<T>& electron_density) {
    Array_3D<T> exchange_energy_density(electron_density.get_vertices());
    Linalg::vector_cbrt(exchange_energy_density.data, electron_density.data, exchange_energy_density.length);
    // exchange parameter
    T C2 = 0.738558766382022; // (3/pi)^(1/3)
    exchange_energy_density *= - C2;
    return exchange_energy_density;
}

/**
 * @brief   slater exchange
 * 
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/exchangeCorrelation.c#L459
 */
template<typename T>
void Exchange_correlation_solver<T>::slater_exchange(const Array_3D<T>& electron_density) {
    Array_3D<T> temp(electron_density.get_vertices());
    Linalg::vector_cbrt(temp.data, electron_density.data, temp.length);
    // exchange parameter
    T C2 = 0.738558766382022;  // 3/4 * (3/pi)^(1/3)
    T C3 = 0.9847450218426965; // (3/pi)^(1/3)
    //calculate
    this->exchange_correlation_energy_density = temp * (-C2);
    this->exchange_correlation_potentials[0] = temp * (-C3);
    return;
}

template<typename T>
void Exchange_correlation_solver<T>::slater_exchange(T const* const electron_density, uint const nd) {
    T* temp = new (std::align_val_t(64)) T [nd];
    Linalg::vector_cbrt(temp, electron_density, nd);
    // exchange parameter
    T C2 = 0.738558766382022;  // 3/4 * (3/pi)^(1/3)
    T C3 = 0.9847450218426965; // (3/pi)^(1/3)
    //calculate
    Linalg::scalar_product_general(this->exchange_correlation_energy_density.data, temp, -C2, nd);
    Linalg::scalar_product_general(this->exchange_correlation_potentials[0].data, temp, -C3, nd);
    ::operator delete[](temp, std::align_val_t(64));
}

template<typename T>
void Exchange_correlation_solver<T>::slater_exchange_mp(T const* const electron_density, uint const nd,
                            Memory_pool<T, Fast_memory>& pool_fast,
                            Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    // T* temp = new (std::align_val_t(64)) T [nd];
    T* temp = pool_fast.allocate(nd);
    Linalg::vector_cbrt(temp, electron_density, nd);
    // exchange parameter
    T C2 = 0.738558766382022;  // 3/4 * (3/pi)^(1/3)
    T C3 = 0.9847450218426965; // (3/pi)^(1/3)
    //calculate
    Linalg::scalar_product_general(this->exchange_correlation_energy_density.data, temp, -C2, nd);
    Linalg::scalar_product_general(this->exchange_correlation_potentials[0].data, temp, -C3, nd);
    return;
}

/**
 * @brief pbe exchange
 *
 * @param   flag=1  J.P.Perdew, K.Burke, M.Ernzerhof, PRL 77, 3865 (1996)
 * @param   flag=2  PBEsol: J.P.Perdew et al., PRL 100, 136406 (2008)
 * @param   flag=3  RPBE: B. Hammer, et al., Phys. Rev. B 59, 7413 (1999)
 * @param   flag=4  Zhang-Yang Revised PBE: Y. Zhang and W. Yang., Phys. Rev. Lett. 80, 890 (1998)
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/exchangeCorrelation.c#L540
 */
template<typename T>
void Exchange_correlation_solver<T>::pbe_exchange_kernel(
                const Array_3D<T>& electron_density, const Array_3D<T>& sigma,
                Array_3D<T>& ex, Array_3D<T>& vx, Array_0D<T>& v2x, const int flag) {
    assert(flag == 1 || flag == 2 || flag == 3 || flag == 4);
    T mu_[4] = {0.2195149727645171, 10.0/81.0, 0.2195149727645171, 0.2195149727645171};
    T kappa_[4] = {0.804, 0.804, 0.804, 1.245};

    // parameters
    T kappa = kappa_[flag-1];
    T mu = mu_[flag-1];
    T mu_divkappa = mu / kappa;
    T threefourth_divpi = (3.0/4.0) / M_PI;
    T third = 1.0/3.0;
    T sixpi2_1_3 = std::pow(6.0*M_PI*M_PI, third);
    T sixpi2m1_3 = 1.0/sixpi2_1_3;

    T const* const& __restrict__ rho = electron_density.data;
    T* const& __restrict__ vx_data = vx.data;
    T* const& __restrict__ v2x_data = v2x.data;
    T* const& __restrict__ ex_data = ex.data;
    const int DMnd = electron_density.length;

    for(int i = 0; i < DMnd; i++){
        T rho_updn = rho[i]/2.0;
        T rho_updnm1_3 = std::pow(rho_updn, -third);

        // First take care of the exchange part of the functional
        T rhomot = rho_updnm1_3;
        T ex_lsd = -threefourth_divpi * sixpi2_1_3 * (rhomot * rhomot * rho_updn);

        // Perdew-Burke-Ernzerhof GGA, exchange part
        T rho_inv = rhomot * rhomot * rhomot;
        T coeffss = (1.0/4.0) * sixpi2m1_3 * sixpi2m1_3 * (rho_inv * rho_inv * rhomot * rhomot);
        T ss = (sigma[i]/4.0) * coeffss; // s^2

        T divss = T(0);
        T dfxdss = T(0);
        if (flag == 1 || flag == 2 || flag == 4) {
            divss = 1.0/(1.0 + mu_divkappa * ss);
            dfxdss = mu * (divss * divss);
        } else if (flag == 3) {
            divss = exp(-mu_divkappa * ss);
            dfxdss = mu * divss;
        }

        T fx = 1.0 + kappa * (1.0 - divss);
        T dssdn = (-8.0/3.0) * (ss * rho_inv);
        T dfxdn = dfxdss * dssdn;
        T dssdg = 2.0 * coeffss;
        T dfxdg = dfxdss * dssdg;

        ex_data[i] = ex_lsd * fx;
        vx_data[i] = ex_lsd * ((4.0/3.0) * fx + rho_updn * dfxdn);
        v2x_data[i] = 0.5 * ex_lsd * rho_updn * dfxdg;
    }
}

/**
 * @brief pbe exchange
 *
 * @param   flag=1  J.P.Perdew, K.Burke, M.Ernzerhof, PRL 77, 3865 (1996)
 * @param   flag=2  PBEsol: J.P.Perdew et al., PRL 100, 136406 (2008)
 * @param   flag=3  RPBE: B. Hammer, et al., Phys. Rev. B 59, 7413 (1999)
 * @param   flag=4  Zhang-Yang Revised PBE: Y. Zhang and W. Yang., Phys. Rev. Lett. 80, 890 (1998)
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/exchangeCorrelation.c#L540
 */
template<typename T>
void Exchange_correlation_solver<T>::pbe_exchange_kernel(
        T const* const electron_density, T const* const sigma,
        T* const ex, T* const vx, T* const v2x, const uint nd, const int flag) {
    assert(flag == 1 || flag == 2 || flag == 3 || flag == 4);
    T mu_[4] = {0.2195149727645171, 10.0/81.0, 0.2195149727645171, 0.2195149727645171};
    T kappa_[4] = {0.804, 0.804, 0.804, 1.245};

    // parameters
    T kappa = kappa_[flag-1];
    T mu = mu_[flag-1];
    T mu_divkappa = mu / kappa;
    T threefourth_divpi = (3.0/4.0) / M_PI;
    T third = 1.0/3.0;
    T sixpi2_1_3 = std::pow(6.0*M_PI*M_PI, third);
    T sixpi2m1_3 = 1.0/sixpi2_1_3;

    T const* const& __restrict__ rho = electron_density;
    T* const& __restrict__ vx_data = vx;
    T* const& __restrict__ v2x_data = v2x;
    T* const& __restrict__ ex_data = ex;
    const int DMnd = nd;

    for(int i = 0; i < DMnd; i++){
        T rho_updn = rho[i]/2.0;
        T rho_updnm1_3 = std::pow(rho_updn, -third);

        // First take care of the exchange part of the functional
        T rhomot = rho_updnm1_3;
        T ex_lsd = -threefourth_divpi * sixpi2_1_3 * (rhomot * rhomot * rho_updn);

        // Perdew-Burke-Ernzerhof GGA, exchange part
        T rho_inv = rhomot * rhomot * rhomot;
        T coeffss = (1.0/4.0) * sixpi2m1_3 * sixpi2m1_3 * (rho_inv * rho_inv * rhomot * rhomot);
        T ss = (sigma[i]/4.0) * coeffss; // s^2

        T divss = T(0);
        T dfxdss = T(0);
        if (flag == 1 || flag == 2 || flag == 4) {
            divss = 1.0/(1.0 + mu_divkappa * ss);
            dfxdss = mu * (divss * divss);
        } else if (flag == 3) {
            divss = exp(-mu_divkappa * ss);
            dfxdss = mu * divss;
        }

        T fx = 1.0 + kappa * (1.0 - divss);
        T dssdn = (-8.0/3.0) * (ss * rho_inv);
        T dfxdn = dfxdss * dssdn;
        T dssdg = 2.0 * coeffss;
        T dfxdg = dfxdss * dssdg;

        ex_data[i] = ex_lsd * fx;
        vx_data[i] = ex_lsd * ((4.0/3.0) * fx + rho_updn * dfxdn);
        v2x_data[i] = 0.5 * ex_lsd * rho_updn * dfxdg;
    }
}

template<typename T>
void Exchange_correlation_solver<T>::pbe_exchange(const Array_3D<T>& electron_density, const Array_3D<T>& sigma, const int flag) {
    // const Vertices_3D vertices = electron_density.get_vertices();
    this->pbe_exchange_kernel(electron_density, sigma,
        this->exchange_correlation_energy_density, this->exchange_correlation_potentials[0],
        this->Dxcdgrho, flag);
    return;
}

template<typename T>
void Exchange_correlation_solver<T>::pbe_exchange(T const* const electron_density, T const* const sigma, const uint nd, const int flag) {
    // const Vertices_3D vertices = electron_density.get_vertices();
    this->pbe_exchange_kernel(electron_density, sigma,
        this->exchange_correlation_energy_density.data, this->exchange_correlation_potentials[0].data,
        this->Dxcdgrho.data, nd, flag);
    return;
}

/**
 * @brief   pbe exchange - spin polarized
 *
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/exchangeCorrelation.c#L887
 * @param   iflag=1  J.P.Perdew, K.Burke, M.Ernzerhof, PRL 77, 3865 (1996)
 * @param   iflag=2  PBEsol: J.P.Perdew et al., PRL 100, 136406 (2008)
 * @param   iflag=3  RPBE: B. Hammer, et al., Phys. Rev. B 59, 7413 (1999)
 * @param   iflag=4  Zhang-Yang Revised PBE: Y. Zhang and W. Yang., Phys. Rev. Lett. 80, 890 (1998)
 */
template<typename T>
void Exchange_correlation_solver<T>::pbe_exchange_spin(const uint& DMnd, T const* const& rho, T const* const& sigma, const int& iflag, T* const& ex, T* const& vx, T* const& v2x) {
    assert(iflag == 1 || iflag == 2 || iflag == 3 || iflag == 4);
    T mu_[4] = {0.2195149727645171, 10.0/81.0, 0.2195149727645171, 0.2195149727645171};
    T kappa_[4] = {0.804, 0.804, 0.804, 1.245};
    
    // parameters 
    T kappa = kappa_[iflag-1];
    T mu = mu_[iflag-1];
    T mu_divkappa = mu / kappa;
    T threefourth_divpi = (3.0/4.0) / M_PI;
    T third = 1.0/3.0;
    T sixpi2_1_3 = std::pow(6.0*M_PI*M_PI, third);
    T sixpi2m1_3 = 1.0/sixpi2_1_3;

    for(uint i = 0; i < DMnd; i++) {
        T rhom1_3 = std::pow(rho[i],-third);
        T rhotot_inv = std::pow(rhom1_3,3.0);

        // First take care of the exchange part of the functional
        T extot = 0.0;
        for(int spn_i = 0; spn_i < 2; spn_i++){
            T rho_updn = rho[DMnd + spn_i*DMnd + i];
            T rho_updnm1_3 = std::pow(rho_updn, -third);
            T rhomot = rho_updnm1_3;
            T ex_lsd = -threefourth_divpi * sixpi2_1_3 * (rhomot * rhomot * rho_updn);
            T rho_inv = rhomot * rhomot * rhomot;
            T coeffss = (1.0/4.0) * sixpi2m1_3 * sixpi2m1_3 * (rho_inv * rho_inv * rhomot * rhomot);
            T ss = sigma[DMnd + spn_i*DMnd + i] * coeffss;
            
            T divss = T(0);
            T dfxdss = T(0);
            if (iflag == 1 || iflag == 2 || iflag == 4) {
                divss = 1.0/(1.0 + mu_divkappa * ss);
                dfxdss = mu * (divss * divss);
            } else if (iflag == 3) {
                divss = std::exp(-mu_divkappa * ss);
                dfxdss = mu * divss;
            }

			T fx = 1.0 + kappa * (1.0 - divss);
            T ex_gga = ex_lsd * fx;
            T dssdn = (-8.0/3.0) * (ss * rho_inv);
            T dfxdn = dfxdss * dssdn;
            vx[spn_i*DMnd + i] = ex_lsd * ((4.0/3.0) * fx + rho_updn * dfxdn);

            T dssdg = 2.0 * coeffss;
            T dfxdg = dfxdss * dssdg;
            v2x[spn_i*DMnd + i] = ex_lsd * rho_updn * dfxdg; // changed to assuming 2 columns 
            extot += ex_gga * rho_updn;
        }
        ex[i] = extot * rhotot_inv;
    }
}

/**
 * @brief   pz correlation
 *          J.P. Perdew and A. Zunger, PRB 23, 5048 (1981).
 * 
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/exchangeCorrelation.c#L505
 */
template<typename T>
Array_3D<T> Exchange_correlation_solver<T>::pz_correlation_potential(const Array_3D<T>& electron_density) {
    // parameters
    T A = 0.0311;
    T B = -0.048 ;
    T C = 0.002 ;
    T D = -0.0116 ;
    T gamma1 = -0.1423 ;
    T beta1 = 1.0529 ;
    T beta2 = 0.3334 ; 
    T C31 = 0.6203504908993999; // (3/4pi)^(1/3)

    Array_3D<T> correlation_potential(electron_density.get_vertices());
    Linalg::vector_cbrt(correlation_potential.data, electron_density.data, correlation_potential.length);
    T* const& __restrict__ correlation_potential_data = correlation_potential.data;
    for (uint i = 0; i < correlation_potential.length; i++) {
        T rs = C31 / correlation_potential_data[i]; // rs = (3/(4*pi*rho))^(1/3)
        if (rs < 1.0) {
            correlation_potential_data[i] = log(rs)*(A+(2.0/3.0)*C*rs) + (B-(1.0/3.0)*A) + (1.0/3.0)*(2.0*D-C)*rs;
        } else {
            T sqrtrs = sqrt(rs);
            correlation_potential_data[i] = (gamma1 + (7.0/6.0)*gamma1*beta1*sqrtrs
                    + (4.0/3.0)*gamma1*beta2*rs)/pow(1+beta1*sqrtrs+beta2*rs,2.0);
        }
    }
    return correlation_potential;
}

/**
 * @brief   pz correlation
 *          J.P. Perdew and A. Zunger, PRB 23, 5048 (1981).
 * 
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/exchangeCorrelation.c#L505
 */
template<typename T>
Array_3D<T> Exchange_correlation_solver<T>::pz_correlation_energy_density(const Array_3D<T>& electron_density) {
    // parameters
    T A = 0.0311;
    T B = -0.048 ;
    T C = 0.002 ;
    T D = -0.0116 ;
    T gamma1 = -0.1423 ;
    T beta1 = 1.0529 ;
    T beta2 = 0.3334 ; 
    T C31 = 0.6203504908993999; // (3/4pi)^(1/3)

    Array_3D<T> correlation_energy_density(electron_density.get_vertices());
    Linalg::vector_cbrt(correlation_energy_density.data, electron_density.data, correlation_energy_density.length);
    T* const& __restrict__ correlation_energy_density_data = correlation_energy_density.data;
    for (uint i = 0; i < correlation_energy_density.length; i++) {
        T rs = C31 / correlation_energy_density_data[i]; // rs = (3/(4*pi*rho))^(1/3)
        if (rs < 1.0) {
            correlation_energy_density_data[i] = A*log(rs) + B + C*rs*log(rs) + D*rs;
        } else {
            T sqrtrs = sqrt(rs);
            correlation_energy_density_data[i] = gamma1/(1.0+beta1*sqrtrs+beta2*rs);
        }
    }
    return correlation_energy_density;
}

/**
 * @brief   pz correlation
 *          J.P. Perdew and A. Zunger, PRB 23, 5048 (1981).
 * 
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/exchangeCorrelation.c#L505
 */
template<typename T>
void Exchange_correlation_solver<T>::pz_correlation(const Array_3D<T>& electron_density) {
    this->pz_correlation(electron_density.data, electron_density.length);
    return;
}

template<typename T>
void Exchange_correlation_solver<T>::pz_correlation(T const* const electron_density, uint const nd) {
    // parameters
    T A = 0.0311;
    T B = -0.048 ;
    T C = 0.002 ;
    T D = -0.0116 ;
    T gamma1 = -0.1423 ;
    T beta1 = 1.0529 ;
    T beta2 = 0.3334 ; 
    T C31 = 0.6203504908993999; // (3/4pi)^(1/3)

    // Array_3D<T> electron_density_cbrt(electron_density.get_vertices());
    T* electron_density_cbrt = new (std::align_val_t(64)) T [nd];
    Linalg::vector_cbrt(electron_density_cbrt, electron_density, nd);
    // T const* const& __restrict__ electron_density_cbrt_data = electron_density_cbrt.data;
    T* const __restrict__ exchange_correlation_potential_data = this->exchange_correlation_potentials[0].data;
    T* const __restrict__ exchange_correlation_energy_density_data = this->exchange_correlation_energy_density.data;

    for (uint i = 0; i < nd; i++) {
        T rs = C31 / electron_density_cbrt[i]; // rs = (3/(4*pi*rho))^(1/3)
        if (rs < 1.0) {
            exchange_correlation_energy_density_data[i] += A*std::log(rs) + B + C*rs*std::log(rs) + D*rs;
            exchange_correlation_potential_data[i] += std::log(rs)*(A+(2.0/3.0)*C*rs) + (B-(1.0/3.0)*A) + (1.0/3.0)*(2.0*D-C)*rs;
        } else {
            T sqrtrs = std::sqrt(rs);
            exchange_correlation_energy_density_data[i] += gamma1/(1.0+beta1*sqrtrs+beta2*rs);
            exchange_correlation_potential_data[i] += (gamma1 + (7.0/6.0)*gamma1*beta1*sqrtrs
                                                    + (4.0/3.0)*gamma1*beta2*rs)/std::pow(1+beta1*sqrtrs+beta2*rs,2.0);
        }
    }
    ::operator delete[](electron_density_cbrt, std::align_val_t(64));
    return;
}

template<typename T>
void Exchange_correlation_solver<T>::pz_correlation_mp(T const* const electron_density, uint const nd,
                                                        Memory_pool<T, Fast_memory>& pool_fast,
                                                        Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    // parameters
    T A = 0.0311;
    T B = -0.048 ;
    T C = 0.002 ;
    T D = -0.0116 ;
    T gamma1 = -0.1423 ;
    T beta1 = 1.0529 ;
    T beta2 = 0.3334 ; 
    T C31 = 0.6203504908993999; // (3/4pi)^(1/3)

    // Array_3D<T> electron_density_cbrt(electron_density.get_vertices());
    // T* electron_density_cbrt = new (std::align_val_t(64)) T [nd];
    T* electron_density_cbrt = pool_fast.allocate(nd);
    Linalg::vector_cbrt(electron_density_cbrt, electron_density, nd);
    // T const* const& __restrict__ electron_density_cbrt_data = electron_density_cbrt.data;
    T* const __restrict__ exchange_correlation_potential_data = this->exchange_correlation_potentials[0].data;
    T* const __restrict__ exchange_correlation_energy_density_data = this->exchange_correlation_energy_density.data;

    for (uint i = 0; i < nd; i++) {
        T rs = C31 / electron_density_cbrt[i]; // rs = (3/(4*pi*rho))^(1/3)
        if (rs < 1.0) {
            exchange_correlation_energy_density_data[i] += A*std::log(rs) + B + C*rs*std::log(rs) + D*rs;
            exchange_correlation_potential_data[i] += std::log(rs)*(A+(2.0/3.0)*C*rs) + (B-(1.0/3.0)*A) + (1.0/3.0)*(2.0*D-C)*rs;
        } else {
            T sqrtrs = std::sqrt(rs);
            exchange_correlation_energy_density_data[i] += gamma1/(1.0+beta1*sqrtrs+beta2*rs);
            exchange_correlation_potential_data[i] += (gamma1 + (7.0/6.0)*gamma1*beta1*sqrtrs
                                                    + (4.0/3.0)*gamma1*beta2*rs)/std::pow(1+beta1*sqrtrs+beta2*rs,2.0);
        }
    }
    return;
}

/**
 * @brief   pbe correlation
 *
 * @param   flag=1  J.P.Perdew, K.Burke, M.Ernzerhof, PRL 77, 3865 (1996)
 * @param   flag=2  PBEsol: J.P.Perdew et al., PRL 100, 136406 (2008)
 * @param   flag=3  RPBE: B. Hammer, et al., Phys. Rev. B 59, 7413 (1999)
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/exchangeCorrelation.c#L540
 */
template<typename T>
void Exchange_correlation_solver<T>::pbe_correlation_kernel(const Array_3D<T>& electron_density, const Array_3D<T>& sigma,
    Array_3D<T>& ec, Array_3D<T>& vc, Array_3D<T>& v2c, const int flag) {
    this->pbe_correlation_kernel(electron_density.data, sigma.data, ec.data, vc.data, v2c.data, electron_density.length, flag);
    return;
}

/**
 * @brief   pbe correlation
 *
 * @param   flag=1  J.P.Perdew, K.Burke, M.Ernzerhof, PRL 77, 3865 (1996)
 * @param   flag=2  PBEsol: J.P.Perdew et al., PRL 100, 136406 (2008)
 * @param   flag=3  RPBE: B. Hammer, et al., Phys. Rev. B 59, 7413 (1999)
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/exchangeCorrelation.c#L540
 */
template<typename T>
void Exchange_correlation_solver<T>::pbe_correlation_kernel(T const* const electron_density, T const* const sigma,
        T* const ec, T* const vc, T* const v2c, const uint nd, const int flag) {
    assert(flag == 1 || flag == 2 || flag == 3);
    T beta_[3] = {0.066725, 0.046, 0.066725};

    // parameters
    T beta = beta_[flag-1];
    T gamma = (1.0 - std::log(2.0)) / (M_PI*M_PI);
    T gamma_inv = 1.0/gamma;
    T phi_zeta_inv = 1.0;
    T phi3_zeta = 1.0;
    T gamphi3inv = gamma_inv;
    T third = 1.0/3.0;
    T twom1_3 = std::pow(2.0,-third);
    T rsfac = 0.6203504908994000;
    T sq_rsfac = std::sqrt(rsfac);
    T sq_rsfac_inv = 1.0/sq_rsfac;
    T coeff_tt = 1.0/(16.0 / M_PI * std::pow(3.0*M_PI*M_PI,third));
    T ec0_aa = 0.031091;
    T ec0_a1 = 0.21370;
    T ec0_b1 = 7.5957;
    T ec0_b2 = 3.5876;
    T ec0_b3 = 1.6382;
    T ec0_b4 = 0.49294;

    const int DMnd = nd;
    T const* const& __restrict__ rho = electron_density;
    T* const& __restrict__ ec_data = ec;
    T* const& __restrict__ vc_data = vc;
    T* const& __restrict__ v2c_data = v2c;

    for(int i = 0; i < DMnd; i++){
        T rho_updn = rho[i]/2.0;
        T rho_updnm1_3 = std::pow(rho_updn, -third);
        T rhom1_3 = twom1_3 * rho_updnm1_3;
        T rhotot_inv = rhom1_3 * rhom1_3 * rhom1_3;
        T rhotmo6 = std::sqrt(rhom1_3);
        T rhoto6 = rho[i] * rhom1_3 * rhom1_3 * rhotmo6;

        // Then takes care of the LSD correlation part of the functional
        T rs = rsfac * rhom1_3;
        T sqr_rs = sq_rsfac * rhotmo6;
        T rsm1_2 = sq_rsfac_inv * rhoto6;

        // Formulas A6-A8 of PW92LSD
        T ec0_q0 = -2.0 * ec0_aa * (1.0 + ec0_a1 * rs);
        T ec0_q1 = 2.0 * ec0_aa * (ec0_b1 * sqr_rs + ec0_b2 * rs + ec0_b3 * rs * sqr_rs + ec0_b4 * rs * rs);
        T ec0_q1p = ec0_aa * (ec0_b1 * rsm1_2 + 2.0 * ec0_b2 + 3.0 * ec0_b3 * sqr_rs + 4.0 * ec0_b4 * rs);
        T ec0_den = 1.0/(ec0_q1 * ec0_q1 + ec0_q1);
        T ec0_log = -std::log(ec0_q1 * ec0_q1 * ec0_den);
        T ecrs0 = ec0_q0 * ec0_log;
        T decrs0_drs = -2.0 * ec0_aa * ec0_a1 * ec0_log - ec0_q0 * ec0_q1p * ec0_den;

        T ecrs = ecrs0;
        T decrs_drs = decrs0_drs;

        // Add LSD correlation functional to GGA exchange functional
        ec_data[i] = ecrs;
        vc_data[i] = ecrs - (rs/3.0) * decrs_drs;

        // Eventually add the GGA correlation part of the PBE functional
        // Note : the computation of the potential in the spin-unpolarized
        // case could be optimized much further. Other optimizations are left to do.

        // From ec to bb
        T bb = ecrs * gamphi3inv;
        T dbb_drs = decrs_drs * gamphi3inv;
        // dbb_dzeta = gamphi3inv * (decrs_dzeta - 3.0 * ecrs * phi_logder);

        // From bb to cc
        T exp_pbe = std::exp(-bb);
        T cc = 1.0/(exp_pbe - 1.0);
        T dcc_dbb = cc * cc * exp_pbe;
        T dcc_drs = dcc_dbb * dbb_drs;
        // dcc_dzeta = dcc_dbb * dbb_dzeta;

        // From cc to aa
        T coeff_aa = beta * gamma_inv * phi_zeta_inv * phi_zeta_inv;
        T aa = coeff_aa * cc;
        T daa_drs = coeff_aa * dcc_drs;
        //daa_dzeta = -2.0 * aa * phi_logder + coeff_aa * dcc_dzeta;

        // Introduce tt : do not assume that the spin-dependent gradients are collinear
        T grrho2 = sigma[i];
        T dtt_dg = 2.0 * rhotot_inv * rhotot_inv * rhom1_3 * coeff_tt;
        // Note that tt is (the t variable of PBE divided by phi) squared
        T tt = 0.5 * grrho2 * dtt_dg;

        // Get xx from aa and tt
        T xx = aa * tt;
        T dxx_drs = daa_drs * tt;
        T dxx_dtt = aa;

        // From xx to pade
        T pade_den = 1.0/(1.0 + xx * (1.0 + xx));
        T pade = (1.0 + xx) * pade_den;
        T dpade_dxx = -xx * (2.0 + xx) * std::pow(pade_den,2);
        T dpade_drs = dpade_dxx * dxx_drs;
        T dpade_dtt = dpade_dxx * dxx_dtt;
        //dpade_dzeta = dpade_dxx * dxx_dzeta;

        // From pade to qq
        T coeff_qq = tt * phi_zeta_inv * phi_zeta_inv;
        T qq = coeff_qq * pade;
        T dqq_drs = coeff_qq * dpade_drs;
        T dqq_dtt = pade * phi_zeta_inv * phi_zeta_inv + coeff_qq * dpade_dtt;
        //dqq_dzeta = coeff_qq * (dpade_dzeta - 2.0 * pade * phi_logder);

        // From qq to rr
        T arg_rr = 1.0 + beta * gamma_inv * qq;
        T div_rr = 1.0/arg_rr;
        T rr = gamma * std::log(arg_rr);
        T drr_dqq = beta * div_rr;
        T drr_drs = drr_dqq * dqq_drs;
        T drr_dtt = drr_dqq * dqq_dtt;
        //drr_dzeta = drr_dqq * dqq_dzeta;

        // From rr to hh
        T hh = phi3_zeta * rr;
        T dhh_drs = phi3_zeta * drr_drs;
        T dhh_dtt = phi3_zeta * drr_dtt;
        //dhh_dzeta = phi3_zeta * (drr_dzeta + 3.0 * rr * phi_logder);

        // The GGA correlation energy is added
        ec_data[i] += hh;

        // From hh to the derivative of the energy wrt the density
        T drhohh_drho = hh - third * rs * dhh_drs - (7.0/3.0) * tt * dhh_dtt; //- zeta * dhh_dzeta
        vc_data[i] += drhohh_drho;

        // From hh to the derivative of the energy wrt to the gradient of the
        // density, divided by the gradient of the density
        // (The v3.3 definition includes the division by the norm of the gradient)
        v2c_data[i] = (rho[i] * dtt_dg * dhh_dtt);
    }
    return;
}

template<typename T>
void Exchange_correlation_solver<T>::pbe_correlation(const Array_3D<T>& electron_density, const Array_3D<T>& sigma, const int flag) {
    const Vertices_3D vertices = electron_density.get_vertices();
    const uint length = vertices.get_size();
    Array_3D<T> ec(vertices);
    Array_3D<T> vc(vertices);
    Array_3D<T> v2c(vertices);
    this->pbe_correlation_kernel(electron_density, sigma, ec, vc, v2c, flag);
    Linalg::hadamard_plus_general(this->exchange_correlation_energy_density.data, ec.data, length);
    Linalg::hadamard_plus_general(this->exchange_correlation_potentials[0].data, vc.data, length);
    Linalg::hadamard_plus_general(this->Dxcdgrho.data, v2c.data, length);
    return;
}

template<typename T>
void Exchange_correlation_solver<T>::pbe_correlation(T const* const electron_density, T const* const sigma, uint const nd, const int flag) {
    T* ec = new (std::align_val_t(64)) T [nd];
    T* vc = new (std::align_val_t(64)) T [nd];
    T* v2c = new (std::align_val_t(64)) T [nd];
    this->pbe_correlation_kernel(electron_density, sigma, ec, vc, v2c, nd, flag);
    Linalg::hadamard_plus_general(this->exchange_correlation_energy_density.data, ec, nd);
    Linalg::hadamard_plus_general(this->exchange_correlation_potentials[0].data, vc, nd);
    Linalg::hadamard_plus_general(this->Dxcdgrho.data, v2c, nd);
    ::operator delete[](ec, std::align_val_t(64));
    ::operator delete[](vc, std::align_val_t(64));
    ::operator delete[](v2c, std::align_val_t(64));
    return;
}

template<typename T>
void Exchange_correlation_solver<T>::pbe_correlation_mp(
    T const* const electron_density, T const* const sigma, uint const nd, const int flag,
                      Memory_pool<T, Fast_memory>& pool_fast,
                      Memory_pool<T, Capacity_memory>& pool_cap) {
                        Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
                        Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    // T* ec = new (std::align_val_t(64)) T [nd];
    // T* vc = new (std::align_val_t(64)) T [nd];
    // T* v2c = new (std::align_val_t(64)) T [nd];
    T* ec = pool_fast.allocate(nd);
    T* vc = pool_fast.allocate(nd);
    T* v2c = pool_fast.allocate(nd);
    this->pbe_correlation_kernel(electron_density, sigma, ec, vc, v2c, nd, flag);
    Linalg::hadamard_plus_general(this->exchange_correlation_energy_density.data, ec, nd);
    Linalg::hadamard_plus_general(this->exchange_correlation_potentials[0].data, vc, nd);
    Linalg::hadamard_plus_general(this->Dxcdgrho.data, v2c, nd);
    return;
}

/**
 * @brief   pbe correlation - spin polarized
 *
 * @param   iflag=1  J.P.Perdew, K.Burke, M.Ernzerhof, PRL 77, 3865 (1996)
 * @param   iflag=2  PBEsol: J.P.Perdew et al., PRL 100, 136406 (2008)
 * @param   iflag=3  RPBE: B. Hammer, et al., Phys. Rev. B 59, 7413 (1999)
 */
// void pbec_spin(int DMnd, double *rho, double *sigma, int iflag, double *ec, double *vc, double *v2c) {
template<typename T>
void Exchange_correlation_solver<T>::pbe_correlation_spin(
    const uint DMnd, T const* const rho, T const* const sigma,
    const int iflag, T* const ec, T* const vc, T* const v2c) {
    assert(iflag == 1 || iflag == 2 || iflag == 3);
    T beta_[3] = {0.066725, 0.046, 0.066725};
    
    // parameters 
    T beta = beta_[iflag-1];
    T gamma = (1.0 - log(2.0)) / (M_PI*M_PI);
    T gamma_inv = 1.0/gamma;    
    T third = 1.0/3.0;
    T alpha_zeta2 = 1.0 - 1.0e-6; 
    T alpha_zeta = 1.0 - 1.0e-6;
    T rsfac = 0.6203504908994000;
    T sq_rsfac = std::sqrt(rsfac);
    T sq_rsfac_inv = 1.0/sq_rsfac;
    T fsec_inv = 1.0/1.709921;
    T factf_zeta = 1.0/(std::pow(2.0,(4.0/3.0)) - 2.0);
    T factfp_zeta = 4.0/3.0 * factf_zeta * alpha_zeta2;
    T coeff_tt = 1.0/(16.0 / M_PI * std::pow(3.0*M_PI*M_PI,third));

    T ec0_aa = 0.031091; T ec1_aa = 0.015545; T mac_aa = 0.016887;
    T ec0_a1 = 0.21370;  T ec1_a1 = 0.20548;  T mac_a1 = 0.11125;
    T ec0_b1 = 7.5957;   T ec1_b1 = 14.1189;  T mac_b1 = 10.357;
    T ec0_b2 = 3.5876;   T ec1_b2 = 6.1977;   T mac_b2 = 3.6231;
    T ec0_b3 = 1.6382;   T ec1_b3 = 3.3662;   T mac_b3 = 0.88026;
    T ec0_b4 = 0.49294;  T ec1_b4 = 0.62517;  T mac_b4 = 0.49671;
    
    for(uint i = 0; i < DMnd; i++) {
        T rhom1_3 = std::pow(rho[i],-third);
        T rhotot_inv = std::pow(rhom1_3,3.0);
        T zeta = (rho[DMnd + i] - rho[2*DMnd + i]) * rhotot_inv;
        T zetp = 1.0 + zeta * alpha_zeta;
        T zetm = 1.0 - zeta * alpha_zeta;
        T zetpm1_3 = std::pow(zetp,-third);
        T zetmm1_3 = std::pow(zetm,-third);
        T rhotmo6 = std::sqrt(rhom1_3);
        T rhoto6 = rho[i] * rhom1_3 * rhom1_3 * rhotmo6;

        // Then takes care of the LSD correlation part of the functional
        T rs = rsfac * rhom1_3;
        T sqr_rs = sq_rsfac * rhotmo6;
        T rsm1_2 = sq_rsfac_inv * rhoto6;

        // Formulas A6-A8 of PW92LSD
        T ec0_q0 = -2.0 * ec0_aa * (1.0 + ec0_a1 * rs);
        T ec0_q1 = 2.0 * ec0_aa *(ec0_b1 * sqr_rs + ec0_b2 * rs + ec0_b3 * rs * sqr_rs + ec0_b4 * rs * rs);
        T ec0_q1p = ec0_aa * (ec0_b1 * rsm1_2 + 2.0 * ec0_b2 + 3.0 * ec0_b3 * sqr_rs + 4.0 * ec0_b4 * rs);
        T ec0_den = 1.0/(ec0_q1 * ec0_q1 + ec0_q1);
        T ec0_log = -std::log(ec0_q1 * ec0_q1 * ec0_den);
        T ecrs0 = ec0_q0 * ec0_log;
        T decrs0_drs = -2.0 * ec0_aa * ec0_a1 * ec0_log - ec0_q0 * ec0_q1p * ec0_den;

        T mac_q0 = -2.0 * mac_aa * (1.0 + mac_a1 * rs);
        T mac_q1 = 2.0 * mac_aa * (mac_b1 * sqr_rs + mac_b2 * rs + mac_b3 * rs * sqr_rs + mac_b4 * rs * rs);
        T mac_q1p = mac_aa * (mac_b1 * rsm1_2 + 2.0 * mac_b2 + 3.0 * mac_b3 * sqr_rs + 4.0 * mac_b4 * rs);
        T mac_den = 1.0/(mac_q1 * mac_q1 + mac_q1);
        T mac_log = -std::log( mac_q1 * mac_q1 * mac_den );
        T macrs = mac_q0 * mac_log;
        T dmacrs_drs = -2.0 * mac_aa * mac_a1 * mac_log - mac_q0 * mac_q1p * mac_den;

        T ec1_q0 = -2.0 * ec1_aa * (1.0 + ec1_a1 * rs);
        T ec1_q1 = 2.0 * ec1_aa * (ec1_b1 * sqr_rs + ec1_b2 * rs + ec1_b3 * rs * sqr_rs + ec1_b4 * rs * rs);
        T ec1_q1p = ec1_aa * (ec1_b1 * rsm1_2 + 2.0 * ec1_b2 + 3.0 * ec1_b3 * sqr_rs + 4.0 * ec1_b4 * rs);
        T ec1_den = 1.0/(ec1_q1 * ec1_q1 + ec1_q1);
        T ec1_log = -std::log( ec1_q1 * ec1_q1 * ec1_den );
        T ecrs1 = ec1_q0 * ec1_log;
        T decrs1_drs = -2.0 * ec1_aa * ec1_a1 * ec1_log - ec1_q0 * ec1_q1p * ec1_den;
        
        // alpha_zeta is introduced in order to remove singularities for fully polarized systems.
        T zetp_1_3 = (1.0 + zeta * alpha_zeta) * std::pow(zetpm1_3,2.0);
        T zetm_1_3 = (1.0 - zeta * alpha_zeta) * std::pow(zetmm1_3,2.0);

        T f_zeta = ( (1.0 + zeta * alpha_zeta2) * zetp_1_3 + (1.0 - zeta * alpha_zeta2) * zetm_1_3 - 2.0 ) * factf_zeta;
        T fp_zeta = ( zetp_1_3 - zetm_1_3 ) * factfp_zeta;
        T zeta4 = std::pow(zeta, 4.0);

        T gcrs = ecrs1 - ecrs0 + macrs * fsec_inv;
        T ecrs = ecrs0 + f_zeta * (zeta4 * gcrs - macrs * fsec_inv);
        T dgcrs_drs = decrs1_drs - decrs0_drs + dmacrs_drs * fsec_inv;
        T decrs_drs = decrs0_drs + f_zeta * (zeta4 * dgcrs_drs - dmacrs_drs * fsec_inv);
        T dfzeta4_dzeta = 4.0 * std::pow(zeta,3.0) * f_zeta + fp_zeta * zeta4;
        T decrs_dzeta = dfzeta4_dzeta * gcrs - fp_zeta * macrs * fsec_inv;

        ec[i] = ecrs;
        T vxcadd = ecrs - rs * third * decrs_drs - zeta * decrs_dzeta;
        vc[i] = vxcadd + decrs_dzeta;
        vc[DMnd+i] = vxcadd - decrs_dzeta;

        // Eventually add the GGA correlation part of the PBE functional
        // The definition of phi has been slightly changed, because
        // the original PBE one gives divergent behaviour for fully polarized points
        
        T phi_zeta = ( zetpm1_3 * (1.0 + zeta * alpha_zeta) + zetmm1_3 * (1.0 - zeta * alpha_zeta)) * 0.5;
        T phip_zeta = (zetpm1_3 - zetmm1_3) * third * alpha_zeta;
        T phi_zeta_inv = 1.0/phi_zeta;
        T phi_logder = phip_zeta * phi_zeta_inv;
        T phi3_zeta = phi_zeta * phi_zeta * phi_zeta;
        T gamphi3inv = gamma_inv * phi_zeta_inv * phi_zeta_inv * phi_zeta_inv;        
        
        // From ec to bb
        T bb = ecrs * gamphi3inv;
        T dbb_drs = decrs_drs * gamphi3inv;
        T dbb_dzeta = gamphi3inv * (decrs_dzeta - 3.0 * ecrs * phi_logder);

        // From bb to cc
        T exp_pbe = std::exp(-bb);
        T cc = 1.0/(exp_pbe - 1.0);
        T dcc_dbb = cc * cc * exp_pbe;
        T dcc_drs = dcc_dbb * dbb_drs;
        T dcc_dzeta = dcc_dbb * dbb_dzeta;

        // From cc to aa
        T coeff_aa = beta * gamma_inv * phi_zeta_inv * phi_zeta_inv;
        T aa = coeff_aa * cc;
        T daa_drs = coeff_aa * dcc_drs;
        T daa_dzeta = -2.0 * aa * phi_logder + coeff_aa * dcc_dzeta;

        // Introduce tt : do not assume that the spin-dependent gradients are collinear
        T grrho2 = sigma[i];
        T dtt_dg = 2.0 * rhotot_inv * rhotot_inv * rhom1_3 * coeff_tt;
        // Note that tt is (the t variable of PBE divided by phi) squared
        T tt = 0.5 * grrho2 * dtt_dg;

        // Get xx from aa and tt
        T xx = aa * tt;
        T dxx_drs = daa_drs * tt;
        T dxx_dzeta = daa_dzeta * tt;
        T dxx_dtt = aa;

        // From xx to pade
        T pade_den = 1.0/(1.0 + xx * (1.0 + xx));
        T pade = (1.0 + xx) * pade_den;
        T dpade_dxx = -xx * (2.0 + xx) * std::pow(pade_den,2.0);
        T dpade_drs = dpade_dxx * dxx_drs;
        T dpade_dtt = dpade_dxx * dxx_dtt;
        T dpade_dzeta = dpade_dxx * dxx_dzeta;

        // From pade to qq
        T coeff_qq = tt * phi_zeta_inv * phi_zeta_inv;
        T qq = coeff_qq * pade;
        T dqq_drs = coeff_qq * dpade_drs;
        T dqq_dtt = pade * phi_zeta_inv * phi_zeta_inv + coeff_qq * dpade_dtt;
        T dqq_dzeta = coeff_qq * (dpade_dzeta - 2.0 * pade * phi_logder);

        // From qq to rr
        T arg_rr = 1.0 + beta * gamma_inv * qq;
        T div_rr = 1.0/arg_rr;
        T rr = gamma * std::log(arg_rr);
        T drr_dqq = beta * div_rr;
        T drr_drs = drr_dqq * dqq_drs;
        T drr_dtt = drr_dqq * dqq_dtt;
        T drr_dzeta = drr_dqq * dqq_dzeta;

        // From rr to hh
        T hh = phi3_zeta * rr;
        T dhh_drs = phi3_zeta * drr_drs;
        T dhh_dtt = phi3_zeta * drr_dtt;
        T dhh_dzeta = phi3_zeta * (drr_dzeta + 3.0 * rr * phi_logder);

        // The GGA correlation energy is added
        ec[i] += hh;

        // From hh to the derivative of the energy wrt the density
        T drhohh_drho = hh - third * rs * dhh_drs - zeta * dhh_dzeta - (7.0/3.0) * tt * dhh_dtt; 
        vc[i] += drhohh_drho + dhh_dzeta;
        vc[DMnd + i] += drhohh_drho - dhh_dzeta;

        // From hh to the derivative of the energy wrt to the gradient of the
        // density, divided by the gradient of the density
        // (The v3.3 definition includes the division by the norm of the gradient)
        v2c[i] = rho[i] * dtt_dg * dhh_dtt;
    }
}

template<typename T>
void Exchange_correlation_solver<T>::Drho_times_v2xc(Array_3D<T>& Drho_x, Array_3D<T>& Drho_y, Array_3D<T>& Drho_z) const {
    this->Drho_times_v2xc(Drho_x.data, Drho_y.data, Drho_z.data, Drho_x.length, 1);
}

template<typename T>
void Exchange_correlation_solver<T>::Drho_times_v2xc(T* const Drho_x, T* const Drho_y, T* const Drho_z, const uint nd, const uint ncol) const {
    if (this->stencil.cell_type <= 2) {
        Linalg::hadamard_product_general(Drho_x, this->Dxcdgrho.data, nd * ncol);
        Linalg::hadamard_product_general(Drho_y, this->Dxcdgrho.data, nd * ncol);
        Linalg::hadamard_product_general(Drho_z, this->Dxcdgrho.data, nd * ncol);
    } else {
        assert(0 && "cell_type is not supported.");
    }
}

template<typename T>
void Exchange_correlation_solver<T>::cal_DDrho(Array_3D<T>& DDrho_x, Array_3D<T>& DDrho_y, Array_3D<T>& DDrho_z,
    const Array_3D<T>& Drho_x, const Array_3D<T>& Drho_y, const Array_3D<T>& Drho_z) const {
    const uint ncol = 1;
    this->cal_DDrho(DDrho_x.data, DDrho_y.data, DDrho_z.data,
        this->exarr_mpi_package.domain_vertices.get_3D_local_vertices(),
        ncol, Drho_x.data, Drho_y.data, Drho_z.data);
}

template<typename T>
void Exchange_correlation_solver<T>::cal_DDrho(T* const DDrho_x, T* const DDrho_y, T* const DDrho_z, const Vertices_3D& vertices, const uint ncol,
                   T const* const Drho_x, T const* const Drho_y, T const* const Drho_z) const {
    if (this->stencil.cell_type <= 2) {
        uint Nd = vertices.get_size();
        const Vertices_3D ex_vertice(vertices.generate_ex_vertices(this->stencil.FDn));
        T* ex_arr = new T [ex_vertice.get_size()]();
        for (uint icol = 0; icol < ncol; icol++) {
            this->exarr_mpi_package.fill_domain_par_ex_arr(Drho_x + icol * Nd, vertices, ex_arr, ex_vertice);
            Stencil_method::calc_gradient(ex_arr, ex_vertice, 0, stencil, vertices, DDrho_x + icol * Nd, vertices);
            this->exarr_mpi_package.fill_domain_par_ex_arr(Drho_y + icol * Nd, vertices, ex_arr, ex_vertice);
            Stencil_method::calc_gradient(ex_arr, ex_vertice, 1, stencil, vertices, DDrho_y + icol * Nd, vertices);
            this->exarr_mpi_package.fill_domain_par_ex_arr(Drho_z + icol * Nd, vertices, ex_arr, ex_vertice);
            Stencil_method::calc_gradient(ex_arr, ex_vertice, 2, stencil, vertices, DDrho_z + icol * Nd, vertices);
        }
        delete [] ex_arr;
    } else {
        assert(0 && "cell_type is not supported.");
    }
}


template<typename T>
void Exchange_correlation_solver<T>::cal_DDrho_mp(
                        T* const DDrho_x, T* const DDrho_y, T* const DDrho_z,
                        const Vertices_3D& vertices, const uint ncol,
                        T const* const Drho_x, T const* const Drho_y, T const* const Drho_z,
                        Memory_pool<T, Fast_memory>& pool_fast,
                        Memory_pool<T, Capacity_memory>& pool_cap) const {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    if (this->stencil.cell_type <= 2) {
        uint Nd = vertices.get_size();
        const Vertices_3D ex_vertice(vertices.generate_ex_vertices(this->stencil.FDn));
        // T* ex_arr = new T [ex_vertice.get_size()]();
        T* ex_arr = pool_fast.allocate(ex_vertice.get_size());
        Linalg::set_value_general(ex_arr, T(0), ex_vertice.get_size());
        for (uint icol = 0; icol < ncol; icol++) {
            this->exarr_mpi_package.fill_domain_par_ex_arr(Drho_x + icol * Nd, vertices, ex_arr, ex_vertice);
            Stencil_method::calc_gradient(ex_arr, ex_vertice, 0, stencil, vertices, DDrho_x + icol * Nd, vertices);
            this->exarr_mpi_package.fill_domain_par_ex_arr(Drho_y + icol * Nd, vertices, ex_arr, ex_vertice);
            Stencil_method::calc_gradient(ex_arr, ex_vertice, 1, stencil, vertices, DDrho_y + icol * Nd, vertices);
            this->exarr_mpi_package.fill_domain_par_ex_arr(Drho_z + icol * Nd, vertices, ex_arr, ex_vertice);
            Stencil_method::calc_gradient(ex_arr, ex_vertice, 2, stencil, vertices, DDrho_z + icol * Nd, vertices);
        }
    } else {
        assert(0 && "cell_type is not supported.");
    }
}

/**
 * @brief Calculate exchange correlation potential
 *
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/exchangeCorrelation.c#L36
 */
template<typename T>
void Exchange_correlation_solver<T>::cal_exchange_correlation_potential(
                    const std::vector<Array_3D<T>>& electron_densities,
                    const Array_3D<T>& electron_density_core) {
    const uint nspin = this->spin.generate_nspin();
    T const* electron_densities_data[2];
    for (uint ispin = 0; ispin < nspin; ispin++) {
        electron_densities_data[ispin] = electron_densities[ispin].data;
    }
    bool const if_add_core = electron_density_core.length == 0 ? false : true;
    this->cal_exchange_correlation_potential(electron_densities_data, electron_density_core.data, if_add_core);
    return;
}

/**
 * @brief Calculate exchange correlation potential
 *
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/exchangeCorrelation.c#L36
 */
template<typename T>
void Exchange_correlation_solver<T>::cal_exchange_correlation_potential(
                        T const* const* const electron_densities,
                        T const* const electron_density_core,
                        bool const if_add_core) {
    const Vertices_3D vertices = this->exarr_mpi_package.domain_vertices.get_3D_local_vertices();
    const uint nd = vertices.get_size();
    const uint nspin = this->spin.generate_nspin();
    assert(this->spin.get_spin_type() == 0 || this->spin.get_spin_type() == 1);
    const uint ncol = this->spin.get_spin_type() == 0
                    ? 1
                    : this->spin.get_spin_type() == 1
                    ? 3
                    : 0;
    // Array_3D<T> treated_electron_densities = this->pretreat_electron_density(electron_densities, electron_density_core);
    T* treated_electron_densities = new (std::align_val_t(64)) T [nd * ncol];
    this->pretreat_electron_density(treated_electron_densities,
                                    electron_densities,
                                    electron_density_core,
                                    if_add_core);
    if (this->spin.get_spin_type() == 0) {
        // Array_3D<T>& treated_electron_density = treated_electron_densities;
        // Array_3D<T>& exchange_correlation_potential = this->exchange_correlation_potentials[0];
        // Array_3D<T> Drho_x;
        // Array_3D<T> Drho_y;
        // Array_3D<T> Drho_z;
        // Array_3D<T> sigma;
        T*& treated_electron_density = treated_electron_densities;
        T*& exchange_correlation_potential = this->exchange_correlation_potentials[0].data;
        T* Drho_x = nullptr;
        T* Drho_y = nullptr;
        T* Drho_z = nullptr;
        T* sigma = nullptr;
        if (this->exchange_correlation_solver_control.exchange_method == 1
            || this->exchange_correlation_solver_control.correlation_method == 1) {
            // Drho_x.reconstructor(vertices);
            // Drho_y.reconstructor(vertices);
            // Drho_z.reconstructor(vertices);
            // sigma.reconstructor(vertices);
            Drho_x = new (std::align_val_t(64)) T [nd];
            Drho_y = new (std::align_val_t(64)) T [nd];
            Drho_z = new (std::align_val_t(64)) T [nd];
            sigma = new (std::align_val_t(64)) T [nd];
            Parallel_vertices::cal_gradient_d3(treated_electron_density, vertices,
                                                this->stencil, this->exarr_mpi_package, Drho_x, Drho_y, Drho_z);
            this->cal_sigma(sigma, nd, ncol, Drho_x, Drho_y, Drho_z);
        }

        //exchange potential
        switch(this->exchange_correlation_solver_control.exchange_method) {
        case 0:         //LDA_PZ
            // this->exchange_correlation_potential = std::move(this->slater_exchange_potential(treated_electron_density));
            // this->exchange_correlation_energy_density = std::move(this->slater_exchange_energy_density(treated_electron_density));
            // this->slater_exchange(treated_electron_density);
            this->slater_exchange(treated_electron_density, nd);
            break;
        case 1:         //GGA_PBE
            this->pbe_exchange(treated_electron_density, sigma, nd, 1);
            break;
        default:
            assert(0 && "exchange_method is not supported.");
        }

        switch(this->exchange_correlation_solver_control.correlation_method) {
        case 0:             //LDA_PZ
            // this->exchange_correlation_potential += this->pz_correlation_potential(treated_electron_density);
            // this->exchange_correlation_energy_density += this->pz_correlation_energy_density(treated_electron_density);
            this->pz_correlation(treated_electron_density, nd);
            break;
        case 1:             //GGA_PBE
            this->pbe_correlation(treated_electron_density, sigma, nd, 1);
            break;
        default:
            assert(0 && "correlation_method is not supported.");
        }

        if (this->exchange_correlation_solver_control.exchange_method == 1
            || this->exchange_correlation_solver_control.correlation_method == 1) {
                T* DDrho_x = new (std::align_val_t(64)) T [nd];
                T* DDrho_y = new (std::align_val_t(64)) T [nd];
                T* DDrho_z = new (std::align_val_t(64)) T [nd];
                this->Drho_times_v2xc(Drho_x, Drho_y, Drho_z, nd, ncol);
                this->cal_DDrho(DDrho_x, DDrho_y, DDrho_z, vertices, ncol, Drho_x, Drho_y, Drho_z);
                Linalg::hadamard_minus_general(exchange_correlation_potential, DDrho_x, nd);
                Linalg::hadamard_minus_general(exchange_correlation_potential, DDrho_y, nd);
                Linalg::hadamard_minus_general(exchange_correlation_potential, DDrho_z, nd);
                ::operator delete[](DDrho_x, std::align_val_t(64));
                ::operator delete[](DDrho_y, std::align_val_t(64));
                ::operator delete[](DDrho_z, std::align_val_t(64));
        }

        if (this->exchange_correlation_solver_control.exchange_method == 1
            || this->exchange_correlation_solver_control.correlation_method == 1) {
            ::operator delete[](Drho_x, std::align_val_t(64));
            ::operator delete[](Drho_y, std::align_val_t(64));
            ::operator delete[](Drho_z, std::align_val_t(64));
            ::operator delete[](sigma, std::align_val_t(64));
        }
    } else if (this->spin.get_spin_type() == 1) {
        // const uint Nd = vertices.get_size();
        // const uint ncol = 3;
        // const uint nspin = this->spin.generate_nspin();
        // Array_3D<T> treated_electron_density = treated_electron_densities[0] + treated_electron_densities[1];
        T* Drho_x = nullptr;
        T* Drho_y = nullptr;
        T* Drho_z = nullptr;
        T* sigma = nullptr;
        if (this->exchange_correlation_solver_control.exchange_method == 1
            || this->exchange_correlation_solver_control.correlation_method == 1) {
            // Vertices_3D vertices_k_boost(vertices);
            // vertices_k_boost.set_nk(vertices_k_boost.get_nk() * ncol);
            // vertices_k_boost.nk *= ncol;
            Drho_x = new (std::align_val_t(64)) T [nd * ncol];
            Drho_y = new (std::align_val_t(64)) T [nd * ncol];
            Drho_z = new (std::align_val_t(64)) T [nd * ncol];
            sigma = new (std::align_val_t(64)) T [nd * ncol];
            Parallel_vertices::cal_gradient_d3(treated_electron_densities, vertices,
                                                this->stencil, this->exarr_mpi_package, Drho_x, Drho_y, Drho_z);
            Parallel_vertices::cal_gradient_d3(treated_electron_densities + nd, vertices,
                                                this->stencil, this->exarr_mpi_package, Drho_x + nd, Drho_y + nd, Drho_z + nd);
            Parallel_vertices::cal_gradient_d3(treated_electron_densities + nd * 2, vertices,
                                                this->stencil, this->exarr_mpi_package, Drho_x + nd * 2, Drho_y + nd * 2, Drho_z + nd * 2);
            this->cal_sigma(sigma, nd, ncol, Drho_x, Drho_y, Drho_z);
        }
        //exchange
        // Array_0D<T> ex(nd);
        // Array_0D<T> vx(nd * 2);
        // Array_0D<T> v2x(nd * 2);
        T* ex = new (std::align_val_t(64)) T [nd];
        T* vx = new (std::align_val_t(64)) T [nd * 2];
        T* v2x = new (std::align_val_t(64)) T [nd * 2];

        //correlation
        // Array_0D<T> ec(nd);
        // Array_0D<T> vc(nd * 2);
        // Array_0D<T> v2c(nd);
        T* ec = new (std::align_val_t(64)) T [nd];
        T* vc = new (std::align_val_t(64)) T [nd * 2];
        T* v2c = new (std::align_val_t(64)) T [nd];

        switch(this->exchange_correlation_solver_control.exchange_method) {
        case 0:     //LDA_PZ
            assert(0 && "LDA_PZ for spin polarized case (spin_typ = 1) is not implemented");
            break;
        case 1:    //GGA_PBE
            // int xcoption[0] = 1;
            this->pbe_exchange_spin(nd, treated_electron_densities, sigma, 1, ex, vx, v2x);
            break;
        default:
            assert(0 && "exchange_method is not supported.");
        }

        switch(this->exchange_correlation_solver_control.correlation_method) {
        case 0:             //LDA_PZ
            assert(0 && "LDA_PZ for spin polarized case (spin_typ = 1) is not implemented");
            break;
        case 1:             //GGA_PBE
            // int xcoption[1] = 1
            this->pbe_correlation_spin(nd, treated_electron_densities, sigma, 1, ec, vc, v2c);
            break;
        default:
            assert(0 && "correlation_method is not supported.");
        }

        Linalg::hadamard_plus_general(this->exchange_correlation_energy_density.data,
                                    ex, ec, nd);
        for (uint ispin = 0; ispin < nspin; ispin++) {
            Linalg::hadamard_plus_general(this->exchange_correlation_potentials[ispin].data,
                                        vx + ispin * nd, vc + ispin * nd, nd);
        }
        if (this->exchange_correlation_solver_control.exchange_method == 1
         || this->exchange_correlation_solver_control.correlation_method == 1) {
            Linalg::set_value_general(this->Dxcdgrho.data, v2c, nd);
            Linalg::set_value_general(this->Dxcdgrho.data + nd, v2x, nd * 2);
            // Array_0D<T> DDrho_x(nd * ncol);
            // Array_0D<T> DDrho_y(nd * ncol);
            // Array_0D<T> DDrho_z(nd * ncol);
            T* DDrho_x = new (std::align_val_t(64)) T [nd * ncol];
            T* DDrho_y = new (std::align_val_t(64)) T [nd * ncol];
            T* DDrho_z = new (std::align_val_t(64)) T [nd * ncol];
            this->Drho_times_v2xc(Drho_x, Drho_y, Drho_z, nd, ncol);
            this->cal_DDrho(DDrho_x, DDrho_y, DDrho_z, vertices, ncol,
                        Drho_x, Drho_y, Drho_z);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[0].data, DDrho_x, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[0].data, DDrho_y, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[0].data, DDrho_z, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[0].data, DDrho_x + nd, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[0].data, DDrho_y + nd, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[0].data, DDrho_z + nd, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[1].data, DDrho_x, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[1].data, DDrho_y, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[1].data, DDrho_z, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[1].data, DDrho_x + nd * 2, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[1].data, DDrho_y + nd * 2, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[1].data, DDrho_z + nd * 2, nd);
            ::operator delete[](DDrho_x, std::align_val_t(64));
            ::operator delete[](DDrho_y, std::align_val_t(64));
            ::operator delete[](DDrho_z, std::align_val_t(64));
        }

        ::operator delete[](ex, std::align_val_t(64));
        ::operator delete[](vx, std::align_val_t(64));
        ::operator delete[](v2x, std::align_val_t(64));
        ::operator delete[](ec, std::align_val_t(64));
        ::operator delete[](vc, std::align_val_t(64));
        ::operator delete[](v2c, std::align_val_t(64));
        if (this->exchange_correlation_solver_control.exchange_method == 1
            || this->exchange_correlation_solver_control.correlation_method == 1) {
            ::operator delete[](Drho_x, std::align_val_t(64));
            ::operator delete[](Drho_y, std::align_val_t(64));
            ::operator delete[](Drho_z, std::align_val_t(64));
            ::operator delete[](sigma, std::align_val_t(64));
        }
        // assert(0 && "spin_type is not supported.");
    } else {
        assert(this->spin.get_spin_type() == 0 || this->spin.get_spin_type() == 1);
    }
    ::operator delete[](treated_electron_densities, std::align_val_t(64));
    return;
}


/**
 * @brief Calculate exchange correlation potential
 *
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/exchangeCorrelation.c#L36
 */
template<typename T>
void Exchange_correlation_solver<T>::cal_exchange_correlation_potential_mp(
                            T const* const* const electron_densities,
                            T const* const electron_density_core,
                            bool const if_add_core,
                            Memory_pool<T, Fast_memory>& pool_fast,
                            Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    const Vertices_3D vertices = this->exarr_mpi_package.domain_vertices.get_3D_local_vertices();
    const uint nd = vertices.get_size();
    const uint nspin = this->spin.generate_nspin();
    assert(this->spin.get_spin_type() == 0 || this->spin.get_spin_type() == 1);
    const uint ncol = this->spin.get_spin_type() == 0
                    ? 1
                    : this->spin.get_spin_type() == 1
                    ? 3
                    : 0;
    // Array_3D<T> treated_electron_densities = this->pretreat_electron_density(electron_densities, electron_density_core);
    // T* treated_electron_densities = new (std::align_val_t(64)) T [nd * ncol];
    T* treated_electron_densities = pool_fast.allocate(nd * ncol);
    this->pretreat_electron_density(treated_electron_densities,
                                    electron_densities,
                                    electron_density_core,
                                    if_add_core);
    if (this->spin.get_spin_type() == 0) {
        Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast2(pool_fast);
        Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap2(pool_cap);
        // Array_3D<T>& treated_electron_density = treated_electron_densities;
        // Array_3D<T>& exchange_correlation_potential = this->exchange_correlation_potentials[0];
        // Array_3D<T> Drho_x;
        // Array_3D<T> Drho_y;
        // Array_3D<T> Drho_z;
        // Array_3D<T> sigma;
        T*& treated_electron_density = treated_electron_densities;
        T*& exchange_correlation_potential = this->exchange_correlation_potentials[0].data;
        T* Drho_x = nullptr;
        T* Drho_y = nullptr;
        T* Drho_z = nullptr;
        T* sigma = nullptr;
        if (this->exchange_correlation_solver_control.exchange_method == 1
            || this->exchange_correlation_solver_control.correlation_method == 1) {
            // Drho_x.reconstructor(vertices);
            // Drho_y.reconstructor(vertices);
            // Drho_z.reconstructor(vertices);
            // sigma.reconstructor(vertices);
            // Drho_x = new (std::align_val_t(64)) T [nd];
            // Drho_y = new (std::align_val_t(64)) T [nd];
            // Drho_z = new (std::align_val_t(64)) T [nd];
            // sigma = new (std::align_val_t(64)) T [nd];
            Drho_x = pool_fast.allocate(nd);
            Drho_y = pool_fast.allocate(nd);
            Drho_z = pool_fast.allocate(nd);
            sigma = pool_fast.allocate(nd);
            Parallel_vertices::cal_gradient_d3(treated_electron_density, vertices,
                                                this->stencil, this->exarr_mpi_package, Drho_x, Drho_y, Drho_z);
            this->cal_sigma(sigma, nd, ncol, Drho_x, Drho_y, Drho_z);
        }

        //exchange potential
        switch(this->exchange_correlation_solver_control.exchange_method) {
        case 0:         //LDA_PZ
            // this->exchange_correlation_potential = std::move(this->slater_exchange_potential(treated_electron_density));
            // this->exchange_correlation_energy_density = std::move(this->slater_exchange_energy_density(treated_electron_density));
            // this->slater_exchange(treated_electron_density);
            this->slater_exchange_mp(treated_electron_density, nd, pool_fast, pool_cap);
            break;
        case 1:         //GGA_PBE
            this->pbe_exchange(treated_electron_density, sigma, nd, 1);
            break;
        default:
            assert(0 && "exchange_method is not supported.");
        }

        switch(this->exchange_correlation_solver_control.correlation_method) {
        case 0:             //LDA_PZ
            // this->exchange_correlation_potential += this->pz_correlation_potential(treated_electron_density);
            // this->exchange_correlation_energy_density += this->pz_correlation_energy_density(treated_electron_density);
            this->pz_correlation_mp(treated_electron_density, nd, pool_fast, pool_cap);
            break;
        case 1:             //GGA_PBE
            this->pbe_correlation_mp(treated_electron_density, sigma, nd, 1, pool_fast, pool_cap);
            break;
        default:
            assert(0 && "correlation_method is not supported.");
        }

        if (this->exchange_correlation_solver_control.exchange_method == 1
            || this->exchange_correlation_solver_control.correlation_method == 1) {
                // T* DDrho_x = new (std::align_val_t(64)) T [nd];
                // T* DDrho_y = new (std::align_val_t(64)) T [nd];
                // T* DDrho_z = new (std::align_val_t(64)) T [nd];
                T* DDrho_x = pool_fast.allocate(nd);
                T* DDrho_y = pool_fast.allocate(nd);
                T* DDrho_z = pool_fast.allocate(nd);
                this->Drho_times_v2xc(Drho_x, Drho_y, Drho_z, nd, ncol);
                this->cal_DDrho_mp(DDrho_x, DDrho_y, DDrho_z, vertices, ncol, Drho_x, Drho_y, Drho_z,
                                    pool_fast, pool_cap);
                Linalg::hadamard_minus_general(exchange_correlation_potential, DDrho_x, nd);
                Linalg::hadamard_minus_general(exchange_correlation_potential, DDrho_y, nd);
                Linalg::hadamard_minus_general(exchange_correlation_potential, DDrho_z, nd);
        }
    } else if (this->spin.get_spin_type() == 1) {
        Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast2(pool_fast);
        Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap2(pool_cap);
        // const uint Nd = vertices.get_size();
        // const uint ncol = 3;
        // const uint nspin = this->spin.generate_nspin();
        // Array_3D<T> treated_electron_density = treated_electron_densities[0] + treated_electron_densities[1];
        T* Drho_x = nullptr;
        T* Drho_y = nullptr;
        T* Drho_z = nullptr;
        T* sigma = nullptr;
        if (this->exchange_correlation_solver_control.exchange_method == 1
            || this->exchange_correlation_solver_control.correlation_method == 1) {
            // Vertices_3D vertices_k_boost(vertices);
            // vertices_k_boost.set_nk(vertices_k_boost.get_nk() * ncol);
            // vertices_k_boost.nk *= ncol;
            // Drho_x = new (std::align_val_t(64)) T [nd * ncol];
            // Drho_y = new (std::align_val_t(64)) T [nd * ncol];
            // Drho_z = new (std::align_val_t(64)) T [nd * ncol];
            // sigma = new (std::align_val_t(64)) T [nd * ncol];
            Drho_x = pool_fast.allocate(nd * ncol);
            Drho_y = pool_fast.allocate(nd * ncol);
            Drho_z = pool_fast.allocate(nd * ncol);
            sigma = pool_fast.allocate(nd * ncol);
            Parallel_vertices::cal_gradient_d3(treated_electron_densities, vertices,
                                                this->stencil, this->exarr_mpi_package, Drho_x, Drho_y, Drho_z);
            Parallel_vertices::cal_gradient_d3(treated_electron_densities + nd, vertices,
                                                this->stencil, this->exarr_mpi_package, Drho_x + nd, Drho_y + nd, Drho_z + nd);
            Parallel_vertices::cal_gradient_d3(treated_electron_densities + nd * 2, vertices,
                                                this->stencil, this->exarr_mpi_package, Drho_x + nd * 2, Drho_y + nd * 2, Drho_z + nd * 2);
            this->cal_sigma(sigma, nd, ncol, Drho_x, Drho_y, Drho_z);
        }
        //exchange
        // Array_0D<T> ex(nd);
        // Array_0D<T> vx(nd * 2);
        // Array_0D<T> v2x(nd * 2);
        // T* ex = new (std::align_val_t(64)) T [nd];
        // T* vx = new (std::align_val_t(64)) T [nd * 2];
        // T* v2x = new (std::align_val_t(64)) T [nd * 2];
        T* ex = pool_fast.allocate(nd);
        T* vx = pool_fast.allocate(nd * 2);
        T* v2x = pool_fast.allocate(nd * 2);

        //correlation
        // Array_0D<T> ec(nd);
        // Array_0D<T> vc(nd * 2);
        // Array_0D<T> v2c(nd);
        // T* ec = new (std::align_val_t(64)) T [nd];
        // T* vc = new (std::align_val_t(64)) T [nd * 2];
        // T* v2c = new (std::align_val_t(64)) T [nd];
        T* ec = pool_fast.allocate(nd);
        T* vc = pool_fast.allocate(nd * 2);
        T* v2c = pool_fast.allocate(nd);

        switch(this->exchange_correlation_solver_control.exchange_method) {
        case 0:     //LDA_PZ
            assert(0 && "LDA_PZ for spin polarized case (spin_typ = 1) is not implemented");
            break;
        case 1:    //GGA_PBE
            // int xcoption[0] = 1;
            this->pbe_exchange_spin(nd, treated_electron_densities, sigma, 1, ex, vx, v2x);
            break;
        default:
            assert(0 && "exchange_method is not supported.");
        }

        switch(this->exchange_correlation_solver_control.correlation_method) {
        case 0:             //LDA_PZ
            assert(0 && "LDA_PZ for spin polarized case (spin_typ = 1) is not implemented");
            break;
        case 1:             //GGA_PBE
            // int xcoption[1] = 1
            this->pbe_correlation_spin(nd, treated_electron_densities, sigma, 1, ec, vc, v2c);
            break;
        default:
            assert(0 && "correlation_method is not supported.");
        }

        Linalg::hadamard_plus_general(this->exchange_correlation_energy_density.data,
                                    ex, ec, nd);
        for (uint ispin = 0; ispin < nspin; ispin++) {
            Linalg::hadamard_plus_general(this->exchange_correlation_potentials[ispin].data,
                                        vx + ispin * nd, vc + ispin * nd, nd);
        }
        if (this->exchange_correlation_solver_control.exchange_method == 1
         || this->exchange_correlation_solver_control.correlation_method == 1) {
            Linalg::set_value_general(this->Dxcdgrho.data, v2c, nd);
            Linalg::set_value_general(this->Dxcdgrho.data + nd, v2x, nd * 2);
            // Array_0D<T> DDrho_x(nd * ncol);
            // Array_0D<T> DDrho_y(nd * ncol);
            // Array_0D<T> DDrho_z(nd * ncol);
            // T* DDrho_x = new (std::align_val_t(64)) T [nd * ncol];
            // T* DDrho_y = new (std::align_val_t(64)) T [nd * ncol];
            // T* DDrho_z = new (std::align_val_t(64)) T [nd * ncol];
            T* DDrho_x = pool_fast.allocate(nd * ncol);
            T* DDrho_y = pool_fast.allocate(nd * ncol);
            T* DDrho_z = pool_fast.allocate(nd * ncol);
            this->Drho_times_v2xc(Drho_x, Drho_y, Drho_z, nd, ncol);
            this->cal_DDrho_mp(DDrho_x, DDrho_y, DDrho_z, vertices, ncol,
                            Drho_x, Drho_y, Drho_z, pool_fast, pool_cap);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[0].data, DDrho_x, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[0].data, DDrho_y, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[0].data, DDrho_z, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[0].data, DDrho_x + nd, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[0].data, DDrho_y + nd, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[0].data, DDrho_z + nd, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[1].data, DDrho_x, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[1].data, DDrho_y, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[1].data, DDrho_z, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[1].data, DDrho_x + nd * 2, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[1].data, DDrho_y + nd * 2, nd);
            Linalg::hadamard_minus_general(exchange_correlation_potentials[1].data, DDrho_z + nd * 2, nd);
        }
        // assert(0 && "spin_type is not supported.");
    } else {
        assert(this->spin.get_spin_type() == 0 || this->spin.get_spin_type() == 1);
    }
    // ::operator delete[](treated_electron_densities, std::align_val_t(64));
    return;
}

template<typename T>
T& Exchange_correlation_solver<T>::evaluate_exchange_correlation_energy(const std::vector<Array_3D<T>>& electron_densities,
                                                                        const Array_3D<T>& electron_density_core,
                                                                        const T& dv, const MPI_Comm& comm) {
    // std::vector<Array_3D<T>> temp = this->pretreat_electron_density(electron_densities);
    Array_3D<T> treated_electron_densities = this->pretreat_electron_density(electron_densities, electron_density_core);
    if (this->spin.get_spin_type() == 0 || this->spin.get_spin_type() == 1) {
        const uint Nd = electron_densities[0].length;
        this->exchange_correlation_energy = Linalg::vector_dot_product(
            treated_electron_densities.data, this->exchange_correlation_energy_density.data, Nd, comm);
    } else {
        assert(this->spin.get_spin_type() == 0 || this->spin.get_spin_type() == 1);
    }
    this->exchange_correlation_energy *= dv;
    return this->exchange_correlation_energy;
}

template<typename T>
void Exchange_correlation_solver<T>::init(const Vertices_3D& vertices) {
    assert(this->spin.get_spin_type() == 0 || this->spin.get_spin_type() == 1);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    const uint nspin = this->spin.generate_nspin();
    this->exchange_correlation_potentials.reserve(nspin);
    for (uint ispin = 0; ispin < nspin; ispin++) {
        this->exchange_correlation_potentials.emplace_back(vertices);
    }
    this->exchange_correlation_energy_density.reconstructor(vertices);
    if (this->exchange_correlation_solver_control.exchange_method == 1
    ||  this->exchange_correlation_solver_control.correlation_method == 1) {
        if (this->spin.get_spin_type() == 0) {
            this->Dxcdgrho.reconstructor(vertices.get_size());
        } else if (this->spin.get_spin_type() == 1) {
            this->Dxcdgrho.reconstructor(vertices.get_size() * 3);
        } else {
            assert(this->spin.get_spin_type() == 0 || this->spin.get_spin_type() == 1);
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "The Exchange_correlation_solver init took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

template<typename T>
template<typename T2>
void Exchange_correlation_solver<T>::init(const Exchange_correlation_solver<T2>& exchange_correlation_solver) {
    T temp = T(0);
    std::vector<Array_3D<T>>().swap(this->exchange_correlation_potentials);
    const uint nspin = this->spin.generate_nspin();
    this->exchange_correlation_potentials.reserve(nspin);
    for (uint ispin = 0; ispin < nspin; ispin++) {
        this->exchange_correlation_potentials.emplace_back(
            exchange_correlation_solver.exchange_correlation_potentials[ispin].as_type(&temp));
    }
    this->Dxcdgrho.deepcopy(
        std::move(exchange_correlation_solver.Dxcdgrho.as_type(
            this->Dxcdgrho.data)));
    this->exchange_correlation_energy_density.deepcopy(
        std::move(exchange_correlation_solver.exchange_correlation_energy_density.as_type(
            this->exchange_correlation_energy_density.data)));
    return;
}
template void Exchange_correlation_solver<float>::init(const Exchange_correlation_solver<float>& exchange_correlation_solver);
template void Exchange_correlation_solver<double>::init(const Exchange_correlation_solver<double>& exchange_correlation_solver);
template void Exchange_correlation_solver<float>::init(const Exchange_correlation_solver<double>& exchange_correlation_solver);
template void Exchange_correlation_solver<double>::init(const Exchange_correlation_solver<float>& exchange_correlation_solver);

template<typename T>
void Exchange_correlation_solver<T>::destructor() {
    std::vector<Array_3D<T>>().swap(this->exchange_correlation_potentials);
    this->exchange_correlation_energy_density.destructor();
    this->Dxcdgrho.destructor();
    return;
}

template<typename T>
void Exchange_correlation_solver<T>::show() const {
    // this->exchange_correlation_potential.show();
    // this->exchange_correlation_energy_density.show();
    return;
}

template class Exchange_correlation_solver<float>;
template class Exchange_correlation_solver<double>;
