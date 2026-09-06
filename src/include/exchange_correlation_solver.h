#ifndef _EXCHANGE_CORRELATION_SOLVER_H_
#define _EXCHANGE_CORRELATION_SOLVER_H_

#include "arr.h"
#include "tools.h"
#include "control.h"
#include "spin.h"

template<typename T>
class Exchange_correlation_solver
{
public:
    const Exchange_correlation_solver_control& exchange_correlation_solver_control;
    const Spin& spin;
    const Stencil<T>& stencil;
    const Exarr_3D_mpi_package& exarr_mpi_package;
    T exchange_correlation_energy = (T)0.0;
    std::vector<Array_3D<T>> exchange_correlation_potentials;
    Array_3D<T> exchange_correlation_energy_density;
    Array_0D<T> Dxcdgrho;
    Exchange_correlation_solver(const Exchange_correlation_solver_control& exchange_correlation_solver_control,
                                const Spin& spin,
                                const Stencil<T>& stencil,
                                const Exarr_3D_mpi_package& exarr_mpi_package);
    ~Exchange_correlation_solver();
    // std::vector<Array_3D<T>> pretreat_electron_density(const std::vector<Array_3D<T>>& electron_densities);
    Array_3D<T> pretreat_electron_density(const std::vector<Array_3D<T>>& electron_densities,
                                          const Array_3D<T>& electron_density_core);
    void pretreat_electron_density(T* const treated_electron_densities,
                                   T const* const* const electron_densities,
                                   T const* const electron_density_core,
                                   bool const if_add_core);                          
    void cal_sigma(Array_3D<T>& sigma, const Array_3D<T>& Drho_x, const Array_3D<T>& Drho_y, const Array_3D<T>& Drho_z);
    void cal_sigma(T* sigma, const uint nd, const uint ncol, T const* const Drho_x, T const* const Drho_y, T const* const Drho_z);
    Array_3D<T> slater_exchange_potential(const Array_3D<T>& electron_density);
    Array_3D<T> slater_exchange_energy_density(const Array_3D<T>& electron_density);
    void slater_exchange(const Array_3D<T>& electron_density);
    void slater_exchange(T const* const electron_density, uint const nd);
    void slater_exchange_mp(T const* const electron_density, uint const nd,
                            Memory_pool<T, Fast_memory>& pool_fast,
                            Memory_pool<T, Capacity_memory>& pool_cap);
    void pbe_exchange_kernel(const Array_3D<T>& electron_density, const Array_3D<T>& sigma,
        Array_3D<T>& ex, Array_3D<T>& vx, Array_0D<T>& v2x, const int flag = 1);
    void pbe_exchange_kernel(T const* const electron_density, T const* const sigma,
        T* const ex, T* const vx, T* const v2x, const uint nd, const int flag);
    void pbe_exchange(const Array_3D<T>& electron_density, const Array_3D<T>& sigma, const int flag = 1);
    void pbe_exchange(T const* const electron_density, T const* const sigma, const uint nd, const int flag);
    void pbe_exchange_spin(const uint& DMnd, T const* const& rho, T const* const& sigma, const int& iflag, T* const& ex, T* const& vx, T* const& v2x);
    Array_3D<T> pz_correlation_potential(const Array_3D<T>& electron_density);
    Array_3D<T> pz_correlation_energy_density(const Array_3D<T>& electron_density);
    void pz_correlation(const Array_3D<T>& electron_density);
    void pz_correlation(T const* const electron_density, uint const nd);
    void pz_correlation_mp(T const* const electron_density, uint const nd,
                            Memory_pool<T, Fast_memory>& pool_fast,
                            Memory_pool<T, Capacity_memory>& pool_cap);
    void pbe_correlation_kernel(const Array_3D<T>& electron_density, const Array_3D<T>& sigma,
        Array_3D<T>& ec, Array_3D<T>& vc, Array_3D<T>& v2c, const int flag = 1);
    void pbe_correlation_kernel(T const* const electron_density, T const* const sigma,
        T* const ec, T* const vc, T* const v2c, const uint nd, const int flag);
    void pbe_correlation(const Array_3D<T>& electron_density, const Array_3D<T>& sigma, const int flag = 1);
    void pbe_correlation(T const* const electron_density, T const* const sigma, uint const nd, const int flag);
    void pbe_correlation_mp(T const* const electron_density, T const* const sigma, uint const nd, const int flag,
                                            Memory_pool<T, Fast_memory>& pool_fast,
                                            Memory_pool<T, Capacity_memory>& pool_cap);
    void pbe_correlation_spin(const uint DMnd, T const* const rho, T const* const sigma, const int iflag, T* const ec, T* const vc, T* const v2c);
    void Drho_times_v2xc(Array_3D<T>& Drho_x, Array_3D<T>& Drho_y, Array_3D<T>& Drho_z) const;
    void Drho_times_v2xc(T* const Drho_x, T* const Drho_y, T* const Drho_z, const uint nd, const uint ncol) const;
    void cal_DDrho(Array_3D<T>& DDrho_x, Array_3D<T>& DDrho_y, Array_3D<T>& DDrho_z,
                   const Array_3D<T>& Drho_x, const Array_3D<T>& Drho_y, const Array_3D<T>& Drho_z) const;
    void cal_DDrho(T* const DDrho_x, T* const DDrho_y, T* const DDrho_z, const Vertices_3D& vertices, const uint ncol,
                   T const* const Drho_x, T const* const Drho_y, T const* const Drho_z) const;
    void cal_DDrho_mp(T* const DDrho_x, T* const DDrho_y, T* const DDrho_z, const Vertices_3D& vertices, const uint ncol,
                    T const* const Drho_x, T const* const Drho_y, T const* const Drho_z,
                    Memory_pool<T, Fast_memory>& pool_fast,
                    Memory_pool<T, Capacity_memory>& pool_cap) const;
    void cal_exchange_correlation_potential(const std::vector<Array_3D<T>>& electron_densities,
                                            const Array_3D<T>& electron_density_core);
    void cal_exchange_correlation_potential(T const* const* const electron_densities,
                                            T const* const electron_density_core,
                                            bool const if_add_core);
    void cal_exchange_correlation_potential_mp(T const* const* const electron_densities,
                                            T const* const electron_density_core,
                                            bool const if_add_core,
                                            Memory_pool<T, Fast_memory>& pool_fast,
                                            Memory_pool<T, Capacity_memory>& pool_cap);
    T& evaluate_exchange_correlation_energy(const std::vector<Array_3D<T>>& electron_densities,
                                            const Array_3D<T>& electron_density_core,
                                            const T& dv, const MPI_Comm& comm);
    void init(const Vertices_3D& vertices);
    template<typename T2> void init(const Exchange_correlation_solver<T2>& exchange_correlation_solver);
    void destructor();
    void show() const;
};

#endif //_EXCHANGE_CORRELATION_SOLVER_H_