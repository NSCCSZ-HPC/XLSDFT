#ifndef _XLSDFT_H_
#define _XLSDFT_H_

#include "eigen_solver.h"
#include "laplacian_head_lottery.h"

#if !defined(ENABLE_XLSDFT_TIMER) && defined(ENABLE_TIMER)
#define ENABLE_XLSDFT_TIMER
#endif

#ifdef ENABLE_XLSDFT_TIMER
#include "timer.h"
class Xlsdft_timer
{
public:
    Timer xlsdft;
    Timer xlsdft_copy_veff;
    #if (defined(LOW_MEMORY))
    Timer xlsdft_low_memory_psi;
    #endif
    Timer xlsdft_eigen_solver;
    Timer xlsdft_barrier1;
    Timer xlsdft_barrier2;
    Timer xlsdft_barrier3;
    Xlsdft_timer();
    ~Xlsdft_timer();
    void reset();
    void show(std::ostream& output = std::cout) const;
};

class Xlsdft_performance
{
public:
    enum Item
    {
        CHEFSI = 0,
        LANCZOS,
        FILTER,
        FILTER_COPY,
        FILTER_PRODUCT,
        FILTER_LAP,
        FILTER_NLOC,
        H_PSI,
        PROJECTION,
        DIAGONALIZATION,
        ROTATION,

        EIGEN_SOLVER,
#if defined(LOW_MEMORY)
        EIGEN_SOLVER_LOW_MEMORY_CHI,
#endif
        EIGEN_SOLVER_KERNEL,

        XLSDFT,
        XLSDFT_COPY_VEFF,
        XLSDFT_LOW_MEMORY_PSI,
        XLSDFT_EIGEN_SOLVER,

        NITEM
    };

    double local_flops[NITEM];
    double flops[NITEM];

    Xlsdft_performance()
    {
        reset();
    }

    void reset()
    {
        for (int i = 0; i < NITEM; ++i) {
            local_flops[i] = 0.0;
            flops[i] = 0.0;
        }
    }

    template<typename T>
    void update(const std::vector<Eigen_solver<T>>& eigen_solvers,
                const Xlsdft_timer& xlsdft_timer)
    {
        reset();

        auto calc_flops = [](const double flop_count,
                             const Timer& timer) -> double
        {
            const double time =
                std::chrono::duration<double>(timer.elapsed).count();

            if (time <= 0.0)
                return 0.0;

            return flop_count / time / 1.0e9;
        };

        Chefsi_timer chefsi_timer_total;
        Chefsi_flop_counter chefsi_flop_counter_total;
        Eigen_solver_timer eigen_solver_timer_total;

        chefsi_timer_total.reset();
        chefsi_flop_counter_total.reset();
        eigen_solver_timer_total.reset();

        for (uint ielement = 0;
             ielement < eigen_solvers.size();
             ++ielement)
        {
            const Chefsi_timer& chefsi_timer =
                eigen_solvers[ielement].chefsi.chefsi_timer;

            const Chefsi_flop_counter& chefsi_flop_counter =
                eigen_solvers[ielement].chefsi.chefsi_flop_counter;

            const Eigen_solver_timer& eigen_solver_timer =
                eigen_solvers[ielement].eigen_solver_timer;

            chefsi_timer_total += chefsi_timer;
            chefsi_flop_counter_total += chefsi_flop_counter;
            eigen_solver_timer_total += eigen_solver_timer;
        }

        /*
         * CheFSI performance
         */
        Chefsi_performance chefsi_performance_total(
            chefsi_timer_total,
            chefsi_flop_counter_total);

        // 必须显式 update，因为构造函数只 reset
        chefsi_performance_total.update();

        local_flops[CHEFSI] =
            chefsi_performance_total.local_flops[
                Chefsi_performance::CHEFSI];

        local_flops[LANCZOS] =
            chefsi_performance_total.local_flops[
                Chefsi_performance::LANCZOS];

        local_flops[FILTER] =
            chefsi_performance_total.local_flops[
                Chefsi_performance::FILTER];

        local_flops[FILTER_COPY] =
            chefsi_performance_total.local_flops[
                Chefsi_performance::FILTER_COPY];

        local_flops[FILTER_PRODUCT] =
            chefsi_performance_total.local_flops[
                Chefsi_performance::FILTER_PRODUCT];

        local_flops[FILTER_LAP] =
            chefsi_performance_total.local_flops[
                Chefsi_performance::FILTER_LAP];

        local_flops[FILTER_NLOC] =
            chefsi_performance_total.local_flops[
                Chefsi_performance::FILTER_NLOC];

        local_flops[H_PSI] =
            chefsi_performance_total.local_flops[
                Chefsi_performance::H_PSI];

        local_flops[PROJECTION] =
            chefsi_performance_total.local_flops[
                Chefsi_performance::PROJECTION];

        local_flops[DIAGONALIZATION] =
            chefsi_performance_total.local_flops[
                Chefsi_performance::DIAGONALIZATION];

        local_flops[ROTATION] =
            chefsi_performance_total.local_flops[
                Chefsi_performance::ROTATION];

        /*
         * Total CheFSI FLOP count
         */
        const double filter_flop_count_total =
              chefsi_flop_counter_total.filter_copy
            + chefsi_flop_counter_total.filter_product
            + chefsi_flop_counter_total.filter_lap
            + chefsi_flop_counter_total.filter_nloc;

        const double chefsi_flop_count_total =
              chefsi_flop_counter_total.lanczos
            + filter_flop_count_total
            + chefsi_flop_counter_total.H_psi
            + chefsi_flop_counter_total.projection
            + chefsi_flop_counter_total.diagonalization
            + chefsi_flop_counter_total.rotation;

        /*
         * Eigen solver
         */
        local_flops[EIGEN_SOLVER] =
            calc_flops(
                chefsi_flop_count_total,
                eigen_solver_timer_total.eigen_solver);

#if defined(LOW_MEMORY)
        local_flops[EIGEN_SOLVER_LOW_MEMORY_CHI] = 0.0;
#endif

        local_flops[EIGEN_SOLVER_KERNEL] =
            calc_flops(
                chefsi_flop_count_total,
                eigen_solver_timer_total.eigen_solver_kernel);

        /*
         * XLSDFT
         */
        local_flops[XLSDFT] =
            calc_flops(
                chefsi_flop_count_total,
                xlsdft_timer.xlsdft);

        local_flops[XLSDFT_COPY_VEFF] = 0.0;

        local_flops[XLSDFT_LOW_MEMORY_PSI] = 0.0;

        local_flops[XLSDFT_EIGEN_SOLVER] =
            calc_flops(
                chefsi_flop_count_total,
                xlsdft_timer.xlsdft_eigen_solver);

        /*
         * Before MPI reduction:
         * flops == local_flops
         */
        for (int i = 0; i < NITEM; ++i)
            flops[i] = local_flops[i];
    }

    void reduce(const MPI_Comm comm, const int root = 0)
    {
        MPI_Reduce(local_flops,
                   flops,
                   NITEM,
                   MPI_DOUBLE,
                   MPI_SUM,
                   root,
                   comm);
    }

    void allreduce(const MPI_Comm comm)
    {
        MPI_Allreduce(local_flops,
                      flops,
                      NITEM,
                      MPI_DOUBLE,
                      MPI_SUM,
                      comm);
    }

    void show(const bool if_print,
              std::ostream& output = std::cout) const
    {
        if (!if_print)
            return;

        static const char* names[NITEM] = {
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

            "Eigen solver",
#if defined(LOW_MEMORY)
            "  Low memory chi",
#endif
            "  Eigen solver kernel",

            "XLSDFT",
            "  Copy Veff",
            "  Low memory psi",
            "  Eigen solver"
        };

        output << '\n';

        output << std::left
               << std::setw(32) << "XLSDFT performance"
               << std::right
               << std::setw(20) << "Local GFLOP/s"
               << std::setw(20) << "Total GFLOP/s"
               << '\n';

        output << std::string(72, '-') << '\n';

        for (int i = 0; i < NITEM; ++i) {
            output << std::left
                   << std::setw(32) << names[i]
                   << std::right
                   << std::setw(20)
                   << std::fixed
                   << std::setprecision(3)
                   << local_flops[i]
                   << std::setw(20)
                   << flops[i]
                   << '\n';
        }

        output << std::string(72, '-') << '\n';
    }
};
#endif //ENABLE_XLSDFT_TIMER

template<typename T>
class Xlsdft
{
public:
    const Xlsdft_control& xlsdft_control;
    const Mesh_control& mesh_control;
    const Geometry& geometry;
    const Stencil<T>& stencil;
    const Domain_parallel_vertices_4D& domain_vertices;
    #ifdef ENABLE_XLSDFT_TIMER
    Xlsdft_timer xlsdft_timer;
    #endif //ENABLE_XLSDFT_TIMER
    // varables in the element comm
    uint local_element_num;
    std::vector<uint> element_indexes;
    std::vector<Vertices_3D> element_verticeses;
    std::vector<Geometry> ex_geometries; // ex element geometries
    // for eigen_solvers
    std::vector<Vertices_3D> shared_verticeses;
    std::vector<Eigen_solver_control> eigen_solver_controls;
    std::vector<Mesh_control> mesh_controls;
    std::vector<Domain_parallel_vertices_4D> domain_verticeses;
    std::vector<Eigen_solver<T>> eigen_solvers;
    std::vector<T> U0s;
    #if defined(LOW_MEMORY)
    #if defined(FP16_FLAG)
    std::vector<Array_4D<__fp16>>
    #else
    std::vector<Array_4D<float>>
    #endif
    eigen_vectorses_lp; //eigen_vectorses low precision
    #endif
    // Persistent packed ψ panels in pool_fast (opt path); see reserve_retained_packed_pool().
    triple_head::Panels triple_panels_;
    double* retained_packed_psi_ = nullptr;
    double* wf_scratch_panel_ = nullptr;
    double* packed_live_ = nullptr;
    double* packed_alt_ = nullptr;
    bool retained_pool_reserved_ = false;
    bool retained_packed_initialized_ = false;
    Xlsdft(const Xlsdft_control& xlsdft_control,
           const Mesh_control& mesh_control,
           const Geometry& geometry,
           const Stencil<T>& stencil,
           const Domain_parallel_vertices_4D& domain_vertices);
    ~Xlsdft();
    void cal_int_density_per_band(std::vector<T>& eigen_value_per_band,
                                  std::vector<T>& int_density_per_band) const;
    void cal_int_density_per_band2(std::vector<T>& eigen_value_per_band,
                                   std::vector<T>& int_density_per_band) const;
    T get_max_eigen_value() const;
    T get_min_eigen_value() const;
    void get_eigen_value_range(T* const range) const;
    Array_3D<T> cal_electron_charge_density(const Smearing& smearing, const T smearing_coef, const T chemical_potential) const;
    void cal_electron_charge_density_mp(T* const electron_density, const Smearing& smearing,
                                        const T smearing_coef, const T chemical_potential,
                                        Memory_pool<T, Fast_memory>& pool_fast,
                                        Memory_pool<T, Capacity_memory>& pool_cap) const;
    Array_3D<T> cal_electron_charge_density_with_fg_electron(const Smearing& smearing, const T& smearing_coef, const T& chemical_potential) const;
    void cal_electron_charge_density_with_fg_electron_mp(T* const electron_density, const Smearing& smearing,
                                        const T smearing_coef, const T chemical_potential,
                                        Memory_pool<T, Fast_memory>& pool_fast,
                                        Memory_pool<T, Capacity_memory>& pool_cap) const;
    T cal_electron_charge(const Smearing& smearing, const T& smearing_coef, const T& chemical_potential) const;
    T evaluate_chemical_potential(const Smearing& smearing, const T electron_charge,
                                  const T smearing_coef, const T trial_chemical_potential) const;
    T evaluate_chemical_potential2(const Smearing& smearing, const T electron_charge,
                                  const T smearing_coef, const T trial_chemical_potential) const;
    T evaluate_chemical_potential_mp(const Smearing& smearing, const T electron_charge,
                                    const T smearing_coef, const T trial_chemical_potential,
                                    Memory_pool<T, Fast_memory>& pool_fast,
                                    Memory_pool<T, Capacity_memory>& pool_cap) const;
    T evaluate_chemical_potential2_mp(const Smearing& smearing, const T electron_charge,
                                    const T smearing_coef, const T trial_chemical_potential,
                                    Memory_pool<T, Fast_memory>& pool_fast,
                                    Memory_pool<T, Capacity_memory>& pool_cap) const;
    T evaluate_chemical_potential3_mp(const Smearing& smearing, const T electron_charge,
                                    const T smearing_coef, const T trial_chemical_potential,
                                    Memory_pool<T, Fast_memory>& pool_fast,
                                    Memory_pool<T, Capacity_memory>& pool_cap) const;
    T evaluate_chemical_potential2_with_fg_electron(const Smearing& smearing, const T electron_charge,
                                  const T smearing_coef, const T trial_chemical_potential) const;
    T evaluate_chemical_potential_with_fg_electron_mp(const Smearing& smearing, const T electron_charge,
                                    const T smearing_coef, const T trial_chemical_potential,
                                    Memory_pool<T, Fast_memory>& pool_fast,
                                    Memory_pool<T, Capacity_memory>& pool_cap) const;
    T evaluate_chemical_potential2_with_fg_electron_mp(const Smearing& smearing, const T electron_charge,
                                    const T smearing_coef, const T trial_chemical_potential,
                                    Memory_pool<T, Fast_memory>& pool_fast,
                                    Memory_pool<T, Capacity_memory>& pool_cap) const;
    T evaluate_chemical_potential3_with_fg_electron_mp(const Smearing& smearing, const T electron_charge,
                                    const T smearing_coef, const T trial_chemical_potential,
                                    Memory_pool<T, Fast_memory>& pool_fast,
                                    Memory_pool<T, Capacity_memory>& pool_cap) const;
    T evaluate_band_energy(const Smearing& smearing, const T& chemical_potential, const T& smearing_coef) const;
    T evaluate_entropy_energy(const Smearing& smearing, const T& chemical_potential, const T& smearing_coef) const;
    const Vertices_3D get_atoms_region(const Vertices_3D& element_vertices) const;
    void cal_nonlocal_forces(Array_2D<T>& nonlocal_forces, const T& chemical_potential, const Smearing& smearing,
                             const Spin& spin, const std::vector<Psp8_file>& psp8_files) const;
    void print_eigens(const T& chemical_potential, const Smearing& smearing, const Spin& spin,
                      const std::string& fname = "EIGENS") const;
    void print_eigens_divided(const T& chemical_potential, const Smearing& smearing, const Spin& spin,
                      const std::string& dir_name = "EIGENS") const;
    void print_pdos(const T& chemical_potential, const Spin& spin, const std::string& dir_name = "PDOS") const;
    #if defined(ENABLE_TIMER)
    void print_timer_statistics(const bool if_print, std::ostream& output = std::cout) const;
    #endif
    double evalutate_flops();
    void run(const Array_3D<T>& ex_effective_potentail_loc, const bool& print_flag = true);
    void run_mp(T const* const ex_effective_potentail_loc, const Vertices_3D& ex_effective_potentail_vertices, const bool print_flag);
    void run_mp(T const* const ex_effective_potentail_loc, const Vertices_3D& ex_effective_potentail_vertices, const bool print_flag,
                Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap);
    void reserve_retained_packed_pool(Memory_pool<T, Fast_memory>& pool_fast);
    void init(const std::vector<Psp8_file>& psp8_files);
    template<typename T2> void init(const Xlsdft<T2>& xlsdft);
    void destructor();
    void show() const;
};

namespace Xlsdft_method {
    template<typename T>
    T fg_electron_number_for_ext_fpmd(const T lambda_max, const T U0, const T upper_limit,
                                      const T chemical_potential, const T volume, const T beta,
                                      const T coef);
    template<typename T>
    T xlsdft_fg_electron_number_for_ext_fpmd(const uint nelement, const T* lambda_maxs, const T* U0s, const T upper_limit,
                                      const T chemical_potential, const T* volumes, const T beta,
                                      const T* coefs);
    template<typename T>
    T evaluate_chemical_potential2_with_fg_electron(T const * const eigen_values, T const* const fracs, const uint length, const MPI_Comm domain_3d_comm, const T lower_bound, const T upper_bound,
                                const Smearing& smearing, const T electron_charge, const T smearing_coef, const uint max_iter, const T tol,
                                const uint nelement, const T* lambda_maxs, const T* U0s, const T* volumes, const T* coefs);
    template<typename T>
    T evaluate_chemical_potential2(T const * const eigen_values, T const* const fracs, const uint length, const MPI_Comm domain_3d_comm, const T lower_bound, const T upper_bound,
                                const Smearing& smearing, const T electron_charge, const T smearing_coef, const uint max_iter, const T tol);
}

#endif //_XLSDFT_H_
