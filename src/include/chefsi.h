#ifndef _CHEFSI_H_
#define _CHEFSI_H_

#include "parallel_vertices.h"
#include "hamiltonian.h"
#include "control.h"
#include "smearing.h"
#include "lanczos.h"
#include "spin.h"

#if !defined(ENABLE_CHEFSI_TIMER) && defined(ENABLE_TIMER)
#define ENABLE_CHEFSI_TIMER
#endif

#ifdef ENABLE_CHEFSI_TIMER
#include "timer.h"
class Chefsi_timer 
{
public:
    Timer chefsi;
    Timer lanczos;
    Timer filter;
    Timer filter_copy;
    Timer filter_product;
    Timer filter_lap;
    Timer filter_nloc;
    Timer H_psi;
    Timer projection;
    Timer diagonalization;
    Timer rotation;
    Chefsi_timer();
    ~Chefsi_timer();
    void reset();
    void show(std::ostream& output = std::cout) const;
    Chefsi_timer& operator+=(const Chefsi_timer& other) {
        chefsi          += other.chefsi;
        lanczos         += other.lanczos;
        filter          += other.filter;
        filter_copy     += other.filter_copy;
        filter_product  += other.filter_product;
        filter_lap      += other.filter_lap;
        filter_nloc     += other.filter_nloc;
        H_psi           += other.H_psi;
        projection      += other.projection;
        diagonalization += other.diagonalization;
        rotation        += other.rotation;

        return *this;
    }
    Chefsi_timer operator+(const Chefsi_timer& other) const {
        Chefsi_timer result = *this;
        result += other;
        return result;
    }
};
class Chefsi_flop_counter
{
public:
    double lanczos;
    double filter_copy;
    double filter_product;
    double filter_lap;
    double filter_nloc;
    double H_psi;
    double projection;
    double diagonalization;
    double rotation;
    Chefsi_flop_counter() {
        reset();
    }
    ~Chefsi_flop_counter() = default;
    void reset() {
        lanczos         = 0.0;
        filter_copy     = 0.0;
        filter_product  = 0.0;
        filter_lap      = 0.0;
        filter_nloc     = 0.0;
        H_psi           = 0.0;
        projection      = 0.0;
        diagonalization = 0.0;
        rotation        = 0.0;
    }
    double get_filter() const {
        return filter_copy
             + filter_product
             + filter_lap
             + filter_nloc;
    }
    double get_chefsi() const {
        return lanczos
             + get_filter()
             + H_psi
             + projection
             + diagonalization
             + rotation;
    }
    Chefsi_flop_counter& operator+=(const Chefsi_flop_counter& other)  {
        lanczos         += other.lanczos;
        filter_copy     += other.filter_copy;
        filter_product  += other.filter_product;
        filter_lap      += other.filter_lap;
        filter_nloc     += other.filter_nloc;
        H_psi           += other.H_psi;
        projection      += other.projection;
        diagonalization += other.diagonalization;
        rotation        += other.rotation;
        return *this;
    }
    template<typename T>
    void set_data(const Vertices_4D& shared_vertices,
                const Effective_potential_nloc<T>& Vnloc,
                const uint niter,
                const uint chefsi_degree) {
        const uint64_t nb = shared_vertices.nb;
        const uint64_t nd = shared_vertices.Vertices_3D::get_size();
        const uint n_nloc_projector = Vnloc.nloc_projectors.size();
        uint ncolnrow = 0;
        for (uint i_nloc_projector = 0; i_nloc_projector < n_nloc_projector; i_nloc_projector++) {
            ncolnrow += Vnloc.nloc_projectors[i_nloc_projector].ncol
                        * Vnloc.nloc_projectors[i_nloc_projector].nrow;
        }

        this->lanczos = 0.0;

        this->filter_copy = 0.0;
        this->filter_product = niter * (2 * nd * nb * (chefsi_degree - 1) + nd * nb);
        this->filter_lap = niter * chefsi_degree * nd * nb * (37 * 2 + 2);
        this->filter_nloc = niter * chefsi_degree * ncolnrow * nb * 4;

        this->H_psi           = niter * (nd * nb * (37 * 2 + 2) + ncolnrow * nb * 4);
        this->projection      = niter * nb * nb * nd * 3;
        this->diagonalization = niter * (17.0 / 3.0) * nb * nb * nb;
        this->rotation        = niter * nb * nb * nd * 2;
    }
};

class Chefsi_performance
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
        NITEM
    };
    const Chefsi_timer& chefsi_timer;
    const Chefsi_flop_counter& chefsi_flop_counter;
    double local_flops[NITEM];
    double flops[NITEM];
    Chefsi_performance(const Chefsi_timer& timer,
                       const Chefsi_flop_counter& flop_counter)
        : chefsi_timer(timer),
          chefsi_flop_counter(flop_counter)
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
    void update()
    {
        auto calc_flops = [](const double flop_count,
                             const Timer& timer) -> double
        {
            const double time =
                std::chrono::duration<double>(timer.elapsed).count();

            if (time <= 0.0)
                return 0.0;

            return flop_count / time / 1.0e9;
        };

        const double filter_flop_count =
              chefsi_flop_counter.filter_copy
            + chefsi_flop_counter.filter_product
            + chefsi_flop_counter.filter_lap
            + chefsi_flop_counter.filter_nloc;

        const double chefsi_flop_count =
              chefsi_flop_counter.lanczos
            + filter_flop_count
            + chefsi_flop_counter.H_psi
            + chefsi_flop_counter.projection
            + chefsi_flop_counter.diagonalization
            + chefsi_flop_counter.rotation;

        local_flops[CHEFSI] =
            calc_flops(chefsi_flop_count,
                       chefsi_timer.chefsi);

        local_flops[LANCZOS] =
            calc_flops(chefsi_flop_counter.lanczos,
                       chefsi_timer.lanczos);

        local_flops[FILTER] =
            calc_flops(filter_flop_count,
                       chefsi_timer.filter);

        local_flops[FILTER_COPY] =
            calc_flops(chefsi_flop_counter.filter_copy,
                       chefsi_timer.filter_copy);

        local_flops[FILTER_PRODUCT] =
            calc_flops(chefsi_flop_counter.filter_product,
                       chefsi_timer.filter_product);

        local_flops[FILTER_LAP] =
            calc_flops(chefsi_flop_counter.filter_lap,
                       chefsi_timer.filter_lap);

        local_flops[FILTER_NLOC] =
            calc_flops(chefsi_flop_counter.filter_nloc,
                       chefsi_timer.filter_nloc);

        local_flops[H_PSI] =
            calc_flops(chefsi_flop_counter.H_psi,
                       chefsi_timer.H_psi);

        local_flops[PROJECTION] =
            calc_flops(chefsi_flop_counter.projection,
                       chefsi_timer.projection);

        local_flops[DIAGONALIZATION] =
            calc_flops(chefsi_flop_counter.diagonalization,
                       chefsi_timer.diagonalization);

        local_flops[ROTATION] =
            calc_flops(chefsi_flop_counter.rotation,
                       chefsi_timer.rotation);

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
            "  Rotation"
        };
        output << '\n';
        output << std::left
               << std::setw(32) << "CheFSI performance"
               << std::right
               << std::setw(20) << "Local GFLOP/s"
               << std::setw(20) << "Total GFLOP/s"
               << '\n';
        output << std::string(68, '-') << '\n';
        for (int i = 0; i < NITEM; ++i) {
            output << std::left
                   << std::setw(28) << names[i]
                   << std::right
                   << std::setw(20)
                   << std::fixed
                   << std::setprecision(3)
                   << local_flops[i]
                   << std::setw(20)
                   << flops[i]
                   << '\n';
        }
        output << std::string(68, '-') << '\n';
    }
};
#endif //ENABLE_CHEFSI_TIMER

template<typename T>
class Chefsi
{
public:
    const Chefsi_control& chefsi_control;
    const Mesh_control& mesh_control;
    const Stencil<T>& stencil;
    const Domain_parallel_vertices_4D& domain_vertices;
    const Exarr_4D_mpi_package& exarr_mpi_package;
    #ifdef ENABLE_CHEFSI_TIMER
    Chefsi_timer chefsi_timer;
    Chefsi_flop_counter chefsi_flop_counter;
    Chefsi_performance chefsi_performance;
    #endif //ENABLE_CHEFSI_TIMER
    bool is_very_first = true;
    #if (defined(USE_MKL) || defined(USE_SCALAPACK))
        int icontxt_whole_comm = -1;  //the whole comm
        int icontxt_intra_band = -1;  //share some local_3d_domain but different bands
        int icontxt_domain_3D = -1;  //share some local_3d_domain but different bands
        int icontxt_2d_reshape = -1;  //2d contxt for diagonalization
        int icontxt_2d_reshape_nprow = -1;
        int icontxt_2d_reshape_npcol = -1;
        int icontxt_2d_reshape_myprow = -1;
        int icontxt_2d_reshape_mypcol = -1;
        int desc_intra_band_eigen_vectors[9];
        int desc_intra_band_hp_mp[9];
        int desc_domain_3D_hp_mp[9];
        int desc_2d_reshape_hp_mp[9];
        int hp_mp_2d_reshape_m = 0;
        int hp_mp_2d_reshape_n = 0;
        int lwork;
        int liwork;
    #endif
    #ifdef USE_OPENMP
        T* eigen_vectors_reshape_temp = nullptr;
        T* eigen_vectors_temp = nullptr;
        Array_4D<T> h_eigen_vectors;
        Array_2D<T> hp;
        Array_2D<T> mp;
        Array_4D<T> eigen_vectors_reshape;
        Stencil<T> stencil_temp;
        T* ex_eigen_vectors = nullptr;
        T* eigen_vectors_m1 = nullptr;
        T* dp_h_eigen_vectors = nullptr;
        #if (defined(USE_MKL) || defined(USE_SCALAPACK))
            T* mp_2d_rashape = nullptr;
            T* hp_2d_rashape = nullptr;
            T* qp_2d_rashape = nullptr;
            T* work = nullptr;
            int* iwork = nullptr;
            int* ifail = nullptr;
            int* iclustr = nullptr;
            T* gap = nullptr;
            int info;
        #endif
    #endif
    Domain_parallel_vertices_3D single_band_domain_vertices;
    Exarr_3D_mpi_package single_band_exarr_mpi_package;
    Lanczos<T> lanczos;
    Domain_parallel_vertices_4D dp_domain_vertices;
    Domain_4D_to_4D_mpi_package dp_mpi_package;
    Chefsi(const Chefsi_control& chefsi_control,
           const Mesh_control& mesh_control,
           const Stencil<T>& stencil,
           const Domain_parallel_vertices_4D& domain_vertices,
           const Exarr_4D_mpi_package& exarr_mpi_package);
    ~Chefsi();
    void chebyshev_filtering(Array_4D<T>& eigen_vectors, const Array_3D<T>& Vloc,
                             const Effective_potential_nloc<T>& Vnloc, const bool& print_flag = true);
    void chebyshev_filtering(T*& eigen_vectors, T const* const& Vloc, const Effective_potential_nloc<T>& Vnloc,
                             const bool& print_flag = true);
    void chebyshev_filtering_column_wise(T* const& eigen_vectors, T const* const& Vloc,
                                         const Effective_potential_nloc<T>& Vnloc, const bool& print_flag = true);
    void chebyshev_filtering_column_wise2(T*& eigen_vectors, T const* const& Vloc,
                                          const Effective_potential_nloc<T>& Vnloc, const bool& print_flag = true);
    void chebyshev_filtering_column_wise2_omp_task_comm_self_with_temp_swap(T*& eigen_vectors,
                                          T const* const& Vloc,
                                          const Effective_potential_nloc<T>& Vnloc,
                                          T*& eigen_vectors_temp,
                                          const bool& print_flag = true);
    #ifdef USE_OPENMP
    inline void chebyshev_filtering_column_wise2_omp(T*& eigen_vectors, T const* const& Vloc,
                                          const Effective_potential_nloc<T>& Vnloc, const bool& print_flag = true);
    inline void chebyshev_filtering_column_wise2_omp_comm_self(T*& eigen_vectors, T const* const& Vloc,
                                          const Effective_potential_nloc<T>& Vnloc, const bool& print_flag = true);
    inline void chebyshev_filtering_column_wise2_omp_task_comm_self(T*& eigen_vectors, T const* const& Vloc,
                                          const Effective_potential_nloc<T>& Vnloc, const bool& print_flag = true);
    #endif
    void project_hamiltonian(const Array_4D<T>& eigen_vectors, const Array_4D<T>& h_eigen_vectors,
                             Array_4D<T>& eigen_vectors_reshape, Array_2D<T>& hp, Array_2D<T>& mp,
                             const bool& print_flag = true);
    void project_hamiltonian(T* const& eigen_vectors, T* const& h_eigen_vectors,
                             T* const& eigen_vectors_reshape, T* const& hp, T* const& mp,
                             const bool& print_flag = true);
    void project_hamiltonian_with_temp_swap(T*& eigen_vectors, T*& h_eigen_vectors,
                                        T* const hp, T* const mp,
                                        const bool print_flag = true);
    void subspace_diagonalization(Array_2D<T>& hp, Array_2D<T>& mp, Array_0D<T>& eigen_values, const bool& print_flag = true);
    void subspace_diagonalization(T* const& hp, T* const& mp, T* const& eigen_values, const bool& print_flag = true);
    void subspace_rotation(Array_4D<T>& eigen_vectors, const Array_4D<T>& eigen_vectors_reshape,
                           const Array_2D<T>& qp, const bool& print_flag = true);
    void subspace_rotation(T*& eigen_vectors, T const* const& eigen_vectors_reshape, T const* const& qp,
                           const bool& print_flag = true);
    void subspace_rotation_specialization(T*& eigen_vectors, T const* const& eigen_vectors_reshape, T const* const& qp,
                           const bool& print_flag = true);
    void cal_int_density_per_band(const Vertices_3D& region, const Array_0D<T>& eigen_values,
                                  const Array_4D<T>& eigen_vector, std::vector<T>& eigen_value_per_band,
                                  std::vector<T>& int_density_per_band) const;
    Array_3D<T> cal_electron_charge_density(const Smearing& smearing, const T& smearing_coef,
                                            const T& chemical_potential, const Vertices_3D& region,
                                            const Array_0D<T>& eigen_values, const Array_4D<T>& eigen_vector) const;
    T cal_electron_charge(const Smearing& smearing, const T& smearing_coef,
                          const T& chemical_potential, const Vertices_3D& region,
                          const Array_0D<T>& eigen_values, const Array_4D<T>& eigen_vector) const;
    T cal_electron_charge(const Smearing& smearing, const T& smearing_coef,
                          const T& chemical_potential, const Array_0D<T>& eigen_values) const;
    T evaluate_chemical_potential(const Smearing& smearing, const Array_0D<T>& eigen_values, const T& electron_charge, const T& smearing_coef) const;
    T evaluate_band_energy(const Smearing& smearing, const T& chemical_potential, const T& smearing_coef, const Vertices_3D& region,
                           const Array_0D<T>& eigen_values, const Array_4D<T>& eigen_vector) const;
    T evaluate_entropy_energy(const Smearing& smearing, const T& chemical_potential, const T& smearing_coef, const Vertices_3D& region,
                              const Array_0D<T>& eigen_values, const Array_4D<T>& eigen_vector) const;
    void run_init(Array_4D<T>& h_eigen_vectors, Array_2D<T>& hp, Array_2D<T>& mp, Array_4D<T>& eigen_vectors_reshape, const bool& print_flag = true);
    void run_finalize(Array_4D<T>& h_eigen_vectors, Array_2D<T>& hp, Array_2D<T>& mp, Array_4D<T>& eigen_vectors_reshape);
    void cal_nonlocal_forces(Array_2D<T>& nonlocal_forces, const T& chemical_potential, const Smearing& smearing,
                             const Spin& spin, const Effective_potential_nloc<T>& effective_potential_nloc,
                             const Array_0D<T> eigen_values, const Array_4D<T> eigen_vectors) const;
    inline void cal_nonlocal_forces1(Array_2D<T>& nonlocal_forces, const T& chemical_potential, const Smearing& smearing,
                                    const Spin& spin, const Effective_potential_nloc<T>& effective_potential_nloc,
                                    const Array_0D<T> eigen_values, const Array_4D<T> eigen_vectors) const;
    inline void cal_nonlocal_forces2(Array_2D<T>& nonlocal_forces, const T& chemical_potential, const Smearing& smearing,
                                    const Spin& spin, const Effective_potential_nloc<T>& effective_potential_nloc,
                                    const Array_0D<T> eigen_values, const Array_4D<T> eigen_vectors) const;
    void run(Array_4D<T>& eigen_vectors, Array_0D<T>& eigen_values, const Array_3D<T>& Vloc,
             const Effective_potential_nloc<T>& Vnloc, const bool& print_flag = true);

    // run_mp
    void run_mp(T*& eigen_vectors, T *const eigen_values, T const *const Vloc,
             const Effective_potential_nloc<T>& Vnloc, const bool print_flag);
    void run_mp(T*& eigen_vectors_in, T*& eigen_vectors_out, T *const eigen_values, T const *const Vloc,
             const Effective_potential_nloc<T>& Vnloc, const bool print_flag,
            Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap);
    void chebyshev_filtering_column_wise_mp(T*& eigen_vectors,
                                            T*& eigen_vectors_buffer,
                                            T const* const Vloc,
                                            const Effective_potential_nloc<T>& Vnloc,
                                            const bool print_flag,
                                            Memory_pool<T, Fast_memory>& pool_fast,
                                            Memory_pool<T, Capacity_memory>& pool_cap);
    void project_hamiltonian_mp(T const* const eigen_vectors, T const* const h_eigen_vectors,
                                T* const hp, T* const mp,
                                const bool print_flag,
                                Memory_pool<T, Fast_memory>& pool_fast,
                                Memory_pool<T, Capacity_memory>& pool_cap);
    void subspace_diagonalization_mp(T* const hp, T* const mp, T* const eigen_values, const bool print_flag = true);
    void subspace_rotation_mp(T const* const eigen_vectors_in, T* const eigen_vectors_out,
                              T const* const qp, const bool print_flag,
                              Memory_pool<T, Fast_memory>& pool_fast,
                              Memory_pool<T, Capacity_memory>& pool_cap);

    double evalutate_flops();
    void init(const bool* is_periodic, const bool& is_rand_fixed);
    template<typename T2> void init(const Chefsi<T2>& chefsi);
    void destructor();
    void show() const;
};

#endif //_CHEFSI_H_
