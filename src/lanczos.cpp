
#include "lanczos.h"
#include "xlsdft_backend.h"
#include <iomanip>

namespace {

int tridiagonal_eigenvalues(const int n, double* diagonal, double* off_diagonal) {
    return Xlsdft_backend::dsterf(n, diagonal, off_diagonal);
}

int tridiagonal_eigenvalues(const int n, float* diagonal, float* off_diagonal) {
    if (n < 0) return -1;
    const std::size_t diagonal_count =
        std::max<std::size_t>(1U, static_cast<std::size_t>(n));
    const std::size_t off_diagonal_count =
        std::max<std::size_t>(1U, n > 1 ? static_cast<std::size_t>(n - 1) : 0U);
    double* diagonal_f64 = new (std::align_val_t(64)) double[diagonal_count];
    double* off_diagonal_f64 =
        new (std::align_val_t(64)) double[off_diagonal_count];
    for (int i = 0; i < n; ++i) {
        diagonal_f64[i] = static_cast<double>(diagonal[i]);
    }
    for (int i = 0; i + 1 < n; ++i) {
        off_diagonal_f64[i] = static_cast<double>(off_diagonal[i]);
    }
    const int info = Xlsdft_backend::dsterf(n, diagonal_f64, off_diagonal_f64);
    for (int i = 0; i < n; ++i) {
        diagonal[i] = static_cast<float>(diagonal_f64[i]);
    }
    for (int i = 0; i + 1 < n; ++i) {
        off_diagonal[i] = static_cast<float>(off_diagonal_f64[i]);
    }
    ::operator delete[](diagonal_f64, std::align_val_t(64));
    ::operator delete[](off_diagonal_f64, std::align_val_t(64));
    return info;
}

template<typename T>
void require_tridiagonal_success(const int status, const int n,
                                 T* diagonal, T* off_diagonal,
                                 const MPI_Comm comm, const char* label) {
    Xlsdft_backend::require_success(
        Xlsdft_backend::Operation::dsterf, status, comm,
        {label, n, n, 0, 1, 1, 0,
         diagonal, off_diagonal, nullptr, nullptr, 0});
}

}  // namespace

template<typename T>
Lanczos<T>::Lanczos() {}

template<typename T>
Lanczos<T>::~Lanczos() {}

template<typename T>
void Lanczos<T>::set_eig_min(const T& eig_min) {
    this->eig_min = eig_min;
    return;
}

template<typename T>
void Lanczos<T>::set_eig_max(const T& eig_max) {
    this->eig_max = eig_max;
    return;
}

template<typename T>
void Lanczos<T>::set_lambda_cutoff(const T& lambda_cutoff) {
    this->lambda_cutoff = lambda_cutoff;
    return;
}

template<typename T>
void Lanczos<T>::set_max_iter(const uint& max_iter) {
    this->max_iter = max_iter;
    return;
}

template<typename T>
void Lanczos<T>::set_tolerance(const double& tolerance) {
    this->tolerance = tolerance;
    return;
}

template<typename T>
void Lanczos<T>::cal_max_and_min_without_nonlocal(T const* const& Vloc, const Vertices_3D& vertices, const Effective_potential_nloc<T>& Vnloc,
                                 const Stencil<T>& stencil,  const T& dv, const Exarr_3D_mpi_package& exarr_mpi_package) {
    uint Nd = vertices.get_size();
    const MPI_Comm& comm = exarr_mpi_package.comm;
    if (Nd == 0 || comm == MPI_COMM_NULL) return;
    #ifdef USE_OPENMP
    (void) dv;
    (void) Vnloc;

    T*& a = this->a;
    T*& b = this->b;
    T*& d = this->d;
    T*& e = this->e;
    T*& V_j = this->V_j;
    T*& V_jp1 = this->V_jp1;
    #pragma omp single nowait
    a = new (std::align_val_t(64)) T [this->max_iter + 1];
    #pragma omp single nowait
    b = new (std::align_val_t(64)) T [this->max_iter + 1];
    #pragma omp single nowait
    d = new (std::align_val_t(64)) T [this->max_iter + 1];
    #pragma omp single nowait
    e = new (std::align_val_t(64)) T [this->max_iter + 1];
    #pragma omp single nowait
    V_j = new (std::align_val_t(64)) T [Nd];
    #pragma omp single nowait
    V_jp1 = new (std::align_val_t(64)) T [Nd];
    #pragma omp barrier

    Stencil<T> stencil_temp(stencil.coeffs_scale(-0.5, 2));

    T*& V_jm1 = this->V.data;
    Linalg::vector_normalize(V_jm1, Nd, comm);

    Stencil_method::Boundary_safe::calc_laplacian_boundary_safe(V_jm1, vertices, stencil_temp, vertices,
                                            V_j, vertices, Vloc);
    #pragma omp barrier
    T a_temp = Linalg::vector_dot_product(V_jm1, V_j, Nd, comm);
    #pragma omp single
    a[0] = a_temp;
    Linalg::accumulate_scalar_product_general(V_j, V_jm1, -a[0], Nd);
    T b_temp = Linalg::vector_2norm(V_j, Nd, comm);
    #pragma omp single
    b[0] = b_temp;

    #pragma omp parallel
    {
    if (!b[0]) {
        #pragma omp single
        Linalg::seededrand(V_j, Nd, (T) -1.0, (T) 1.0, 1);
        T a_temp = Linalg::vector_dot_product(V_jm1, V_j, Nd, comm);
        #pragma omp single
        a[0] = a_temp;
        Linalg::accumulate_scalar_product_general(V_j, V_jm1, -a[0], Nd);
        T b_temp = Linalg::vector_2norm(V_j, Nd, comm);
        #pragma omp single
        b[0] = b_temp;
    }
    if (b[0] != 0.0) {
        Linalg::scalar_product_general(V_j, (T)1.0/b[0], Nd);
    }
    }

    T eigmin_pre = 0.0;
    T eigmax_pre = 0.0;
    T err_eigmin = (T)this->tolerance + 1.0;
    T err_eigmax = (T)this->tolerance + 1.0;
    this->eig_min = 0.0;
    this->eig_max = 0.0;

    uint j = 0;
    while ((err_eigmin > (T)this->tolerance || err_eigmax > (T)this->tolerance) && j < this->max_iter)
    {
        #pragma omp parallel
        {
        #pragma omp barrier
        Stencil_method::Boundary_safe::calc_laplacian_boundary_safe(V_j, vertices, stencil_temp, vertices,
                                            V_jp1, vertices, Vloc);
        #pragma omp barrier
        T a_temp = Linalg::vector_dot_product(V_j, V_jp1, Nd, comm);
        #pragma omp single
        a[j+1] = a_temp;

        Linalg::accumulate_scalar_product_general(V_jp1, V_j, -a[j+1], Nd, V_jm1, -b[j]);
        #pragma omp barrier
        #pragma omp single
        std::swap(V_jm1, V_j);

        T b_temp =Linalg::vector_2norm(V_jp1, Nd, comm);
        #pragma omp single
        b[j+1] = b_temp;

        Linalg::scalar_product_general(V_j, V_jp1, (T)1.0/b[j+1], Nd);

        #pragma omp barrier
        Linalg::set_value_general(d, a, j+2);
        Linalg::set_value_general(e, b, j+2);
        }
        #ifdef USE_LAPACK
        #pragma omp barrier
        int info = tridiagonal_eigenvalues(static_cast<int>(j + 2), d, e);
        if (!info) {
            #pragma omp barrier
            this->eig_min = d[0];
            this->eig_max = d[j+1];
        } else {
            int rank;
            MPI_Comm_rank(comm, &rank);
            if (rank == 0) { printf("WARNING: Tridiagonal matrix eigensolver (?sterf) failed!\n");}
            exit(0);
        }
        #else
        assert(!"LAPACKE_sterf should be involved with LAPACKE loaded");
        #endif //USE_LAPACK

        #pragma omp barrier
        err_eigmin = std::fabs(this->eig_min - eigmin_pre);
        err_eigmax = std::fabs(this->eig_max - eigmax_pre);
        eigmin_pre = this->eig_min;
        eigmax_pre = this->eig_max;
        j++;
    }

    #pragma omp barrier
    #pragma omp single nowait
    {
        ::operator delete[](a, std::align_val_t(64));
        a = nullptr;
    }
    #pragma omp single nowait
    {
        ::operator delete[](b, std::align_val_t(64));
        b = nullptr;
    }
    #pragma omp single nowait
    {
        ::operator delete[](d, std::align_val_t(64));
        d = nullptr;
    }
    #pragma omp single nowait
    {
        ::operator delete[](e, std::align_val_t(64));
        e = nullptr;
    }
    #pragma omp single nowait
    {
        ::operator delete[](V_j, std::align_val_t(64));
        V_j = nullptr;
    }
    #pragma omp single nowait
    {
        ::operator delete[](V_jp1, std::align_val_t(64));
        V_jp1 = nullptr;
    }

    #else //USE_OPENMP
    (void) Vloc;
    (void) vertices;
    (void) Vnloc;
    (void) stencil;
    (void) dv;
    (void) exarr_mpi_package;
    assert(false);
    #endif //USE_OPENMP
    return;
}

template<typename T>
void Lanczos<T>::cal_max_and_min_without_nonlocal_mp(T const* const Vloc, const Vertices_3D& vertices, const Effective_potential_nloc<T>& Vnloc,
                                 const Stencil<T>& stencil,  const T dv, const Exarr_3D_mpi_package& exarr_mpi_package,
                                Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    uint Nd = vertices.get_size();
    const MPI_Comm comm = exarr_mpi_package.comm;
    if (Nd == 0 || comm == MPI_COMM_NULL) return;
    (void) dv;
    (void) Vnloc;
                
    // T* a = new (std::align_val_t(64)) T [this->max_iter + 1];
    // T* b = new (std::align_val_t(64)) T [this->max_iter + 1];
    // T* d = new (std::align_val_t(64)) T [this->max_iter + 1];
    // T* e = new (std::align_val_t(64)) T [this->max_iter + 1];
    // T* V_j = new (std::align_val_t(64)) T [Nd];
    // T* V_jp1 = new (std::align_val_t(64)) T [Nd];
    // T* V_jm1 = new (std::align_val_t(64)) T [Nd];
    T* a = pool_fast.allocate(this->max_iter + 1);
    T* b = pool_fast.allocate(this->max_iter + 1);
    T* d = pool_fast.allocate(this->max_iter + 1);
    T* e = pool_fast.allocate(this->max_iter + 1);
    T* V_j = pool_fast.allocate(Nd);
    T* V_jp1 = pool_fast.allocate(Nd);
    T* V_jm1 = pool_fast.allocate(Nd);
    constexpr bool is_rand_fixed = true;
    if (is_rand_fixed) {
        Parallel_vertices::domain_vertices_rand<T>(V_jm1, exarr_mpi_package.domain_vertices, 0.0, 1.0);
    }

    Stencil<T> stencil_temp(stencil.coeffs_scale(-0.5, 2));

    // T*& V_jm1 = this->V.data;
    #pragma omp parallel
    {
        Linalg::vector_normalize(V_jm1, Nd, comm);
        #pragma omp barrier
        Stencil_method::Boundary_safe::calc_laplacian_boundary_safe(V_jm1, vertices, stencil_temp, vertices,
                                                V_j, vertices, Vloc);
        #pragma omp barrier
        T a_temp = Linalg::vector_dot_product(V_jm1, V_j, Nd, comm);
        #pragma omp single
        a[0] = a_temp;
        Linalg::accumulate_scalar_product_general(V_j, V_jm1, -a_temp, Nd);
        T b_temp = Linalg::vector_2norm(V_j, Nd, comm);
        #pragma omp single
        b[0] = b_temp;
    }

    #pragma omp parallel
    {
        if (!b[0]) {
            #pragma omp single
            Linalg::seededrand(V_j, Nd, (T) -1.0, (T) 1.0, 1);
            T a_temp = Linalg::vector_dot_product(V_jm1, V_j, Nd, comm);
            #pragma omp single
            a[0] = a_temp;
            Linalg::accumulate_scalar_product_general(V_j, V_jm1, -a_temp, Nd);
            T b_temp = Linalg::vector_2norm(V_j, Nd, comm);
            #pragma omp single
            b[0] = b_temp;
        }
        if (b[0] != 0.0) {
            // V_j *= ((T)1.0/b[0]);
            Linalg::scalar_product_general(V_j, (T)1.0/b[0], Nd);
        }
    }

    T eigmin_pre = 0.0;
    T eigmax_pre = 0.0;
    T err_eigmin = (T)this->tolerance + 1.0;
    T err_eigmax = (T)this->tolerance + 1.0;
    this->eig_min = 0.0;
    this->eig_max = 0.0;

    // Array_3D<T> V_jp1(vertices);

    uint j = 0;
    while ((err_eigmin > (T)this->tolerance || err_eigmax > (T)this->tolerance) && j < this->max_iter)
    {
        #pragma omp parallel
        {
            Stencil_method::Boundary_safe::calc_laplacian_boundary_safe(V_j, vertices, stencil_temp, vertices,
                                            V_jp1, vertices, Vloc);
            #pragma omp barrier
            // a[j+1] = <V_j, V_{j+1}>
            T a_temp = Linalg::vector_dot_product(V_j, V_jp1, Nd, comm);
            #pragma omp single
            a[j+1] = a_temp;
            // V_{j+1} = V_{j+1} - a[j+1] * V_j - b[j] * V_{j-1}
            Linalg::accumulate_scalar_product_general(V_jp1, V_j, -a_temp, Nd, V_jm1, -b[j]);
        }
        // update V_{j-1}, i.e., V_{j-1} := V_j
        std::swap(V_jm1, V_j);

        #pragma omp parallel
        {
            T b_temp =Linalg::vector_2norm(V_jp1, Nd, comm);
            #pragma omp single
            b[j+1] = b_temp;
            // update V_j := V_{j+1} / ||V_{j+1}||
            // V_j = V_jp1 * ((T)1.0/b[j+1]);
            Linalg::scalar_product_general(V_j, V_jp1, (T)1.0/b_temp, Nd);
            #pragma omp barrier

            // solve for eigenvalues of the (j+2) x (j+2) tridiagonal matrix T = tridiag(b,a,b)
            Linalg::set_value_general(d, a, j+2);
            Linalg::set_value_general(e, b, j+2);
        }

        int info = tridiagonal_eigenvalues(static_cast<int>(j + 2), d, e);
        require_tridiagonal_success(info, static_cast<int>(j + 2),
                                    d, e, comm, "lanczos_ompunnested");
        this->eig_min = d[0];
        this->eig_max = d[j+1];

        err_eigmin = std::fabs(this->eig_min - eigmin_pre);
        err_eigmax = std::fabs(this->eig_max - eigmax_pre);
        eigmin_pre = this->eig_min;
        eigmax_pre = this->eig_max;
        j++;
    }

        // ::operator delete[](a, std::align_val_t(64));
        // a = nullptr;
        // ::operator delete[](b, std::align_val_t(64));
        // b = nullptr;
        // ::operator delete[](d, std::align_val_t(64));
        // d = nullptr;
        // ::operator delete[](e, std::align_val_t(64));
        // e = nullptr;
        // ::operator delete[](V_j, std::align_val_t(64));
        // V_j = nullptr;
        // ::operator delete[](V_jp1, std::align_val_t(64));
        // V_jp1 = nullptr;
        // ::operator delete[](V_jm1, std::align_val_t(64));
        // V_jm1 = nullptr;
    return;
}

template<typename T>
void Lanczos<T>::run(const MPI_Comm& comm, const Array_3D<T>& Vloc, const Effective_potential_nloc<T>& Vnloc,
                     const Stencil<T>& stencil, const T& dv, const Exarr_3D_mpi_package& exarr_mpi_package,
                     const bool& calculate_flag, const T& eig_max, const T& eig_min,
                     const bool& print_flag) {
    this->run(comm, Vloc.data, Vloc.get_vertices(), Vnloc, stencil, dv, exarr_mpi_package, calculate_flag, eig_max,
              eig_min, print_flag);
    return;
}

template<typename T>
void Lanczos<T>::run(const MPI_Comm& comm, T const* const Vloc, const Vertices_3D vertices_3d, const Effective_potential_nloc<T>& Vnloc,
                     const Stencil<T>& stencil, const T& dv, const Exarr_3D_mpi_package& exarr_mpi_package,
                     const bool& calculate_flag, const T& eig_max, const T& eig_min,
                     const bool& print_flag) {
    #ifdef USE_OPENMP

    if (calculate_flag) {
        this->cal_max_and_min_without_nonlocal(Vloc, vertices_3d, Vnloc, stencil, dv, exarr_mpi_package);
        #pragma omp barrier
        #pragma omp master
        {
        this->eig_max *= (T)1.01;
        this->eig_min -= (T)0.1;
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        MPI_Bcast(&(this->eig_max), 1, mpi_datatype, 0, comm);
        MPI_Bcast(&(this->eig_min), 1, mpi_datatype, 0, comm);
        this->lambda_cutoff = (T) 0.5 * (this->eig_max + this->eig_min);
        }
        #pragma omp barrier
    } else {
        #pragma omp single
        {
        this->set_lambda_cutoff(eig_max + 0.1);
        this->set_eig_min(eig_min);
        }
    }
    #pragma omp single nowait
    {
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0 && print_flag) {
        std::cout << "Lanczos, eig_min = " << std::setprecision(12) << std::fixed << this->eig_min
                  << ", eig_max = " << std::setprecision(12) << std::fixed << this->eig_max
                  << ", lambda_cutoff = " << std::setprecision(12) << std::fixed << this->lambda_cutoff
                  << "." << std::endl;
    }
    }

    #else //USE_OPENMP

    if (calculate_flag) {
        this->cal_max_and_min_without_nonlocal(Vloc, vertices_3d, Vnloc, stencil, dv, exarr_mpi_package);
        this->eig_max *= (T)1.01;
        this->eig_min -= (T)0.1;
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        MPI_Bcast(&(this->eig_max), 1, mpi_datatype, 0, comm);
        MPI_Bcast(&(this->eig_min), 1, mpi_datatype, 0, comm);
        this->lambda_cutoff = (T) 0.5 * (this->eig_max + this->eig_min);
    } else {
        this->set_lambda_cutoff(eig_max + 0.1);
        this->set_eig_min(eig_min);
    }
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0 && print_flag) {
        std::cout << "Lanczos, eig_min = " << std::setprecision(12) << std::fixed << this->eig_min
                  << ", eig_max = " << std::setprecision(12) << std::fixed << this->eig_max
                  << ", lambda_cutoff = " << std::setprecision(12) << std::fixed << this->lambda_cutoff
                  << "." << std::endl;
    }

    #endif //USE_OPENMP
    return;
}

template<typename T>
void Lanczos<T>::run_mp(const MPI_Comm comm, T const* const Vloc, const Vertices_3D vertices_3d, const Effective_potential_nloc<T>& Vnloc,
                     const Stencil<T>& stencil, const T dv, const Exarr_3D_mpi_package& exarr_mpi_package,
                     const bool calculate_flag, const T eig_max, const T eig_min,
                     const bool print_flag,
                    Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap) {
    (void) print_flag;
    if (calculate_flag) {
        this->cal_max_and_min_without_nonlocal_mp(Vloc, vertices_3d, Vnloc, stencil, dv, exarr_mpi_package,
                                pool_fast, pool_cap);
        this->eig_max *= (T)1.01; //add 1% buffer
        this->eig_min -= (T)0.1;  //for safety
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        MPI_Bcast(&(this->eig_max), 1, mpi_datatype, 0, comm);
        MPI_Bcast(&(this->eig_min), 1, mpi_datatype, 0, comm);
        this->lambda_cutoff = (T) 0.5 * (this->eig_max + this->eig_min);
    } else {
        this->set_lambda_cutoff(eig_max + 0.1);
        this->set_eig_min(eig_min);
    }
    // int rank;
    // MPI_Comm_rank(comm, &rank);
    // if (rank == 0 && print_flag) {
    //     std::cout << "Lanczos, eig_min = " << std::setprecision(12) << std::fixed << this->eig_min
    //               << ", eig_max = " << std::setprecision(12) << std::fixed << this->eig_max
    //               << ", lambda_cutoff = " << std::setprecision(12) << std::fixed << this->lambda_cutoff
    //               << "." << std::endl;
    // }
    return;
}

template<typename T>
void Lanczos<T>::init(const Domain_parallel_vertices_3D& domain_vertices, const bool& is_rand_fixed) {
    this->set_max_iter(100);
    this->set_tolerance(1e-2);
    if (domain_vertices.is_active) {
        this->V.reconstructor(domain_vertices.get_3D_local_vertices());
        if (is_rand_fixed) {
            Parallel_vertices::domain_vertices_rand<T>(this->V, domain_vertices, 0.0, 1.0);
        } else {
            this->V.seededrand(domain_vertices.get_comm_rank() * 100 + 1, 0.0, 1.0);
        }
    }
    return;
}

template<typename T>
template<typename T2>
void Lanczos<T>::init(const Lanczos<T2>& lanczos) {
    this->eig_min = (T)lanczos.eig_min;
    this->eig_max = (T)lanczos.eig_max;
    this->lambda_cutoff = (T)lanczos.lambda_cutoff;
    this->max_iter = lanczos.max_iter;
    this->tolerance = lanczos.tolerance;
    this->V.deepcopy(std::move(lanczos.V.as_type(this->V.data)));
    return;
}
template void Lanczos<float>::init(const Lanczos<float>& lanczos);
template void Lanczos<double>::init(const Lanczos<double>& lanczos);
template void Lanczos<float>::init(const Lanczos<double>& lanczos);
template void Lanczos<double>::init(const Lanczos<float>& lanczos);

template<typename T>
void Lanczos<T>::destructor() {
    this->V.destructor();
    return;
}

template<typename T>
void Lanczos<T>::show() const {
    std::cout << "eig_min = " << this->eig_min << std::endl;
    std::cout << "eig_max = " << this->eig_max << std::endl;
    std::cout << "lambda_cutoff = " << this->lambda_cutoff << std::endl;
    std::cout << "max_iter = " << this->max_iter << std::endl;
    std::cout << "tolerance = " << this->tolerance << std::endl;
    this->V.show();
    return;
}

template class Lanczos<float>;
template class Lanczos<double>;
