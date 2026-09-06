
#include "lanczos.h"

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

/**
 * @brief Lanczos algorithm for calculating min and max eigenvalues
 *        for the Hamiltonian.  
 * 
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/eigenSolver.c#L1918
 */
template<typename T>
void Lanczos<T>::cal_max_and_min(const Array_3D<T>& Vloc, const Effective_potential_nloc<T>& Vnloc,
                                 const Stencil<T>& stencil, const T& dv, const Exarr_3D_mpi_package& exarr_mpi_package) {
    const MPI_Comm& comm = exarr_mpi_package.comm;
    if (Vloc.length == 0 || comm == MPI_COMM_NULL) return;

    Vertices_3D local_vertices = this->V.get_vertices();
    uint Nd = this->V.length;
    
    Array_0D<T> a(this->max_iter + 1);
    Array_0D<T> b(this->max_iter + 1);
    Array_0D<T> d(this->max_iter + 1);
    Array_0D<T> e(this->max_iter + 1);

    Array_3D<T>& V_jm1 = this->V;
    Linalg::vector_normalize(V_jm1.data, Nd, comm);

    Array_3D<T> V_j(local_vertices);
    Hamiltonian::hamiltonian_product_vectors(V_j, V_jm1, Vloc, Vnloc, stencil, dv, exarr_mpi_package);
    a[0] = Linalg::vector_dot_product(V_jm1.data, V_j.data, Nd, comm);
    Linalg::accumulate_scalar_product_general(V_j.data, V_jm1.data, -a[0], Nd);
    b[0] = Linalg::vector_2norm(V_j.data, Nd, comm);
    if (!b[0]) {
        Linalg::seededrand(V_j.data, Nd, (T) -1.0, (T) 1.0, 1);
        a[0] = Linalg::vector_dot_product(V_jm1.data, V_j.data, Nd, comm);
        Linalg::accumulate_scalar_product_general(V_j.data, V_jm1.data, -a[0], Nd);
        b[0] = Linalg::vector_2norm(V_j.data, Nd, comm);
    }
    if (b[0] != 0.0) {
        V_j *= ((T)1.0/b[0]);
    }

    T eigmin_pre = this->eig_min = 0.0;
    T eigmax_pre = this->eig_max = 0.0;
    T err_eigmin = (T)this->tolerance + 1.0;
    T err_eigmax = (T)this->tolerance + 1.0;

    Array_3D<T> V_jp1(local_vertices);
    uint j = 0;
    while ((err_eigmin > (T)this->tolerance || err_eigmax > (T)this->tolerance) && j < this->max_iter) 
    {
        // V_{j+1} = H * V_j
        Hamiltonian::hamiltonian_product_vectors(V_jp1, V_j, Vloc, Vnloc, stencil, dv, exarr_mpi_package);

        // a[j+1] = <V_j, V_{j+1}>
        a[j+1] = Linalg::vector_dot_product(V_j.data, V_jp1.data, Nd, comm);

        // V_{j+1} = V_{j+1} - a[j+1] * V_j - b[j] * V_{j-1}
        Linalg::accumulate_scalar_product_general(V_jp1.data, V_j.data, -a[j+1], Nd, V_jm1.data, -b[j]);
        // update V_{j-1}, i.e., V_{j-1} := V_j
        std::swap(V_jm1.data, V_j.data);
        
        b[j+1] = Linalg::vector_2norm(V_jp1.data, Nd, comm);
        
        // update V_j := V_{j+1} / ||V_{j+1}||
        V_j = V_jp1 * ((T)1.0/b[j+1]);

        // solve for eigenvalues of the (j+2) x (j+2) tridiagonal matrix T = tridiag(b,a,b)
        Linalg::set_value_general(d.data, a.data, j+2);
        Linalg::set_value_general(e.data, b.data, j+2);
        
        #ifdef USE_LAPACK
        if (!Linalg::LAPACKE__sterf(j+2, d.data, e.data)) {
            this->eig_min = d[0];
            this->eig_max = d[j+1];
        } else {
            int rank;
            MPI_Comm_rank(comm, &rank);
            if (rank == 0) { printf("WARNING: Tridiagonal matrix eigensolver (?sterf) failed!\n");}
            break;
        }
        #else 
        assert(!"LAPACKE_sterf should be involved with LAPACKE loaded");
        #endif //USE_LAPACK

        err_eigmin = fabs(this->eig_min - eigmin_pre);
        err_eigmax = fabs(this->eig_max - eigmax_pre);

        eigmin_pre = this->eig_min;
        eigmax_pre = this->eig_max;

        j++;
    }
    return;
}

/**
 * @brief Lanczos algorithm for calculating min and max eigenvalues
 *        for the Hamiltonian.  
 * 
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/eigenSolver.c#L1918
 */
template<typename T>
void Lanczos<T>::cal_max_and_min(T const* const& Vloc, const Vertices_3D& vertices, const Effective_potential_nloc<T>& Vnloc,
                                 const Stencil<T>& stencil,  const T& dv, const Exarr_3D_mpi_package& exarr_mpi_package) {
    uint Nd = vertices.get_size();
    const MPI_Comm& comm = exarr_mpi_package.comm;
    if (Nd == 0 || comm == MPI_COMM_NULL) return;
    #ifdef USE_OPENMP

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


    T*& V_jm1 = this->V.data;
    Linalg::vector_normalize(V_jm1, Nd, comm);

    // Array_3D<T> V_j(vertices);
    Hamiltonian::hamiltonian_product_vectors(V_j, vertices, V_jm1, Vloc, Vnloc, stencil, dv, exarr_mpi_package);
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
        // V_{j+1} = H * V_j
        #pragma omp barrier
        Hamiltonian::hamiltonian_product_vectors(V_jp1, vertices, V_j, Vloc, Vnloc, stencil, dv, exarr_mpi_package);

        // a[j+1] = <V_j, V_{j+1}>
        #pragma omp barrier
        T a_temp = Linalg::vector_dot_product(V_j, V_jp1, Nd, comm);
        #pragma omp single
        a[j+1] = a_temp;

        // V_{j+1} = V_{j+1} - a[j+1] * V_j - b[j] * V_{j-1}
        Linalg::accumulate_scalar_product_general(V_jp1, V_j, -a[j+1], Nd, V_jm1, -b[j]);
        // update V_{j-1}, i.e., V_{j-1} := V_j
        #pragma omp barrier
        #pragma omp single
        std::swap(V_jm1, V_j);

        T b_temp =Linalg::vector_2norm(V_jp1, Nd, comm);
        #pragma omp single
        b[j+1] = b_temp;

        // update V_j := V_{j+1} / ||V_{j+1}||
        // V_j = V_jp1 * ((T)1.0/b[j+1]);
        Linalg::scalar_product_general(V_j, V_jp1, (T)1.0/b[j+1], Nd);

        // solve for eigenvalues of the (j+2) x (j+2) tridiagonal matrix T = tridiag(b,a,b)
        #pragma omp barrier
        Linalg::set_value_general(d, a, j+2);
        Linalg::set_value_general(e, b, j+2);
        }
        #ifdef USE_LAPACK
        #pragma omp barrier
        int info = Linalg::LAPACKE__sterf(j+2, d, e);
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

    // Array_0D<T> a(this->max_iter + 1);
    // Array_0D<T> b(this->max_iter + 1);
    // Array_0D<T> d(this->max_iter + 1);
    // Array_0D<T> e(this->max_iter + 1);
    T* a = new (std::align_val_t(64)) T[this->max_iter + 1];
    T* b = new (std::align_val_t(64)) T[this->max_iter + 1];
    T* d = new (std::align_val_t(64)) T[this->max_iter + 1];
    T* e = new (std::align_val_t(64)) T[this->max_iter + 1];

    T*& V_jm1 = this->V.data;
    Linalg::vector_normalize(V_jm1, Nd, comm);

    // Array_3D<T> V_j(vertices);
    T* V_j = new (std::align_val_t(64)) T [Nd];
    Hamiltonian::hamiltonian_product_vectors(V_j, vertices, V_jm1, Vloc, Vnloc, stencil, dv, exarr_mpi_package);
    a[0] = Linalg::vector_dot_product(V_jm1, V_j, Nd, comm);
    Linalg::accumulate_scalar_product_general(V_j, V_jm1, -a[0], Nd);
    b[0] = Linalg::vector_2norm(V_j, Nd, comm);
    if (!b[0]) {
        Linalg::seededrand(V_j, Nd, (T) -1.0, (T) 1.0, 1);
        a[0] = Linalg::vector_dot_product(V_jm1, V_j, Nd, comm);
        Linalg::accumulate_scalar_product_general(V_j, V_jm1, -a[0], Nd);
        b[0] = Linalg::vector_2norm(V_j, Nd, comm);
    }
    if (b[0] != 0.0) {
        // V_j *= ((T)1.0/b[0]);
        Linalg::scalar_product_general(V_j, (T)1.0/b[0], Nd);
    }

    T eigmin_pre = 0.0;
    T eigmax_pre = 0.0;
    T err_eigmin = (T)this->tolerance + 1.0;
    T err_eigmax = (T)this->tolerance + 1.0;
    this->eig_min = 0.0;
    this->eig_max = 0.0;

    // Array_3D<T> V_jp1(vertices);
    T* V_jp1 = new (std::align_val_t(64)) T [Nd];
    uint j = 0;
    while ((err_eigmin > (T)this->tolerance || err_eigmax > (T)this->tolerance) && j < this->max_iter)
    {
        // V_{j+1} = H * V_j
        Hamiltonian::hamiltonian_product_vectors(V_jp1, vertices, V_j, Vloc, Vnloc, stencil, dv, exarr_mpi_package);

        // a[j+1] = <V_j, V_{j+1}>
        a[j+1] = Linalg::vector_dot_product(V_j, V_jp1, Nd, comm);

        // V_{j+1} = V_{j+1} - a[j+1] * V_j - b[j] * V_{j-1}
        Linalg::accumulate_scalar_product_general(V_jp1, V_j, -a[j+1], Nd, V_jm1, -b[j]);
        // update V_{j-1}, i.e., V_{j-1} := V_j
        std::swap(V_jm1, V_j);
        
        b[j+1] = Linalg::vector_2norm(V_jp1, Nd, comm);
        
        // update V_j := V_{j+1} / ||V_{j+1}||
        // V_j = V_jp1 * ((T)1.0/b[j+1]);
        Linalg::scalar_product_general(V_j, V_jp1, (T)1.0/b[j+1], Nd);

        // solve for eigenvalues of the (j+2) x (j+2) tridiagonal matrix T = tridiag(b,a,b)
        Linalg::set_value_general(d, a, j+2);
        Linalg::set_value_general(e, b, j+2);
        
        #ifdef USE_LAPACK
        if (!Linalg::LAPACKE__sterf(j+2, d, e)) {
            this->eig_min = d[0];
            this->eig_max = d[j+1];
        } else {
            int rank;
            MPI_Comm_rank(comm, &rank);
            if (rank == 0) { printf("WARNING: Tridiagonal matrix eigensolver (?sterf) failed!\n");}
            break;
        }
        #else
        assert(!"LAPACKE_sterf should be involved with LAPACKE loaded");
        #endif //USE_LAPACK

        err_eigmin = fabs(this->eig_min - eigmin_pre);
        err_eigmax = fabs(this->eig_max - eigmax_pre);

        eigmin_pre = this->eig_min;
        eigmax_pre = this->eig_max;

        j++;
    }
    ::operator delete[](a, std::align_val_t(64));
    ::operator delete[](b, std::align_val_t(64));
    ::operator delete[](d, std::align_val_t(64));
    ::operator delete[](e, std::align_val_t(64));
    ::operator delete[](V_j, std::align_val_t(64));
    ::operator delete[](V_jp1, std::align_val_t(64));

    #endif //USE_OPENMP
    return;
}

/**
 * @brief Lanczos algorithm for calculating min and max eigenvalues
 *        for the Hamiltonian.  
 * 
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/eigenSolver.c#L1918
 */
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

    // Array_3D<T> V_j(vertices);
    // Hamiltonian::hamiltonian_product_vectors(V_j, vertices, V_jm1, Vloc, Vnloc, stencil, dv, exarr_mpi_package);
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
        // V_{j+1} = H * V_j
        #pragma omp barrier
        // Hamiltonian::hamiltonian_product_vectors(V_jp1, vertices, V_j, Vloc, Vnloc, stencil, dv, exarr_mpi_package);
        Stencil_method::Boundary_safe::calc_laplacian_boundary_safe(V_j, vertices, stencil_temp, vertices,
                                            V_jp1, vertices, Vloc);
        // a[j+1] = <V_j, V_{j+1}>
        #pragma omp barrier
        T a_temp = Linalg::vector_dot_product(V_j, V_jp1, Nd, comm);
        #pragma omp single
        a[j+1] = a_temp;

        // V_{j+1} = V_{j+1} - a[j+1] * V_j - b[j] * V_{j-1}
        Linalg::accumulate_scalar_product_general(V_jp1, V_j, -a[j+1], Nd, V_jm1, -b[j]);
        // update V_{j-1}, i.e., V_{j-1} := V_j
        #pragma omp barrier
        #pragma omp single
        std::swap(V_jm1, V_j);

        T b_temp =Linalg::vector_2norm(V_jp1, Nd, comm);
        #pragma omp single
        b[j+1] = b_temp;

        // update V_j := V_{j+1} / ||V_{j+1}||
        // V_j = V_jp1 * ((T)1.0/b[j+1]);
        Linalg::scalar_product_general(V_j, V_jp1, (T)1.0/b[j+1], Nd);

        // solve for eigenvalues of the (j+2) x (j+2) tridiagonal matrix T = tridiag(b,a,b)
        #pragma omp barrier
        Linalg::set_value_general(d, a, j+2);
        Linalg::set_value_general(e, b, j+2);
        }
        #ifdef USE_LAPACK
        #pragma omp barrier
        int info = Linalg::LAPACKE__sterf(j+2, d, e);
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
        // Array_3D<T> V_j(vertices);
        // Hamiltonian::hamiltonian_product_vectors(V_j, vertices, V_jm1, Vloc, Vnloc, stencil, dv, exarr_mpi_package);
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
        // V_{j+1} = H * V_j
        // Hamiltonian::hamiltonian_product_vectors(V_jp1, vertices, V_j, Vloc, Vnloc, stencil, dv, exarr_mpi_package);
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

        #ifdef USE_LAPACK
        int info = Linalg::LAPACKE__sterf_org(j+2, d, e);
        if (!info) {
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
        // this->cal_max_and_min(Vloc, Vnloc, stencil, dv, exarr_mpi_package);
        // this->cal_max_and_min(Vloc.data, Vloc.get_vertices(), Vnloc, stencil, dv, exarr_mpi_package);
        this->cal_max_and_min_without_nonlocal(Vloc, vertices_3d, Vnloc, stencil, dv, exarr_mpi_package);
        #pragma omp barrier
        #pragma omp master
        {
        this->eig_max *= (T)1.01; //add 1% buffer
        this->eig_min -= (T)0.1;  //for safety
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
        // this->cal_max_and_min(Vloc, Vnloc, stencil, dv, exarr_mpi_package);
        // this->cal_max_and_min(Vloc.data, Vloc.get_vertices(), Vnloc, stencil, dv, exarr_mpi_package);
        this->cal_max_and_min_without_nonlocal(Vloc, vertices_3d, Vnloc, stencil, dv, exarr_mpi_package);
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
        // this->cal_max_and_min(Vloc, Vnloc, stencil, dv, exarr_mpi_package);
        // this->cal_max_and_min(Vloc.data, Vloc.get_vertices(), Vnloc, stencil, dv, exarr_mpi_package);
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
