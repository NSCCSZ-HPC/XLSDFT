
#include "chefsi.h"

#ifdef ENABLE_CHEFSI_TIMER
#pragma message("Building with ENABLE_CHEFSI_TIMER.")
Chefsi_timer::Chefsi_timer() {}
Chefsi_timer::~Chefsi_timer() {}
void Chefsi_timer::reset() {
    this->chefsi.reset();
    this->lanczos.reset();
    this->filter.reset();
    this->filter_copy.reset();
    this->filter_product.reset();
    this->filter_lap.reset();
    this->filter_nloc.reset();
    this->H_psi.reset();
    this->projection_gemm.reset();
    this->projection_syrk.reset();
    this->diagonalization.reset();
    this->rotation.reset();
    return;
}
void Chefsi_timer::show(std::ostream& output) const {
    output << std::left << std::setw(20) << "chefsi"             << ": " << this->chefsi.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "  lanczos"           << ": " << this->lanczos.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "  filter"           << ": " << this->filter.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "    filter_copy"    << ": " << this->filter_copy.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "    filter_product"    << ": " << this->filter_product.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "    filter_lap"     << ": " << this->filter_lap.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "    filter_nloc"    << ": " << this->filter_nloc.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "  H_psi"            << ": " << this->H_psi.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "  projection_gemm"  << ": " << this->projection_gemm.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "  projection_syrk"  << ": " << this->projection_syrk.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "  diagonalization"  << ": " << this->diagonalization.time_cost_millisecond() << " [ms]" << std::endl;
    output << std::left << std::setw(20) << "  rotation "        << ": " << this->rotation.time_cost_millisecond() << " [ms]" << std::endl;
    return;
}
#endif //ENABLE_CHEFSI_TIMER

template<typename T>
Chefsi<T>::Chefsi(const Chefsi_control& chefsi_control,
                  const Mesh_control& mesh_control,
                  const Stencil<T>& stencil,
                  const Domain_parallel_vertices_4D& domain_vertices,
                  const Exarr_4D_mpi_package& exarr_mpi_package)
                : chefsi_control(chefsi_control),
                  mesh_control(mesh_control),
                  stencil(stencil),
                  domain_vertices(domain_vertices),
                  exarr_mpi_package(exarr_mpi_package),
                  #ifdef ENABLE_CHEFSI_TIMER
                  chefsi_performance(this->chefsi_timer, this->chefsi_flop_counter),
                  #endif //ENABLE_CHEFSI_TIMER
                  single_band_exarr_mpi_package(this->single_band_domain_vertices),
                  dp_mpi_package(this->domain_vertices, this->dp_domain_vertices)
                    {}

template<typename T>
Chefsi<T>::~Chefsi() {
    this->destructor();
}

/**
 * @brief Perform Chebyshev filtering.
 *
 * 
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/eigenSolver.c#L722
 */
template<typename T>
void Chefsi<T>::chebyshev_filtering(Array_4D<T>& eigen_vectors, const Array_3D<T>& Vloc,
                                    const Effective_potential_nloc<T>& Vnloc, const bool& print_flag) {
    if (eigen_vectors.length == 0) return;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->chefsi_control.chebyshev_filter_degree == 0) return;

    const T e = 0.5 * (this->lanczos.eig_max - this->lanczos.lambda_cutoff);
    const T c = 0.5 * (this->lanczos.eig_max + this->lanczos.lambda_cutoff);
    T sigma = e / (this->lanczos.eig_min - c);
    T sigma1 = sigma;
    T gamma = 2.0 / sigma1;
    T sigma2;
    const Vertices_4D& local_vertices = this->domain_vertices.get_4D_local_vertices();
    Array_4D<T> eigen_vectors_temp(local_vertices);
    Stencil<T> stencil_temp = this->stencil.coeffs_scale(-0.5, 2);
    stencil_temp.shift_D2_coeffs(-c);
    Vertices_4D ex_vertice(local_vertices.Vertices_3D::get_vertices().generate_ex_vertices(stencil_temp.FDn),
                           local_vertices.bs, local_vertices.get_be());
    Array_4D<T> ex_eigen_vectors(ex_vertice, 0);
    this->exarr_mpi_package.fill_domain_par_ex_arr(eigen_vectors, ex_eigen_vectors);

    // eigen_vectors_temp = (-0.5 \nabla - cI + Vloc ) eigen_vectors
    Stencil_method::Special::calc_laplacian_d4(ex_eigen_vectors.data, ex_vertice, stencil_temp, local_vertices,
                                               eigen_vectors_temp.data, local_vertices, Vloc.data);
    // eigen_vectors_temp += Vnloc*eigen_vectors;
    Hamiltonian::nloc_project_vectors<T>(eigen_vectors_temp, eigen_vectors, Vnloc,
                                         (T)this->mesh_control.delta_V, exarr_mpi_package.comm);

    T vscal = sigma1 / e;
    eigen_vectors_temp *= vscal;

    if (this->chefsi_control.chebyshev_filter_degree == 1) {
        // eigen_vectors.deepcopy(std::move(eigen_vectors_temp));
        std::swap(eigen_vectors.data, eigen_vectors_temp.data);
        return;
    }
    Array_4D<T> eigen_vectors_m1(local_vertices);
    // eigen_vectors_m1.deepcopy(std::move(eigen_vectors));
    // eigen_vectors.deepcopy(std::move(eigen_vectors_temp));
    std::swap(eigen_vectors_m1.data, eigen_vectors.data);
    std::swap(eigen_vectors.data, eigen_vectors_temp.data);
    T vscal2;
    for (uint j = 2; j <= this->chefsi_control.chebyshev_filter_degree; j++) {
        sigma2 = 1.0 / (gamma - sigma);

        this->exarr_mpi_package.fill_domain_par_ex_arr(eigen_vectors, ex_eigen_vectors);
        // eigen_vectors_temp = (-0.5 \nabla - cI + Vloc ) eigen_vectors
        Stencil_method::Special::calc_laplacian_d4(ex_eigen_vectors.data, ex_vertice, stencil_temp, local_vertices,
                                                   eigen_vectors_temp.data, local_vertices, Vloc.data);
        // TODO:: nonlocal
        // eigen_vectors_temp += Vnloc*eigen_vectors;
        Hamiltonian::nloc_project_vectors<T>(eigen_vectors_temp, eigen_vectors, Vnloc,
                                             (T)this->mesh_control.delta_V, this->exarr_mpi_package.comm);
        
        vscal = 2.0 * sigma2 / e;
        vscal2 = sigma * sigma2;
        eigen_vectors_temp.scalar_product_general(vscal, eigen_vectors_m1, -vscal2);

        // eigen_vectors_m1.deepcopy(std::move(eigen_vectors));
        // eigen_vectors.deepcopy(std::move(eigen_vectors_temp));
        std::swap(eigen_vectors_m1.data, eigen_vectors.data);
        std::swap(eigen_vectors.data, eigen_vectors_temp.data);

        sigma = sigma2;
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
        std::cout << "The chebyshev_filtering took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

template<typename T>
void Chefsi<T>::chebyshev_filtering(T*& __restrict__ eigen_vectors, T const* const& __restrict__ Vloc,
                                    const Effective_potential_nloc<T>& Vnloc, const bool& print_flag) {
    // uint Nd_Nb = this->domain_vertices.get_4D_local_vertices().get_size();
    const Vertices_4D& local_vertices = this->domain_vertices.get_4D_local_vertices();
    const uint Nd_Nb = local_vertices.get_size();
    if (Nd_Nb == 0) return;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->chefsi_control.chebyshev_filter_degree == 0) return;

    const T e = 0.5 * (this->lanczos.eig_max - this->lanczos.lambda_cutoff);
    const T c = 0.5 * (this->lanczos.eig_max + this->lanczos.lambda_cutoff);
    T sigma = e / (this->lanczos.eig_min - c);
    T sigma1 = sigma;
    T gamma = 2.0 / sigma1;
    T sigma2;

    // Array_4D<T> eigen_vectors_temp(this->domain_vertices.local_vertices);
    T* eigen_vectors_temp = new (std::align_val_t(64)) T [Nd_Nb];

    Stencil<T> stencil_temp = this->stencil.coeffs_scale(-0.5, 2);
    stencil_temp.shift_D2_coeffs(-c);

    Vertices_4D ex_vertice(local_vertices.Vertices_3D::get_vertices().generate_ex_vertices(stencil_temp.FDn),
                           local_vertices.bs, local_vertices.get_be());
    uint ex_Nd_Nb = ex_vertice.get_size();

    // Array_4D<T> ex_eigen_vectors(ex_vertice, 0);
    T* ex_eigen_vectors = new (std::align_val_t(64)) T [ex_Nd_Nb]();
    this->exarr_mpi_package.fill_domain_par_ex_arr(eigen_vectors, local_vertices, ex_eigen_vectors, ex_vertice);

    // eigen_vectors_temp = (-0.5 \nabla - cI + Vloc ) eigen_vectors
    Stencil_method::Special::calc_laplacian_d4(ex_eigen_vectors, ex_vertice, stencil_temp, this->domain_vertices.local_vertices,
                                               eigen_vectors_temp, this->domain_vertices.local_vertices, Vloc);
    // TODO:: nonlocal
    // eigen_vectors_temp += Vnloc*eigen_vectors;
    Hamiltonian::nloc_project_vectors<T>(eigen_vectors_temp, local_vertices, eigen_vectors, Vnloc,
                                         (T)this->mesh_control.delta_V, exarr_mpi_package.comm);

    T vscal = sigma1 / e;
    // eigen_vectors_temp *= vscal;
    Linalg::scalar_product_general(eigen_vectors_temp, vscal, Nd_Nb);

    if (this->chefsi_control.chebyshev_filter_degree == 1) {
        std::swap(eigen_vectors, eigen_vectors_temp);
        delete [] eigen_vectors_temp;
        delete [] ex_eigen_vectors;
        return;
    }
    // Array_4D<T> eigen_vectors_m1(this->domain_vertices.local_vertices);
    T* eigen_vectors_m1 = new (std::align_val_t(64)) T [Nd_Nb];

    std::swap(eigen_vectors_m1, eigen_vectors);
    std::swap(eigen_vectors, eigen_vectors_temp);
    T vscal2;
    for (uint j = 2; j <= this->chefsi_control.chebyshev_filter_degree; j++) {
        sigma2 = 1.0 / (gamma - sigma);

        this->exarr_mpi_package.fill_domain_par_ex_arr(eigen_vectors, local_vertices,
                                                       ex_eigen_vectors, ex_vertice);
        // eigen_vectors_temp = (-0.5 \nabla - cI + Vloc ) eigen_vectors
        Stencil_method::Special::calc_laplacian_d4(ex_eigen_vectors, ex_vertice, stencil_temp, this->domain_vertices.local_vertices,
                                                   eigen_vectors_temp, this->domain_vertices.local_vertices, Vloc);
        // TODO:: nonlocal
        // eigen_vectors_temp += Vnloc*eigen_vectors;
        Hamiltonian::nloc_project_vectors<T>(eigen_vectors_temp, local_vertices, eigen_vectors, Vnloc,
                                             (T)this->mesh_control.delta_V, this->exarr_mpi_package.comm);

        vscal = 2.0 * sigma2 / e;
        vscal2 = sigma * sigma2;
        // eigen_vectors_temp.scalar_product_general(vscal, eigen_vectors_m1, -vscal2);
        Linalg::scalar_product_general(eigen_vectors_temp, vscal, Nd_Nb, eigen_vectors_m1, -vscal2);

        // eigen_vectors_m1.deepcopy(std::move(eigen_vectors));
        // eigen_vectors.deepcopy(std::move(eigen_vectors_temp));
        std::swap(eigen_vectors_m1, eigen_vectors);
        std::swap(eigen_vectors, eigen_vectors_temp);

        sigma = sigma2;
    }

    delete [] eigen_vectors_m1;
    delete [] eigen_vectors_temp;
    delete [] ex_eigen_vectors;
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
        std::cout << "The chebyshev_filtering took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

template<typename T>
void Chefsi<T>::chebyshev_filtering_column_wise(T* const& __restrict__ eigen_vectors, T const* const& __restrict__ Vloc,
                                                const Effective_potential_nloc<T>& Vnloc, const bool& print_flag) {
    // uint Nd_Nb = this->domain_vertices.get_4D_local_vertices().get_size();
    const Vertices_3D& local_vertices_3d = this->domain_vertices.get_3D_local_vertices();
    const uint Nd = local_vertices_3d.get_size();
    const uint& nb = this->domain_vertices.get_4D_local_vertices().nb;
    if (Nd == 0 || nb == 0 || this->chefsi_control.chebyshev_filter_degree == 0) return;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    const T e = 0.5 * (this->lanczos.eig_max - this->lanczos.lambda_cutoff);
    const T c = 0.5 * (this->lanczos.eig_max + this->lanczos.lambda_cutoff);

    Stencil<T> stencil_temp = this->stencil.coeffs_scale(-0.5, 2);
    stencil_temp.shift_D2_coeffs(-c);
    Vertices_3D ex_vertice(local_vertices_3d.generate_ex_vertices(stencil_temp.FDn));
    uint ex_Nd = ex_vertice.get_size();

    T* eigen_vectors_temp = new (std::align_val_t(64)) T [Nd];
    T* ex_eigen_vectors = new (std::align_val_t(64)) T [ex_Nd]();
    T* eigen_vectors_m1 = new (std::align_val_t(64)) T [Nd];
    T* eigen_vectors_column_wise = new (std::align_val_t(64)) T [Nd];

    for (uint ib = 0; ib < nb; ib++) {
        const uint offset = ib * Nd;
        T sigma = e / (this->lanczos.eig_min - c);
        T sigma1 = sigma;
        T gamma = 2.0 / sigma1;
        T sigma2;

        this->single_band_exarr_mpi_package.fill_domain_par_ex_arr(eigen_vectors + offset,
                                                local_vertices_3d, ex_eigen_vectors, ex_vertice);

        // eigen_vectors_temp = (-0.5 \nabla - cI + Vloc ) eigen_vectors
        Stencil_method::calc_laplacian(ex_eigen_vectors, ex_vertice, stencil_temp, local_vertices_3d,
                                       eigen_vectors_temp, local_vertices_3d, Vloc);
        // eigen_vectors_temp += Vnloc*eigen_vectors;
        Hamiltonian::nloc_project_vectors<T>(eigen_vectors_temp, eigen_vectors + offset, Vnloc,
                                            (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package.comm);

        T vscal = sigma1 / e;
        // eigen_vectors_temp *= vscal;
        Linalg::scalar_product_general(eigen_vectors_temp, vscal, Nd);

        if (this->chefsi_control.chebyshev_filter_degree == 1) {
            Linalg::set_value_general(eigen_vectors + offset, eigen_vectors_temp, Nd);
            continue;
        }
        Linalg::set_value_general(eigen_vectors_m1, eigen_vectors + offset, Nd);
        std::swap(eigen_vectors_column_wise, eigen_vectors_temp);

        T vscal2;
        for (uint j = 2; j <= this->chefsi_control.chebyshev_filter_degree; j++) {
            sigma2 = 1.0 / (gamma - sigma);

            this->single_band_exarr_mpi_package.fill_domain_par_ex_arr(eigen_vectors_column_wise,
                                                local_vertices_3d, ex_eigen_vectors, ex_vertice);
            // eigen_vectors_temp = (-0.5 \nabla - cI + Vloc ) eigen_vectors
            Stencil_method::calc_laplacian(ex_eigen_vectors, ex_vertice, stencil_temp, local_vertices_3d,
                                           eigen_vectors_temp, local_vertices_3d, Vloc);
            // eigen_vectors_temp += Vnloc*eigen_vectors;
            Hamiltonian::nloc_project_vectors<T>(eigen_vectors_temp, eigen_vectors_column_wise, Vnloc,
                                            (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package.comm);

            vscal = 2.0 * sigma2 / e;
            vscal2 = sigma * sigma2;
            Linalg::scalar_product_general(eigen_vectors_temp, vscal, Nd, eigen_vectors_m1, -vscal2);

            std::swap(eigen_vectors_m1, eigen_vectors_column_wise);
            std::swap(eigen_vectors_column_wise, eigen_vectors_temp);

            sigma = sigma2;
        }
        Linalg::set_value_general(eigen_vectors + offset, eigen_vectors_column_wise, Nd);
    }
    delete [] eigen_vectors_m1;
    delete [] eigen_vectors_temp;
    delete [] ex_eigen_vectors;
    delete [] eigen_vectors_column_wise;
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
        std::cout << "The chebyshev_filtering_column_wise took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

template<typename T>
void Chefsi<T>::chebyshev_filtering_column_wise2(T*& __restrict__ eigen_vectors, T const* const& __restrict__ Vloc,
                                                 const Effective_potential_nloc<T>& Vnloc, const bool& print_flag) {
    #ifdef USE_OPENMP

    // this->chebyshev_filtering_column_wise2_omp(eigen_vectors, Vloc, Vnloc, print_flag);
    // this->chebyshev_filtering_column_wise2_omp_task_comm_self(eigen_vectors, Vloc, Vnloc, print_flag);
    this->chebyshev_filtering_column_wise2_omp_comm_self(eigen_vectors, Vloc, Vnloc, print_flag);
    // if (unlikely(this->single_band_domain_vertices.get_active_comm_size() > 1)) {
    //     this->chebyshev_filtering_column_wise2_omp(eigen_vectors, Vloc, Vnloc, print_flag);
    // } else {
    //     this->chebyshev_filtering_column_wise2_omp_task_comm_self(eigen_vectors, Vloc, Vnloc, print_flag);
        // this->chebyshev_filtering_column_wise2_omp_comm_self(eigen_vectors, Vloc, Vnloc, print_flag);
    // }

    #else //USE_OPENMP

    // uint Nd_Nb = this->domain_vertices.get_4D_local_vertices().get_size();
    const Vertices_4D& local_vertices = this->domain_vertices.get_4D_local_vertices();
    const Vertices_3D& local_vertices_3d = this->domain_vertices.get_3D_local_vertices();
    const uint Nd = local_vertices_3d.get_size();
    const uint& nb = this->domain_vertices.get_4D_local_vertices().nb;
    if (Nd == 0 || nb == 0 || this->chefsi_control.chebyshev_filter_degree == 0) return;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    const T e = 0.5 * (this->lanczos.eig_max - this->lanczos.lambda_cutoff);
    const T c = 0.5 * (this->lanczos.eig_max + this->lanczos.lambda_cutoff);

    Stencil<T> stencil_temp = this->stencil.coeffs_scale(-0.5, 2);
    stencil_temp.shift_D2_coeffs(-c);
    Vertices_3D ex_vertice(local_vertices_3d.generate_ex_vertices(stencil_temp.FDn));
    uint ex_Nd = ex_vertice.get_size();

    T* eigen_vectors_temp = new (std::align_val_t(64)) T [Nd * nb];
    T* ex_eigen_vectors = new (std::align_val_t(64)) T [ex_Nd]();

    T sigma = e / (this->lanczos.eig_min - c);
    T sigma1 = sigma;
    T gamma = 2.0 / sigma1;
    T sigma2;

    for (uint ib = 0; ib < nb; ib++) {
        const uint offset = ib * Nd;
        this->single_band_exarr_mpi_package.fill_domain_par_ex_arr(eigen_vectors + offset,
                                                local_vertices_3d, ex_eigen_vectors, ex_vertice);
        // eigen_vectors_temp = (-0.5 \nabla - cI + Vloc ) eigen_vectors
        Stencil_method::calc_laplacian(ex_eigen_vectors, ex_vertice, stencil_temp, local_vertices_3d,
                                       eigen_vectors_temp + offset, local_vertices_3d, Vloc);
    }
    // eigen_vectors_temp += Vnloc*eigen_vectors;
    Hamiltonian::nloc_project_vectors<T>(eigen_vectors_temp, local_vertices, eigen_vectors, Vnloc,
                                        (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package.comm);

    T vscal = sigma1 / e;
    // eigen_vectors_temp *= vscal;
    Linalg::scalar_product_general(eigen_vectors_temp, vscal, Nd * nb);

    if (this->chefsi_control.chebyshev_filter_degree == 1) {
        Linalg::set_value_general(eigen_vectors, eigen_vectors_temp, Nd * nb);
        delete [] eigen_vectors_temp;
        delete [] ex_eigen_vectors;
        return;
    }

    T* eigen_vectors_m1 = new (std::align_val_t(64)) T [Nd * nb];

    std::swap(eigen_vectors_m1, eigen_vectors);
    std::swap(eigen_vectors, eigen_vectors_temp);

    T vscal2;
    for (uint j = 2; j <= this->chefsi_control.chebyshev_filter_degree; j++) {
        sigma2 = 1.0 / (gamma - sigma);
        for (uint ib = 0; ib < nb; ib++) {
            const uint offset = ib * Nd;
            this->single_band_exarr_mpi_package.fill_domain_par_ex_arr(eigen_vectors + offset,
                                                local_vertices_3d, ex_eigen_vectors, ex_vertice);
            // eigen_vectors_temp = (-0.5 \nabla - cI + Vloc ) eigen_vectors
            Stencil_method::calc_laplacian(ex_eigen_vectors, ex_vertice, stencil_temp, local_vertices_3d,
                                            eigen_vectors_temp + offset, local_vertices_3d, Vloc);
        }
        // eigen_vectors_temp += Vnloc*eigen_vectors;
        Hamiltonian::nloc_project_vectors<T>(eigen_vectors_temp, local_vertices, eigen_vectors, Vnloc,
                                        (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package.comm);

        vscal = 2.0 * sigma2 / e;
        vscal2 = sigma * sigma2;
        Linalg::scalar_product_general(eigen_vectors_temp, vscal, Nd * nb, eigen_vectors_m1, -vscal2);

        std::swap(eigen_vectors_m1, eigen_vectors);
        std::swap(eigen_vectors, eigen_vectors_temp);

        sigma = sigma2;
    }

    delete [] eigen_vectors_m1;
    delete [] eigen_vectors_temp;
    delete [] ex_eigen_vectors;
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
        std::cout << "The chebyshev_filtering_column_wise took " << Tools::time_cost(begin, end) << "." << std::endl;
    }

    #endif //USE_OPENMP

    return;
}

#ifdef USE_OPENMP
template<typename T>
inline void Chefsi<T>::chebyshev_filtering_column_wise2_omp(T*& __restrict__ eigen_vectors, T const* const& __restrict__ Vloc,
                                                 const Effective_potential_nloc<T>& Vnloc, const bool& print_flag) {

    #ifdef ENABLE_CHEFSI_TIMER
        #pragma omp master
        this->chefsi_timer.filter.start();
    #endif //ENABLE_CHEFSI_TIMER

    // uint Nd_Nb = this->domain_vertices.get_4D_local_vertices().get_size();
    const Vertices_4D& local_vertices = this->domain_vertices.get_4D_local_vertices();
    const Vertices_3D& local_vertices_3d = this->domain_vertices.get_3D_local_vertices();
    const uint Nd = local_vertices_3d.get_size();
    const uint& nb = this->domain_vertices.get_4D_local_vertices().nb;
    if (Nd == 0 || nb == 0 || this->chefsi_control.chebyshev_filter_degree == 0) return;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    const T e = 0.5 * (this->lanczos.eig_max - this->lanczos.lambda_cutoff);
    const T c = 0.5 * (this->lanczos.eig_max + this->lanczos.lambda_cutoff);

    Stencil<T>& stencil_temp = this->stencil_temp;
    #pragma omp single
    stencil_temp.deepcopy(stencil);
    stencil_temp.coeffs_scale_self(-0.5);
    #pragma omp barrier
    #pragma omp single nowait
    stencil_temp.shift_D2_coeffs(-c);
    Vertices_3D ex_vertice(local_vertices_3d.generate_ex_vertices(stencil_temp.FDn));
    uint ex_Nd = ex_vertice.get_size();

    T*& eigen_vectors_temp = this->eigen_vectors_temp;
    T*& ex_eigen_vectors = this->ex_eigen_vectors;
    #pragma omp single nowait
    eigen_vectors_temp = new (std::align_val_t(64)) T [Nd * nb];
    #pragma omp single nowait
    ex_eigen_vectors = new (std::align_val_t(64)) T [ex_Nd]();

    T sigma = e / (this->lanczos.eig_min - c);
    T sigma1 = sigma;
    T gamma = 2.0 / sigma1;
    T sigma2;

    for (uint ib = 0; ib < nb; ib++) {
        const uint offset = ib * Nd;
        #ifdef ENABLE_CHEFSI_TIMER
            #pragma omp master
            {
                this->chefsi_timer.filter_copy.start();
            }
        #endif //ENABLE_CHEFSI_TIMER
        #pragma omp barrier
        this->single_band_exarr_mpi_package.fill_domain_par_ex_arr(eigen_vectors + offset,
                                                local_vertices_3d, ex_eigen_vectors, ex_vertice);
        #ifdef ENABLE_CHEFSI_TIMER
            #pragma omp master
            {
                this->chefsi_timer.filter_copy.stop();
                this->chefsi_timer.filter_lap.start();
            }
        #endif //ENABLE_CHEFSI_TIMER
        #pragma omp barrier
        // eigen_vectors_temp = (-0.5 \nabla - cI + Vloc ) eigen_vectors
        Stencil_method::calc_laplacian(ex_eigen_vectors, ex_vertice, stencil_temp, local_vertices_3d,
                                       eigen_vectors_temp + offset, local_vertices_3d, Vloc);
        #ifdef ENABLE_CHEFSI_TIMER
            #pragma omp master
            {
                this->chefsi_timer.filter_lap.stop();
            }
        #endif //ENABLE_CHEFSI_TIMER
    }
    #ifdef ENABLE_CHEFSI_TIMER
        #pragma omp master
        {
            this->chefsi_timer.filter_nloc.start();
        }
    #endif //ENABLE_CHEFSI_TIMER
    // eigen_vectors_temp += Vnloc*eigen_vectors;
    #pragma omp barrier
    Hamiltonian::nloc_project_vectors<T>(eigen_vectors_temp, local_vertices, eigen_vectors, Vnloc,
                                        (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package.comm);
    #ifdef ENABLE_CHEFSI_TIMER
        #pragma omp master
        {
            this->chefsi_timer.filter_nloc.stop();
        }
    #endif //ENABLE_CHEFSI_TIMER
    T vscal = sigma1 / e;
    // eigen_vectors_temp *= vscal;
    #pragma omp barrier
    Linalg::scalar_product_general(eigen_vectors_temp, vscal, Nd * nb);

    if (this->chefsi_control.chebyshev_filter_degree == 1) {
        Linalg::set_value_general(eigen_vectors, eigen_vectors_temp, Nd * nb);
        #pragma omp barrier
        #pragma omp single nowait
        {
            stencil_temp.destructor();
        }
        #pragma omp single nowait
        {
            delete [] eigen_vectors_temp;
            eigen_vectors_temp = nullptr;
        }
        #pragma omp single nowait
        {
            delete [] ex_eigen_vectors;
            ex_eigen_vectors = nullptr;
        }
        return;
    }

    T*& eigen_vectors_m1 = this->eigen_vectors_m1;
    #pragma omp single
    eigen_vectors_m1 = new (std::align_val_t(64)) T [Nd * nb];

    #pragma omp barrier
    #pragma omp single nowait
    {
    std::swap(eigen_vectors_m1, eigen_vectors);
    std::swap(eigen_vectors, eigen_vectors_temp);
    }
    // Linalg::set_value_general(eigen_vectors_m1, eigen_vectors, Nd * nb);
    // Linalg::set_value_general(eigen_vectors, eigen_vectors_temp, Nd * nb);

    T vscal2;
    for (uint j = 2; j <= this->chefsi_control.chebyshev_filter_degree; j++) {
        sigma2 = 1.0 / (gamma - sigma);
        for (uint ib = 0; ib < nb; ib++) {
            const uint offset = ib * Nd;
            #ifdef ENABLE_CHEFSI_TIMER
                #pragma omp master
                {
                    this->chefsi_timer.filter_copy.start();
                }
            #endif //ENABLE_CHEFSI_TIMER
            #pragma omp barrier
            this->single_band_exarr_mpi_package.fill_domain_par_ex_arr(eigen_vectors + offset,
                                                local_vertices_3d, ex_eigen_vectors, ex_vertice);
            #ifdef ENABLE_CHEFSI_TIMER
                #pragma omp master
                {
                    this->chefsi_timer.filter_copy.stop();
                    this->chefsi_timer.filter_lap.start();
                }
            #endif //ENABLE_CHEFSI_TIMER
            #pragma omp barrier
            // eigen_vectors_temp = (-0.5 \nabla - cI + Vloc ) eigen_vectors
            Stencil_method::calc_laplacian(ex_eigen_vectors, ex_vertice, stencil_temp, local_vertices_3d,
                                            eigen_vectors_temp + offset, local_vertices_3d, Vloc);
            #ifdef ENABLE_CHEFSI_TIMER
                #pragma omp master
                {
                    this->chefsi_timer.filter_lap.stop();
                }
            #endif //ENABLE_CHEFSI_TIMER
        }
        #ifdef ENABLE_CHEFSI_TIMER
            #pragma omp master
            {
                this->chefsi_timer.filter_nloc.start();
            }
        #endif //ENABLE_CHEFSI_TIMER
        // eigen_vectors_temp += Vnloc*eigen_vectors;
        #pragma omp barrier
        Hamiltonian::nloc_project_vectors<T>(eigen_vectors_temp, local_vertices, eigen_vectors, Vnloc,
                                        (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package.comm);
        #ifdef ENABLE_CHEFSI_TIMER
            #pragma omp master
            {
                this->chefsi_timer.filter_nloc.stop();
            }
        #endif //ENABLE_CHEFSI_TIMER

        vscal = 2.0 * sigma2 / e;
        vscal2 = sigma * sigma2;
        #pragma omp barrier
        Linalg::scalar_product_general(eigen_vectors_temp, vscal, Nd * nb, eigen_vectors_m1, -vscal2);

        #pragma omp barrier
        #pragma omp single nowait
        {
        std::swap(eigen_vectors_m1, eigen_vectors);
        std::swap(eigen_vectors, eigen_vectors_temp);
        }
        // Linalg::set_value_general(eigen_vectors_m1, eigen_vectors, Nd * nb);
        // Linalg::set_value_general(eigen_vectors, eigen_vectors_temp, Nd * nb);

        sigma = sigma2;
    }

    #pragma omp barrier
    #pragma omp single nowait
    {
        delete [] eigen_vectors_m1;
        eigen_vectors_m1 = nullptr;
    }
    #pragma omp single nowait
    {
        delete [] eigen_vectors_temp;
        eigen_vectors_temp = nullptr;
    }
    #pragma omp single nowait
    {
        delete [] ex_eigen_vectors;
        ex_eigen_vectors = nullptr;
    }
    #pragma omp single nowait
    {
        stencil_temp.destructor();
    }
    #pragma omp single nowait
    {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
            std::cout << "The chebyshev_filtering_column_wise took " << Tools::time_cost(begin, end) << "." << std::endl;
        }
    }

    #ifdef ENABLE_CHEFSI_TIMER
        #pragma omp master
        this->chefsi_timer.filter.stop();
    #endif //ENABLE_CHEFSI_TIMER

    return;
}

template<typename T>
inline void Chefsi<T>::chebyshev_filtering_column_wise2_omp_comm_self(T*& __restrict__ eigen_vectors, T const* const& __restrict__ Vloc,
                                                 const Effective_potential_nloc<T>& Vnloc, const bool& print_flag) {
    // uint Nd_Nb = this->domain_vertices.get_4D_local_vertices().get_size();
    if (unlikely(this->single_band_domain_vertices.get_active_comm_size() > 1)) {
        if (unlikely(print_flag)) {
            std::cout << RED << "WARNING: chebyshev_filtering_column_wise2_omp_task_comm_self only support single MPI process in the communicator." << RESET << std::endl;
            std::cout << RED << "WARNING: calling chebyshev_filtering_column_wise2_omp instead." << RESET << std::endl;
        }
        this->chebyshev_filtering_column_wise2_omp(eigen_vectors, Vloc, Vnloc, print_flag);
        return;
    }

    #ifdef ENABLE_CHEFSI_TIMER
        #pragma omp master
        this->chefsi_timer.filter.start();
    #endif //ENABLE_CHEFSI_TIMER

    const Vertices_4D& local_vertices = this->domain_vertices.get_4D_local_vertices();
    const Vertices_3D& local_vertices_3d = this->domain_vertices.get_3D_local_vertices();
    const uint Nd = local_vertices_3d.get_size();
    const uint& nb = this->domain_vertices.get_4D_local_vertices().nb;
    if (Nd == 0 || nb == 0 || this->chefsi_control.chebyshev_filter_degree == 0) return;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    const T e = 0.5 * (this->lanczos.eig_max - this->lanczos.lambda_cutoff);
    const T c = 0.5 * (this->lanczos.eig_max + this->lanczos.lambda_cutoff);

    Stencil<T>& stencil_temp = this->stencil_temp;
    #pragma omp single
    stencil_temp.deepcopy(stencil);
    stencil_temp.coeffs_scale_self(-0.5);
    #pragma omp barrier
    #pragma omp single nowait
    stencil_temp.shift_D2_coeffs(-c);


    T*& eigen_vectors_temp = this->eigen_vectors_temp;
    #pragma omp single nowait
    eigen_vectors_temp = new (std::align_val_t(64)) T [Nd * nb];
    #pragma omp barrier

    T sigma = e / (this->lanczos.eig_min - c);
    T sigma1 = sigma;
    T gamma = 2.0 / sigma1;
    // T sigma2;

    #ifdef ENABLE_CHEFSI_TIMER
    #pragma omp master
            this->chefsi_timer.filter_lap.start();
    #endif //ENABLE_CHEFSI_TIMER
    Stencil_method::Boundary_safe::calc_laplacian_boundary_safe(eigen_vectors, local_vertices, stencil_temp, local_vertices,
                                            eigen_vectors_temp, local_vertices, Vloc);

    #ifdef ENABLE_CHEFSI_TIMER
        #pragma omp master
        {
            this->chefsi_timer.filter_lap.stop();
            this->chefsi_timer.filter_nloc.start();
        }
    #endif //ENABLE_CHEFSI_TIMER
    // eigen_vectors_temp += Vnloc*eigen_vectors;
    #pragma omp barrier
    Hamiltonian::nloc_project_vectors<T>(eigen_vectors_temp, local_vertices, eigen_vectors, Vnloc,
                                        (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package.comm);
    #ifdef ENABLE_CHEFSI_TIMER
        #pragma omp master
        {
            this->chefsi_timer.filter_nloc.stop();
        }
    #endif //ENABLE_CHEFSI_TIMER

    T vscal = sigma1 / e;
    // eigen_vectors_temp *= vscal;
    #pragma omp barrier
    Linalg::scalar_product_general(eigen_vectors_temp, vscal, Nd * nb);

    if (unlikely(this->chefsi_control.chebyshev_filter_degree == 1)) {
        Linalg::set_value_general(eigen_vectors, eigen_vectors_temp, Nd * nb);
        #pragma omp barrier
        #pragma omp single nowait
        {
            stencil_temp.destructor();
        }
        #pragma omp single nowait
        {
            delete [] eigen_vectors_temp;
            eigen_vectors_temp = nullptr;
        }
        return;
    }

    T*& eigen_vectors_m1 = this->eigen_vectors_m1;
    #pragma omp single
    eigen_vectors_m1 = new (std::align_val_t(64)) T [Nd * nb];

    // #pragma omp barrier
    // #pragma omp single nowait
    // {
    // std::swap(eigen_vectors_m1, eigen_vectors);
    // std::swap(eigen_vectors, eigen_vectors_temp);
    // }
    Linalg::set_value_general(eigen_vectors_m1, eigen_vectors, Nd * nb);
    Linalg::set_value_general(eigen_vectors, eigen_vectors_temp, Nd * nb);

    T vscal2;
    for (uint j = 2; j <= this->chefsi_control.chebyshev_filter_degree; j++) {
        #pragma omp barrier
        T sigma2 = 1.0 / (gamma - sigma);

        #ifdef ENABLE_CHEFSI_TIMER
        #pragma omp master
                this->chefsi_timer.filter_lap.start();
        #endif //ENABLE_CHEFSI_TIMER
        Stencil_method::Boundary_safe::calc_laplacian_boundary_safe(eigen_vectors, local_vertices, stencil_temp, local_vertices,
                                            eigen_vectors_temp, local_vertices, Vloc);

        #ifdef ENABLE_CHEFSI_TIMER
            #pragma omp master
            {
                this->chefsi_timer.filter_lap.stop();
                this->chefsi_timer.filter_nloc.start();
            }
        #endif //ENABLE_CHEFSI_TIMER
        // eigen_vectors_temp += Vnloc*eigen_vectors;
        #pragma omp barrier
        Hamiltonian::nloc_project_vectors<T>(eigen_vectors_temp, local_vertices, eigen_vectors, Vnloc,
                                        (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package.comm);
        #ifdef ENABLE_CHEFSI_TIMER
            #pragma omp master
            {
                this->chefsi_timer.filter_nloc.stop();
            }
        #endif //ENABLE_CHEFSI_TIMER

        vscal = 2.0 * sigma2 / e;
        vscal2 = sigma * sigma2;
        #pragma omp barrier
        Linalg::scalar_product_general(eigen_vectors_temp, vscal, Nd * nb, eigen_vectors_m1, -vscal2);

        // #pragma omp barrier
        // #pragma omp single nowait
        // {
        // std::swap(eigen_vectors_m1, eigen_vectors);
        // std::swap(eigen_vectors, eigen_vectors_temp);
        // }
        Linalg::set_value_general(eigen_vectors_m1, eigen_vectors, Nd * nb);
        Linalg::set_value_general(eigen_vectors, eigen_vectors_temp, Nd * nb);

        sigma = sigma2;
    }

    #pragma omp barrier
    #pragma omp single nowait
    {
        delete [] eigen_vectors_m1;
        eigen_vectors_m1 = nullptr;
    }
    #pragma omp single nowait
    {
        delete [] eigen_vectors_temp;
        eigen_vectors_temp = nullptr;
    }
    #pragma omp single nowait
    {
        stencil_temp.destructor();
    }
    #pragma omp single nowait
    {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
            std::cout << "The chebyshev_filtering_column_wise took " << Tools::time_cost(begin, end) << "." << std::endl;
        }
    }

    #ifdef ENABLE_CHEFSI_TIMER
        #pragma omp master
        this->chefsi_timer.filter.stop();
    #endif //ENABLE_CHEFSI_TIMER

    return;
}

template<typename T>
inline void Chefsi<T>::chebyshev_filtering_column_wise2_omp_task_comm_self(T*& __restrict__ eigen_vectors, T const* const& __restrict__ Vloc,
                                                 const Effective_potential_nloc<T>& Vnloc, const bool& print_flag) {
    // uint Nd_Nb = this->domain_vertices.get_4D_local_vertices().get_size();
    if (unlikely(this->single_band_domain_vertices.get_active_comm_size() > 1)) {
        if (unlikely(print_flag)) {
            std::cout << RED << "WARNING: chebyshev_filtering_column_wise2_omp_task_comm_self only support single MPI process in the communicator." << RESET << std::endl;
            std::cout << RED << "WARNING: calling chebyshev_filtering_column_wise2_omp instead." << RESET << std::endl;
        }
        this->chebyshev_filtering_column_wise2_omp(eigen_vectors, Vloc, Vnloc, print_flag);
        return;
    }
    const Vertices_4D& local_vertices = this->domain_vertices.get_4D_local_vertices();
    const Vertices_3D& local_vertices_3d = this->domain_vertices.get_3D_local_vertices();
    const uint Nd = local_vertices_3d.get_size();
    const uint nb = this->domain_vertices.get_4D_local_vertices().nb;
    if (Nd == 0 || nb == 0 || this->chefsi_control.chebyshev_filter_degree == 0) return;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    const T e = 0.5 * (this->lanczos.eig_max - this->lanczos.lambda_cutoff);
    const T c = 0.5 * (this->lanczos.eig_max + this->lanczos.lambda_cutoff);

    Stencil<T>& stencil_temp = this->stencil_temp;
    #pragma omp single
    stencil_temp.deepcopy(stencil);
    stencil_temp.coeffs_scale_self(-0.5);
    #pragma omp barrier
    #pragma omp single nowait
    stencil_temp.shift_D2_coeffs(-c);
    const Vertices_3D ex_vertice(local_vertices_3d.generate_ex_vertices(stencil_temp.FDn));
    const uint ex_Nd = ex_vertice.get_size();

    T*& eigen_vectors_temp = this->eigen_vectors_temp;
    // T*& ex_eigen_vectors = this->ex_eigen_vectors;
    #pragma omp single nowait
    eigen_vectors_temp = new (std::align_val_t(64)) T [Nd * nb];
    #pragma omp barrier

    T sigma = e / (this->lanczos.eig_min - c);
    T sigma1 = sigma;
    T gamma = 2.0 / sigma1;
    T sigma2;

    #pragma omp master
    {
        for (uint ib = 0; ib < nb; ib++) {
            #pragma omp task shared(eigen_vectors, local_vertices_3d, stencil_temp, eigen_vectors_temp, Vloc)
            {
                #pragma omp parallel if(0)
                {
                    T* ex_eigen_vectors = new (std::align_val_t(64)) T [ex_Nd]();
                    const uint offset = ib * Nd;
                    this->single_band_exarr_mpi_package.fill_domain_par_ex_arr(eigen_vectors + offset,
                                                            local_vertices_3d, ex_eigen_vectors, ex_vertice);
                    // eigen_vectors_temp = (-0.5 \nabla - cI + Vloc ) eigen_vectors
                    Stencil_method::calc_laplacian(ex_eigen_vectors, ex_vertice, stencil_temp, local_vertices_3d,
                                                eigen_vectors_temp + offset, local_vertices_3d, Vloc);
                    delete [] ex_eigen_vectors;
                }
            }
        }
    }
    // eigen_vectors_temp += Vnloc*eigen_vectors;
    #pragma omp barrier
    Hamiltonian::nloc_project_vectors<T>(eigen_vectors_temp, local_vertices, eigen_vectors, Vnloc,
                                        (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package.comm);

    T vscal = sigma1 / e;
    // eigen_vectors_temp *= vscal;
    #pragma omp barrier
    Linalg::scalar_product_general(eigen_vectors_temp, vscal, Nd * nb);

    if (unlikely(this->chefsi_control.chebyshev_filter_degree == 1)) {
        Linalg::set_value_general(eigen_vectors, eigen_vectors_temp, Nd * nb);
        #pragma omp barrier
        #pragma omp single nowait
        {
            stencil_temp.destructor();
        }
        #pragma omp single nowait
        {
            delete [] eigen_vectors_temp;
            eigen_vectors_temp = nullptr;
        }
        return;
    }

    T*& eigen_vectors_m1 = this->eigen_vectors_m1;
    #pragma omp single
    eigen_vectors_m1 = new (std::align_val_t(64)) T [Nd * nb];

    #pragma omp barrier
    #pragma omp single nowait
    {
    std::swap(eigen_vectors_m1, eigen_vectors);
    std::swap(eigen_vectors, eigen_vectors_temp);
    }
    // Linalg::set_value_general(eigen_vectors_m1, eigen_vectors, Nd * nb);
    // Linalg::set_value_general(eigen_vectors, eigen_vectors_temp, Nd * nb);

    T vscal2;
    for (uint j = 2; j <= this->chefsi_control.chebyshev_filter_degree; j++) {
        #pragma omp barrier
        sigma2 = 1.0 / (gamma - sigma);
        #pragma omp master
        {
            for (uint ib = 0; ib < nb; ib++) {
                #pragma omp task shared(eigen_vectors, local_vertices_3d, stencil_temp, eigen_vectors_temp, Vloc)
                {
                    #pragma omp parallel if(0)
                    {
                        T* ex_eigen_vectors = new (std::align_val_t(64)) T [ex_Nd]();
                        const uint offset = ib * Nd;
                        this->single_band_exarr_mpi_package.fill_domain_par_ex_arr(eigen_vectors + offset,
                                                                local_vertices_3d, ex_eigen_vectors, ex_vertice);
                        // eigen_vectors_temp = (-0.5 \nabla - cI + Vloc ) eigen_vectors
                        Stencil_method::calc_laplacian(ex_eigen_vectors, ex_vertice, stencil_temp, local_vertices_3d,
                                                    eigen_vectors_temp + offset, local_vertices_3d, Vloc);
                        delete [] ex_eigen_vectors;
                    }
                }
            }
        }
        // eigen_vectors_temp += Vnloc*eigen_vectors;
        #pragma omp barrier
        Hamiltonian::nloc_project_vectors<T>(eigen_vectors_temp, local_vertices, eigen_vectors, Vnloc,
                                        (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package.comm);

        vscal = 2.0 * sigma2 / e;
        vscal2 = sigma * sigma2;
        #pragma omp barrier
        Linalg::scalar_product_general(eigen_vectors_temp, vscal, Nd * nb, eigen_vectors_m1, -vscal2);

        #pragma omp barrier
        #pragma omp single nowait
        {
        std::swap(eigen_vectors_m1, eigen_vectors);
        std::swap(eigen_vectors, eigen_vectors_temp);
        }
        // Linalg::set_value_general(eigen_vectors_m1, eigen_vectors, Nd * nb);
        // Linalg::set_value_general(eigen_vectors, eigen_vectors_temp, Nd * nb);

        sigma = sigma2;
    }

    #pragma omp barrier
    #pragma omp single nowait
    {
        delete [] eigen_vectors_m1;
        eigen_vectors_m1 = nullptr;
    }
    #pragma omp single nowait
    {
        delete [] eigen_vectors_temp;
        eigen_vectors_temp = nullptr;
    }
    #pragma omp single nowait
    {
        stencil_temp.destructor();
    }
    #pragma omp single nowait
    {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
            std::cout << "The chebyshev_filtering_column_wise took " << Tools::time_cost(begin, end) << "." << std::endl;
        }
    }
    return;
}
#endif //USE_OPENMP

template<typename T>
void Chefsi<T>::project_hamiltonian(const Array_4D<T>& eigen_vectors, const Array_4D<T>& h_eigen_vectors,
                                    Array_4D<T>& dp_eigen_vectors, Array_2D<T>& hp, Array_2D<T>& mp,
                                    const bool& print_flag) {
    this->project_hamiltonian(eigen_vectors.data, h_eigen_vectors.data, dp_eigen_vectors.data, hp.data, mp.data, print_flag);
}

template<typename T>
void Chefsi<T>::project_hamiltonian(T* const& __restrict__ eigen_vectors, T* const& __restrict__ h_eigen_vectors,
                                    T* const& __restrict__ dp_eigen_vectors, T* const& __restrict__ hp, T* const& __restrict__ mp,
                                    const bool& print_flag) {
    #ifdef USE_OPENMP

    #ifdef ENABLE_CHEFSI_TIMER
        #pragma omp master
        {
            this->chefsi_timer.projection_gemm.start();
        }
    #endif //ENABLE_CHEFSI_TIMER

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->chefsi_control.projection_method == 0
     || this->chefsi_control.projection_method == 2) {
        const uint m = this->dp_domain_vertices.shared_vertices.get_nb();
        const uint k = this->dp_domain_vertices.local_vertices.Vertices_3D::get_size();
        if (this->domain_vertices.get_domain_4d_comm_size() == 1) {
#if defined(XLSDFT_BACKEND_FREE)
            Linalg::matrix_product(eigen_vectors, 0, h_eigen_vectors, 1, hp, 1, m, m, k);
            Linalg::matrix_product(eigen_vectors, 0, eigen_vectors, 1, mp, 1, m, m, k);
#else
            Linalg::set_kblas_nthread();
            Linalg::cblas__gemm<T>(CblasColMajor, CblasTrans, CblasNoTrans, m, m, k, T(1.0),
                                eigen_vectors, k, h_eigen_vectors, k, T(0.0), hp, m);
            #ifdef ENABLE_CHEFSI_TIMER
                #pragma omp master
                {
                    this->chefsi_timer.projection_gemm.stop();
                    this->chefsi_timer.projection_syrk.start();
                }
            #endif //ENABLE_CHEFSI_TIMER
            Linalg::cblas__syrk(CblasColMajor, CblasUpper, CblasTrans, m, k, T(1.0),
                                     eigen_vectors, k, T(0.0), mp, m);
            #ifdef ENABLE_CHEFSI_TIMER
                #pragma omp master
                {
                    this->chefsi_timer.projection_syrk.stop();
                }
            #endif //ENABLE_CHEFSI_TIMER
            Linalg::set_kblas_1();
#endif
            // Linalg::matrix_product(eigen_vectors, 0, h_eigen_vectors, 1,
            //                                 hp, 1, m, m, k);
            // Linalg::matrix_product(eigen_vectors, 0, eigen_vectors, 1,
            //                                 mp, 1, m, m, k);
            #pragma omp parallel
            Linalg::set_value_general(dp_eigen_vectors, eigen_vectors, m * k);
            #pragma omp barrier
        } else {
            T*& dp_h_eigen_vectors = this->dp_h_eigen_vectors;
            #pragma omp single
            dp_h_eigen_vectors = new (std::align_val_t(64)) T [this->dp_domain_vertices.get_4D_local_vertices().get_size()];
            this->dp_mpi_package.send_data(eigen_vectors, this->domain_vertices.get_4D_local_vertices(),
                                       dp_eigen_vectors, this->dp_domain_vertices.get_4D_local_vertices());
            #pragma omp barrier
            this->dp_mpi_package.send_data(h_eigen_vectors, this->domain_vertices.get_4D_local_vertices(),
                                        dp_h_eigen_vectors, this->dp_domain_vertices.get_4D_local_vertices());
            if (likely(k > 0 && m > 0)) {
                MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
                #pragma omp barrier
                Linalg::matrix_product(dp_eigen_vectors, 0, dp_h_eigen_vectors, 1,
                                            hp, 1, m, m, k);
                if (this->dp_domain_vertices.get_domain_3d_comm_size() > 1) {
                    if (this->dp_domain_vertices.get_active_comm_i() == 0
                    && this->dp_domain_vertices.get_active_comm_j() == 0
                    && this->dp_domain_vertices.get_active_comm_k() == 0) {
                        #pragma omp barrier
                        #pragma omp master
                        MPI_Reduce(MPI_IN_PLACE, hp, (int) m * m, mpi_datatype, MPI_SUM, 0, this->dp_domain_vertices.domain_3d_comm);
                    } else {
                        #pragma omp barrier
                        #pragma omp master
                        MPI_Reduce(hp, hp, (int) m * m, mpi_datatype, MPI_SUM, 0, this->dp_domain_vertices.domain_3d_comm);
                    }
                }
                Linalg::matrix_product(dp_eigen_vectors, 0, dp_eigen_vectors, 1,
                                   mp, 1, m, m, k);
                // #pragma omp single
                // Linalg::__cblas_syrk(CblasColMajor, CblasUpper, CblasTrans, m, k, (T)1.0,
                //                      dp_eigen_vectors, k, (T)0.0, mp, m);
                if (this->dp_domain_vertices.get_domain_3d_comm_size() > 1) {
                    if (this->dp_domain_vertices.get_active_comm_i() == 0
                    && this->dp_domain_vertices.get_active_comm_j() == 0
                    && this->dp_domain_vertices.get_active_comm_k() == 0) {
                        #pragma omp barrier
                        #pragma omp master
                        MPI_Reduce(MPI_IN_PLACE, mp, (int) m * m, mpi_datatype, MPI_SUM, 0, this->dp_domain_vertices.domain_3d_comm);
                    } else {
                        #pragma omp barrier
                        #pragma omp master
                        MPI_Reduce(mp, mp, (int) m * m, mpi_datatype, MPI_SUM, 0, this->dp_domain_vertices.domain_3d_comm);
                    }
                }
            }
            #pragma omp barrier
            #pragma omp single nowait
            {
                delete [] dp_h_eigen_vectors;
                dp_h_eigen_vectors = nullptr;
            }
        }
    } else if (this->chefsi_control.projection_method == 1) {
        #if (defined(USE_MKL) || defined(USE_SCALAPACK))
        const Vertices_4D& local_vertices = this->domain_vertices.get_4D_local_vertices();
        const Vertices_4D& shared_vertices = this->domain_vertices.get_4D_shared_vertices();
        int nstates = (int)shared_vertices.get_nb();
        int local_nstates = (int)local_vertices.get_nb();
        int Nd = local_vertices.Vertices_3D::get_size();
        T alpha = (T)1.0;
        T beta = (T)0.0;
        int one = 1;
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        if (this->icontxt_intra_band != -1 && this->domain_vertices.is_active) {
            Linalg::p_syrk_("U", "T", &nstates, &Nd, &alpha, eigen_vectors, &one, &one,
                             this->desc_intra_band_eigen_vectors, &beta, mp, &one, &one,
                             this->desc_intra_band_hp_mp);
        }
        // to generalize when icontxt_domain_3D smaller than the active comm
        #pragma omp barrier
        #pragma omp master
        if (this->domain_vertices.local_vertices.get_size() > 0
         && this->domain_vertices.get_domain_3d_comm_size() > 1) {
            if (this->domain_vertices.get_active_comm_i() == 0
             && this->domain_vertices.get_active_comm_j() == 0
             && this->domain_vertices.get_active_comm_k() == 0) {
                MPI_Reduce(MPI_IN_PLACE, mp, nstates * local_nstates, mpi_datatype,
                           MPI_SUM, 0, this->domain_vertices.domain_3d_comm);
            } else {
                MPI_Reduce(mp, mp, nstates * local_nstates, mpi_datatype,
                           MPI_SUM, 0, this->domain_vertices.domain_3d_comm);
            }
        }
        #pragma omp barrier
        if (this->icontxt_intra_band != -1 && this->domain_vertices.is_active) {
            Linalg::p_gemm_("T", "N", &nstates, &nstates, &Nd, &alpha, eigen_vectors,
                             &one, &one, this->desc_intra_band_eigen_vectors, h_eigen_vectors,
                             &one, &one, this->desc_intra_band_eigen_vectors, &beta, hp,
                             &one, &one, this->desc_intra_band_hp_mp);
        }
        // to generalize when icontxt_domain_3D smaller than the active comm
        #pragma omp barrier
        #pragma omp master
        if (this->domain_vertices.local_vertices.get_size() > 0
         && this->domain_vertices.get_domain_3d_comm_size() > 1) {
            if (this->domain_vertices.get_active_comm_i() == 0
             && this->domain_vertices.get_active_comm_j() == 0
             && this->domain_vertices.get_active_comm_k() == 0) {
                MPI_Reduce(MPI_IN_PLACE, hp, nstates * local_nstates, mpi_datatype,
                           MPI_SUM, 0, this->domain_vertices.domain_3d_comm);
            } else {
                MPI_Reduce(hp, hp, nstates * local_nstates, mpi_datatype,
                           MPI_SUM, 0, this->domain_vertices.domain_3d_comm);
            }
        }
        #endif
    } else {
        assert(this->chefsi_control.projection_method == 0
            || this->chefsi_control.projection_method == 1
            || this->chefsi_control.projection_method == 2);
    }
    #pragma omp barrier
    #pragma omp single nowait
    {
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
        std::cout << "The project_hamiltonian took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    }

    #else //USE_OPENMP

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->chefsi_control.projection_method == 0
     || this->chefsi_control.projection_method == 2) {
        const uint m = this->dp_domain_vertices.shared_vertices.get_nb();
        const uint k = this->dp_domain_vertices.local_vertices.Vertices_3D::get_size();
        if (this->domain_vertices.get_domain_4d_comm_size() == 1) {
            Linalg::matrix_product(eigen_vectors, 0, h_eigen_vectors, 1,
                                            hp, 1, m, m, k);
            // Linalg::matrix_product(eigen_vectors, 0, eigen_vectors, 1,
            //                                 mp, 1, m, m, k);
            Linalg::cblas__syrk(CblasColMajor, CblasUpper, CblasTrans, m, k, T(1.0),
                                     eigen_vectors, k, T(0.0), mp, m);
            Linalg::set_value_general(dp_eigen_vectors, eigen_vectors, m * k);
        } else {
            T* dp_h_eigen_vectors = new (std::align_val_t(64)) T [this->dp_domain_vertices.get_4D_local_vertices().get_size()];
            this->dp_mpi_package.send_data(eigen_vectors, this->domain_vertices.get_4D_local_vertices(),
                                       dp_eigen_vectors, this->dp_domain_vertices.get_4D_local_vertices());
            this->dp_mpi_package.send_data(h_eigen_vectors, this->domain_vertices.get_4D_local_vertices(),
                                       dp_h_eigen_vectors, this->dp_domain_vertices.get_4D_local_vertices());
            if (k > 0 && m > 0) {
                MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
                Linalg::matrix_product(dp_eigen_vectors, 0, dp_h_eigen_vectors, 1,
                                            hp, 1, m, m, k);
                if (this->dp_domain_vertices.get_domain_3d_comm_size() > 1) {
                    if (this->dp_domain_vertices.get_active_comm_i() == 0
                    && this->dp_domain_vertices.get_active_comm_j() == 0
                    && this->dp_domain_vertices.get_active_comm_k() == 0) {
                        MPI_Reduce(MPI_IN_PLACE, hp, (int) m * m, mpi_datatype, MPI_SUM, 0, this->dp_domain_vertices.domain_3d_comm);
                    } else {
                        MPI_Reduce(hp, hp, (int) m * m, mpi_datatype, MPI_SUM, 0, this->dp_domain_vertices.domain_3d_comm);
                    }
                }
                #ifdef USE_CBLAS
                Linalg::cblas__syrk(CblasColMajor, CblasUpper, CblasTrans, m, k, (T)1.0,
                                    dp_eigen_vectors, k, (T)0.0, mp, m);
                #else
                Linalg::matrix_product(dp_eigen_vectors, 0, dp_eigen_vectors, 1,
                                    mp, 1, m, m, k);
                #endif
                if (this->dp_domain_vertices.get_domain_3d_comm_size() > 1) {
                    if (this->dp_domain_vertices.get_active_comm_i() == 0
                    && this->dp_domain_vertices.get_active_comm_j() == 0
                    && this->dp_domain_vertices.get_active_comm_k() == 0) {
                        MPI_Reduce(MPI_IN_PLACE, mp, (int) m * m, mpi_datatype, MPI_SUM, 0, this->dp_domain_vertices.domain_3d_comm);
                    } else {
                        MPI_Reduce(mp, mp, (int) m * m, mpi_datatype, MPI_SUM, 0, this->dp_domain_vertices.domain_3d_comm);
                    }
                }
            }
            delete [] dp_h_eigen_vectors;
        }
    } else if (this->chefsi_control.projection_method == 1) {
        #if (defined(USE_MKL) || defined(USE_SCALAPACK))
        const Vertices_4D& local_vertices = this->domain_vertices.get_4D_local_vertices();
        const Vertices_4D& shared_vertices = this->domain_vertices.get_4D_shared_vertices();
        int nstates = (int)shared_vertices.get_nb();
        int local_nstates = (int)local_vertices.get_nb();
        int Nd = local_vertices.Vertices_3D::get_size();
        T alpha = (T)1.0;
        T beta = (T)0.0;
        int one = 1;
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        if (this->icontxt_intra_band != -1 && this->domain_vertices.is_active) {
            // Linalg::p_syrk_("U", "T", &nstates, &Nd, &alpha, eigen_vectors, &one, &one,
            //                  this->desc_intra_band_eigen_vectors, &beta, mp, &one, &one,
            //                  this->desc_intra_band_hp_mp);
            //p_gemm_ is faster than p_syrk_, i dont know why
            Linalg::p_gemm_("T", "N", &nstates, &nstates, &Nd, &alpha, eigen_vectors,
                             &one, &one, this->desc_intra_band_eigen_vectors, eigen_vectors,
                             &one, &one, this->desc_intra_band_eigen_vectors, &beta, mp,
                             &one, &one, this->desc_intra_band_hp_mp);
        }
        // to generalize when icontxt_domain_3D smaller than the active comm
        if (this->domain_vertices.local_vertices.get_size() > 0
         && this->domain_vertices.get_domain_3d_comm_size() > 1) {
            if (this->domain_vertices.get_active_comm_i() == 0
             && this->domain_vertices.get_active_comm_j() == 0
             && this->domain_vertices.get_active_comm_k() == 0) {
                MPI_Reduce(MPI_IN_PLACE, mp, nstates * local_nstates, mpi_datatype,
                           MPI_SUM, 0, this->domain_vertices.domain_3d_comm);
            } else {
                MPI_Reduce(mp, mp, nstates * local_nstates, mpi_datatype,
                           MPI_SUM, 0, this->domain_vertices.domain_3d_comm);
            }
        }
        if (this->icontxt_intra_band != -1 && this->domain_vertices.is_active) {
            Linalg::p_gemm_("T", "N", &nstates, &nstates, &Nd, &alpha, eigen_vectors,
                             &one, &one, this->desc_intra_band_eigen_vectors, h_eigen_vectors,
                             &one, &one, this->desc_intra_band_eigen_vectors, &beta, hp,
                             &one, &one, this->desc_intra_band_hp_mp);
        }
        // to generalize when icontxt_domain_3D smaller than the active comm
        if (this->domain_vertices.local_vertices.get_size() > 0
         && this->domain_vertices.get_domain_3d_comm_size() > 1) {
            if (this->domain_vertices.get_active_comm_i() == 0
             && this->domain_vertices.get_active_comm_j() == 0
             && this->domain_vertices.get_active_comm_k() == 0) {
                MPI_Reduce(MPI_IN_PLACE, hp, nstates * local_nstates, mpi_datatype,
                           MPI_SUM, 0, this->domain_vertices.domain_3d_comm);
            } else {
                MPI_Reduce(hp, hp, nstates * local_nstates, mpi_datatype,
                           MPI_SUM, 0, this->domain_vertices.domain_3d_comm);
            }
        }
        #endif
    } else {
        assert(this->chefsi_control.projection_method == 0
            || this->chefsi_control.projection_method == 1
            || this->chefsi_control.projection_method == 2);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
        std::cout << "The project_hamiltonian took " << Tools::time_cost(begin, end) << "." << std::endl;
    }

    #endif //USE_OPENMP
    return;
}

template<typename T>
void Chefsi<T>::subspace_diagonalization(Array_2D<T>& hp, Array_2D<T>& mp, Array_0D<T>& eigen_values, const bool& print_flag) {
    this->subspace_diagonalization(hp.data, mp.data, eigen_values.data, print_flag);
    return;
}

template<typename T>
void Chefsi<T>::subspace_diagonalization(T* const& __restrict__ hp, T* const& __restrict__ mp,
                                         T* const& __restrict__ eigen_values, const bool& print_flag) {
    #ifdef USE_OPENMP

    #ifdef ENABLE_CHEFSI_TIMER
        #pragma omp master
        {
            this->chefsi_timer.diagonalization.start();
        }
    #endif //ENABLE_CHEFSI_TIMER

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->chefsi_control.projection_method == 0) {
        uint nstate = this->dp_domain_vertices.shared_vertices.get_nb();
        if (this->dp_domain_vertices.get_comm_rank() == 0) {
            
            #ifdef USE_LAPACK
            int info = Linalg::LAPACKE__sygvd<T>(LAPACK_COL_MAJOR, 1, 'V', 'U', nstate,
                                               hp, nstate, mp, nstate, eigen_values);
            if (info != 0) {
                assert(false);
            }
            #else
            assert(!"LAPACKE__sygvd should be involved with LAPACKE loaded");
            (void) mp;
            #endif
        }
        #pragma omp master
        {
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        MPI_Bcast(eigen_values, nstate, mpi_datatype, 0, this->dp_domain_vertices.comm);
        if (this->dp_domain_vertices.local_vertices.Vertices_3D::get_size() > 0) {
            MPI_Bcast(hp, nstate * nstate, mpi_datatype, 0, this->dp_domain_vertices.domain_3d_comm);
        }
        }
    } else if (this->chefsi_control.projection_method == 1) {
        #if (defined(USE_MKL) || defined(USE_SCALAPACK))
        const Vertices_4D& local_vertices = this->domain_vertices.get_4D_local_vertices();
        const Vertices_4D& shared_vertices = this->domain_vertices.get_4D_shared_vertices();
        int nstates = (int)shared_vertices.get_nb();
        int local_nstates = (int)local_vertices.get_nb();
        if (this->icontxt_intra_band != -1 && this->domain_vertices.get_domain_3d_comm_rank() == 0) {
            T*& mp_2d_rashape = this->mp_2d_rashape;
            T*& hp_2d_rashape = this->hp_2d_rashape;
            T*& qp_2d_rashape = this->qp_2d_rashape;
            #pragma omp single nowait
            mp_2d_rashape = new (std::align_val_t(64)) T [this->hp_mp_2d_reshape_m * this->hp_mp_2d_reshape_n];
            #pragma omp single nowait
            hp_2d_rashape = new (std::align_val_t(64)) T [this->hp_mp_2d_reshape_m * this->hp_mp_2d_reshape_n];
            #pragma omp single nowait
            qp_2d_rashape = new (std::align_val_t(64)) T [this->hp_mp_2d_reshape_m * this->hp_mp_2d_reshape_n];
            #pragma omp barrier

            int one = 1;
            Linalg::p_gemr2d_(&nstates, &nstates, hp, &one, &one, this->desc_intra_band_hp_mp, hp_2d_rashape, &one, &one,
                            this->desc_2d_reshape_hp_mp, &(this->icontxt_intra_band));
            #pragma omp barrier
            Linalg::p_gemr2d_(&nstates, &nstates, mp, &one, &one, this->desc_intra_band_hp_mp, mp_2d_rashape, &one, &one,
                            this->desc_2d_reshape_hp_mp, &(this->icontxt_intra_band));
            if (this->icontxt_2d_reshape != -1) {
                int lwork = this->lwork;
                int liwork = this->liwork;
                T*& work = this->work;
                int*& iwork = this->iwork;
                int*& ifail = this->ifail;
                int*& iclustr = this->iclustr;
                T*& gap = this->gap;
                #pragma omp single nowait
                work = new (std::align_val_t(64)) T [lwork];
                #pragma omp single nowait
                iwork = new (std::align_val_t(64)) int [liwork];
                #pragma omp single nowait
                ifail = new (std::align_val_t(64)) int [nstates];
                #pragma omp single nowait
                iclustr = new (std::align_val_t(64)) int [2 * this->icontxt_2d_reshape_nprow * this->icontxt_2d_reshape_npcol];
                #pragma omp single nowait
                gap = new (std::align_val_t(64)) T [this->icontxt_2d_reshape_nprow * this->icontxt_2d_reshape_npcol];
                #pragma omp barrier
                int ibtype = 1;
                char jobz = 'V';
                char range = 'A';
                char uplo = 'U';
                T vl = this->lanczos.eig_min;
                T vu = this->lanczos.lambda_cutoff + (T)0.1;
                T abstol = (T)this->chefsi_control.abstol;
                T orfac = this->chefsi_control.orfac;
                int m;
                int nz;
                int& info = this->info;
                Linalg::p_sygvx_(&ibtype, &jobz, &range, &uplo, &nstates,
                                hp_2d_rashape, &one, &one, this->desc_2d_reshape_hp_mp,
                                mp_2d_rashape, &one, &one, this->desc_2d_reshape_hp_mp,
                                &vl, &vu, &one, &nstates, &abstol, &m, &nz, eigen_values, &orfac,
                                qp_2d_rashape, &one, &one, this->desc_2d_reshape_hp_mp,
                                work, &lwork, iwork, &liwork, ifail, iclustr, gap, &info);
                #pragma omp barrier
                #pragma omp single nowait
                if (info != 0) {
                assert(false);
            }
                #pragma omp single nowait
                {
                    delete [] work;
                    work = nullptr;
                }
                #pragma omp single nowait
                {
                    delete [] iwork;
                    iwork = nullptr;
                }
                #pragma omp single nowait
                {
                    delete [] ifail;
                    ifail = nullptr;
                }
                #pragma omp single nowait
                {
                    delete [] iclustr;
                    iclustr = nullptr;
                }
                #pragma omp single nowait
                {
                    delete [] gap;
                    gap = nullptr;
                }
            }
            #pragma omp barrier
            Linalg::p_gemr2d_(&nstates, &nstates, qp_2d_rashape, &one, &one, this->desc_2d_reshape_hp_mp,
                            hp, &one, &one, this->desc_intra_band_hp_mp, &(this->icontxt_intra_band));
            #pragma omp single nowait
            {
                delete [] mp_2d_rashape;
                mp_2d_rashape = nullptr;
            }
            #pragma omp single nowait
            {
                delete [] hp_2d_rashape;
                hp_2d_rashape = nullptr;
            }
            #pragma omp barrier
            #pragma omp single nowait
            {
                delete [] qp_2d_rashape;
                qp_2d_rashape = nullptr;
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            // to generalize when icontxt_domain_3D smaller than the active comm
            MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
            MPI_Bcast(eigen_values, nstates, mpi_datatype, 0, this->domain_vertices.comm);
            #pragma omp master
            if (this->domain_vertices.local_vertices.get_size() > 0
            && this->domain_vertices.get_domain_3d_comm_size() > 1) {
                MPI_Bcast(hp, nstates * local_nstates, mpi_datatype, 0,
                        this->domain_vertices.get_domain_3d_comm());
            }
        }
        #endif
    } else if (this->chefsi_control.projection_method == 2) {
        #if (defined(USE_MKL) || defined(USE_SCALAPACK))
        T*& mp_2d_rashape = this->mp_2d_rashape;
        T*& hp_2d_rashape = this->hp_2d_rashape;
        T*& qp_2d_rashape = this->qp_2d_rashape;
        #pragma omp single nowait
        mp_2d_rashape = new (std::align_val_t(64)) T [this->hp_mp_2d_reshape_m * this->hp_mp_2d_reshape_n];
        #pragma omp single nowait
        hp_2d_rashape = new (std::align_val_t(64)) T [this->hp_mp_2d_reshape_m * this->hp_mp_2d_reshape_n];
        #pragma omp single nowait
        qp_2d_rashape = new (std::align_val_t(64)) T [this->hp_mp_2d_reshape_m * this->hp_mp_2d_reshape_n];
        #pragma omp barrier
        const Vertices_4D& shared_vertices = this->domain_vertices.get_4D_shared_vertices();
        int nstates = (int)shared_vertices.get_nb();
        int one = 1;
        if (this->icontxt_2d_reshape != -1) {
            #pragma omp barrier
            #pragma omp single
            Linalg::p_gemr2d_(&nstates, &nstates, hp, &one, &one, this->desc_domain_3D_hp_mp, hp_2d_rashape, &one, &one,
                               this->desc_2d_reshape_hp_mp, &(this->icontxt_2d_reshape));
            #pragma omp single nowait
            Linalg::p_gemr2d_(&nstates, &nstates, mp, &one, &one, this->desc_domain_3D_hp_mp, mp_2d_rashape, &one, &one,
                               this->desc_2d_reshape_hp_mp, &(this->icontxt_2d_reshape));
            int lwork = this->lwork;
            int liwork = this->liwork;
            T*& work = this->work;
            int*& iwork = this->iwork;
            int*& ifail = this->ifail;
            int*& iclustr = this->iclustr;
            T*& gap = this->gap;
            #pragma omp single nowait
            work = new (std::align_val_t(64)) T [lwork];
            #pragma omp single nowait
            iwork = new (std::align_val_t(64)) int [liwork];
            #pragma omp single nowait
            ifail = new (std::align_val_t(64)) int [nstates];
            #pragma omp single nowait
            iclustr = new (std::align_val_t(64)) int [2 * this->icontxt_2d_reshape_nprow * this->icontxt_2d_reshape_npcol];
            #pragma omp single nowait
            gap = new (std::align_val_t(64)) T [this->icontxt_2d_reshape_nprow * this->icontxt_2d_reshape_npcol];
            #pragma omp barrier
            int ibtype = 1;
            char jobz = 'V';
            char range = 'A';
            char uplo = 'U';
            T vl = this->lanczos.eig_min;
            T vu = this->lanczos.lambda_cutoff + (T)0.1;
            T abstol = (T)this->chefsi_control.abstol;
            T orfac = this->chefsi_control.orfac;
            int m;
            int nz;
            int& info = this->info;
            #pragma omp single
            Linalg::p_sygvx_(&ibtype, &jobz, &range, &uplo, &nstates,
                            hp_2d_rashape, &one, &one, this->desc_2d_reshape_hp_mp,
                            mp_2d_rashape, &one, &one, this->desc_2d_reshape_hp_mp,
                            &vl, &vu, &one, &nstates, &abstol, &m, &nz, eigen_values, &orfac,
                            qp_2d_rashape, &one, &one, this->desc_2d_reshape_hp_mp,
                            work, &lwork, iwork, &liwork, ifail, iclustr, gap, &info);
            // #pragma omp barrier
            #pragma omp single nowait
            if (info != 0) {
                assert(false);
            }
            #pragma omp single nowait
            {
                delete [] work;
                work = nullptr;
            }
            #pragma omp single nowait
            {
                delete [] iwork;
                iwork = nullptr;
            }
            #pragma omp single nowait
            {
                delete [] ifail;
                ifail = nullptr;
            }
            #pragma omp single nowait
            {
                delete [] iclustr;
                iclustr = nullptr;
            }
            #pragma omp single nowait
            {
                delete [] gap;
                gap = nullptr;
            }
            #pragma omp barrier
            #pragma omp single nowait
            Linalg::p_gemr2d_(&nstates, &nstates, qp_2d_rashape, &one, &one, this->desc_2d_reshape_hp_mp,
                              hp, &one, &one, this->desc_domain_3D_hp_mp, &(this->icontxt_2d_reshape));
        }
        #pragma omp barrier
        #pragma omp master
        {
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        MPI_Bcast(eigen_values, nstates, mpi_datatype, 0, this->dp_domain_vertices.comm);
        if (this->dp_domain_vertices.local_vertices.Vertices_3D::get_size() > 0) {
            MPI_Bcast(hp, nstates * nstates, mpi_datatype, 0, this->dp_domain_vertices.domain_3d_comm);
        }
        }
        #pragma omp single nowait
        {
            delete [] mp_2d_rashape;
            mp_2d_rashape = nullptr;
        }
        #pragma omp single nowait
        {
            delete [] hp_2d_rashape;
            hp_2d_rashape = nullptr;
        }
        #pragma omp single nowait
        {
            delete [] qp_2d_rashape;
            qp_2d_rashape = nullptr;
        }
        #endif
    } else {
        assert(this->chefsi_control.projection_method == 0
            || this->chefsi_control.projection_method == 1
            || this->chefsi_control.projection_method == 2);
    }
    #pragma omp barrier
    #pragma omp single nowait
    {
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
        std::cout << "The subspace_diagonalization took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    }

    #ifdef ENABLE_CHEFSI_TIMER
        #pragma omp master
        {
            this->chefsi_timer.diagonalization.stop();
        }
    #endif //ENABLE_CHEFSI_TIMER

    #else //USE_OPENMP

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->chefsi_control.projection_method == 0) {
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        uint nstate = this->dp_domain_vertices.shared_vertices.get_nb();
        if (this->dp_domain_vertices.get_comm_rank() == 0) {
            #ifdef USE_LAPACK
            int info = Linalg::LAPACKE__sygvd(LAPACK_COL_MAJOR, 1, 'V', 'U', nstate,
                                               hp, nstate, mp, nstate, eigen_values);
            if (info != 0) {
                assert(false);
            }
            #else
            assert(!"LAPACKE__sygvd should be involved with LAPACKE loaded");
            std::cout << "mp[0] = " << mp[0] << std::endl;
            #endif
        }
        MPI_Bcast(eigen_values, nstate, mpi_datatype, 0, this->dp_domain_vertices.comm);
        if (this->dp_domain_vertices.local_vertices.Vertices_3D::get_size() > 0) {
            MPI_Bcast(hp, nstate * nstate, mpi_datatype, 0, this->dp_domain_vertices.domain_3d_comm);
        }
    } else if (this->chefsi_control.projection_method == 1) {
        #if (defined(USE_MKL) || defined(USE_SCALAPACK))

        const Vertices_4D& local_vertices = this->domain_vertices.get_4D_local_vertices();
        const Vertices_4D& shared_vertices = this->domain_vertices.get_4D_shared_vertices();
        int nstates = (int)shared_vertices.get_nb();
        int local_nstates = (int)local_vertices.get_nb();
        if (this->icontxt_intra_band != -1 && this->domain_vertices.get_domain_3d_comm_rank() == 0) {
            T* mp_2d_rashape = new (std::align_val_t(64)) T [this->hp_mp_2d_reshape_m * this->hp_mp_2d_reshape_n];
            T* hp_2d_rashape = new (std::align_val_t(64)) T [this->hp_mp_2d_reshape_m * this->hp_mp_2d_reshape_n];
            T* qp_2d_rashape = new (std::align_val_t(64)) T [this->hp_mp_2d_reshape_m * this->hp_mp_2d_reshape_n];
            int one = 1;
            Linalg::p_gemr2d_(&nstates, &nstates, hp, &one, &one, this->desc_intra_band_hp_mp, hp_2d_rashape, &one, &one,
                            this->desc_2d_reshape_hp_mp, &(this->icontxt_intra_band));
            Linalg::p_gemr2d_(&nstates, &nstates, mp, &one, &one, this->desc_intra_band_hp_mp, mp_2d_rashape, &one, &one,
                            this->desc_2d_reshape_hp_mp, &(this->icontxt_intra_band));
            if (this->icontxt_2d_reshape != -1) {
                int ibtype = 1;
                char jobz = 'V';
                char range = 'A';
                char uplo = 'U';
                T vl = this->lanczos.eig_min;
                T vu = this->lanczos.lambda_cutoff + (T)0.1;
                T abstol = (T)this->chefsi_control.abstol;
                T orfac = this->chefsi_control.orfac;
                int lwork = this->lwork;
                int liwork = this->liwork;
                T* work = new (std::align_val_t(64)) T [lwork];
                int* iwork = new (std::align_val_t(64)) int [liwork];
                int m;
                int nz;
                int* ifail = new (std::align_val_t(64)) int [nstates];
                int* iclustr = new (std::align_val_t(64)) int [2 * this->icontxt_2d_reshape_nprow * this->icontxt_2d_reshape_npcol];
                T* gap = new (std::align_val_t(64)) T [this->icontxt_2d_reshape_nprow * this->icontxt_2d_reshape_npcol];
                int info;
                Linalg::p_sygvx_(&ibtype, &jobz, &range, &uplo, &nstates,
                                hp_2d_rashape, &one, &one, this->desc_2d_reshape_hp_mp,
                                mp_2d_rashape, &one, &one, this->desc_2d_reshape_hp_mp,
                                &vl, &vu, &one, &nstates, &abstol, &m, &nz, eigen_values, &orfac,
                                qp_2d_rashape, &one, &one, this->desc_2d_reshape_hp_mp,
                                work, &lwork, iwork, &liwork, ifail, iclustr, gap, &info);
                delete [] ifail;
                delete [] work;
                delete [] iwork;
                delete [] iclustr;
                delete [] gap;
                if (info != 0) {
                assert(false);
            }
            }

            Linalg::p_gemr2d_(&nstates, &nstates, qp_2d_rashape, &one, &one, this->desc_2d_reshape_hp_mp,
                            hp, &one, &one, this->desc_intra_band_hp_mp, &(this->icontxt_intra_band));
            delete [] mp_2d_rashape;
            delete [] hp_2d_rashape;
            delete [] qp_2d_rashape;
        }
        // to generalize when icontxt_domain_3D smaller than the active comm
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        MPI_Bcast(eigen_values, nstates, mpi_datatype, 0, this->domain_vertices.comm);
        if (this->domain_vertices.local_vertices.get_size() > 0
         && this->domain_vertices.get_domain_3d_comm_size() > 1) {
            MPI_Bcast(hp, nstates * local_nstates, mpi_datatype, 0,
                      this->domain_vertices.get_domain_3d_comm());
        }
        #endif
    } else if (this->chefsi_control.projection_method == 2) {
        #if (defined(USE_MKL) || defined(USE_SCALAPACK))
        T* mp_2d_rashape = new (std::align_val_t(64)) T [this->hp_mp_2d_reshape_m * this->hp_mp_2d_reshape_n];
        T* hp_2d_rashape = new (std::align_val_t(64)) T [this->hp_mp_2d_reshape_m * this->hp_mp_2d_reshape_n];
        T* qp_2d_rashape = new (std::align_val_t(64)) T [this->hp_mp_2d_reshape_m * this->hp_mp_2d_reshape_n];
        const Vertices_4D& shared_vertices = this->domain_vertices.get_4D_shared_vertices();
        int nstates = (int)shared_vertices.get_nb();
        int one = 1;
        if (this->icontxt_2d_reshape != -1) {
            Linalg::p_gemr2d_(&nstates, &nstates, hp, &one, &one, this->desc_domain_3D_hp_mp, hp_2d_rashape, &one, &one,
                               this->desc_2d_reshape_hp_mp, &(this->icontxt_2d_reshape));
            Linalg::p_gemr2d_(&nstates, &nstates, mp, &one, &one, this->desc_domain_3D_hp_mp, mp_2d_rashape, &one, &one,
                               this->desc_2d_reshape_hp_mp, &(this->icontxt_2d_reshape));
            int ibtype = 1;
            char jobz = 'V';
            char range = 'A';
            char uplo = 'U';
            T vl = this->lanczos.eig_min;
            T vu = this->lanczos.lambda_cutoff + (T)0.1;
            T abstol = (T)this->chefsi_control.abstol;
            T orfac = this->chefsi_control.orfac;
            int lwork = this->lwork;
            int liwork = this->liwork;
            T* work = new (std::align_val_t(64)) T [lwork];
            int* iwork = new (std::align_val_t(64)) int [liwork];
            int m;
            int nz;
            int* ifail = new (std::align_val_t(64)) int [nstates];
            int* iclustr = new (std::align_val_t(64)) int [2 * this->icontxt_2d_reshape_nprow * this->icontxt_2d_reshape_npcol];
            T* gap = new (std::align_val_t(64)) T [this->icontxt_2d_reshape_nprow * this->icontxt_2d_reshape_npcol];
            int info;
            Linalg::p_sygvx_(&ibtype, &jobz, &range, &uplo, &nstates,
                            hp_2d_rashape, &one, &one, this->desc_2d_reshape_hp_mp,
                            mp_2d_rashape, &one, &one, this->desc_2d_reshape_hp_mp,
                            &vl, &vu, &one, &nstates, &abstol, &m, &nz, eigen_values, &orfac,
                            qp_2d_rashape, &one, &one, this->desc_2d_reshape_hp_mp,
                            work, &lwork, iwork, &liwork, ifail, iclustr, gap, &info);
            delete [] ifail;
            delete [] work;
            delete [] iwork;
            delete [] iclustr;
            delete [] gap;
            if (info != 0) {
                assert(false);
            }
            Linalg::p_gemr2d_(&nstates, &nstates, qp_2d_rashape, &one, &one, this->desc_2d_reshape_hp_mp,
                              hp, &one, &one, this->desc_domain_3D_hp_mp, &(this->icontxt_2d_reshape));
        }
        // to generalize when icontxt_domain_3D smaller than the active comm
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        MPI_Bcast(eigen_values, nstates, mpi_datatype, 0, this->dp_domain_vertices.comm);
        if (this->dp_domain_vertices.local_vertices.Vertices_3D::get_size() > 0) {
            MPI_Bcast(hp, nstates * nstates, mpi_datatype, 0, this->dp_domain_vertices.domain_3d_comm);
        }
        delete [] mp_2d_rashape;
        delete [] hp_2d_rashape;
        delete [] qp_2d_rashape;
        #endif
    } else {
        assert(this->chefsi_control.projection_method == 0
            || this->chefsi_control.projection_method == 1
            || this->chefsi_control.projection_method == 2);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
        std::cout << "The subspace_diagonalization took " << Tools::time_cost(begin, end) << "." << std::endl;
    }

    #endif //USE_OPENMP

    return;
}

template<typename T>
void Chefsi<T>::subspace_rotation(Array_4D<T>& eigen_vectors, const Array_4D<T>& eigen_vectors_reshape,
                                  const Array_2D<T>& qp, const bool& print_flag) {
    this->subspace_rotation(eigen_vectors.data, eigen_vectors_reshape.data, qp.data, print_flag);
    return;
}

template<typename T>
void Chefsi<T>::subspace_rotation(T*& __restrict__ eigen_vectors, T const* const& __restrict__ eigen_vectors_reshape,
                                  T const* const& __restrict__ qp, const bool& print_flag) {
    #ifdef USE_OPENMP

    #ifdef ENABLE_CHEFSI_TIMER
        #pragma omp master
        {
            this->chefsi_timer.rotation.start();
        }
    #endif //ENABLE_CHEFSI_TIMER

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->chefsi_control.projection_method == 0
     || this->chefsi_control.projection_method == 2) {
        uint m = this->dp_domain_vertices.local_vertices.Vertices_3D::get_size();
        uint n = this->dp_domain_vertices.local_vertices.get_nb();
        if (this->domain_vertices.get_domain_4d_comm_size() == 1) {
            if (m > 0) {
#if defined(XLSDFT_BACKEND_FREE)
                Linalg::matrix_product(eigen_vectors_reshape, 1, qp, 1, eigen_vectors, 1, m, n, n);
#else
                Linalg::set_kblas_nthread();
                Linalg::cblas__gemm<T>(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, n, T(1.0),
                                eigen_vectors_reshape, m, qp, n, T(0.0), eigen_vectors, m);
                Linalg::set_kblas_1();
#endif
            }
            #pragma omp barrier
        } else {
            T*& eigen_vectors_reshape_temp = this->eigen_vectors_reshape_temp;
            #pragma omp single
            eigen_vectors_reshape_temp = new (std::align_val_t(64)) T [this->dp_domain_vertices.get_4D_local_vertices().get_size()];
            if (m > 0) {
                Linalg::matrix_product(eigen_vectors_reshape, 1, qp, 1, eigen_vectors_reshape_temp, 1, m, n, n);
            }
            #pragma omp barrier
            this->dp_mpi_package.recv_data(eigen_vectors, this->domain_vertices.get_4D_local_vertices(),
                                       eigen_vectors_reshape_temp, this->dp_domain_vertices.get_4D_local_vertices());
            #pragma omp barrier
            #pragma omp single nowait
            {
                delete [] eigen_vectors_reshape_temp;
                eigen_vectors_reshape_temp = nullptr;
            }
        }
    } else if (this->chefsi_control.projection_method == 1) {
        #if (defined(USE_MKL) || defined(USE_SCALAPACK))
        const Vertices_4D& local_vertices = this->domain_vertices.get_4D_local_vertices();
        const Vertices_4D& shared_vertices = this->domain_vertices.get_4D_shared_vertices();
        int nstates = (int)shared_vertices.get_nb();
        int local_nstates = (int)local_vertices.get_nb();
        int Nd = local_vertices.Vertices_3D::get_size();
        T*& eigen_vectors_temp = this->eigen_vectors_temp;
        #pragma omp single
        eigen_vectors_temp = new (std::align_val_t(64)) T [Nd * local_nstates];
        T alpha = (T)1.0;
        T beta = (T)0.0;
        int one = 1;
        if (this->icontxt_intra_band != -1 && this->domain_vertices.is_active) {
            Linalg::p_gemm_("N", "N", &Nd, &nstates, &nstates, &alpha, eigen_vectors,
                             &one, &one, this->desc_intra_band_eigen_vectors, qp,
                             &one, &one, this->desc_intra_band_hp_mp, &beta, eigen_vectors_temp,
                             &one, &one, this->desc_intra_band_eigen_vectors);
            #pragma omp barrier
            #pragma omp single nowait
            std::swap(eigen_vectors, eigen_vectors_temp);
        }
        #pragma omp barrier
        #pragma omp single nowait
        {
            delete [] eigen_vectors_temp;
            eigen_vectors_temp = nullptr;
        }
        #endif
    } else {
        assert(this->chefsi_control.projection_method == 0);
    }
    #pragma omp barrier
    #pragma omp single nowait
    {
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
        std::cout << "The subspace_rotation took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    }

    #ifdef ENABLE_CHEFSI_TIMER
        #pragma omp master
        {
            this->chefsi_timer.rotation.stop();
        }
    #endif //ENABLE_CHEFSI_TIMER

    #else //USE_OPENMP

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->chefsi_control.projection_method == 0
     || this->chefsi_control.projection_method == 2) {
        const uint m = this->dp_domain_vertices.local_vertices.Vertices_3D::get_size();
        const uint n = this->dp_domain_vertices.local_vertices.get_nb();
        if (this->domain_vertices.get_domain_4d_comm_size() == 1) {
            if (likely(m > 0)) {
                Linalg::matrix_product(eigen_vectors_reshape, 1, qp, 1, eigen_vectors, 1, m, n, n);
            }
        } else {
            if (likely(m > 0)) {
                T* eigen_vectors_reshape_temp = new (std::align_val_t(64)) T [this->dp_domain_vertices.get_4D_local_vertices().get_size()];
                Linalg::matrix_product(eigen_vectors_reshape, 1, qp, 1, eigen_vectors_reshape_temp, 1, m, n, n);
                this->dp_mpi_package.recv_data(eigen_vectors, this->domain_vertices.get_4D_local_vertices(),
                                        eigen_vectors_reshape_temp, this->dp_domain_vertices.get_4D_local_vertices());
                delete [] eigen_vectors_reshape_temp;
            }
        }
    } else if (this->chefsi_control.projection_method == 1) {
        #if (defined(USE_MKL) || defined(USE_SCALAPACK))
        const Vertices_4D& local_vertices = this->domain_vertices.get_4D_local_vertices();
        const Vertices_4D& shared_vertices = this->domain_vertices.get_4D_shared_vertices();
        int nstates = (int)shared_vertices.get_nb();
        int local_nstates = (int)local_vertices.get_nb();
        int Nd = local_vertices.Vertices_3D::get_size();
        T* eigen_vectors_temp = new (std::align_val_t(64)) T [Nd * local_nstates];
        T alpha = (T)1.0;
        T beta = (T)0.0;
        int one = 1;
        if (this->icontxt_intra_band != -1 && this->domain_vertices.is_active) {
            Linalg::p_gemm_("N", "N", &Nd, &nstates, &nstates, &alpha, eigen_vectors,
                             &one, &one, this->desc_intra_band_eigen_vectors, qp,
                             &one, &one, this->desc_intra_band_hp_mp, &beta, eigen_vectors_temp,
                             &one, &one, this->desc_intra_band_eigen_vectors);
            std::swap(eigen_vectors, eigen_vectors_temp);
        }
        delete [] eigen_vectors_temp;
        #endif
    } else {
        assert(this->chefsi_control.projection_method == 0);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
        std::cout << "The subspace_rotation took " << Tools::time_cost(begin, end) << "." << std::endl;
    }

    #endif //USE_OPENMP

    return;
}

template<typename T>
void Chefsi<T>::cal_int_density_per_band(const Vertices_3D& region, const Array_0D<T>& eigen_values,
                                         const Array_4D<T>& eigen_vector, std::vector<T>& eigen_value_per_band,
                                         std::vector<T>& int_density_per_band) const {
    if (this->domain_vertices.is_active) {
        const Vertices_4D& local_vertices = this->domain_vertices.local_vertices;
        uint nstates = local_vertices.nb;
        eigen_value_per_band.resize(nstates);
        Linalg::set_value_general(eigen_value_per_band.data(),
                                  eigen_values.data + local_vertices.bs,
                                  nstates);
        int_density_per_band.reserve(nstates);
        Array_4D<T> sub_eigen_vector = eigen_vector.sub_arr(Vertices_4D(region,
                                       local_vertices.bs, local_vertices.get_be()));
        uint Nd = region.get_size();
        for (uint i = 0; i < nstates; i++) {
            int_density_per_band.emplace_back(Linalg::vector_norm_square_sum(sub_eigen_vector.data + i * Nd, Nd));
        }
    } else {
        eigen_value_per_band.resize(0);
        int_density_per_band.resize(0);
    }
    return;
}

template<typename T>
Array_3D<T> Chefsi<T>::cal_electron_charge_density(const Smearing& smearing, const T& smearing_coef,
                                 const T& chemical_potential, const Vertices_3D& region,
                                 const Array_0D<T>& eigen_values, const Array_4D<T>& eigen_vector) const {
    Array_3D<T> electron_charge_density(region, 0);
    if (this->domain_vertices.is_active) {
        const Vertices_4D& local_vertices = this->domain_vertices.local_vertices;
        uint nstates = local_vertices.nb;
        Array_0D<T> occ(nstates);
        Smearing_method::smear(eigen_values.data + local_vertices.bs,
                               occ.data, chemical_potential, smearing, nstates);
        // if (unlikely(nstates > 0
        //           && local_vertices.get_be() == this->domain_vertices.shared_vertices.get_be()
        //           && occ[nstates-1] > 1e-10)) {
        //     std::cout << "WARNING:: occ[nstates-1] = " << occ[nstates-1]
        //               << " when chemical_potential = " << chemical_potential
        //               << " with nstates = " << this->domain_vertices.shared_vertices.get_nb()
        //               <<", which is bigger than 1e-10." << std::endl;
        // }
        occ *= smearing_coef;
        Array_4D<T> sub_eigen_vector = eigen_vector.sub_arr(Vertices_4D(region,
                                            local_vertices.bs, local_vertices.get_be()));
        uint Nd = region.get_size();
        for (uint i = 0; i < nstates; i++) {
            Linalg::accumulate_vector_norm_square(electron_charge_density.data, sub_eigen_vector.data + i * Nd, Nd, occ[i]);
        }
        if (this->domain_vertices.get_active_comm_nb() > 1) {
            // MPI_Allreduce(MPI_IN_PLACE, electron_charge_density.data, Nd,
            //               Linalg::get_mpi_datatype(electron_charge_density.data),
            //               MPI_SUM, this->domain_vertices.band_comm);
            if (this->domain_vertices.get_active_comm_b() == 0) {
                MPI_Reduce(MPI_IN_PLACE, electron_charge_density.data, Nd,
                           Linalg::get_mpi_datatype<T>(),
                           MPI_SUM, 0, this->domain_vertices.band_comm);
            } else {
                MPI_Reduce(electron_charge_density.data, electron_charge_density.data,
                           Nd, Linalg::get_mpi_datatype<T>(),
                           MPI_SUM, 0, this->domain_vertices.band_comm);
            }
        }
    }
    return electron_charge_density;
}

template<typename T>
T Chefsi<T>::cal_electron_charge(const Smearing& smearing, const T& smearing_coef,
                                 const T& chemical_potential, const Vertices_3D& region,
                                 const Array_0D<T>& eigen_values, const Array_4D<T>& eigen_vector) const {
    T electron_charge = 0.0;
    if (this->domain_vertices.is_active) {
        const Vertices_4D& local_vertices = this->domain_vertices.local_vertices;
        uint nstates = local_vertices.nb;
        Array_0D<T> occ(nstates);
        Smearing_method::smear(eigen_values.data + local_vertices.bs,
                               occ.data, chemical_potential, smearing, nstates);
        // occ *= smearing_coef;
        Array_4D<T> sub_eigen_vector = eigen_vector.sub_arr(Vertices_4D(region,
                                        local_vertices.bs, local_vertices.get_be()));
        uint Nd = region.get_size();
        for (uint i = 0; i < nstates; i++) {
            electron_charge += occ[i] * Linalg::vector_norm_square_sum(sub_eigen_vector.data + i * Nd, Nd);
        }
        electron_charge *= smearing_coef;
    }
    if (this->domain_vertices.get_comm_size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &electron_charge, 1, Linalg::get_mpi_datatype<T>(),
                      MPI_SUM, this->domain_vertices.comm);
    }
    return electron_charge;
}

template<typename T>
T Chefsi<T>::cal_electron_charge(const Smearing& smearing, const T& smearing_coef,
                                 const T& chemical_potential, const Array_0D<T>& eigen_values) const {
    T electron_charge = 0.0;
    if (this->domain_vertices.get_comm_rank() == 0) {
        uint nstates = this->domain_vertices.shared_vertices.nb;
        Array_0D<T> occ(nstates);
        Smearing_method::smear(eigen_values.data, occ.data, chemical_potential, smearing, nstates);
        occ *= smearing_coef;
        electron_charge = occ.vector_sum();
    }
    if (this->domain_vertices.get_comm_size() > 0) {
        MPI_Bcast(&electron_charge, 1, Linalg::get_mpi_datatype<T>(), 0, this->domain_vertices.comm);
    }
    return electron_charge;
}

template<typename T>
T Chefsi<T>::evaluate_chemical_potential(const Smearing& smearing, const Array_0D<T>& eigen_values, const T& electron_charge, const T& smearing_coef) const {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->chefsi_control.projection_method == 0) {
        T tol = 1e-12;
        if (Linalg::get_type_id<T>() == 1) tol = 1e-4;
        uint max_it = 100;
        T chemical_potential;
        if (this->dp_domain_vertices.get_comm_rank() == 0) {
            uint nstates = this->dp_domain_vertices.get_4D_local_vertices().get_nb();
            T lower_bound = eigen_values[0] - 1.0;
            T upper_bound = eigen_values[nstates-1] + 1.0;
            Array_0D<T> occ(nstates);
            T n = -1.0;
            uint count = 0;
            while (fabs(electron_charge - n) > tol) {
                chemical_potential = 0.5 * (lower_bound + upper_bound);
                Smearing_method::smear(eigen_values.data, occ.data, chemical_potential, smearing, nstates);
                occ *= smearing_coef;
                n = occ.vector_sum();
                if (n > electron_charge) {
                    upper_bound = chemical_potential;
                } else {
                    lower_bound = chemical_potential;
                }
                count++;
                if (count == max_it) {
                    std::cout << "WARNING:: CANNOT FIND chemical_potential AFTER " << count << " iteration." << std::endl;
                    std::cout << "WARNING:: WTIH chemical_potential = " << chemical_potential
                              << ", n = " << n
                              << ", electron_charge = " << electron_charge
                              << "." << std::endl;
                    break;
                }
            }
        }
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        MPI_Bcast(&chemical_potential, 1, mpi_datatype, 0, this->dp_domain_vertices.comm);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (this->domain_vertices.get_comm_rank() == 0) {
            std::cout << "The evaluate_chemical_potential took " << Tools::time_cost(begin, end) << "." << std::endl;
        }
        return chemical_potential;
    } else {
        assert(this->chefsi_control.projection_method == 0);
        return (T) 0.0;
    }
}

template<typename T>
T Chefsi<T>::evaluate_band_energy(const Smearing& smearing, const T& chemical_potential, const T& smearing_coef, const Vertices_3D& region,
                                  const Array_0D<T>& eigen_values, const Array_4D<T>& eigen_vector) const {
    T band_energy = 0.0;
    if (this->domain_vertices.is_active) {
        const Vertices_4D& local_vertices = this->domain_vertices.local_vertices;
        uint nstates = local_vertices.nb;
        Array_0D<T> occ(nstates);
        Smearing_method::smear(eigen_values.data + local_vertices.bs,
                               occ.data, chemical_potential, smearing, nstates);
        // occ *= smearing_coef;
        Array_4D<T> sub_eigen_vector = eigen_vector.sub_arr(Vertices_4D(region,
                                        local_vertices.bs, local_vertices.get_be()));
        uint Nd = region.get_size();
        for (uint i = 0; i < nstates; i++) {
            band_energy += occ[i] * Linalg::vector_norm_square_sum(sub_eigen_vector.data + i * Nd, Nd)
                         * (eigen_values.data + local_vertices.bs)[i];
        }
        band_energy *= smearing_coef;
    }
    if (this->domain_vertices.get_comm_size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &band_energy, 1, Linalg::get_mpi_datatype<T>(),
                      MPI_SUM, this->domain_vertices.comm);
    }
    return band_energy;                                
}

template<typename T>
T Chefsi<T>::evaluate_entropy_energy(const Smearing& smearing, const T& chemical_potential, const T& smearing_coef, const Vertices_3D& region,
                                     const Array_0D<T>& eigen_values, const Array_4D<T>& eigen_vector) const {
    T entropy_energy = 0.0;
    if (this->domain_vertices.is_active) {
        const Vertices_4D& local_vertices = this->domain_vertices.local_vertices;
        uint nstates = local_vertices.nb;
        Array_0D<T> occ(nstates);
        Smearing_method::smear(eigen_values.data + local_vertices.bs,
                               occ.data, chemical_potential, smearing, nstates);
        // occ *= smearing_coef;
        Array_4D<T> sub_eigen_vector = eigen_vector.sub_arr(Vertices_4D(region,
                                        local_vertices.bs, local_vertices.get_be()));
        uint Nd = region.get_size();
        Array_0D<T> entropy_energy_arr(nstates);
        Smearing_method::generate_entropy_energy_arr(occ.data, entropy_energy_arr.data, smearing, smearing_coef, nstates);
        for (uint i = 0; i < nstates; i++) {
            entropy_energy += entropy_energy_arr[i] * Linalg::vector_norm_square_sum(sub_eigen_vector.data + i * Nd, Nd);
        }
    }
    if (this->domain_vertices.get_comm_size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &entropy_energy, 1, Linalg::get_mpi_datatype<T>(),
                      MPI_SUM, this->domain_vertices.comm);
    }
    return entropy_energy;
}

template<typename T>
void Chefsi<T>::run_init(Array_4D<T>& h_eigen_vectors, Array_2D<T>& hp, Array_2D<T>& mp,
                         Array_4D<T>& eigen_vectors_reshape, const bool& print_flag) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    const Vertices_4D& local_vertices = this->domain_vertices.get_4D_local_vertices();
    const Vertices_4D& shared_vertices = this->domain_vertices.get_4D_shared_vertices();
    if (this->chefsi_control.projection_method == 0
     || this->chefsi_control.projection_method == 2) {
        if (local_vertices.get_size() > 0) {
            h_eigen_vectors.reconstructor(local_vertices);
        }
        if (this->dp_domain_vertices.local_vertices.Vertices_3D::get_size() > 0) {
            uint nstates = this->dp_domain_vertices.shared_vertices.get_nb();
            hp.reconstructor(nstates, nstates);
            mp.reconstructor(nstates, nstates);
            eigen_vectors_reshape.reconstructor(this->dp_domain_vertices.local_vertices);
        }
    } else if (this->chefsi_control.projection_method == 1) {
        #if (defined(USE_MKL) || defined(USE_SCALAPACK))
        uint nstates = shared_vertices.get_nb();
        uint local_nstates = local_vertices.get_nb();
        if (local_vertices.get_size() > 0) {
            h_eigen_vectors.reconstructor(local_vertices);
        }
        if (this->icontxt_intra_band != -1) {
            hp.reconstructor(nstates, local_nstates);
            mp.reconstructor(nstates, local_nstates);
        }
        // eigen_vectors_reshape is not needed
        #endif
    } else {
        assert(this->chefsi_control.projection_method == 0
            || this->chefsi_control.projection_method == 1
            || this->chefsi_control.projection_method == 2);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
        std::cout << "The chefsi allocation of temps took " << Tools::time_cost(begin, end) << "." << std::endl;
        std::cout << "Eigenvectors: nband = " << local_vertices.get_nb()
                  << " in " << shared_vertices.get_nb()
                  << ", grids = "  << local_vertices.get_size()
                  << " in " << shared_vertices.get_size()
                  << ", ([" << local_vertices.get_ni()
                  << ", " << local_vertices.get_nj()
                  << ", " << local_vertices.get_nk()
                  << "] in [" << shared_vertices.get_ni()
                  << ", " << shared_vertices.get_nj()
                  << ", " << shared_vertices.get_nk()
                  << "])."<< std::endl;
    }
    return;
}

template<typename T>
void Chefsi<T>::run_finalize(Array_4D<T>& h_eigen_vectors, Array_2D<T>& hp, Array_2D<T>& mp,
                            Array_4D<T>& eigen_vectors_reshape) {
    if (this->chefsi_control.projection_method == 0
     || this->chefsi_control.projection_method == 2) {
        h_eigen_vectors.destructor();
        hp.destructor();
        mp.destructor();
        eigen_vectors_reshape.destructor();
    } else if (this->chefsi_control.projection_method == 1) {
        #if (defined(USE_MKL) || defined(USE_SCALAPACK))
        h_eigen_vectors.destructor();
        hp.destructor();
        mp.destructor();
        // eigen_vectors_reshape is not needed
        #endif
    } else {
        assert(this->chefsi_control.projection_method == 0
            || this->chefsi_control.projection_method == 1
            || this->chefsi_control.projection_method == 2);
    }
    return;
}

template<typename T>
void Chefsi<T>::cal_nonlocal_forces(Array_2D<T>& nonlocal_forces, const T& chemical_potential, const Smearing& smearing,
                             const Spin& spin, const Effective_potential_nloc<T>& effective_potential_nloc,
                             const Array_0D<T> eigen_values, const Array_4D<T> eigen_vectors) const {
    this->cal_nonlocal_forces1(nonlocal_forces, chemical_potential, smearing,
                               spin, effective_potential_nloc,
                               eigen_values, eigen_vectors);
    // this->cal_nonlocal_forces2(nonlocal_forces, chemical_potential, smearing,
                            //    spin, effective_potential_nloc,
                            //    eigen_values, eigen_vectors);
    return;
}

template<typename T>
inline void Chefsi<T>::cal_nonlocal_forces1(Array_2D<T>& nonlocal_forces, const T& chemical_potential, const Smearing& smearing,
                             const Spin& spin, const Effective_potential_nloc<T>& effective_potential_nloc,
                             const Array_0D<T> eigen_values, const Array_4D<T> eigen_vectors) const {
    const Vertices_4D& local_vertices = eigen_vectors.get_vertices();
    const uint& nb = local_vertices.nb;
    Array_0D<T> alpha(nb * *(effective_potential_nloc.offsets.end()-1) * 4, 0);
    T* const int_chi_psi = alpha.data;
    T* const int_chi_Dpsi_x = int_chi_psi + nb * *(effective_potential_nloc.offsets.end()-1);
    T* const int_chi_Dpsi_y = int_chi_Dpsi_x + nb * *(effective_potential_nloc.offsets.end()-1);
    T* const int_chi_Dpsi_z = int_chi_Dpsi_y + nb * *(effective_potential_nloc.offsets.end()-1);
    //\int chi psi
    std::vector<uint>::const_iterator it_offset = effective_potential_nloc.offsets.cbegin();
    for (typename std::vector<Nloc_projector<T>>::const_iterator it_nloc_projector = effective_potential_nloc.nloc_projectors.cbegin();
        it_nloc_projector != effective_potential_nloc.nloc_projectors.cend(); ++it_nloc_projector) {
        Nloc_projector_method::nloc_projector_product_vectors(int_chi_psi + (*it_offset) * nb, eigen_vectors.data,
                                                                local_vertices, *it_nloc_projector, (T)this->mesh_control.delta_V);
        ++it_offset;
    }
    //\int chi Dpsi
    Vertices_4D ex_local_vertices(local_vertices.Vertices_3D::get_vertices().generate_ex_vertices(this->stencil.FDn), local_vertices.bs, local_vertices.get_be());
    Array_4D<T> ex_eigen_vectors(ex_local_vertices, 0);
    Array_4D<T> Deigen_vectors(local_vertices);
    // Vertices_3D ex_local_vertices_3d = ex_local_vertices.Vertices_3D::get_vertices();
    // Vertices_3D local_vertices_3d = local_vertices.Vertices_3D::get_vertices();
    // uint ex_length_3d = ex_local_vertices_3d.get_size();
    // uint length_3d = local_vertices_3d.get_size();
    this->exarr_mpi_package.fill_domain_par_ex_arr(eigen_vectors.data,
                                                local_vertices, ex_eigen_vectors.data, ex_local_vertices);
    for (uint idir = 0; idir < 3; idir++) {
        // for (uint b = 0; b < nb; b++) {
        //     Stencil_method::calc_gradient(ex_eigen_vectors.data + ex_length_3d * b, ex_local_vertices_3d, idir, this->stencil,
        //                                 local_vertices_3d, Deigen_vectors.data + length_3d * b, local_vertices_3d);
        // }
        Stencil_method::calc_gradient_d4(ex_eigen_vectors.data, ex_local_vertices, idir, this->stencil,
                                            local_vertices, Deigen_vectors.data, local_vertices);
        T* const int_chi_Dpsi_idir = alpha.data + (idir + 1) * nb * *(effective_potential_nloc.offsets.end()-1);
        std::vector<uint>::const_iterator it_offset = effective_potential_nloc.offsets.cbegin();
        for (typename std::vector<Nloc_projector<T>>::const_iterator it_nloc_projector = effective_potential_nloc.nloc_projectors.cbegin();
            it_nloc_projector != effective_potential_nloc.nloc_projectors.cend(); ++it_nloc_projector) {
            // Nloc_projector_method::nloc_projector_product_vectors(int_chi_Dpsi_idir + (*it_offset) * nb, Deigen_vectors.data,
            //                                                       local_vertices, *it_nloc_projector, (T)this->mesh_control.delta_V);
            Nloc_projector_method::nloc_projector_product_vectors(int_chi_Dpsi_idir + (*it_offset) * nb, Deigen_vectors.data,
                                                                    local_vertices, *it_nloc_projector, (T)1.0);
            ++it_offset;
        }
    }
    if (this->exarr_mpi_package.need_comm) {
        MPI_Allreduce(MPI_IN_PLACE, alpha.data,
                      nb * *(effective_potential_nloc.offsets.end()-1) * 4,
                      Linalg::get_mpi_datatype<T>(), MPI_SUM,
                      this->exarr_mpi_package.comm);
    }
    // nonlocal_forces
    // Array_2D<T> nonlocal_forces(forces.get_vertices(), 0);
    Array_0D<T> coef(nb);
    Smearing_method::smear(eigen_values.data + local_vertices.bs,
                            coef.data, chemical_potential, smearing, nb);
    Linalg::scalar_product_general(coef.data, (T)(spin.generate_smearing_coef() * 2.0), nb);
    it_offset = effective_potential_nloc.offsets.cbegin();
    for (typename std::vector<Nloc_projector<T>>::const_iterator it_nloc_projector = effective_potential_nloc.nloc_projectors.cbegin();
        it_nloc_projector != effective_potential_nloc.nloc_projectors.cend(); ++it_nloc_projector) {
        if (!(it_nloc_projector->is_real && it_nloc_projector->is_in_domain)) {
            ++it_offset;
            continue;
        }
        const Array_0D<T>& gamma = it_nloc_projector->gamma;
        const uint& ncol = it_nloc_projector->ncol;
        T* const __restrict__ atom_int_chi_psi = int_chi_psi + (*it_offset) * nb;
        T* const __restrict__ atom_int_chi_Dpsi_x = int_chi_Dpsi_x + (*it_offset) * nb;
        T* const __restrict__ atom_int_chi_Dpsi_y = int_chi_Dpsi_y + (*it_offset) * nb;
        T* const __restrict__ atom_int_chi_Dpsi_z = int_chi_Dpsi_z + (*it_offset) * nb;
        T* const __restrict__ atom_nonlocal_forces = nonlocal_forces.data + 3 * it_nloc_projector->atom_index;
        uint count = 0;
        for (uint b = 0; b < nb; b++) {
            for (uint icol = 0; icol < ncol; icol++) {
                atom_nonlocal_forces[0] -= atom_int_chi_psi[count] * atom_int_chi_Dpsi_x[count] * coef[b] * gamma[icol];
                atom_nonlocal_forces[1] -= atom_int_chi_psi[count] * atom_int_chi_Dpsi_y[count] * coef[b] * gamma[icol];
                atom_nonlocal_forces[2] -= atom_int_chi_psi[count] * atom_int_chi_Dpsi_z[count] * coef[b] * gamma[icol];
                ++count;
            }
        }
        ++it_offset;
    }
    const MPI_Comm& domain_3d_comm = this->domain_vertices.get_domain_3d_comm();
    int domain_3d_comm_size;
    MPI_Comm_size(domain_3d_comm, &domain_3d_comm_size);
    if (domain_3d_comm_size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, nonlocal_forces.data, nonlocal_forces.length,
                                Linalg::get_mpi_datatype<T>(), MPI_SUM,
                                domain_3d_comm);
    }
    int bandcomm_size;
    MPI_Comm_size(this->domain_vertices.band_comm, &bandcomm_size);
    if (bandcomm_size > 0) {
        MPI_Allreduce(MPI_IN_PLACE, nonlocal_forces.data, nonlocal_forces.length,
                        Linalg::get_mpi_datatype<T>(), MPI_SUM,
                        this->domain_vertices.band_comm);
    }
    // if (this->domain_vertices.get_comm_rank() == 0) {
    //     std::cout << "nonlocal_forces = " << std::endl;
    //     nonlocal_forces.print();
    // }
    // forces += nonlocal_forces;
    return;
}

template<typename T>
inline void Chefsi<T>::cal_nonlocal_forces2(Array_2D<T>& nonlocal_forces, const T& chemical_potential, const Smearing& smearing,
                             const Spin& spin, const Effective_potential_nloc<T>& effective_potential_nloc,
                             const Array_0D<T> eigen_values, const Array_4D<T> eigen_vectors) const {
    const Vertices_4D& local_vertices = eigen_vectors.get_vertices();
    const uint nb = local_vertices.nb;
    const uint n_chi_reduce = *(effective_potential_nloc.offsets.end()-1);
    const Vertices_3D local_vertices_3d = local_vertices.Vertices_3D::get_vertices();
    const uint nd = local_vertices_3d.get_size();

    Array_0D<T> chis(nd * n_chi_reduce, 0);
    const Vertices_3D ex_local_vertices_3d(local_vertices_3d.generate_ex_vertices(this->stencil.FDn));
    const uint ex_nd = ex_local_vertices_3d.get_size();
    Array_0D<T> ex_chis(ex_nd * n_chi_reduce, 0);
    for (uint i_nloc_projector = 0; i_nloc_projector < effective_potential_nloc.nloc_projectors.size(); i_nloc_projector++) {
        const Nloc_projector<T>& nloc_projector = effective_potential_nloc.nloc_projectors[i_nloc_projector];
        const uint ncol = nloc_projector.ncol;
        const uint nrow = nloc_projector.nrow;
        uint const* const __restrict__ index = nloc_projector.index_data();
        for (uint icol = 0; icol < ncol; icol++) {
            T const* const __restrict__ chi = nloc_projector.chi.data + icol * nrow;
            T* const chis_data_icol = chis.data + (effective_potential_nloc.offsets[i_nloc_projector] + icol) * nd;
            for (uint irol = 0; irol < nrow; irol++) {
                chis_data_icol[index[irol]] += chi[irol];
            }
        }
    }
    for (uint i_chi_reduce = 0; i_chi_reduce < n_chi_reduce; i_chi_reduce++) {
        this->single_band_exarr_mpi_package.fill_domain_par_ex_arr(chis.data + nd * i_chi_reduce,
                                                local_vertices_3d, ex_chis.data + ex_nd * i_chi_reduce, ex_local_vertices_3d);
    }
    Array_0D<T> alpha(nb * n_chi_reduce * 4, 0);
    T* const int_chi_psi = alpha.data;
    T* const int_Dchi_psi_x = int_chi_psi + nb * n_chi_reduce;
    T* const int_Dchi_psi_y = int_Dchi_psi_x + nb * n_chi_reduce;
    T* const int_Dchi_psi_z = int_Dchi_psi_y + nb * n_chi_reduce;
    Vertices_4D local_vertices_3d_nchi = Vertices_4D(local_vertices_3d, n_chi_reduce);
    Array_4D<T> Dchi(local_vertices_3d_nchi);
    //\int chi psi
    int atom_index = effective_potential_nloc.nloc_projectors[0].atom_index + 1;
    for (uint i_nloc_projector = 0; i_nloc_projector < effective_potential_nloc.nloc_projectors.size(); i_nloc_projector++) {
        const Nloc_projector<T>& nloc_projector = effective_potential_nloc.nloc_projectors[i_nloc_projector];
        if (nloc_projector.atom_index == atom_index) continue;
        atom_index = nloc_projector.atom_index;
        const uint offset = effective_potential_nloc.offsets[i_nloc_projector];
        Linalg::matrix_product(chis.data + offset * nd, 0, eigen_vectors.data, 1, alpha.data + offset * nb, 1, nloc_projector.ncol, nb, nd, T(this->mesh_control.delta_V));
    }

    //\int Dchi psi
    for (uint idir = 0; idir < 3; idir++) {
        Stencil_method::calc_gradient_d4(ex_chis.data, Vertices_4D(ex_local_vertices_3d, n_chi_reduce), idir, this->stencil,
                                            local_vertices_3d_nchi, Dchi.data, local_vertices_3d_nchi);
        T* const int_Dchi_psi_idir = alpha.data + (idir + 1) * nb * n_chi_reduce;
        int atom_index = effective_potential_nloc.nloc_projectors[0].atom_index + 1;
        for (uint i_nloc_projector = 0; i_nloc_projector < effective_potential_nloc.nloc_projectors.size(); i_nloc_projector++) {
            const Nloc_projector<T>& nloc_projector = effective_potential_nloc.nloc_projectors[i_nloc_projector];
            if (nloc_projector.atom_index == atom_index) continue;
            atom_index = nloc_projector.atom_index;
            const uint offset = effective_potential_nloc.offsets[i_nloc_projector];
            Linalg::matrix_product(Dchi.data + offset * nd, 0, eigen_vectors.data, 1, int_Dchi_psi_idir + offset * nb, 1, nloc_projector.ncol, nb, nd);
        }
    }

    if (this->exarr_mpi_package.need_comm) {
        MPI_Allreduce(MPI_IN_PLACE, alpha.data,
                      nb * *(effective_potential_nloc.offsets.end()-1) * 4,
                      Linalg::get_mpi_datatype<T>(), MPI_SUM,
                      this->exarr_mpi_package.comm);
    }
    // nonlocal_forces
    // Array_2D<T> nonlocal_forces(forces.get_vertices(), 0);
    Array_0D<T> coef(nb);
    Smearing_method::smear(eigen_values.data + local_vertices.bs,
                            coef.data, chemical_potential, smearing, nb);
    Linalg::scalar_product_general(coef.data, (T)(spin.generate_smearing_coef() * 2.0), nb);
    std::vector<uint>::const_iterator it_offset = effective_potential_nloc.offsets.cbegin();
    for (typename std::vector<Nloc_projector<T>>::const_iterator it_nloc_projector = effective_potential_nloc.nloc_projectors.cbegin();
        it_nloc_projector != effective_potential_nloc.nloc_projectors.cend(); ++it_nloc_projector) {
        if (!(it_nloc_projector->is_real && it_nloc_projector->is_in_domain)) {
            ++it_offset;
            continue;
        }
        const Array_0D<T>& gamma = it_nloc_projector->gamma;
        const uint& ncol = it_nloc_projector->ncol;
        T* const __restrict__ atom_int_chi_psi = int_chi_psi + (*it_offset) * nb;
        T* const __restrict__ atom_int_chi_Dpsi_x = int_Dchi_psi_x + (*it_offset) * nb;
        T* const __restrict__ atom_int_chi_Dpsi_y = int_Dchi_psi_y + (*it_offset) * nb;
        T* const __restrict__ atom_int_chi_Dpsi_z = int_Dchi_psi_z + (*it_offset) * nb;
        T* const __restrict__ atom_nonlocal_forces = nonlocal_forces.data + 3 * it_nloc_projector->atom_index;
        uint count = 0;
        for (uint b = 0; b < nb; b++) {
            for (uint icol = 0; icol < ncol; icol++) {
                atom_nonlocal_forces[0] += atom_int_chi_psi[count] * atom_int_chi_Dpsi_x[count] * coef[b] * gamma[icol];
                atom_nonlocal_forces[1] += atom_int_chi_psi[count] * atom_int_chi_Dpsi_y[count] * coef[b] * gamma[icol];
                atom_nonlocal_forces[2] += atom_int_chi_psi[count] * atom_int_chi_Dpsi_z[count] * coef[b] * gamma[icol];
                ++count;
            }
        }
        ++it_offset;
    }
    const MPI_Comm& domain_3d_comm = this->domain_vertices.get_domain_3d_comm();
    int domain_3d_comm_size;
    MPI_Comm_size(domain_3d_comm, &domain_3d_comm_size);
    if (domain_3d_comm_size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, nonlocal_forces.data, nonlocal_forces.length,
                                Linalg::get_mpi_datatype<T>(), MPI_SUM,
                                domain_3d_comm);
    }
    int bandcomm_size;
    MPI_Comm_size(this->domain_vertices.band_comm, &bandcomm_size);
    if (bandcomm_size > 0) {
        MPI_Allreduce(MPI_IN_PLACE, nonlocal_forces.data, nonlocal_forces.length,
                        Linalg::get_mpi_datatype<T>(), MPI_SUM,
                        this->domain_vertices.band_comm);
    }
    // if (this->domain_vertices.get_comm_rank() == 0) {
    //     std::cout << "nonlocal_forces = " << std::endl;
    //     nonlocal_forces.print();
    // }
    // forces += nonlocal_forces;
    return;
}

template<typename T>
void Chefsi<T>::run(Array_4D<T>& eigen_vectors, Array_0D<T>& eigen_values, const Array_3D<T>& Vloc,
                    const Effective_potential_nloc<T>& Vnloc, const bool& print_flag) {
    if (this->domain_vertices.get_4D_shared_vertices().get_size() == 0) return;

    #ifdef USE_OPENMP

    #ifdef ENABLE_CHEFSI_TIMER
        #pragma omp master
        {
            this->chefsi_timer.reset();
            this->chefsi_timer.chefsi.start();
        }
    #endif //ENABLE_CHEFSI_TIMER

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    //init 
    Array_4D<T>& h_eigen_vectors = this->h_eigen_vectors;
    Array_2D<T>& hp = this->hp;
    Array_2D<T>& mp = this->mp;
    const Array_2D<T>& qp = hp;
    Array_4D<T>& eigen_vectors_reshape = this->eigen_vectors_reshape;
    #pragma omp single nowait
    this->run_init(h_eigen_vectors, hp, mp, eigen_vectors_reshape, print_flag);
    //run
    for (uint iter = 0; iter < (unlikely(this->is_very_first) ? this->chefsi_control.rho_trigger : this->chefsi_control.max_iter); iter++) {
        #pragma omp barrier
        this->lanczos.run(this->domain_vertices.comm, Vloc, Vnloc, this->stencil,
                      (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package,
                      this->is_very_first && iter == 0, eigen_values[eigen_values.length - 1], eigen_values.data[0],
                      print_flag);
        // this->chebyshev_filtering(eigen_vectors, Vloc, Vnloc);
        // this->chebyshev_filtering(eigen_vectors.data, Vloc.data, Vnloc);
        // this->chebyshev_filtering_column_wise(eigen_vectors.data, Vloc.data, Vnloc);
        #pragma omp parallel
        {
        #pragma omp barrier
        this->chebyshev_filtering_column_wise2(eigen_vectors.data, Vloc.data, Vnloc, print_flag);
        // Hamiltonian::hamiltonian_product_vectors(h_eigen_vectors, eigen_vectors, Vloc, Vnloc, this->stencil,
                                                // (T)this->mesh_control.delta_V, this->exarr_mpi_package);
        // Hamiltonian::hamiltonian_product_vectors(h_eigen_vectors.data, this->domain_vertices.get_4D_local_vertices(),
        //                                          eigen_vectors.data, Vloc.data, Vnloc, this->stencil,
        //                                          (T)this->mesh_control.delta_V, this->exarr_mpi_package);
        // Hamiltonian::hamiltonian_product_vectors_column_wise(h_eigen_vectors.data, this->domain_vertices.get_4D_local_vertices(),
        //                                                      eigen_vectors.data, Vloc.data, Vnloc, this->stencil,
        //                                                      (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package);
        #ifdef ENABLE_CHEFSI_TIMER
            #pragma omp master
            {
                this->chefsi_timer.H_psi.start();
            }
        #endif //ENABLE_CHEFSI_TIMER
        #pragma omp barrier
        Hamiltonian::hamiltonian_product_vectors_column_wise2(h_eigen_vectors.data, this->domain_vertices.get_4D_local_vertices(),
                                                              eigen_vectors.data, Vloc.data, Vnloc, this->stencil,
                                                              (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package,
                                                              print_flag);
        #ifdef ENABLE_CHEFSI_TIMER
            #pragma omp master
            {
                this->chefsi_timer.H_psi.stop();
            }
        #endif //ENABLE_CHEFSI_TIMER
        }
        // this->project_hamiltonian(eigen_vectors, h_eigen_vectors, eigen_vectors_reshape, hp, mp);
        this->project_hamiltonian(eigen_vectors.data, h_eigen_vectors.data, eigen_vectors_reshape.data, hp.data, mp.data, print_flag);
        // this->subspace_diagonalization(hp, mp, eigen_values);
        this->subspace_diagonalization(hp.data, mp.data, eigen_values.data, print_flag);
        // this->subspace_rotation(eigen_vectors, eigen_vectors_reshape, qp);
        this->subspace_rotation(eigen_vectors.data, eigen_vectors_reshape.data, qp.data, print_flag);
    }
    #pragma omp barrier
    #pragma omp single nowait
    if (unlikely(this->is_very_first)) {this->is_very_first = false;}
    #pragma omp single nowait
    this->run_finalize(h_eigen_vectors, hp, mp, eigen_vectors_reshape);
    #pragma omp single nowait
    {
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
        std::cout << "The Chefsi run took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    }

    #ifdef ENABLE_CHEFSI_TIMER
        #pragma omp master
        {
            this->chefsi_timer.chefsi.stop();
            if (this->domain_vertices.get_comm_rank() == 0 && print_flag) this->chefsi_timer.show();
        }
    #endif //ENABLE_CHEFSI_TIMER

    #else //USE_OPENMP

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    //init 
    Array_4D<T> h_eigen_vectors;
    Array_2D<T> hp;
    Array_2D<T> mp;
    const Array_2D<T>& qp = hp;
    Array_4D<T> eigen_vectors_reshape;
    this->run_init(h_eigen_vectors, hp, mp, eigen_vectors_reshape, print_flag);
    //run
    for (uint iter = 0; iter < (unlikely(this->is_very_first) ? this->chefsi_control.rho_trigger : this->chefsi_control.max_iter); iter++) {
        this->lanczos.run(this->domain_vertices.comm, Vloc, Vnloc, this->stencil,
                      (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package,
                      this->is_very_first && iter == 0, eigen_values[eigen_values.length - 1], eigen_values.data[0],
                      print_flag);
        // this->chebyshev_filtering(eigen_vectors, Vloc, Vnloc);
        // this->chebyshev_filtering(eigen_vectors.data, Vloc.data, Vnloc);
        // this->chebyshev_filtering_column_wise(eigen_vectors.data, Vloc.data, Vnloc);
        this->chebyshev_filtering_column_wise2(eigen_vectors.data, Vloc.data, Vnloc, print_flag);
        // Hamiltonian::hamiltonian_product_vectors(h_eigen_vectors, eigen_vectors, Vloc, Vnloc, this->stencil,
                                                // (T)this->mesh_control.delta_V, this->exarr_mpi_package);
        // Hamiltonian::hamiltonian_product_vectors(h_eigen_vectors.data, this->domain_vertices.get_4D_local_vertices(),
        //                                          eigen_vectors.data, Vloc.data, Vnloc, this->stencil,
        //                                          (T)this->mesh_control.delta_V, this->exarr_mpi_package);
        // Hamiltonian::hamiltonian_product_vectors_column_wise(h_eigen_vectors.data, this->domain_vertices.get_4D_local_vertices(),
        //                                                      eigen_vectors.data, Vloc.data, Vnloc, this->stencil,
        //                                                      (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package);
        Hamiltonian::hamiltonian_product_vectors_column_wise2(h_eigen_vectors.data, this->domain_vertices.get_4D_local_vertices(),
                                                              eigen_vectors.data, Vloc.data, Vnloc, this->stencil,
                                                              (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package,
                                                              print_flag);
        // this->project_hamiltonian(eigen_vectors, h_eigen_vectors, eigen_vectors_reshape, hp, mp);
        this->project_hamiltonian(eigen_vectors.data, h_eigen_vectors.data, eigen_vectors_reshape.data, hp.data, mp.data, print_flag);
        // this->subspace_diagonalization(hp, mp, eigen_values);
        this->subspace_diagonalization(hp.data, mp.data, eigen_values.data, print_flag);
        // this->subspace_rotation(eigen_vectors, eigen_vectors_reshape, qp);
        this->subspace_rotation(eigen_vectors.data, eigen_vectors_reshape.data, qp.data, print_flag);
    }
    if (unlikely(this->is_very_first)) {this->is_very_first = false;}
    this->run_finalize(h_eigen_vectors, hp, mp, eigen_vectors_reshape);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
        std::cout << "The Chefsi run took " << Tools::time_cost(begin, end) << "." << std::endl;
    }

    #endif //USE_OPENMP
    return;
}

template<typename T>
double Chefsi<T>::evalutate_flops() {
    double flops = 0.0;
    const uint64_t ns = this->domain_vertices.shared_vertices.get_nb();
    const uint64_t nd = this->domain_vertices.shared_vertices.Vertices_3D::get_vertices().get_size();
    // filter
    // filter local
    flops += (6.0 * this->stencil.FDn + 2.0) * double(ns) * double(nd) * double(this->chefsi_control.chebyshev_filter_degree);
    // filter nonlocal
    flops += 0.0 * double(this->chefsi_control.chebyshev_filter_degree);
    // Hv
    flops += (6.0 * this->stencil.FDn + 2.0) * double(ns) * double(nd);
    flops += 0.0;
    // projection
    flops += double(nd) * double(ns * ns) * 2.0 + double(nd) * double(ns * ns) * 1.0;
    // diagonalization
    flops += double(ns * ns * ns) * 7.0;
    // rotation
    flops += double(nd) * double(ns * ns) * 2.0;
    return flops;
}

template<typename T>
void Chefsi<T>::init(const bool* is_periodic, const bool& is_rand_fixed) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->domain_vertices.shared_vertices.get_size() == 0) {
        if (this->chefsi_control.projection_method == 0) {                     // 0: DP_SUBEIG
        // do nothing
        } else if (this->chefsi_control.projection_method == 1) {
            #if (defined(USE_MKL) || defined(USE_SCALAPACK))
            Cblacs_get(0, 0, &(this->icontxt_whole_comm));
            Cblacs_get(0, 0, &(this->icontxt_intra_band));
            Cblacs_get(0, 0, &(this->icontxt_domain_3D));
            Cblacs_get(0, 0, &(this->icontxt_2d_reshape));
            Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_whole_comm), 1, 1, 1, false,
                                                    this->domain_vertices.get_mpi_comm());
            Cblacs_gridexit(this->icontxt_whole_comm);
            this->icontxt_whole_comm = -1;
            Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_intra_band), 1, 1, 1, false,
                                                    this->domain_vertices.band_comm);
            Cblacs_gridexit(this->icontxt_intra_band);
            this->icontxt_intra_band = -1;
            Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_domain_3D), 1, 1, 1, false,
                                                    this->domain_vertices.get_domain_4d_comm());
            Cblacs_gridexit(this->icontxt_domain_3D);
            this->icontxt_domain_3D = -1;
            Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_2d_reshape), 1, 1, 1, false,
                                                    this->domain_vertices.get_mpi_comm());
            Cblacs_gridexit(this->icontxt_2d_reshape);
            this->icontxt_2d_reshape = -1;
            #endif
        } else if (this->chefsi_control.projection_method == 2) {
            #if (defined(USE_MKL) || defined(USE_SCALAPACK))
            // Cblacs_get(0, 0, &(this->icontxt_2d_reshape));
            // Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_2d_reshape), 1, 1, 1, false,
            //                                         this->domain_vertices.get_mpi_comm());
            // Cblacs_gridexit(this->icontxt_2d_reshape);
            this->icontxt_2d_reshape = -1;
            #endif
        } else {
            assert(this->chefsi_control.projection_method == 0
                || this->chefsi_control.projection_method == 1
                || this->chefsi_control.projection_method == 2);
        }
    } else {
        if (this->chefsi_control.projection_method == 0) {                     // 0: DP_SUBEIG
            Vertices_4D shared_vertices(this->domain_vertices.get_4D_shared_vertices());
            this->dp_domain_vertices.set_chunk_size_b(shared_vertices.nb);
            uint np_max = 0;
            this->dp_domain_vertices.init(shared_vertices, this->domain_vertices.get_mpi_comm(), np_max);
            this->dp_mpi_package.init();
        } else if (this->chefsi_control.projection_method == 1) {
            #if (defined(USE_MKL) || defined(USE_SCALAPACK))
            assert(this->domain_vertices.is_Col_Maj);

            uint comm_nb = this->domain_vertices.get_active_comm_nb();
            uint comm_ninjnk = this->domain_vertices.get_active_comm_ni()
                            * this->domain_vertices.get_active_comm_nj()
                            * this->domain_vertices.get_active_comm_nk();
            Cblacs_get(0, 0, &(this->icontxt_whole_comm));
            Cblacs_get(0, 0, &(this->icontxt_intra_band));
            Cblacs_get(0, 0, &(this->icontxt_domain_3D));
            Cblacs_get(0, 0, &(this->icontxt_2d_reshape));
            int comm_size = this->domain_vertices.get_comm_size();
            Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_whole_comm),
                                                    comm_size, comm_size,
                                                    1, true,
                                                    this->domain_vertices.get_mpi_comm());
            Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_intra_band), 1, 1, comm_nb,
                                                    this->domain_vertices.is_active,
                                                    this->domain_vertices.band_comm);
            Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_domain_3D), comm_ninjnk,
                                                    comm_ninjnk, comm_nb, true,
                                                    this->domain_vertices.get_domain_4d_comm());
            // int size_d2 = std::floor(std::sqrt(comm_size));
            // Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_2d_reshape),
            //                                         size_d2, size_d2, size_d2, true,
            //                                         this->domain_vertices.get_mpi_comm());
            Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_2d_reshape),
                                                    1, 1, comm_nb, this->domain_vertices.is_active,
                                                    this->domain_vertices.band_comm);
            if (this->icontxt_intra_band != -1 && !this->domain_vertices.is_active) {
                Cblacs_gridexit(this->icontxt_intra_band);
                this->icontxt_intra_band = -1;
            }
            if (this->icontxt_2d_reshape != -1) {
                Cblacs_gridinfo(this->icontxt_2d_reshape, &(this->icontxt_2d_reshape_nprow), &(this->icontxt_2d_reshape_npcol),
                            &(this->icontxt_2d_reshape_myprow), &(this->icontxt_2d_reshape_mypcol));
            }

            if (this->icontxt_intra_band != -1 && this->domain_vertices.is_active) {
                // desc_intra_band_eigen_vectors
                int m = this->domain_vertices.get_3D_local_vertices().get_size();
                int n = this->domain_vertices.get_4D_shared_vertices().nb;
                int mb = m;
                int nb = this->domain_vertices.get_chunk_size_b();
                int irsrc = 0;
                int icsrc = 0;
                int lld = std::max(1,m);
                int info;
                descinit_(this->desc_intra_band_eigen_vectors,
                                &m, &n, &mb, &nb, &irsrc, &icsrc, &(this->icontxt_intra_band), &lld, &info);
            } else {
                this->desc_intra_band_eigen_vectors[1] = -1;
            }
            if (this->icontxt_intra_band != -1 && this->domain_vertices.is_active) {
                // desc_intra_band_hp_mp
                int m = this->domain_vertices.get_4D_shared_vertices().nb;
                int n = this->domain_vertices.get_4D_shared_vertices().nb;
                int mb = m;
                int nb = this->domain_vertices.get_chunk_size_b();
                int irsrc = 0;
                int icsrc = 0;
                int lld = std::max(1,m);
                int info;
                descinit_(this->desc_intra_band_hp_mp,
                                &m, &n, &mb, &nb, &irsrc, &icsrc, &(this->icontxt_intra_band), &lld, &info);
            } else {
                this->desc_intra_band_hp_mp[1] = -1;
            }
            if (this->icontxt_domain_3D != -1) {
                // desc_domain_3D_hp_mp
                int m = this->domain_vertices.get_4D_shared_vertices().nb;
                int n = this->domain_vertices.get_4D_shared_vertices().nb;
                int mb = m;
                int nb = this->domain_vertices.get_chunk_size_b();
                int irsrc = 0;
                int icsrc = 0;
                int lld = std::max(1,m);
                int info;
                descinit_(this->desc_domain_3D_hp_mp,
                                &m, &n, &mb, &nb, &irsrc, &icsrc, &(this->icontxt_domain_3D), &lld, &info);
            } else {
                this->desc_domain_3D_hp_mp[1] = -1;
            }
            if (this->icontxt_2d_reshape != -1) {
                // desc_2d_reshape_hp_mp
                int zero = 0;
                int m = this->domain_vertices.get_4D_shared_vertices().nb;
                int n = this->domain_vertices.get_4D_shared_vertices().nb;
                int mb = 128;
                int nb = 128;
                int irsrc = 0;
                int icsrc = 0;
                this->hp_mp_2d_reshape_m = numroc_(&m, &mb, &(this->icontxt_2d_reshape_myprow), &zero, &(this->icontxt_2d_reshape_nprow));
                this->hp_mp_2d_reshape_n = numroc_(&n, &nb, &(this->icontxt_2d_reshape_mypcol), &zero, &(this->icontxt_2d_reshape_npcol));
                int lld = std::max(1,this->hp_mp_2d_reshape_m);
                int info;
                descinit_(this->desc_2d_reshape_hp_mp,
                                &m, &n, &mb, &nb, &irsrc, &icsrc, &(this->icontxt_2d_reshape), &lld, &info);
            } else {
                this->desc_2d_reshape_hp_mp[1] = -1;
            }
            if (this->icontxt_2d_reshape != -1) {
                int ibtype = 1;
                char jobz = 'V';
                char range = 'A';
                char uplo = 'U';
                T abstol = (T)this->chefsi_control.abstol;
                T orfac = (T)this->chefsi_control.orfac;
                // T work[1];
                T* work = new (std::align_val_t(64)) T [1];
                int lwork = -1;
                // int iwork[1];
                int* iwork = new (std::align_val_t(64)) int [1];
                int liwork = -1;
                int nstates = this->domain_vertices.get_4D_shared_vertices().get_nb();
                // T a[1];
                T* a = new (std::align_val_t(64)) T [1];
                int one = 1;
                int temp;
                Linalg::p_sygvx_(&ibtype, &jobz, &range, &uplo, &nstates,
                                a, &one, &one, this->desc_2d_reshape_hp_mp,
                                a, &one, &one, this->desc_2d_reshape_hp_mp,
                                a, a, &one, &nstates, &abstol, &one, &one, a, &orfac,
                                a, &one, &one, this->desc_2d_reshape_hp_mp,
                                work, &lwork, iwork, &liwork, &temp, &temp, a, &temp);
                // lwork
                // this->lwork = (int) std::fabs(work[0]) * 5;
                //lwork ≥ 5*n + max(5*nn, np0*mq0 + 2*nb*nb) + iceil(neig, NPROW*NPCOL)*nn
                this->lwork = (int) std::fabs(work[0]);
                int NPROW = this->icontxt_2d_reshape_nprow;
                int NPCOL = this->icontxt_2d_reshape_npcol;
                int nb = this->desc_2d_reshape_hp_mp[4];
                int nn = std::max(std::max(nstates, nb),2);
                int zero = 0;
                int np0 = numroc_(&nstates, &nb, &zero, &zero, &NPROW);
                // int mq0  = numroc_(&nstates, &nb, &zero, &zero, &(this->icontxt_2d_reshape_npcol));
                int max_neig_nb_2 = std::max(std::max(nstates, nb),2);
                int mq0 = numroc_(&max_neig_nb_2, &nb, &zero, &zero, &NPCOL);
                int lwork_min = 5*nstates + std::max(5*nn, np0*mq0 + 2*nb*nb)
                              + std::ceil((nstates-1)/(NPROW*NPCOL)+1) * nn;
                this->lwork = std::max(this->lwork, lwork_min);
                //the line below is from sparc and i dont know why
                this->lwork += std::max(nstates*nstates, std::min(10*this->lwork,2000000));
                //liwork
                // this->liwork = iwork[0] * 5;
                //liwork ≥ 6*nnp
                //nnp = max(n, NPROW*NPCOL + 1, 4)
                this->liwork = iwork[0];
                int nnp = std::max(std::max(nstates, NPROW*NPCOL + 1),4);
                this->liwork = std::max(this->liwork, nnp);
                delete [] work;
                delete [] iwork;
                delete [] a;
            }
            #endif
        } else if (this->chefsi_control.projection_method == 2) {
            Vertices_4D shared_vertices(this->domain_vertices.get_4D_shared_vertices());
            this->dp_domain_vertices.set_chunk_size_b(shared_vertices.nb);
            uint np_max = 0;
            this->dp_domain_vertices.init(shared_vertices, this->domain_vertices.get_mpi_comm(), np_max);
            this->dp_mpi_package.init();
            #if (defined(USE_MKL) || defined(USE_SCALAPACK))
            Cblacs_get(0, 0, &(this->icontxt_2d_reshape));
            int comm_size = this->domain_vertices.get_comm_size();
            int size_d2 = std::floor(std::sqrt(comm_size));
            // if (size_d2 > 6) size_d2 = 6;
            // Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_2d_reshape),
            //                                         size_d2, size_d2, size_d2, true,
            //                                         this->domain_vertices.get_mpi_comm());
            this->icontxt_2d_reshape = Csys2blacs_handle(this->domain_vertices.get_mpi_comm());
            // char order[] = "Col";
            char order[] = "Row";
            Cblacs_gridinit(&(this->icontxt_2d_reshape), order, size_d2, size_d2);
            if (this->icontxt_2d_reshape != -1) {
                Cblacs_gridinfo(this->icontxt_2d_reshape, &(this->icontxt_2d_reshape_nprow), &(this->icontxt_2d_reshape_npcol),
                            &(this->icontxt_2d_reshape_myprow), &(this->icontxt_2d_reshape_mypcol));
            }
            if (this->icontxt_2d_reshape != -1) {
                // desc_domain_3D_hp_mp
                int m = this->domain_vertices.get_4D_shared_vertices().nb;
                int n = this->domain_vertices.get_4D_shared_vertices().nb;
                int mb = m;
                int nb = n;
                int irsrc = 0;
                int icsrc = 0;
                int lld = std::max(1,m);
                int info;
                descinit_(this->desc_domain_3D_hp_mp,
                                &m, &n, &mb, &nb, &irsrc, &icsrc, &(this->icontxt_2d_reshape), &lld, &info);
            } else {
                this->desc_domain_3D_hp_mp[1] = -1;
            }
            if (this->icontxt_2d_reshape != -1) {
                // desc_2d_reshape_hp_mp
                int zero = 0;
                int m = this->domain_vertices.get_4D_shared_vertices().nb;
                int n = this->domain_vertices.get_4D_shared_vertices().nb;
                int mb = 128;
                int nb = 128;
                int irsrc = 0;
                int icsrc = 0;
                this->hp_mp_2d_reshape_m = numroc_(&m, &mb, &(this->icontxt_2d_reshape_myprow), &zero, &(this->icontxt_2d_reshape_nprow));
                this->hp_mp_2d_reshape_n = numroc_(&n, &nb, &(this->icontxt_2d_reshape_mypcol), &zero, &(this->icontxt_2d_reshape_npcol));
                int lld = std::max(1,this->hp_mp_2d_reshape_m);
                int info;
                descinit_(this->desc_2d_reshape_hp_mp,
                                &m, &n, &mb, &nb, &irsrc, &icsrc, &(this->icontxt_2d_reshape), &lld, &info);
            } else {
                this->desc_2d_reshape_hp_mp[1] = -1;
            }
            if (this->icontxt_2d_reshape != -1) {
                int ibtype = 1;
                char jobz = 'V';
                char range = 'A';
                char uplo = 'U';
                T abstol = (T)this->chefsi_control.abstol;
                T orfac = (T)this->chefsi_control.orfac;
                // T work[1];
                T* work = new (std::align_val_t(64)) T [1];
                int lwork = -1;
                // int iwork[1];
                int* iwork = new (std::align_val_t(64)) int [1];
                int liwork = -1;
                int nstates = this->domain_vertices.get_4D_shared_vertices().get_nb();
                // T a[1];
                T* a = new (std::align_val_t(64)) T [1];
                int one = 1;
                int temp;
                Linalg::p_sygvx_(&ibtype, &jobz, &range, &uplo, &nstates,
                                a, &one, &one, this->desc_2d_reshape_hp_mp,
                                a, &one, &one, this->desc_2d_reshape_hp_mp,
                                a, a, &one, &nstates, &abstol, &one, &one, a, &orfac,
                                a, &one, &one, this->desc_2d_reshape_hp_mp,
                                work, &lwork, iwork, &liwork, &temp, &temp, a, &temp);
                // lwork
                // this->lwork = (int) std::fabs(work[0]) * 5;
                //lwork ≥ 5*n + max(5*nn, np0*mq0 + 2*nb*nb) + iceil(neig, NPROW*NPCOL)*nn
                this->lwork = (int) std::fabs(work[0]);
                int NPROW = this->icontxt_2d_reshape_nprow;
                int NPCOL = this->icontxt_2d_reshape_npcol;
                int nb = this->desc_2d_reshape_hp_mp[4];
                int nn = std::max(std::max(nstates, nb),2);
                int zero = 0;
                int np0 = numroc_(&nstates, &nb, &zero, &zero, &NPROW);
                // int mq0  = numroc_(&nstates, &nb, &zero, &zero, &(this->icontxt_2d_reshape_npcol));
                int max_neig_nb_2 = std::max(std::max(nstates, nb),2);
                int mq0 = numroc_(&max_neig_nb_2, &nb, &zero, &zero, &NPCOL);
                int lwork_min = 5*nstates + std::max(5*nn, np0*mq0 + 2*nb*nb)
                              + std::ceil((nstates-1)/(NPROW*NPCOL)+1) * nn;
                this->lwork = std::max(this->lwork, lwork_min);
                //the line below is from sparc and i dont know why
                this->lwork += std::max(nstates*nstates, std::min(10*this->lwork,2000000));
                //liwork
                // this->liwork = iwork[0] * 5;
                //liwork ≥ 6*nnp
                //nnp = max(n, NPROW*NPCOL + 1, 4)
                this->liwork = iwork[0];
                int nnp = std::max(std::max(nstates, NPROW*NPCOL + 1),4);
                this->liwork = std::max(this->liwork, nnp);
                delete [] work;
                delete [] iwork;
                delete [] a;
            }
            #endif
        } else {
            assert(this->chefsi_control.projection_method == 0
                || this->chefsi_control.projection_method == 1
                || this->chefsi_control.projection_method == 2);
        }

        if (this->domain_vertices.is_active) {
            this->single_band_domain_vertices.set_chunk_size_i(this->domain_vertices.get_chunk_size_i());
            this->single_band_domain_vertices.set_chunk_size_j(this->domain_vertices.get_chunk_size_j());
            this->single_band_domain_vertices.set_chunk_size_k(this->domain_vertices.get_chunk_size_k());
            this->single_band_domain_vertices.init(this->domain_vertices.get_3D_shared_vertices(), this->domain_vertices.domain_3d_comm);
            int FDn[3] = {this->stencil.FDn, this->stencil.FDn, this->stencil.FDn};
            this->single_band_exarr_mpi_package.init(is_periodic, FDn);
        } else {
            this->single_band_domain_vertices.init(this->domain_vertices.get_3D_shared_vertices(), this->domain_vertices.domain_3d_comm);
            this->single_band_domain_vertices.set_is_active(false);
            this->single_band_domain_vertices.set_local_vertices(Vertices_3D(0,0,0));
            this->single_band_exarr_mpi_package.comm = this->single_band_domain_vertices.get_domain_3d_comm();
        }

        this->lanczos.init(this->single_band_domain_vertices, is_rand_fixed);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
        // if (this->domain_vertices.get_comm_rank() == 0) {
            std::cout << "The chefsi init took " << Tools::time_cost(begin, end) << "." << std::endl;
        }
    }
    return;
}

template<typename T>
template<typename T2>
void Chefsi<T>::init(const Chefsi<T2>& chefsi) {
    if (this->domain_vertices.shared_vertices.get_size() == 0) {
        if (this->chefsi_control.projection_method == 0) {                     // 0: DP_SUBEIG
        // do nothing
        } else if (this->chefsi_control.projection_method == 1) {
            #if (defined(USE_MKL) || defined(USE_SCALAPACK))
            Cblacs_get(0, 0, &(this->icontxt_whole_comm));
            Cblacs_get(0, 0, &(this->icontxt_intra_band));
            Cblacs_get(0, 0, &(this->icontxt_domain_3D));
            Cblacs_get(0, 0, &(this->icontxt_2d_reshape));
            Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_whole_comm), 1, 1, 1, false,
                                                    this->domain_vertices.get_mpi_comm());
            Cblacs_gridexit(this->icontxt_whole_comm);
            this->icontxt_whole_comm = -1;
            Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_intra_band), 1, 1, 1, false,
                                                    this->domain_vertices.band_comm);
            Cblacs_gridexit(this->icontxt_intra_band);
            this->icontxt_intra_band = -1;
            Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_domain_3D), 1, 1, 1, false,
                                                    this->domain_vertices.get_domain_4d_comm());
            Cblacs_gridexit(this->icontxt_domain_3D);
            this->icontxt_domain_3D = -1;
            Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_2d_reshape), 1, 1, 1, false,
                                                    this->domain_vertices.get_mpi_comm());
            Cblacs_gridexit(this->icontxt_2d_reshape);
            this->icontxt_2d_reshape = -1;
            #endif
        } else if (this->chefsi_control.projection_method == 2) {
            #if (defined(USE_MKL) || defined(USE_SCALAPACK))
            Cblacs_get(0, 0, &(this->icontxt_2d_reshape));
            Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_2d_reshape), 1, 1, 1, false,
                                                    this->domain_vertices.get_mpi_comm());
            Cblacs_gridexit(this->icontxt_2d_reshape);
            this->icontxt_2d_reshape = -1;
            #endif
        } else {
            assert(this->chefsi_control.projection_method == 0
                || this->chefsi_control.projection_method == 1
                || this->chefsi_control.projection_method == 2);
        }
    } else {
        this->is_very_first = chefsi.is_very_first;
        this->single_band_domain_vertices.init(chefsi.single_band_domain_vertices);
        this->single_band_exarr_mpi_package.init(chefsi.single_band_exarr_mpi_package);
        this->lanczos.init(chefsi.lanczos);
        if (this->chefsi_control.projection_method == 0) {
            this->dp_domain_vertices.init(chefsi.dp_domain_vertices);
            this->dp_mpi_package.init(chefsi.dp_mpi_package);
        } else if (this->chefsi_control.projection_method == 1) {
            #if (defined(USE_MKL) || defined(USE_SCALAPACK))
            assert(this->domain_vertices.is_Col_Maj);

            uint comm_nb = this->domain_vertices.get_active_comm_nb();
            uint comm_ninjnk = this->domain_vertices.get_active_comm_ni()
                            * this->domain_vertices.get_active_comm_nj()
                            * this->domain_vertices.get_active_comm_nk();
            Cblacs_get(0, 0, &(this->icontxt_whole_comm));
            Cblacs_get(0, 0, &(this->icontxt_intra_band));
            Cblacs_get(0, 0, &(this->icontxt_domain_3D));
            Cblacs_get(0, 0, &(this->icontxt_2d_reshape));
            int comm_size = this->domain_vertices.get_comm_size();
            Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_whole_comm),
                                                    comm_size, comm_size,
                                                    1, true,
                                                    this->domain_vertices.get_mpi_comm());
            Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_intra_band), 1, 1, comm_nb,
                                                    this->domain_vertices.is_active,
                                                    this->domain_vertices.band_comm);
            Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_domain_3D), comm_ninjnk,
                                                    comm_ninjnk, comm_nb, true,
                                                    this->domain_vertices.get_domain_4d_comm());
            // int size_d2 = std::floor(std::sqrt(comm_size));
            // Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_2d_reshape),
            //                                         size_d2, size_d2, size_d2, true,
            //                                         this->domain_vertices.get_mpi_comm());
            Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_2d_reshape),
                                                    1, 1, comm_nb, this->domain_vertices.is_active,
                                                    this->domain_vertices.band_comm);
            if (this->icontxt_intra_band != -1 && !this->domain_vertices.is_active) {
                Cblacs_gridexit(this->icontxt_intra_band);
                this->icontxt_intra_band = -1;
            }
            if (this->icontxt_2d_reshape != -1) {
                Cblacs_gridinfo(this->icontxt_2d_reshape, &(this->icontxt_2d_reshape_nprow), &(this->icontxt_2d_reshape_npcol),
                            &(this->icontxt_2d_reshape_myprow), &(this->icontxt_2d_reshape_mypcol));
            }

            if (this->icontxt_intra_band != -1 && this->domain_vertices.is_active) {
                // desc_intra_band_eigen_vectors
                int m = this->domain_vertices.get_3D_local_vertices().get_size();
                int n = this->domain_vertices.get_4D_shared_vertices().nb;
                int mb = m;
                int nb = this->domain_vertices.get_chunk_size_b();
                int irsrc = 0;
                int icsrc = 0;
                int lld = std::max(1,m);
                int info;
                descinit_(this->desc_intra_band_eigen_vectors,
                                &m, &n, &mb, &nb, &irsrc, &icsrc, &(this->icontxt_intra_band), &lld, &info);
            } else {
                this->desc_intra_band_eigen_vectors[1] = -1;
            }
            if (this->icontxt_intra_band != -1 && this->domain_vertices.is_active) {
                // desc_intra_band_hp_mp
                int m = this->domain_vertices.get_4D_shared_vertices().nb;
                int n = this->domain_vertices.get_4D_shared_vertices().nb;
                int mb = m;
                int nb = this->domain_vertices.get_chunk_size_b();
                int irsrc = 0;
                int icsrc = 0;
                int lld = std::max(1,m);
                int info;
                descinit_(this->desc_intra_band_hp_mp,
                                &m, &n, &mb, &nb, &irsrc, &icsrc, &(this->icontxt_intra_band), &lld, &info);
            } else {
                this->desc_intra_band_hp_mp[1] = -1;
            }
            if (this->icontxt_domain_3D != -1) {
                // desc_domain_3D_hp_mp
                int m = this->domain_vertices.get_4D_shared_vertices().nb;
                int n = this->domain_vertices.get_4D_shared_vertices().nb;
                int mb = m;
                int nb = this->domain_vertices.get_chunk_size_b();
                int irsrc = 0;
                int icsrc = 0;
                int lld = std::max(1,m);
                int info;
                descinit_(this->desc_domain_3D_hp_mp,
                                &m, &n, &mb, &nb, &irsrc, &icsrc, &(this->icontxt_domain_3D), &lld, &info);
            } else {
                this->desc_domain_3D_hp_mp[1] = -1;
            }
            if (this->icontxt_2d_reshape != -1) {
                // desc_2d_reshape_hp_mp
                int zero = 0;
                int m = this->domain_vertices.get_4D_shared_vertices().nb;
                int n = this->domain_vertices.get_4D_shared_vertices().nb;
                int mb = 128;
                int nb = 128;
                int irsrc = 0;
                int icsrc = 0;
                this->hp_mp_2d_reshape_m = numroc_(&m, &mb, &(this->icontxt_2d_reshape_myprow), &zero, &(this->icontxt_2d_reshape_nprow));
                this->hp_mp_2d_reshape_n = numroc_(&n, &nb, &(this->icontxt_2d_reshape_mypcol), &zero, &(this->icontxt_2d_reshape_npcol));
                int lld = std::max(1,this->hp_mp_2d_reshape_m);
                int info;
                descinit_(this->desc_2d_reshape_hp_mp,
                                &m, &n, &mb, &nb, &irsrc, &icsrc, &(this->icontxt_2d_reshape), &lld, &info);
            } else {
                this->desc_2d_reshape_hp_mp[1] = -1;
            }
            if (this->icontxt_2d_reshape != -1) {
                int ibtype = 1;
                char jobz = 'V';
                char range = 'A';
                char uplo = 'U';
                T abstol = (T)this->chefsi_control.abstol;
                T orfac = (T)this->chefsi_control.orfac;
                // T work[1];
                T* work = new (std::align_val_t(64)) T [1];
                int lwork = -1;
                // int iwork[1];
                int* iwork = new (std::align_val_t(64)) int [1];
                int liwork = -1;
                int nstates = this->domain_vertices.get_4D_shared_vertices().get_nb();
                // T a[1];
                T* a = new (std::align_val_t(64)) T [1];
                int one = 1;
                int temp;
                Linalg::p_sygvx_(&ibtype, &jobz, &range, &uplo, &nstates,
                                a, &one, &one, this->desc_2d_reshape_hp_mp,
                                a, &one, &one, this->desc_2d_reshape_hp_mp,
                                a, a, &one, &nstates, &abstol, &one, &one, a, &orfac,
                                a, &one, &one, this->desc_2d_reshape_hp_mp,
                                work, &lwork, iwork, &liwork, &temp, &temp, a, &temp);
                // lwork
                // this->lwork = (int) std::fabs(work[0]) * 5;
                //lwork ≥ 5*n + max(5*nn, np0*mq0 + 2*nb*nb) + iceil(neig, NPROW*NPCOL)*nn
                this->lwork = (int) std::fabs(work[0]);
                int NPROW = this->icontxt_2d_reshape_nprow;
                int NPCOL = this->icontxt_2d_reshape_npcol;
                int nb = this->desc_2d_reshape_hp_mp[4];
                int nn = std::max(std::max(nstates, nb),2);
                int zero = 0;
                int np0 = numroc_(&nstates, &nb, &zero, &zero, &NPROW);
                // int mq0  = numroc_(&nstates, &nb, &zero, &zero, &(this->icontxt_2d_reshape_npcol));
                int max_neig_nb_2 = std::max(std::max(nstates, nb),2);
                int mq0 = numroc_(&max_neig_nb_2, &nb, &zero, &zero, &NPCOL);
                int lwork_min = 5*nstates + std::max(5*nn, np0*mq0 + 2*nb*nb)
                              + std::ceil((nstates-1)/(NPROW*NPCOL)+1) * nn;
                this->lwork = std::max(this->lwork, lwork_min);
                //the line below is from sparc and i dont know why
                this->lwork += std::max(nstates*nstates, std::min(10*this->lwork,2000000));
                //liwork
                // this->liwork = iwork[0] * 5;
                //liwork ≥ 6*nnp
                //nnp = max(n, NPROW*NPCOL + 1, 4)
                this->liwork = iwork[0];
                int nnp = std::max(std::max(nstates, NPROW*NPCOL + 1),4);
                this->liwork = std::max(this->liwork, nnp);
                delete [] work;
                delete [] iwork;
                delete [] a;
            }
            #endif
        } else if (this->chefsi_control.projection_method == 2) {
            this->dp_domain_vertices.init(chefsi.dp_domain_vertices);
            this->dp_mpi_package.init(chefsi.dp_mpi_package);
            #if (defined(USE_MKL) || defined(USE_SCALAPACK))
            Cblacs_get(0, 0, &(this->icontxt_2d_reshape));
            int comm_size = this->domain_vertices.get_comm_size();
            int size_d2 = std::floor(std::sqrt(comm_size));
            Parallel_vertices::Cblacs_gridmap_subcomm(&(this->icontxt_2d_reshape),
                                                    size_d2, size_d2, size_d2, true,
                                                    this->domain_vertices.get_mpi_comm());
            if (this->icontxt_2d_reshape != -1) {
                Cblacs_gridinfo(this->icontxt_2d_reshape, &(this->icontxt_2d_reshape_nprow), &(this->icontxt_2d_reshape_npcol),
                            &(this->icontxt_2d_reshape_myprow), &(this->icontxt_2d_reshape_mypcol));
            }
            if (this->icontxt_2d_reshape != -1) {
                // desc_domain_3D_hp_mp
                int m = this->domain_vertices.get_4D_shared_vertices().nb;
                int n = this->domain_vertices.get_4D_shared_vertices().nb;
                int mb = m;
                int nb = n;
                int irsrc = 0;
                int icsrc = 0;
                int lld = std::max(1,m);
                int info;
                descinit_(this->desc_domain_3D_hp_mp,
                                &m, &n, &mb, &nb, &irsrc, &icsrc, &(this->icontxt_2d_reshape), &lld, &info);
            } else {
                this->desc_domain_3D_hp_mp[1] = -1;
            }
            if (this->icontxt_2d_reshape != -1) {
                // desc_2d_reshape_hp_mp
                int zero = 0;
                int m = this->domain_vertices.get_4D_shared_vertices().nb;
                int n = this->domain_vertices.get_4D_shared_vertices().nb;
                int mb = 128;
                int nb = 128;
                int irsrc = 0;
                int icsrc = 0;
                this->hp_mp_2d_reshape_m = numroc_(&m, &mb, &(this->icontxt_2d_reshape_myprow), &zero, &(this->icontxt_2d_reshape_nprow));
                this->hp_mp_2d_reshape_n = numroc_(&n, &nb, &(this->icontxt_2d_reshape_mypcol), &zero, &(this->icontxt_2d_reshape_npcol));
                int lld = std::max(1,this->hp_mp_2d_reshape_m);
                int info;
                descinit_(this->desc_2d_reshape_hp_mp,
                                &m, &n, &mb, &nb, &irsrc, &icsrc, &(this->icontxt_2d_reshape), &lld, &info);
            } else {
                this->desc_2d_reshape_hp_mp[1] = -1;
            }
            if (this->icontxt_2d_reshape != -1) {
                int ibtype = 1;
                char jobz = 'V';
                char range = 'A';
                char uplo = 'U';
                T abstol = (T)this->chefsi_control.abstol;
                T orfac = (T)this->chefsi_control.orfac;
                // T work[1];
                T* work = new (std::align_val_t(64)) T [1];
                int lwork = -1;
                // int iwork[1];
                int* iwork = new (std::align_val_t(64)) int [1];
                int liwork = -1;
                int nstates = this->domain_vertices.get_4D_shared_vertices().get_nb();
                // T a[1];
                T* a = new (std::align_val_t(64)) T [1];
                int one = 1;
                int temp;
                Linalg::p_sygvx_(&ibtype, &jobz, &range, &uplo, &nstates,
                                a, &one, &one, this->desc_2d_reshape_hp_mp,
                                a, &one, &one, this->desc_2d_reshape_hp_mp,
                                a, a, &one, &nstates, &abstol, &one, &one, a, &orfac,
                                a, &one, &one, this->desc_2d_reshape_hp_mp,
                                work, &lwork, iwork, &liwork, &temp, &temp, a, &temp);
                // lwork
                // this->lwork = (int) std::fabs(work[0]) * 5;
                //lwork ≥ 5*n + max(5*nn, np0*mq0 + 2*nb*nb) + iceil(neig, NPROW*NPCOL)*nn
                this->lwork = (int) std::fabs(work[0]);
                int NPROW = this->icontxt_2d_reshape_nprow;
                int NPCOL = this->icontxt_2d_reshape_npcol;
                int nb = this->desc_2d_reshape_hp_mp[4];
                int nn = std::max(std::max(nstates, nb),2);
                int zero = 0;
                int np0 = numroc_(&nstates, &nb, &zero, &zero, &NPROW);
                // int mq0  = numroc_(&nstates, &nb, &zero, &zero, &(this->icontxt_2d_reshape_npcol));
                int max_neig_nb_2 = std::max(std::max(nstates, nb),2);
                int mq0 = numroc_(&max_neig_nb_2, &nb, &zero, &zero, &NPCOL);
                int lwork_min = 5*nstates + std::max(5*nn, np0*mq0 + 2*nb*nb)
                              + std::ceil((nstates-1)/(NPROW*NPCOL)+1) * nn;
                this->lwork = std::max(this->lwork, lwork_min);
                //the line below is from sparc and i dont know why
                this->lwork += std::max(nstates*nstates, std::min(10*this->lwork,2000000));
                //liwork
                // this->liwork = iwork[0] * 5;
                //liwork ≥ 6*nnp
                //nnp = max(n, NPROW*NPCOL + 1, 4)
                this->liwork = iwork[0];
                int nnp = std::max(std::max(nstates, NPROW*NPCOL + 1),4);
                this->liwork = std::max(this->liwork, nnp);
                delete [] work;
                delete [] iwork;
                delete [] a;
            }
            #endif
        } else {
            assert(this->chefsi_control.projection_method == 0
                || this->chefsi_control.projection_method == 1
                || this->chefsi_control.projection_method == 2);
        }
    }
    return;
}
template void Chefsi<float>::init(const Chefsi<float>& chefsi);
template void Chefsi<double>::init(const Chefsi<double>& chefsi);
template void Chefsi<float>::init(const Chefsi<double>& chefsi);
template void Chefsi<double>::init(const Chefsi<float>& chefsi);

template<typename T>
void Chefsi<T>::destructor() {
    this->single_band_exarr_mpi_package.destructor();
    this->lanczos.destructor();
    if (this->chefsi_control.projection_method == 0) {
        this->dp_mpi_package.destructor();
    } else if (this->chefsi_control.projection_method == 1) {
        #if (defined(USE_MKL) || defined(USE_SCALAPACK))
        if (this->icontxt_whole_comm != -1) {
            Cblacs_gridexit(this->icontxt_whole_comm);
            this->icontxt_whole_comm = -1;
        }
        if (this->icontxt_intra_band != -1) {
            Cblacs_gridexit(this->icontxt_intra_band);
            this->icontxt_intra_band = -1;
        }
        if (this->icontxt_domain_3D != -1) {
            Cblacs_gridexit(this->icontxt_domain_3D);
            this->icontxt_domain_3D = -1;
        }
        if (this->icontxt_2d_reshape != -1) {
            Cblacs_gridexit(this->icontxt_2d_reshape);
            this->icontxt_2d_reshape = -1;
        }
        #endif
    } else if (this->chefsi_control.projection_method == 2) {
        #if (defined(USE_MKL) || defined(USE_SCALAPACK))
        this->dp_mpi_package.destructor();
        if (this->icontxt_2d_reshape != -1) {
            Cblacs_gridexit(this->icontxt_2d_reshape);
            this->icontxt_2d_reshape = -1;
        }
        #endif
    } else {
        assert(this->chefsi_control.projection_method == 0
            || this->chefsi_control.projection_method == 1
            || this->chefsi_control.projection_method == 2);
    }
    return;
}

template<typename T>
void Chefsi<T>::show() const {
    this->chefsi_control.show();
    this->stencil.show();
    this->domain_vertices.show();
    this->exarr_mpi_package.show();
    if (this->chefsi_control.projection_method == 0) {                     // 0: DP_SUBEIG
        this->dp_domain_vertices.show();
        this->dp_mpi_package.show();
    }
    this->lanczos.show();
    return;
}

template class Chefsi<float>;
template class Chefsi<double>;
