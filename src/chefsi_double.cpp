#include "chefsi.h"
#include "xlsdft_chefsi_lvtx.hpp"
#include "xlsdft_backend.h"
#pragma message("Building with chefsi_double.cpp.")

#include <type_traits>
#include <unistd.h>

#ifndef stencil_nthread
#define stencil_nthread 36
#endif
#ifndef nloc_nthread
#define nloc_nthread 36
#endif
#ifndef projection_nthread
#define projection_nthread 24
#endif
#ifndef dia_nthread
#define dia_nthread 24
#endif
#ifndef rotation_nthread
#define rotation_nthread 24
#endif

// #ifdef ENABLE_CHEFSI_TIMER
// #pragma message("Building with ENABLE_CHEFSI_DOUBLE_TIMER.")
// Chefsi_timer::Chefsi_timer() {}
// // Chefsi_timer::~Chefsi_timer() {}
// void Chefsi_timer::reset() {
//     this->chefsi.reset();
//     this->lanczos.reset();
//     this->filter.reset();
//     this->filter_copy.reset();
//     this->filter_product.reset();
//     this->filter_lap.reset();
//     this->filter_nloc.reset();
//     this->H_psi.reset();
//     this->projection.reset();
//     this->diagonalization.reset();
//     this->rotation.reset();
//     return;
// }
// void Chefsi_timer::show(std::ostream& output) const {
//     output << std::left << std::setw(20) << "chefsi"             << ": " << this->chefsi.time_cost_millisecond() << " [ms]" << std::endl;
//     output << std::left << std::setw(20) << "  lanczos"           << ": " << this->lanczos.time_cost_millisecond() << " [ms]" << std::endl;
//     output << std::left << std::setw(20) << "  filter"           << ": " << this->filter.time_cost_millisecond() << " [ms]" << std::endl;
//     output << std::left << std::setw(20) << "    filter_copy"    << ": " << this->filter_copy.time_cost_millisecond() << " [ms]" << std::endl;
//     output << std::left << std::setw(20) << "    filter_product"    << ": " << this->filter_product.time_cost_millisecond() << " [ms]" << std::endl;
//     output << std::left << std::setw(20) << "    filter_lap"     << ": " << this->filter_lap.time_cost_millisecond() << " [ms]" << std::endl;
//     output << std::left << std::setw(20) << "    filter_nloc"    << ": " << this->filter_nloc.time_cost_millisecond() << " [ms]" << std::endl;
//     output << std::left << std::setw(20) << "  H_psi"            << ": " << this->H_psi.time_cost_millisecond() << " [ms]" << std::endl;
//     output << std::left << std::setw(20) << "  projection"       << ": " << this->projection.time_cost_millisecond() << " [ms]" << std::endl;
//     output << std::left << std::setw(20) << "  diagonalization"  << ": " << this->diagonalization.time_cost_millisecond() << " [ms]" << std::endl;
//     output << std::left << std::setw(20) << "  rotation "        << ": " << this->rotation.time_cost_millisecond() << " [ms]" << std::endl;
//     return;
// }
// #endif //ENABLE_CHEFSI_TIMER

template<typename T>
inline void Chefsi<T>::chebyshev_filtering_column_wise2_omp_task_comm_self_with_temp_swap(
                    T*& __restrict__ eigen_vectors,
                    T const* const& __restrict__ Vloc,
                    const Effective_potential_nloc<T>& Vnloc,
                    T*& eigen_vectors_temp,
                    const bool& print_flag) {
    
    #ifdef ENABLE_CHEFSI_TIMER
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

    Stencil<T> stencil_temp;
    stencil_temp.deepcopy(stencil);
    stencil_temp.coeffs_scale_self(-0.5);
    stencil_temp.shift_D2_coeffs(-c);

    T sigma = e / (this->lanczos.eig_min - c);
    T sigma1 = sigma;
    T gamma = 2.0 / sigma1;
    // T sigma2;

    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.filter_lap.start();
    #endif //ENABLE_CHEFSI_TIMER
    #pragma omp parallel num_threads(stencil_nthread)
    Stencil_method::Boundary_safe::calc_laplacian_boundary_safe(eigen_vectors, local_vertices, stencil_temp, local_vertices,
                                            eigen_vectors_temp, local_vertices, Vloc);

    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.filter_lap.stop();
        this->chefsi_timer.filter_nloc.start();
    #endif //ENABLE_CHEFSI_TIMER
    #pragma omp parallel num_threads(nloc_nthread)
    Hamiltonian::nloc_project_vectors<T>(eigen_vectors_temp, local_vertices, eigen_vectors, Vnloc,
                                        (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package.comm);
    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.filter_nloc.stop();
    #endif //ENABLE_CHEFSI_TIMER

    T vscal = sigma1 / e;
    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.filter_product.start();
    #endif //ENABLE_CHEFSI_TIMER
    #pragma omp parallel
    Linalg::scalar_product_general(eigen_vectors_temp, vscal, Nd * nb);
    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.filter_product.stop();
    #endif //ENABLE_CHEFSI_TIMER

    if (unlikely(this->chefsi_control.chebyshev_filter_degree == 1)) {
        #pragma omp parallel
        Linalg::set_value_general(eigen_vectors, eigen_vectors_temp, Nd * nb);
        stencil_temp.destructor();
        return;
    }

    #ifdef USE_HBM
    #ifdef USE_HUGEPAGE_HBM
    int cpu = sched_getcpu();
    unsigned long nodeid = numa_node_of_cpu(cpu);
    unsigned long hbm_node = nodeid + (numa_num_configured_nodes() / 2);
    T* eigen_vectors_m1;
    if (Nd * nb != 0) {
        unsigned long size = Nd * nb * sizeof(T);
        eigen_vectors_m1 = static_cast<T*>(hbm_alloc(size, hbm_node, false));
    }
    #else
    T* eigen_vectors_m1= static_cast<T*>(hbw_malloc(Nd * nb * sizeof(T)));
    #endif
    #else
    T* eigen_vectors_m1 = new (std::align_val_t(64)) T [Nd * nb];
    #endif

    std::swap(eigen_vectors_m1, eigen_vectors);
    std::swap(eigen_vectors, eigen_vectors_temp);

    T vscal2;
    for (uint j = 2; j <= this->chefsi_control.chebyshev_filter_degree; j++) {
        T sigma2 = 1.0 / (gamma - sigma);

        #ifdef ENABLE_CHEFSI_TIMER
            this->chefsi_timer.filter_lap.start();
        #endif //ENABLE_CHEFSI_TIMER
        #pragma omp parallel num_threads(stencil_nthread)
        Stencil_method::Boundary_safe::calc_laplacian_boundary_safe(eigen_vectors, local_vertices, stencil_temp, local_vertices,
                                            eigen_vectors_temp, local_vertices, Vloc);

        #ifdef ENABLE_CHEFSI_TIMER
            this->chefsi_timer.filter_lap.stop();
            this->chefsi_timer.filter_nloc.start();
        #endif //ENABLE_CHEFSI_TIMER
        #pragma omp parallel num_threads(nloc_nthread)
        Hamiltonian::nloc_project_vectors<T>(eigen_vectors_temp, local_vertices, eigen_vectors, Vnloc,
                                        (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package.comm);
        #ifdef ENABLE_CHEFSI_TIMER
            this->chefsi_timer.filter_nloc.stop();
        #endif //ENABLE_CHEFSI_TIMER

        vscal = 2.0 * sigma2 / e;
        vscal2 = sigma * sigma2;
        #ifdef ENABLE_CHEFSI_TIMER
            this->chefsi_timer.filter_product.start();
        #endif //ENABLE_CHEFSI_TIMER
        #pragma omp parallel
        Linalg::scalar_product_general(eigen_vectors_temp, vscal, Nd * nb, eigen_vectors_m1, -vscal2);
        #ifdef ENABLE_CHEFSI_TIMER
            this->chefsi_timer.filter_product.stop();
        #endif //ENABLE_CHEFSI_TIMER

        std::swap(eigen_vectors_m1, eigen_vectors);
        std::swap(eigen_vectors, eigen_vectors_temp);

        sigma = sigma2;
    }

    #ifdef USE_HBM
    #ifdef USE_HUGEPAGE_HBM
    if (Nd * nb != 0) {
        unsigned long size = Nd * nb * sizeof(T);
        hbm_free(eigen_vectors_m1, size, false);
    }
    #else
    hbw_free(eigen_vectors_m1);
    #endif
    #else
    ::operator delete[](eigen_vectors_m1, std::align_val_t(64));
    #endif 
    eigen_vectors_m1 = nullptr;
    stencil_temp.destructor();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
        std::cout << "The chebyshev_filtering_column_wise took " << Tools::time_cost(begin, end) << "." << std::endl;
    }

    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.filter.stop();
    #endif //ENABLE_CHEFSI_TIMER

    return;
}


template<typename T>
void Chefsi<T>::project_hamiltonian_with_temp_swap(T*& eigen_vectors, T*& h_eigen_vectors, 
                                        T* const hp, T* const mp,
                                        const bool print_flag) {

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
            if (Xlsdft_chefsi_lvtx::projection_contract_matches<T>(m, k)) {
                const std::size_t workspace_doubles =
                    Xlsdft_backend::projection_workspace_doubles();
                double* workspace =
                    new (std::align_val_t(64)) double[workspace_doubles];
                const int status = Xlsdft_chefsi_lvtx::project_hamiltonian(
                    eigen_vectors, h_eigen_vectors, hp, mp, workspace,
                    workspace_doubles);
                Xlsdft_backend::require_success(
                    Xlsdft_backend::Operation::dgemm_poj, status,
                    this->domain_vertices.comm);
                ::operator delete[](workspace, std::align_val_t(64));
            } else {
                Xlsdft_chefsi_lvtx::project_hamiltonian_fallback(
                    m, k, eigen_vectors, h_eigen_vectors, hp, mp);
            }
            // Linalg::set_kblas_1();
            // Linalg::matrix_product(eigen_vectors, 0, h_eigen_vectors, 1,
            //                                 hp, 1, m, m, k);
            // Linalg::matrix_product(eigen_vectors, 0, eigen_vectors, 1,
            //                                 mp, 1, m, m, k);
            // #pragma omp parallel
            // Linalg::set_value_general(h_eigen_vectors, eigen_vectors, m * k);
            // #pragma omp barrier
            std::swap(h_eigen_vectors, eigen_vectors);
        } else {
            assert(this->domain_vertices.get_domain_4d_comm_size() == 1);
        }
    } else if (this->chefsi_control.projection_method == 1) {
        assert(this->chefsi_control.projection_method == 0
            || this->chefsi_control.projection_method == 2);
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

    return;
}

template<typename T>
void Chefsi<T>::subspace_rotation_specialization(T*& __restrict__ eigen_vectors, T const* const& __restrict__ eigen_vectors_reshape,
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
                if (Xlsdft_chefsi_lvtx::rotation_contract_matches<T>(m, n)) {
                    const std::size_t workspace_doubles =
                        Xlsdft_backend::rotation_workspace_doubles();
                    double* workspace =
                        new (std::align_val_t(64)) double[workspace_doubles];
                    const int status = Xlsdft_chefsi_lvtx::rotate_subspace(
                        eigen_vectors_reshape, qp, eigen_vectors, workspace,
                        workspace_doubles);
                    Xlsdft_backend::require_success(
                        Xlsdft_backend::Operation::dgemm_rotation, status,
                        this->domain_vertices.comm);
                    ::operator delete[](workspace, std::align_val_t(64));
                } else {
                    Xlsdft_chefsi_lvtx::rotate_subspace_fallback(
                        m, n, eigen_vectors_reshape, qp, eigen_vectors);
                }
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

template<>
void Chefsi<double>::run(Array_4D<double>& eigen_vectors, Array_0D<double>& eigen_values, const Array_3D<double>& Vloc,
                    const Effective_potential_nloc<double>& Vnloc, const bool& print_flag) {
    this->run_mp(eigen_vectors.data, eigen_values.data, Vloc.data,
             Vnloc, print_flag);
    // if (this->domain_vertices.get_4D_shared_vertices().get_size() == 0) return;
    // #ifdef ENABLE_CHEFSI_TIMER
    //     #pragma omp master
    //     {
    //         this->chefsi_timer.reset();
    //         this->chefsi_timer.chefsi.start();
    //     }
    // #endif //ENABLE_CHEFSI_TIMER

    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // //init 
    // Array_4D<double> h_eigen_vectors;
    // Array_4D<double>& eigen_vectors_temp = h_eigen_vectors;
    // Array_2D<double> hp;
    // Array_2D<double> mp;
    // const Array_2D<double>& qp = hp;
    // Array_4D<double>& eigen_vectors_reshape = h_eigen_vectors;

    // #ifdef USE_HBM
    // const Vertices_4D& local_vertices = this->domain_vertices.get_4D_local_vertices();
    // uint nstates = local_vertices.nb;
    // Array_3D<double> Vloc_loc;
    // Vloc_loc.reconstructor_hbm(local_vertices.Vertices_3D::get_vertices());
    // #pragma omp parallel
    // Linalg::set_value_general(Vloc_loc.data, Vloc.data, Vloc_loc.length);
    // h_eigen_vectors.reconstructor_hbm(local_vertices);
    // hp.reconstructor_hbm(nstates, nstates);
    // mp.reconstructor_hbm(nstates, nstates);
    // // eigen_vectors_reshape.reconstructor(local_vertices);
    // #else
    // const Vertices_4D& local_vertices = this->domain_vertices.get_4D_local_vertices();
    // uint nstates = local_vertices.nb;
    // const Array_3D<double>& Vloc_loc = Vloc;
    // h_eigen_vectors.reconstructor(local_vertices);
    // hp.reconstructor(nstates, nstates);
    // mp.reconstructor(nstates, nstates);
    // // eigen_vectors_reshape.reconstructor(local_vertices);
    // #endif

    // if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
    //     const Vertices_4D& shared_vertices = this->domain_vertices.get_4D_shared_vertices();
    //     std::cout << "Eigenvectors: nband = " << local_vertices.get_nb()
    //               << " in " << shared_vertices.get_nb()
    //               << ", grids = "  << local_vertices.get_size()
    //               << " in " << shared_vertices.get_size()
    //               << ", ([" << local_vertices.get_ni()
    //               << ", " << local_vertices.get_nj()
    //               << ", " << local_vertices.get_nk()
    //               << "] in [" << shared_vertices.get_ni()
    //               << ", " << shared_vertices.get_nj()
    //               << ", " << shared_vertices.get_nk()
    //               << "])."<< std::endl;
    // }

    // //run
    // for (uint iter = 0; iter < (unlikely(this->is_very_first) ? this->chefsi_control.rho_trigger : this->chefsi_control.max_iter); iter++) {
    //     #ifdef ENABLE_CHEFSI_TIMER
    //         this->chefsi_timer.lanczos.start();
    //     #endif //ENABLE_CHEFSI_TIMER
    //     this->lanczos.run(this->domain_vertices.comm, Vloc_loc, Vnloc, this->stencil,
    //                   (double)this->mesh_control.delta_V, this->single_band_exarr_mpi_package,
    //                   this->is_very_first && iter == 0, eigen_values[eigen_values.length - 1], eigen_values.data[0],
    //                 //   true, eigen_values[eigen_values.length - 1], eigen_values.data[0],
    //                   print_flag);
    //     #ifdef ENABLE_CHEFSI_TIMER
    //         this->chefsi_timer.lanczos.stop();
    //     #endif //ENABLE_CHEFSI_TIMER
    //     // #pragma omp parallel
    //     this->chebyshev_filtering_column_wise2_omp_task_comm_self_with_temp_swap(eigen_vectors.data, Vloc_loc.data, Vnloc, eigen_vectors_temp.data, print_flag);
    //     #ifdef ENABLE_CHEFSI_TIMER
    //         this->chefsi_timer.H_psi.start();
    //     #endif //ENABLE_CHEFSI_TIMER
    //     Hamiltonian::hamiltonian_product_vectors_column_wise2_specialization(h_eigen_vectors.data, this->domain_vertices.get_4D_local_vertices(),
    //                                                           eigen_vectors.data, Vloc_loc.data, Vnloc, this->stencil,
    //                                                           (double)this->mesh_control.delta_V, this->single_band_exarr_mpi_package,
    //                                                           print_flag);
    //     #ifdef ENABLE_CHEFSI_TIMER
    //         this->chefsi_timer.H_psi.stop();
    //     #endif //ENABLE_CHEFSI_TIMER
    //     #ifdef USE_KML
    //     BlasSetNumThreadsLocal(projection_nthread);
    //     #endif
    //     this->project_hamiltonian_with_temp_swap(eigen_vectors.data, h_eigen_vectors.data, hp.data, mp.data, print_flag);
    //     #ifdef USE_KML
    //     BlasSetNumThreadsLocal(dia_nthread);
    //     #endif
    //     this->subspace_diagonalization(hp.data, mp.data, eigen_values.data, print_flag);
    //     #ifdef USE_KML
    //     BlasSetNumThreadsLocal(rotation_nthread);
    //     #endif
    //     this->subspace_rotation_specialization(eigen_vectors.data, eigen_vectors_reshape.data, qp.data, print_flag);
    //     #ifdef USE_KML
    //     BlasSetNumThreadsLocal(1);
    //     #endif
    // }

    // if (unlikely(this->is_very_first)) {this->is_very_first = false;}

    // #ifdef USE_HBM
    // Vloc_loc.destructor_hbm();
    // h_eigen_vectors.destructor_hbm();
    // hp.destructor_hbm();
    // mp.destructor_hbm();
    // // eigen_vectors_reshape.destructor();
    // #else
    // h_eigen_vectors.destructor();
    // hp.destructor();
    // mp.destructor();
    // // eigen_vectors_reshape.destructor();
    // #endif


    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
    //     std::cout << "The Chefsi run took " << Tools::time_cost(begin, end) << "." << std::endl;
    // }

    // #ifdef ENABLE_CHEFSI_TIMER
    //     this->chefsi_timer.chefsi.stop();
    //     if (this->domain_vertices.get_comm_rank() == 0 && print_flag) this->chefsi_timer.show();
    //     // int rank;
    //     // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //     // std::string fname = "chefsi_timer_" + std::to_string(rank) + ".txt";
    //     // std::ofstream ofs(fname, std::ios::app);
    //     // this->chefsi_timer.show(ofs);
    //     // ofs.close();
    //     // const Vertices_4D& shared_vertices = this->domain_vertices.get_4D_shared_vertices();
    //     // ofs << "Eigenvectors: nband = " << local_vertices.get_nb()
    //     //         << " in " << shared_vertices.get_nb()
    //     //         << ", grids = "  << local_vertices.get_size()
    //     //         << " in " << shared_vertices.get_size()
    //     //         << ", ([" << local_vertices.get_ni()
    //     //         << ", " << local_vertices.get_nj()
    //     //         << ", " << local_vertices.get_nk()
    //     //         << "] in [" << shared_vertices.get_ni()
    //     //         << ", " << shared_vertices.get_nj()
    //     //         << ", " << shared_vertices.get_nk()
    //     //         << "])."<< std::endl;
    //     // Vnloc.show(ofs);
    // #endif //ENABLE_CHEFSI_TIMER

    return;
}

template<typename T>
void Chefsi<T>::run_mp(T*& eigen_vectors, T *const eigen_values, T const *const Vloc,
             const Effective_potential_nloc<T>& Vnloc, const bool print_flag) {
    constexpr size_t GB = 1024 * 1024 * 1024 / sizeof(T);
    Memory_pool<T, Fast_memory> pool_fast(2 * GB);
    Memory_pool<T, Capacity_memory> pool_cap(2 * GB);
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    const uint local_vertices_size = this->domain_vertices.get_4D_local_vertices().get_size();
    if (local_vertices_size || this->chefsi_control.chebyshev_filter_degree == 0) return;
    T*& eigen_vectors_in = eigen_vectors;
    T* eigen_vectors_out = pool_fast.allocate(local_vertices_size);
    this->run_mp(eigen_vectors_in, eigen_vectors_out, eigen_values,
                Vloc, Vnloc, print_flag, pool_fast, pool_cap);
    Linalg::set_value_general(eigen_vectors, eigen_vectors_out, local_vertices_size);
    return;
}

template<typename T>
void Chefsi<T>::run_mp(T*& eigen_vectors_in, T*& eigen_vectors_out, T *const eigen_values,
                    T const *const Vloc, const Effective_potential_nloc<T>& Vnloc, const bool print_flag,
            Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    if (this->domain_vertices.get_4D_shared_vertices().get_size() == 0) return;
     #ifdef ENABLE_CHEFSI_TIMER
        #pragma omp master
        {
            this->chefsi_timer.reset();
            this->chefsi_timer.chefsi.start();
        }
    #endif //ENABLE_CHEFSI_TIMER
    const Vertices_4D& local_vertices_4d = this->domain_vertices.get_4D_local_vertices();
    const Vertices_3D& local_vertices_3d = this->domain_vertices.get_3D_local_vertices();
    const Vertices_4D& shared_vertices = this->domain_vertices.get_4D_shared_vertices();
    const uint nstates_shared = shared_vertices.get_nb();

    const uint niter = unlikely(this->is_very_first) ? this->chefsi_control.rho_trigger : this->chefsi_control.max_iter;
    if (this->domain_vertices.get_comm_rank() == 0 && print_flag) {
        std::cout << "CheFSI: niter = " << niter
                  << ", Eigenvectors: nband = " << local_vertices_4d.get_nb()
                  << " in " << shared_vertices.get_nb()
                  << ", grids = "  << local_vertices_4d.get_size()
                  << " in " << shared_vertices.get_size()
                  << ", ([" << local_vertices_4d.get_ni()
                  << ", " << local_vertices_4d.get_nj()
                  << ", " << local_vertices_4d.get_nk()
                  << "] in [" << shared_vertices.get_ni()
                  << ", " << shared_vertices.get_nj()
                  << ", " << shared_vertices.get_nk()
                  << "])."<< std::endl;
    }

    const uint wf_elems = local_vertices_4d.get_size();
    T* eigen_vectors_m1 = nullptr;
    if (this->chefsi_control.chebyshev_filter_degree > 1) {
        eigen_vectors_m1 = pool_cap.allocate(wf_elems);
    }

    for (uint iter = 0; iter < niter; iter++) {
        Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast2(pool_fast);
        Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap2(pool_cap);
        #ifdef ENABLE_CHEFSI_TIMER
            this->chefsi_timer.lanczos.start();
        #endif //ENABLE_CHEFSI_TIMER
        const bool if_calculate = std::getenv("CAL_LIGEPS") != nullptr
                                ? true
                                : this->is_very_first && iter == 0;
        this->lanczos.run_mp(this->domain_vertices.comm, Vloc, local_vertices_3d, Vnloc, this->stencil,
                      T(this->mesh_control.delta_V), this->single_band_exarr_mpi_package,
                      if_calculate, eigen_values[nstates_shared - 1], eigen_values[0],
                      print_flag, pool_fast, pool_cap);
        #ifdef ENABLE_CHEFSI_TIMER
            this->chefsi_timer.lanczos.stop();
        #endif //ENABLE_CHEFSI_TIMER
        // #pragma omp parallel
        this->chebyshev_filtering_column_wise_mp(eigen_vectors_in, eigen_vectors_out, eigen_vectors_m1,
                                                  Vloc, Vnloc, print_flag, pool_fast, pool_cap);
        // Degree>=2 filter can leave in/out aliased; H must not overwrite psi.
        if (eigen_vectors_in == eigen_vectors_out) {
            assert(eigen_vectors_m1 != nullptr);
            std::swap(eigen_vectors_out, eigen_vectors_m1);
        }
        #ifdef ENABLE_CHEFSI_TIMER
            this->chefsi_timer.H_psi.start();
        #endif //ENABLE_CHEFSI_TIMER
        T*& h_eigen_vectors = eigen_vectors_out;
        Hamiltonian::hamiltonian_product_vectors_column_wise2_specialization_mp(h_eigen_vectors, local_vertices_4d,
                                                              eigen_vectors_in, Vloc, Vnloc, this->stencil,
                                                              T(this->mesh_control.delta_V), this->single_band_exarr_mpi_package,
                                                              print_flag, pool_fast, pool_cap);
        #ifdef ENABLE_CHEFSI_TIMER
            this->chefsi_timer.H_psi.stop();
        #endif //ENABLE_CHEFSI_TIMER
        T* hp = pool_cap.allocate(nstates_shared * nstates_shared);
        T* mp = pool_cap.allocate(nstates_shared * nstates_shared);
        #ifdef USE_KML
        BlasSetNumThreadsLocal(projection_nthread);
        #endif
        this->project_hamiltonian_mp(eigen_vectors_in, h_eigen_vectors, hp, mp, print_flag, pool_fast, pool_cap);
        #ifdef USE_KML
        BlasSetNumThreadsLocal(dia_nthread);
        #endif
        this->subspace_diagonalization_mp(hp, mp, eigen_values, print_flag,
                                          pool_fast, pool_cap);
        #ifdef USE_KML
        BlasSetNumThreadsLocal(rotation_nthread);
        #endif
        T*& qp = hp;
        this->subspace_rotation_mp(eigen_vectors_in, eigen_vectors_out, qp, print_flag, pool_fast, pool_cap);
        #ifdef USE_KML
        BlasSetNumThreadsLocal(1);
        #endif
    }

    if (unlikely(this->is_very_first)) {this->is_very_first = false;}

    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.chefsi.stop();
        if (this->domain_vertices.get_comm_rank() == 0 && print_flag) this->chefsi_timer.show();
        this->chefsi_flop_counter.set_data(this->domain_vertices.shared_vertices, Vnloc, niter,
                                    this->chefsi_control.chebyshev_filter_degree);
        this->chefsi_performance.update();
        this->chefsi_performance.show(print_flag, std::cout);
        // int rank;
        // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        // std::string fname = "chefsi_timer_" + std::to_string(rank) + ".txt";
        // std::ofstream ofs(fname, std::ios::app);
        // this->chefsi_timer.show(ofs);
        // ofs.close();
        // const Vertices_4D& shared_vertices = this->domain_vertices.get_4D_shared_vertices();
        // ofs << "Eigenvectors: nband = " << local_vertices.get_nb()
        //         << " in " << shared_vertices.get_nb()
        //         << ", grids = "  << local_vertices.get_size()
        //         << " in " << shared_vertices.get_size()
        //         << ", ([" << local_vertices.get_ni()
        //         << ", " << local_vertices.get_nj()
        //         << ", " << local_vertices.get_nk()
        //         << "] in [" << shared_vertices.get_ni()
        //         << ", " << shared_vertices.get_nj()
        //         << ", " << shared_vertices.get_nk()
        //         << "])."<< std::endl;
        // Vnloc.show(ofs);
    #endif //ENABLE_CHEFSI_TIMER

    return;
}

template<typename T>
void Chefsi<T>::chebyshev_filtering_column_wise_mp(
                    T*& __restrict__ eigen_vectors,
                    T*& __restrict__ eigen_vectors_buffer,
                    T* const eigen_vectors_m1_panel,
                    T const* const __restrict__ Vloc,
                    const Effective_potential_nloc<T>& Vnloc,
                    const bool print_flag,
                    Memory_pool<T, Fast_memory>& pool_fast,
                    Memory_pool<T, Capacity_memory>& pool_cap) {
    (void) print_flag;
    T* const eigen_vectors_in = eigen_vectors;

    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.filter.start();
    #endif //ENABLE_CHEFSI_TIMER

    const Vertices_4D& local_vertices_4d = this->domain_vertices.get_4D_local_vertices();
    const Vertices_3D& local_vertices_3d = this->domain_vertices.get_3D_local_vertices();
    const uint Nd = local_vertices_3d.get_size();
    const uint& nb = this->domain_vertices.get_4D_local_vertices().nb;
    if (Nd == 0 || nb == 0 || this->chefsi_control.chebyshev_filter_degree == 0) return;

    T*& eigen_vectors_temp = eigen_vectors_buffer;

    const T e = 0.5 * (this->lanczos.eig_max - this->lanczos.lambda_cutoff);
    const T c = 0.5 * (this->lanczos.eig_max + this->lanczos.lambda_cutoff);

    Stencil<T> stencil_temp;
    stencil_temp.deepcopy_mp(stencil, pool_fast);
    stencil_temp.coeffs_scale_self(-0.5);
    stencil_temp.shift_D2_coeffs(-c);

    T sigma = e / (this->lanczos.eig_min - c);
    T sigma1 = sigma;
    T gamma = 2.0 / sigma1;
    // T sigma2;

    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.filter_lap.start();
    #endif //ENABLE_CHEFSI_TIMER
    #pragma omp parallel num_threads(stencil_nthread)
    Stencil_method::Boundary_safe::calc_laplacian_boundary_safe(eigen_vectors, local_vertices_4d, stencil_temp, local_vertices_4d,
                                            eigen_vectors_temp, local_vertices_4d, Vloc);

    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.filter_lap.stop();
        this->chefsi_timer.filter_nloc.start();
    #endif //ENABLE_CHEFSI_TIMER
    // #pragma omp parallel num_threads(nloc_nthread)
    Hamiltonian::nloc_project_vectors_omp_for_comm_self_with_chunk_mp<T>(eigen_vectors_temp, local_vertices_4d, eigen_vectors, Vnloc,
                                        (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package.comm, pool_fast, pool_cap);
    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.filter_nloc.stop();
    #endif //ENABLE_CHEFSI_TIMER

    T vscal = sigma1 / e;
    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.filter_product.start();
    #endif //ENABLE_CHEFSI_TIMER
    #pragma omp parallel
    Linalg::scalar_product_general(eigen_vectors_temp, vscal, Nd * nb);
    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.filter_product.stop();
    #endif //ENABLE_CHEFSI_TIMER

    if (unlikely(this->chefsi_control.chebyshev_filter_degree == 1)) {
        #pragma omp parallel
        Linalg::set_value_general(eigen_vectors, eigen_vectors_temp, Nd * nb);
        stencil_temp.destructor_mp();
        return;
    }

    assert(eigen_vectors_m1_panel != nullptr);
    T* eigen_vectors_m1 = eigen_vectors_m1_panel;

    std::swap(eigen_vectors_m1, eigen_vectors);
    std::swap(eigen_vectors, eigen_vectors_temp);

    T vscal2;
    for (uint j = 2; j <= this->chefsi_control.chebyshev_filter_degree; j++) {
        T sigma2 = 1.0 / (gamma - sigma);

        #ifdef ENABLE_CHEFSI_TIMER
            this->chefsi_timer.filter_lap.start();
        #endif //ENABLE_CHEFSI_TIMER
        #pragma omp parallel num_threads(stencil_nthread)
        Stencil_method::Boundary_safe::calc_laplacian_boundary_safe(eigen_vectors, local_vertices_4d, stencil_temp, local_vertices_4d,
                                            eigen_vectors_temp, local_vertices_4d, Vloc);

        #ifdef ENABLE_CHEFSI_TIMER
            this->chefsi_timer.filter_lap.stop();
            this->chefsi_timer.filter_nloc.start();
        #endif //ENABLE_CHEFSI_TIMER
        // #pragma omp parallel num_threads(nloc_nthread)
        Hamiltonian::nloc_project_vectors_omp_for_comm_self_with_chunk_mp<T>(eigen_vectors_temp, local_vertices_4d, eigen_vectors, Vnloc,
                                        (T)this->mesh_control.delta_V, this->single_band_exarr_mpi_package.comm, pool_fast, pool_cap);
        #ifdef ENABLE_CHEFSI_TIMER
            this->chefsi_timer.filter_nloc.stop();
        #endif //ENABLE_CHEFSI_TIMER

        vscal = 2.0 * sigma2 / e;
        vscal2 = sigma * sigma2;
        #ifdef ENABLE_CHEFSI_TIMER
            this->chefsi_timer.filter_product.start();
        #endif //ENABLE_CHEFSI_TIMER
        #pragma omp parallel
        Linalg::scalar_product_general(eigen_vectors_temp, vscal, Nd * nb, eigen_vectors_m1, -vscal2);
        #ifdef ENABLE_CHEFSI_TIMER
            this->chefsi_timer.filter_product.stop();
        #endif //ENABLE_CHEFSI_TIMER

        std::swap(eigen_vectors_m1, eigen_vectors);
        std::swap(eigen_vectors, eigen_vectors_temp);

        sigma = sigma2;
    }

    stencil_temp.destructor_mp();

    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.filter.stop();
    #endif //ENABLE_CHEFSI_TIMER

    if (eigen_vectors != eigen_vectors_in) {
        #pragma omp parallel
        Linalg::set_value_general(eigen_vectors_in, eigen_vectors, Nd * nb);
        eigen_vectors = eigen_vectors_in;
    }
    assert(eigen_vectors == eigen_vectors_in);
    return;
}

template<typename T>
void Chefsi<T>::project_hamiltonian_mp(T const* const eigen_vectors, T const* const h_eigen_vectors, 
                                        T* const hp, T* const mp,
                                        const bool print_flag,
                                        Memory_pool<T, Fast_memory>& pool_fast,
                                        Memory_pool<T, Capacity_memory>& pool_cap) {
    (void) print_flag;
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.projection_gemm.start();
    #endif //ENABLE_CHEFSI_TIMER

    if (this->chefsi_control.projection_method == 0) {
        assert(this->domain_vertices.comm == MPI_COMM_SELF);
        const uint m = this->dp_domain_vertices.shared_vertices.get_nb();
        const uint k = this->dp_domain_vertices.local_vertices.Vertices_3D::get_size();
        if (Xlsdft_chefsi_lvtx::projection_contract_matches<T>(m, k)) {
            if constexpr (std::is_same_v<T, double>) {
                const std::size_t workspace_doubles =
                    Xlsdft_backend::projection_workspace_doubles();
                double* workspace = pool_fast.allocate(workspace_doubles);
                const int status = Xlsdft_chefsi_lvtx::project_hamiltonian(
                    eigen_vectors, h_eigen_vectors, hp, mp, workspace,
                    workspace_doubles);
                Xlsdft_backend::require_success(
                    Xlsdft_backend::Operation::dgemm_poj, status,
                    this->domain_vertices.comm,
                    {"projected_hamiltonian", m, m, k,
                     static_cast<std::int64_t>(k), static_cast<std::int64_t>(k),
                     static_cast<std::int64_t>(m), eigen_vectors, h_eigen_vectors,
                     hp, workspace, workspace_doubles});
            } else {
                Xlsdft_chefsi_lvtx::project_hamiltonian_fallback(
                    m, k, eigen_vectors, h_eigen_vectors, hp, mp);
            }
        } else {
            Xlsdft_chefsi_lvtx::project_hamiltonian_fallback(
                m, k, eigen_vectors, h_eigen_vectors, hp, mp);
        }
        #ifdef ENABLE_CHEFSI_TIMER
            this->chefsi_timer.projection_gemm.stop();
            this->chefsi_timer.projection_syrk.stop();
        #endif //ENABLE_CHEFSI_TIMER
    }  else {
        assert(this->chefsi_control.projection_method == 0);
    }

    return;
}

template<typename T>
void Chefsi<T>::subspace_diagonalization_mp(T* const __restrict__ hp, T* const __restrict__ mp,
                                         T* const __restrict__ eigen_values, const bool print_flag,
                                         Memory_pool<T, Fast_memory>& pool_fast,
                                         Memory_pool<T, Capacity_memory>& pool_cap) {
    (void) print_flag;
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    (void)pool_cap;

    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.diagonalization.start();
    #endif //ENABLE_CHEFSI_TIMER

    if (this->chefsi_control.projection_method == 0) {
        uint nstate = this->dp_domain_vertices.shared_vertices.get_nb();
        if (this->dp_domain_vertices.get_comm_rank() == 0) {
            const std::size_t workspace_doubles =
                Xlsdft_backend::dsygvd_workspace_doubles(
                    static_cast<int>(nstate));
            if constexpr (std::is_same_v<T, double>) {
                double* workspace = pool_fast.allocate(workspace_doubles);
                const int info = Xlsdft_chefsi_lvtx::dsygvd_upper(
                    static_cast<int>(nstate), hp, mp, eigen_values, workspace,
                    workspace_doubles);
                Xlsdft_backend::require_success(
                    Xlsdft_backend::Operation::dsygvd_upper, info,
                    this->dp_domain_vertices.comm,
                    {"generalized_upper_eigensolve", nstate, nstate, 0, nstate,
                     nstate, 0, hp, mp, eigen_values, workspace,
                     workspace_doubles});
            } else {
                const int info = Xlsdft_chefsi_lvtx::dsygvd_upper(
                    static_cast<int>(nstate), hp, mp, eigen_values, nullptr, 0);
                if (info != 0) {
                    assert(false);
                }
            }
        }
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        MPI_Bcast(eigen_values, nstate, mpi_datatype, 0, this->dp_domain_vertices.comm);
        if (this->dp_domain_vertices.local_vertices.Vertices_3D::get_size() > 0) {
            MPI_Bcast(hp, nstate * nstate, mpi_datatype, 0, this->dp_domain_vertices.domain_3d_comm);
        }
    } else {
        assert(this->chefsi_control.projection_method == 0);
    }

    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.diagonalization.stop();
    #endif //ENABLE_CHEFSI_TIMER

    return;
}

template<typename T>
void Chefsi<T>::subspace_diagonalization_mp_opt(
    T* const __restrict__ hp, T* const __restrict__ mp,
    T* const __restrict__ eigen_values, const bool print_flag,
    Memory_pool<T, Fast_memory>& pool_fast,
    Memory_pool<T, Capacity_memory>& pool_cap) {
    (void)print_flag;
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    (void)pool_cap;

#ifdef ENABLE_CHEFSI_TIMER
    this->chefsi_timer.diagonalization.start();
#endif

    if (this->chefsi_control.projection_method == 0) {
        uint nstate = this->dp_domain_vertices.shared_vertices.get_nb();
        if (this->dp_domain_vertices.get_comm_rank() == 0) {
            const std::size_t workspace_doubles =
                Xlsdft_backend::dsygvd_workspace_doubles(
                    static_cast<int>(nstate));
            if constexpr (std::is_same_v<T, double>) {
                double* workspace = pool_fast.allocate(workspace_doubles);
                const int info = Xlsdft_chefsi_lvtx::dsygvd_upper_opt(
                    static_cast<int>(nstate), hp, mp, eigen_values, workspace,
                    workspace_doubles);
                Xlsdft_backend::require_success(
                    Xlsdft_backend::Operation::dsygvd_upper, info,
                    this->dp_domain_vertices.comm,
                    {"generalized_upper_eigensolve_opt", nstate, nstate, 0,
                     nstate, nstate, 0, hp, mp, eigen_values, workspace,
                     workspace_doubles});
            } else {
                const int info = Xlsdft_chefsi_lvtx::dsygvd_upper_opt(
                    static_cast<int>(nstate), hp, mp, eigen_values, nullptr, 0);
                if (info != 0) {
                    assert(false);
                }
            }
        }
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        MPI_Bcast(eigen_values, nstate, mpi_datatype, 0,
                  this->dp_domain_vertices.comm);
        if (this->dp_domain_vertices.local_vertices.Vertices_3D::get_size() >
            0) {
            MPI_Bcast(hp, nstate * nstate, mpi_datatype, 0,
                      this->dp_domain_vertices.domain_3d_comm);
        }
    } else {
        assert(this->chefsi_control.projection_method == 0);
    }

#ifdef ENABLE_CHEFSI_TIMER
    this->chefsi_timer.diagonalization.stop();
#endif

    return;
}

template<typename T>
void Chefsi<T>::subspace_rotation_mp(T const* const __restrict__ eigen_vectors_in, T* const __restrict__ eigen_vectors_out,
                                    T const* const __restrict__ qp, const bool print_flag,
                                    Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap) {
    (void) print_flag;
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);

    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.rotation.start();
    #endif //ENABLE_CHEFSI_TIMER

    if (this->chefsi_control.projection_method == 0) {
        uint m = this->dp_domain_vertices.local_vertices.Vertices_3D::get_size();
        uint n = this->dp_domain_vertices.local_vertices.get_nb();
        assert(this->domain_vertices.comm == MPI_COMM_SELF);
        if (m > 0) {
            if (Xlsdft_chefsi_lvtx::rotation_contract_matches<T>(m, n)) {
                if constexpr (std::is_same_v<T, double>) {
                    const std::size_t workspace_doubles =
                        Xlsdft_backend::rotation_workspace_doubles();
                    double* const workspace =
                        pool_fast.allocate(workspace_doubles);
                    const int status = Xlsdft_chefsi_lvtx::rotate_subspace(
                        eigen_vectors_in, qp, eigen_vectors_out, workspace,
                        workspace_doubles);
                    Xlsdft_backend::require_success(
                        Xlsdft_backend::Operation::dgemm_rotation, status,
                        this->domain_vertices.comm,
                        {"subspace_rotation", m, n, n, m, n, m, eigen_vectors_in,
                         qp, eigen_vectors_out, workspace, workspace_doubles});
                } else {
                    Xlsdft_chefsi_lvtx::rotate_subspace_fallback(
                        m, n, eigen_vectors_in, qp, eigen_vectors_out);
                }
            } else {
                Xlsdft_chefsi_lvtx::rotate_subspace_fallback(
                    m, n, eigen_vectors_in, qp, eigen_vectors_out);
            }
        }
    } else {
        assert(this->chefsi_control.projection_method == 0);
    }

    #ifdef ENABLE_CHEFSI_TIMER
        this->chefsi_timer.rotation.stop();
    #endif //ENABLE_CHEFSI_TIMER

    return;
}

#define CHEFSI_DOUBLE_EXPLICIT_INST(T)                                                          \
  template void Chefsi<T>::run_mp(                                                              \
      T*&, T* const, T const* const, Effective_potential_nloc<T> const&, bool);                \
  template void Chefsi<T>::run_mp(                                                              \
      T*&, T*&, T* const, T const* const, Effective_potential_nloc<T> const&, bool,            \
      Memory_pool<T, Fast_memory>&, Memory_pool<T, Capacity_memory>&);                         \
  template void Chefsi<T>::chebyshev_filtering_column_wise_mp(                                  \
      T*&, T*&, T* const, T const* const, Effective_potential_nloc<T> const&, bool,            \
      Memory_pool<T, Fast_memory>&, Memory_pool<T, Capacity_memory>&);                          \
  template void Chefsi<T>::project_hamiltonian_mp(                                              \
      T const* const, T const* const, T* const, T* const, bool,                                \
      Memory_pool<T, Fast_memory>&, Memory_pool<T, Capacity_memory>&);                         \
  template void Chefsi<T>::subspace_diagonalization_mp(                                         \
      T* const, T* const, T* const, bool, Memory_pool<T, Fast_memory>&,                           \
      Memory_pool<T, Capacity_memory>&);                                                          \
  template void Chefsi<T>::subspace_diagonalization_mp_opt(                                     \
      T* const, T* const, T* const, bool, Memory_pool<T, Fast_memory>&,                           \
      Memory_pool<T, Capacity_memory>&);                                                          \
  template void Chefsi<T>::subspace_rotation_mp(                                                \
      T const* const, T* const, T const* const, bool,                                          \
      Memory_pool<T, Fast_memory>&, Memory_pool<T, Capacity_memory>&);

CHEFSI_DOUBLE_EXPLICIT_INST(float)
CHEFSI_DOUBLE_EXPLICIT_INST(double)
#undef CHEFSI_DOUBLE_EXPLICIT_INST
