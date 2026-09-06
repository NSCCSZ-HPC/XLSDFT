#include "aar.h"

template<typename T> Aar<T>::Aar(const Domain_parallel_vertices_3D& domain_vertice)
                               : domain_vertice(domain_vertice),
                                 res_exarr_mpi_package(this->domain_vertice) {}

template<typename T> Aar<T>::~Aar() {}

// template<typename T> void Aar<T>::run(Array_3D<T>& result, const Array_3D<T>& rhs, const MPI_Comm& comm) {
//     MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype(result.data);
//     const Vertices_3D local_vertices = result.get_vertices();
//     T tol = (T)this->aar_control.tolerance;
//     const int Nd = (int) local_vertices.get_size();
//     T r_2norm;

//     Array_3D<T> r(local_vertices);            // residual vector, r = b - Ax
//     Array_3D<T> x_old(local_vertices);
//     Array_3D<T> f(local_vertices, 0);            // preconditioned residual vector, f = inv(M) * r
//     Array_3D<T> f_old(local_vertices);

//     Array_4D<T> X(Vertices_4D(local_vertices, 0, this->aar_control.mixing_history - 1), 0);
//     Array_4D<T> F(X.get_vertices(), 0);

//     x_old = result;

//     T rhs_2norm = rhs.vector_norm_square_sum();
//     MPI_Allreduce(MPI_IN_PLACE, &rhs_2norm, 1, mpi_datatype, MPI_SUM, comm);
//     rhs_2norm = std::sqrt(rhs_2norm);

//     // init 
//     Array_3D<T> ex_result;
//     const Vertices_3D ex_vertice = local_vertices.generate_ex_vertices(this->res_stencil.FDn);

//     if (this->aar_control.residual_method == 0) {
//         ex_result.reconstructor(ex_vertice, 0);
//     }

//     // res
//     if (this->aar_control.residual_method == 0) {
//         this->res_exarr_mpi_package.fill_domain_par_ex_arr(result, ex_result);
//         Stencil_method::calc_laplacian(ex_result.data, ex_vertice, this->res_stencil, local_vertices, r.data, local_vertices,
//                                        rhs.data, (T)0.0, rhs.data, (T)1.0);
//         // r = ex_result.calc_laplacian(local_vertices, this->res_stencil, rhs, (T)0.0, rhs, (T)1.0);
//         // Stencil_method::calc_laplacian(ex_result.data, ex_vertice, this->res_stencil, local_vertices, r.data, local_vertices);
//         // Linalg::hadamard_plus_general(r.data, rhs.data, Nd);
//     }
    
//     tol *= rhs_2norm;
//     r_2norm = tol + 1.0;
//     uint iter = 0;
//     while (r_2norm > tol && iter < this->aar_control.max_iter) {
//         // *** calculate preconditioned residual f *** //
//         if (this->aar_control.precondition_method == 0) {
//             Stencil_method::jacobi_preconditioner(this->pre_stencil, Nd, (T)0.0, r.data, f.data);
//         } 
//         // *** store residual & iteration history *** //
//         if (iter > 0) {
//             int i_hist = (iter - 1) % this->aar_control.mixing_history;
//             //X.data + i_hist * Nd = result.data - x_old.data;
//             //F.data + i_hist * Nd = f.data - f_old.data;
//             Linalg::hadamard_minus_general(X.data + i_hist * Nd, result.data, x_old.data, Nd);
//             Linalg::hadamard_minus_general(F.data + i_hist * Nd, f.data, f_old.data, Nd);
//         }

//         x_old = result;
//         f_old = f;

//         if((iter + 1) % this->aar_control.anderson_frequency == 0) {
//             /***********************************
//              *  Anderson extrapolation update  *
//              ***********************************/
//             Mixing_method::AndersonExtrapolation(Nd, this->aar_control.mixing_history, result.data, x_old.data,
//                                                  f_old.data, X.data, F.data, this->aar_control.anderson_beta, comm);

//             if (this->aar_control.residual_method == 0) {
//                 this->res_exarr_mpi_package.fill_domain_par_ex_arr(result, ex_result);
//                 Stencil_method::calc_laplacian(ex_result.data, ex_vertice, this->res_stencil, local_vertices, r.data, local_vertices,
//                                                rhs.data, (T)0.0, rhs.data, (T)1.0);
//                 // r = ex_result.calc_laplacian(local_vertices, this->res_stencil, rhs, (T)0.0, rhs, (T)1.0);
//                 // Stencil_method::calc_laplacian(ex_result.data, ex_vertice, this->res_stencil, local_vertices, r.data, local_vertices);
//                 // Linalg::hadamard_plus_general(r.data, rhs.data, Nd);
//             }
            
//             r_2norm = r.vector_norm_square_sum();
//             MPI_Allreduce(MPI_IN_PLACE, &r_2norm, 1, mpi_datatype, MPI_SUM, comm);
//             r_2norm = std::sqrt(r_2norm);

//         } else {
//             /***********************
//              *  Richardson update  *
//              ***********************/
//             // result.data = f.data * (T)this->aar_control.omega + x_old.data
//             Linalg::scalar_product_general(result.data, f.data, (T)this->aar_control.richardson_omega, Nd, x_old.data);
//             // update residual r = b - Ax
//             if (this->aar_control.residual_method == 0) {
//                 this->res_exarr_mpi_package.fill_domain_par_ex_arr(result, ex_result);
//                 Stencil_method::calc_laplacian(ex_result.data, ex_vertice, this->res_stencil, local_vertices, r.data, local_vertices,
//                                                rhs.data, (T)0.0, rhs.data, (T)1.0);
//                 // r = ex_result.calc_laplacian(local_vertices, this->res_stencil, rhs, (T)0.0, rhs, (T)1.0);
//                 // Stencil_method::calc_laplacian(ex_result.data, ex_vertice, this->res_stencil, local_vertices, r.data, local_vertices);
//                 // Linalg::hadamard_plus_general(r.data, rhs.data, Nd);
//             }
//         }
//         iter++;
//     }
//     int rank;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     if (rank == 0) std::cout << "AAR solver took iter: " << iter << std::endl;
//     return;
// }

template<typename T> void Aar<T>::run(T* const __restrict__ result, T const* const __restrict__ rhs,
                                      const Vertices_3D& local_vertices, const MPI_Comm comm) {
    #ifdef USE_OPENMP
    this->run_ompunnested(result, rhs, local_vertices, comm);
    // T tol = (T)this->aar_control.tolerance;
    // const uint Nd = local_vertices.get_size();
    // T r_2norm;

    // T*& r = this->r;
    // T*& x_old = this->x_old;
    // T*& f = this->f;
    // T*& f_old = this->f_old;
    // T*& X = this->X;
    // T*& F = this->F;
    // #pragma omp single nowait
    // r = new T [Nd];
    // #pragma omp single nowait
    // x_old = new T [Nd];
    // #pragma omp single nowait
    // f = new T [Nd]();
    // #pragma omp single nowait
    // f_old = new T [Nd];
    // #pragma omp single nowait
    // X = new T [Nd * this->aar_control.mixing_history]();
    // #pragma omp single nowait
    // F = new T [Nd * this->aar_control.mixing_history]();
    // #pragma omp barrier

    // Vertices_3D ex_vertices;
    // ex_vertices.set_vertices(local_vertices.generate_ex_vertices(this->res_stencil.FDn));
    // #pragma omp parallel
    // {
    //     // x_old = result;
    //     Linalg::set_value_general(x_old, result, Nd);
    //     T rhs_2norm = Linalg::vector_norm_square_sum(rhs, Nd, comm);
    //     rhs_2norm = std::sqrt(rhs_2norm);

    //     // init
    //     T*& ex_result = this->ex_result;
    //     if (this->aar_control.residual_method == 0) {
    //         // ex_result.reconstructor(ex_vertice, 0);
    //         #pragma omp single
    //         ex_result = new T [ex_vertices.get_size()]();
    //     }

    //     // res
    //     if (this->aar_control.residual_method == 0) {
    //         this->res_exarr_mpi_package.fill_domain_par_ex_arr(result, local_vertices, ex_result, ex_vertices);
    //         #pragma omp barrier
    //         Stencil_method::calc_laplacian(ex_result, ex_vertices, this->res_stencil, local_vertices, r, local_vertices,
    //                                     rhs, (T)0.0, rhs, (T)1.0);
    //     }
    //     #pragma omp single
    //     {
    //         tol *= rhs_2norm;
    //         r_2norm = tol + 1.0;
    //     }
    // }

    // uint iter = 0;
    // while (r_2norm > tol && iter < this->aar_control.max_iter) {
    //     if (this->aar_control.precondition_method == 0 || this->aar_control.precondition_method == 1) {
    //         #pragma omp parallel
    //         {
    //         // *** calculate preconditioned residual f *** //
    //         if (this->aar_control.precondition_method == 0) {
    //             #pragma omp barrier
    //             Stencil_method::jacobi_preconditioner(this->pre_stencil, Nd, (T)0.0, r, f);
    //         } else if (this->aar_control.precondition_method == 1) {
    //             #pragma omp barrier
    //             Stencil_method::dst_poisson_preconditioner(this->dst_pre_data, Nd, r, f, this->pre_stencil);
    //         }
    //         // *** store residual & iteration history *** //
    //         if (iter > 0) {
    //             int i_hist = (iter - 1) % this->aar_control.mixing_history;
    //             //X.data + i_hist * Nd = result.data - x_old.data;
    //             //F.data + i_hist * Nd = f.data - f_old.data;
    //             Linalg::hadamard_minus_general(X + i_hist * Nd, result, x_old, Nd);
    //             Linalg::hadamard_minus_general(F + i_hist * Nd, f, f_old, Nd);
    //         }

    //         // x_old = result;
    //         // f_old = f;
    //         Linalg::set_value_general(x_old, result, Nd);
    //         // Linalg::set_value_general(f_old, f, Nd);
    //         #pragma omp barrier
    //         #pragma omp single
    //         {
    //         // std::swap(x_old, result);
    //         std::swap(f_old, f);
    //         }
    //         }
    //     } else if (this->aar_control.precondition_method == 2) {
    //         #ifdef USE_SSTRUCTMG
    //         this->sstruct_mg.solve(r, f);  // handle OpenMP by itself

    //         #pragma omp parallel
    //         {
    //         // *** store residual & iteration history *** //
    //         if (iter > 0) {
    //             int i_hist = (iter - 1) % this->aar_control.mixing_history;
    //             //X.data + i_hist * Nd = result.data - x_old.data;
    //             //F.data + i_hist * Nd = f.data - f_old.data;
    //             Linalg::hadamard_minus_general(X + i_hist * Nd, result, x_old, Nd);
    //             Linalg::hadamard_minus_general(F + i_hist * Nd, f, f_old, Nd);
    //         }

    //         // x_old = result;
    //         // f_old = f;
    //         Linalg::set_value_general(x_old, result, Nd);
    //         // Linalg::set_value_general(f_old, f, Nd);
    //         #pragma omp barrier
    //         #pragma omp single
    //         {
    //         // std::swap(x_old, result);
    //         std::swap(f_old, f);
    //         }
    //         }
    //         #endif  // USE_SSTRUCTMG
    //     }

    //     if((iter + 1) % this->aar_control.anderson_frequency == 0) {
    //         /***********************************
    //          *  Anderson extrapolation update  *
    //          ***********************************/
    //         Mixing_method::AndersonExtrapolation(Nd, this->aar_control.mixing_history, result, x_old,
    //                                              f_old, X, F, this->aar_control.anderson_beta, comm);
    //         #pragma omp parallel
    //         {
    //             if (this->aar_control.residual_method == 0) {
    //                 #pragma omp barrier
    //                 this->res_exarr_mpi_package.fill_domain_par_ex_arr(result, local_vertices, ex_result, ex_vertices);
    //                 #pragma omp barrier
    //                 Stencil_method::calc_laplacian(ex_result, ex_vertices, this->res_stencil, local_vertices, r, local_vertices,
    //                                             rhs, (T)0.0, rhs, (T)1.0);
    //             }
    //             #pragma omp barrier
    //             // r_2norm = r.vector_square_sum();
    //             T r_2norm_2 = Linalg::vector_norm_square_sum(r, Nd, comm);
    //             #pragma omp single
    //             r_2norm = std::sqrt(r_2norm_2);
    //         }
    //     } else {
    //         #pragma omp parallel
    //         {
    //             /***********************
    //              *  Richardson update  *
    //              ***********************/
    //             // result.data = f_old.data * (T)this->aar_control.omega + x_old.data
    //             Linalg::scalar_product_general(result, f_old, (T)this->aar_control.richardson_omega, Nd, x_old);

    //             // update residual r = b - Ax
    //             if (this->aar_control.residual_method == 0) {
    //                 #pragma omp barrier
    //                 this->res_exarr_mpi_package.fill_domain_par_ex_arr(result, local_vertices, ex_result, ex_vertices);
    //                 #pragma omp barrier
    //                 Stencil_method::calc_laplacian(ex_result, ex_vertices, this->res_stencil, local_vertices, r, local_vertices,
    //                                             rhs, (T)0.0, rhs, (T)1.0);
    //                 #pragma omp barrier
    //                 T r_2norm_2 = Linalg::vector_norm_square_sum(r, Nd, comm);
    //                 #pragma omp single
    //                 r_2norm = std::sqrt(r_2norm_2);
    //             }
    //         }
    //     }
    //     iter++;
    // }


    // // finalize
    // #pragma omp barrier
    // if (this->aar_control.residual_method == 0) {
    //     #pragma omp single nowait
    //     {
    //         delete [] ex_result;
    //         ex_result = nullptr;
    //     }
    // }
    // #pragma omp single nowait
    // {
    //     delete [] r;
    //     r = nullptr;
    // }
    // #pragma omp single nowait
    // {
    //     delete [] x_old;
    //     x_old = nullptr;
    // }
    // #pragma omp single nowait
    // {
    //     delete [] f;
    //     f = nullptr;
    // }
    // #pragma omp single nowait
    // {
    //     delete [] f_old;
    //     f_old = nullptr;
    // }
    // #pragma omp single nowait
    // {
    //     delete [] X;
    //     X = nullptr;
    // }
    // #pragma omp single nowait
    // {
    //     delete [] F;
    //     F = nullptr;
    // }
    // #pragma omp single nowait
    // {
    //     int rank;
    //     MPI_Comm_rank(comm, &rank);
    //     if (rank == 0) {
    //         this->print_runtime_result(iter, r_2norm);
    //     }
    // }
    // #pragma omp barrier

    #else //USE_OPENMP

    MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
    T tol = (T)this->aar_control.tolerance;
    const uint Nd = local_vertices.get_size();
    T r_2norm;
    
    // Array_3D<T> r(local_vertices);            // residual vector, r = b - Ax
    // Array_3D<T> x_old(local_vertices);
    // Array_3D<T> f(local_vertices, 0);            // preconditioned residual vector, f = inv(M) * r
    // Array_3D<T> f_old(local_vertices);

    // Array_4D<T> X(Vertices_4D(local_vertices, 0, this->aar_control.mixing_history - 1), 0);
    // Array_4D<T> F(X.get_vertices(), 0);

    T* r = new T[Nd];    // residual vector, r = b - Ax
    T* x_old = new T[Nd];
    T* f = new T[Nd] ();   // preconditioned residual vector, f = inv(M) * r
    T* f_old = new T[Nd];

    T* X = new T[Nd * this->aar_control.mixing_history] ();
    T* F = new T[Nd * this->aar_control.mixing_history] ();

    // x_old = result;
    Linalg::set_value_general(x_old, result, Nd);

    // T rhs_2norm = rhs.vector_square_sum();
    // MPI_Allreduce(MPI_IN_PLACE, &rhs_2norm, 1, mpi_datatype, MPI_SUM, comm);
    // rhs_2norm = std::sqrt(rhs_2norm);
    T rhs_2norm = Linalg::vector_norm_square_sum(rhs, Nd, MPI_COMM_NULL);
    MPI_Allreduce(MPI_IN_PLACE, &rhs_2norm, 1, mpi_datatype, MPI_SUM, comm);
    rhs_2norm = std::sqrt(rhs_2norm);

    // init 
    Vertices_3D ex_vertices;
    T* ex_result;

    if (this->aar_control.residual_method == 0) {
        ex_vertices.set_vertices(local_vertices.generate_ex_vertices(this->res_stencil.FDn));
        // ex_result.reconstructor(ex_vertice, 0);
        ex_result = new T [ex_vertices.get_size()] ();
    }

    // res
    if (this->aar_control.residual_method == 0) {
        this->res_exarr_mpi_package.fill_domain_par_ex_arr(result, local_vertices, ex_result, ex_vertices);
        Stencil_method::calc_laplacian(ex_result, ex_vertices, this->res_stencil, local_vertices, r, local_vertices,
                                       rhs, (T)0.0, rhs, (T)1.0);
    }

    tol *= rhs_2norm;
    r_2norm = tol + 1.0;
    uint iter = 0;
    while (r_2norm > tol && iter < this->aar_control.max_iter) {
        // *** calculate preconditioned residual f *** //
        if (this->aar_control.precondition_method == 0) {
            Stencil_method::jacobi_preconditioner(this->pre_stencil, Nd, (T)0.0, r, f);
        } else if (this->aar_control.precondition_method == 1) {
            Stencil_method::dst_poisson_preconditioner(this->dst_pre_data, Nd, r, f, this->pre_stencil);
        } else if (this->aar_control.precondition_method == 2) {
            #ifdef USE_SSTRUCTMG
            this->sstruct_mg.solve(r, f);
            #endif
        }
        // *** store residual & iteration history *** //
        if (iter > 0) {
            int i_hist = (iter - 1) % this->aar_control.mixing_history;
            //X.data + i_hist * Nd = result.data - x_old.data;
            //F.data + i_hist * Nd = f.data - f_old.data;
            Linalg::hadamard_minus_general(X + i_hist * Nd, result, x_old, Nd);
            Linalg::hadamard_minus_general(F + i_hist * Nd, f, f_old, Nd);
        }

        // x_old = result;
        // f_old = f;
        Linalg::set_value_general(x_old, result, Nd);
        // Linalg::set_value_general(f_old, f, Nd);
        // std::swap(x_old, result);
        std::swap(f_old, f);

        if((iter + 1) % this->aar_control.anderson_frequency == 0) {
            /***********************************
             *  Anderson extrapolation update  *
             ***********************************/
            Mixing_method::AndersonExtrapolation(Nd, this->aar_control.mixing_history, result, x_old,
                                                 f_old, X, F, T(this->aar_control.anderson_beta), comm);

            if (this->aar_control.residual_method == 0) {
                this->res_exarr_mpi_package.fill_domain_par_ex_arr(result, local_vertices, ex_result, ex_vertices);
                Stencil_method::calc_laplacian(ex_result, ex_vertices, this->res_stencil, local_vertices, r, local_vertices,
                                               rhs, (T)0.0, rhs, (T)1.0);
            }
            
            // r_2norm = r.vector_square_sum();
            r_2norm = Linalg::vector_norm_square_sum(r, Nd, MPI_COMM_NULL);
            MPI_Allreduce(MPI_IN_PLACE, &r_2norm, 1, mpi_datatype, MPI_SUM, comm);
            r_2norm = std::sqrt(r_2norm);

        } else {
            /***********************
             *  Richardson update  *
             ***********************/
            // result.data = f_old.data * (T)this->aar_control.omega + x_old.data
            Linalg::scalar_product_general(result, f_old, (T)this->aar_control.richardson_omega, Nd, x_old);
            // update residual r = b - Ax
            if (this->aar_control.residual_method == 0) {
                this->res_exarr_mpi_package.fill_domain_par_ex_arr(result, local_vertices, ex_result, ex_vertices);
                Stencil_method::calc_laplacian(ex_result, ex_vertices, this->res_stencil, local_vertices, r, local_vertices,
                                               rhs, (T)0.0, rhs, (T)1.0);
                r_2norm = Linalg::vector_norm_square_sum(r, Nd, comm);
                r_2norm = std::sqrt(r_2norm);
            }
        }
        iter++;
    }

    // finalize
    if (this->aar_control.residual_method == 0) {
        delete [] ex_result;
    }
    delete [] r;
    delete [] x_old;
    delete [] f;
    delete [] f_old;
    delete [] X;
    delete [] F;
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        this->print_runtime_result(iter, r_2norm);
    }
    // if (rank == 0) std::cout << "AAR solver took iter: " << iter << std::endl;
    #endif //USE_OPENMP
    return;
}

template<typename T> void Aar<T>::run_ompunnested(T* const __restrict__ result, T const* const __restrict__ rhs,
                                      const Vertices_3D& local_vertices, const MPI_Comm comm) {

    int rank;
    MPI_Comm_rank(comm, &rank);

    T tol = (T)this->aar_control.tolerance;
    const uint Nd = local_vertices.get_size();
    T r_2norm;

    T* r = new T [Nd];
    T* x_old = new T [Nd];
    T* f = new T [Nd]();
    T* f_old = new T [Nd];
    T* X = new T [Nd * this->aar_control.mixing_history]();
    T* F = new T [Nd * this->aar_control.mixing_history]();
    T* ex_result = nullptr;
    Vertices_3D ex_vertices;
    ex_vertices.set_vertices(local_vertices.generate_ex_vertices(this->res_stencil.FDn));
    if (this->aar_control.residual_method == 0) {
        ex_result = new T [ex_vertices.get_size()]();
    }
    
    #pragma omp parallel
    {
        // x_old = result;
        Linalg::set_value_general(x_old, result, Nd);
        T rhs_2norm = Linalg::vector_norm_square_sum(rhs, Nd, comm);
        rhs_2norm = std::sqrt(rhs_2norm);

        // res
        if (this->aar_control.residual_method == 0) {
            this->res_exarr_mpi_package.fill_domain_par_ex_arr(result, local_vertices, ex_result, ex_vertices);
            #pragma omp barrier
            Stencil_method::calc_laplacian(ex_result, ex_vertices, this->res_stencil, local_vertices, r, local_vertices,
                                        rhs, (T)0.0, rhs, (T)1.0);
        }
        #pragma omp single
        {
            tol *= rhs_2norm;
            r_2norm = tol + 1.0;
        }
    }

    uint iter = 0;
    while (r_2norm > tol && iter < this->aar_control.max_iter) {
        if (this->aar_control.precondition_method == 0 || this->aar_control.precondition_method == 1) {
            #pragma omp parallel
            {
            // *** calculate preconditioned residual f *** //
            if (this->aar_control.precondition_method == 0) {
                #pragma omp barrier
                Stencil_method::jacobi_preconditioner(this->pre_stencil, Nd, (T)0.0, r, f);
            } else if (this->aar_control.precondition_method == 1) {
                #pragma omp barrier
                Stencil_method::dst_poisson_preconditioner(this->dst_pre_data, Nd, r, f, this->pre_stencil);
            }
            // *** store residual & iteration history *** //
            if (iter > 0) {
                int i_hist = (iter - 1) % this->aar_control.mixing_history;
                //X.data + i_hist * Nd = result.data - x_old.data;
                //F.data + i_hist * Nd = f.data - f_old.data;
                Linalg::hadamard_minus_general(X + i_hist * Nd, result, x_old, Nd);
                Linalg::hadamard_minus_general(F + i_hist * Nd, f, f_old, Nd);
            }

            // x_old = result;
            // f_old = f;
            Linalg::set_value_general(x_old, result, Nd);
            // Linalg::set_value_general(f_old, f, Nd);
            #pragma omp barrier
            #pragma omp single
            {
            // std::swap(x_old, result);
            std::swap(f_old, f);
            }
            }
        } else if (this->aar_control.precondition_method == 2) {
            #ifdef USE_SSTRUCTMG
            this->sstruct_mg.solve(r, f);  // handle OpenMP by itself

            #pragma omp parallel
            {
            // *** store residual & iteration history *** //
            if (iter > 0) {
                int i_hist = (iter - 1) % this->aar_control.mixing_history;
                //X.data + i_hist * Nd = result.data - x_old.data;
                //F.data + i_hist * Nd = f.data - f_old.data;
                Linalg::hadamard_minus_general(X + i_hist * Nd, result, x_old, Nd);
                Linalg::hadamard_minus_general(F + i_hist * Nd, f, f_old, Nd);
            }

            // x_old = result;
            // f_old = f;
            Linalg::set_value_general(x_old, result, Nd);
            // Linalg::set_value_general(f_old, f, Nd);
            #pragma omp barrier
            #pragma omp single
            {
            // std::swap(x_old, result);
            std::swap(f_old, f);
            }
            }
            #endif  // USE_SSTRUCTMG
        }

        if((iter + 1) % this->aar_control.anderson_frequency == 0) {
            /***********************************
             *  Anderson extrapolation update  *
             ***********************************/
            Mixing_method::AndersonExtrapolation_ompunnested(Nd, this->aar_control.mixing_history, result, x_old,
                                                 f_old, X, F, T(this->aar_control.anderson_beta), comm);
            #pragma omp parallel
            {
                if (this->aar_control.residual_method == 0) {
                    #pragma omp barrier
                    this->res_exarr_mpi_package.fill_domain_par_ex_arr(result, local_vertices, ex_result, ex_vertices);
                    #pragma omp barrier
                    Stencil_method::calc_laplacian(ex_result, ex_vertices, this->res_stencil, local_vertices, r, local_vertices,
                                                rhs, (T)0.0, rhs, (T)1.0);
                }
                #pragma omp barrier
                // r_2norm = r.vector_square_sum();
                T r_2norm_2 = Linalg::vector_norm_square_sum(r, Nd, comm);
                #pragma omp single
                r_2norm = std::sqrt(r_2norm_2);
            }
        } else {
            #pragma omp parallel
            {
                /***********************
                 *  Richardson update  *
                 ***********************/
                // result.data = f_old.data * (T)this->aar_control.omega + x_old.data
                Linalg::scalar_product_general(result, f_old, (T)this->aar_control.richardson_omega, Nd, x_old);

                // update residual r = b - Ax
                if (this->aar_control.residual_method == 0) {
                    #pragma omp barrier
                    this->res_exarr_mpi_package.fill_domain_par_ex_arr(result, local_vertices, ex_result, ex_vertices);
                    #pragma omp barrier
                    Stencil_method::calc_laplacian(ex_result, ex_vertices, this->res_stencil, local_vertices, r, local_vertices,
                                                rhs, (T)0.0, rhs, (T)1.0);
                    #pragma omp barrier
                    T r_2norm_2 = Linalg::vector_norm_square_sum(r, Nd, comm);
                    #pragma omp single
                    r_2norm = std::sqrt(r_2norm_2);
                }
            }
        }
        // if (rank == 0) {
        //     std::string s = "_" + std::to_string(iter);
        //     Linalg::print_vectorX(r, Nd, "r" + s);
        //     Linalg::print_vectorX(x_old, Nd, "x_old" + s);
        //     Linalg::print_vectorX(f, Nd, "f" + s);
        //     Linalg::print_vectorX(f_old, Nd, "f_old" + s);
        //     Linalg::print_vectorX(X, Nd * this->aar_control.mixing_history, "X" + s);
        //     Linalg::print_vectorX(F, Nd * this->aar_control.mixing_history, "F" + s);
        //     Linalg::print_vectorX(ex_result, ex_vertices.get_size(), "ex_result" + s);
        // }
        iter++;
    }


    // finalize
    if (this->aar_control.residual_method == 0) {
        delete [] ex_result;
        ex_result = nullptr;
    }
    delete [] r;
    r = nullptr;
    delete [] x_old;
    x_old = nullptr;
    delete [] f;
    f = nullptr;
    delete [] f_old;
    f_old = nullptr;
    delete [] X;
    X = nullptr;
    delete [] F;
    F = nullptr;

    if (rank == 0) {
        this->print_runtime_result(iter, r_2norm);
    }

    return;
}

template<typename T> void Aar<T>::run_ompunnested_mp(T* const __restrict__ result, T const* const __restrict__ rhs,
                                      const Vertices_3D& local_vertices, const MPI_Comm comm,
                                    Memory_pool<T, Fast_memory>& pool_fast, Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);

    int rank;
    MPI_Comm_rank(comm, &rank);

    T tol = (T)this->aar_control.tolerance;
    const uint Nd = local_vertices.get_size();
    T r_2norm;

    // T* r = new T [Nd];
    // T* x_old = new T [Nd];
    // T* f = new T [Nd]();
    // T* f_old = new T [Nd];
    // T* X = new T [Nd * this->aar_control.mixing_history]();
    // T* F = new T [Nd * this->aar_control.mixing_history]();
    T* r = pool_fast.allocate(Nd);
    T* x_old = pool_fast.allocate(Nd);
    T* f = pool_fast.allocate(Nd);
    #pragma omp parallel
    Linalg::set_value_general(f, T(0), Nd);
    T* f_old = pool_fast.allocate(Nd);

    const uint X_length = Nd * this->aar_control.mixing_history;
    T* X = pool_fast.allocate(X_length);
    T* F = pool_fast.allocate(X_length);
    #pragma omp parallel
    {
        Linalg::set_value_general(X, T(0), X_length);
        Linalg::set_value_general(F, T(0), X_length);
    }

    T* ex_result = nullptr;
    const Vertices_3D ex_vertices(local_vertices.generate_ex_vertices(this->res_stencil.FDn));
    if (this->aar_control.residual_method == 0) {
        // ex_result = new T [ex_vertices.get_size()]();
        ex_result = pool_fast.allocate(ex_vertices.get_size());
        #pragma omp parallel
        Linalg::set_value_general(ex_result, T(0), ex_vertices.get_size());
    }

    #pragma omp parallel
    {
        // x_old = result;
        Linalg::set_value_general(x_old, result, Nd);
        T rhs_2norm = Linalg::vector_norm_square_sum(rhs, Nd, comm);
        rhs_2norm = std::sqrt(rhs_2norm);

        // res
        if (this->aar_control.residual_method == 0) {
            this->res_exarr_mpi_package.fill_domain_par_ex_arr(result, local_vertices, ex_result, ex_vertices);
            #pragma omp barrier
            Stencil_method::calc_laplacian(ex_result, ex_vertices, this->res_stencil, local_vertices, r, local_vertices,
                                        rhs, (T)0.0, rhs, (T)1.0);
        }
        #pragma omp single
        {
            tol *= rhs_2norm;
            r_2norm = tol + 1.0;
        }
    }

    uint iter = 0;
    while (r_2norm > tol && iter < this->aar_control.max_iter) {
        if (this->aar_control.precondition_method == 0 || this->aar_control.precondition_method == 1) {
            #pragma omp parallel
            {
            // *** calculate preconditioned residual f *** //
            if (this->aar_control.precondition_method == 0) {
                #pragma omp barrier
                Stencil_method::jacobi_preconditioner(this->pre_stencil, Nd, (T)0.0, r, f);
            } else if (this->aar_control.precondition_method == 1) {
                #pragma omp barrier
                Stencil_method::dst_poisson_preconditioner(this->dst_pre_data, Nd, r, f, this->pre_stencil);
            }
            // *** store residual & iteration history *** //
            if (iter > 0) {
                int i_hist = (iter - 1) % this->aar_control.mixing_history;
                //X.data + i_hist * Nd = result.data - x_old.data;
                //F.data + i_hist * Nd = f.data - f_old.data;
                Linalg::hadamard_minus_general(X + i_hist * Nd, result, x_old, Nd);
                Linalg::hadamard_minus_general(F + i_hist * Nd, f, f_old, Nd);
            }

            // x_old = result;
            // f_old = f;
            Linalg::set_value_general(x_old, result, Nd);
            // Linalg::set_value_general(f_old, f, Nd);
            #pragma omp barrier
            #pragma omp single
            {
            // std::swap(x_old, result);
            std::swap(f_old, f);
            }
            }
        } else if (this->aar_control.precondition_method == 2) {
            #ifdef USE_SSTRUCTMG
            this->sstruct_mg.solve(r, f);  // handle OpenMP by itself

            #pragma omp parallel
            {
            // *** store residual & iteration history *** //
            if (iter > 0) {
                int i_hist = (iter - 1) % this->aar_control.mixing_history;
                //X.data + i_hist * Nd = result.data - x_old.data;
                //F.data + i_hist * Nd = f.data - f_old.data;
                Linalg::hadamard_minus_general(X + i_hist * Nd, result, x_old, Nd);
                Linalg::hadamard_minus_general(F + i_hist * Nd, f, f_old, Nd);
            }

            // x_old = result;
            // f_old = f;
            Linalg::set_value_general(x_old, result, Nd);
            // Linalg::set_value_general(f_old, f, Nd);
            #pragma omp barrier
            #pragma omp single
            {
            // std::swap(x_old, result);
            std::swap(f_old, f);
            }
            }
            #endif  // USE_SSTRUCTMG
        }

        if((iter + 1) % this->aar_control.anderson_frequency == 0) {
            /***********************************
             *  Anderson extrapolation update  *
             ***********************************/
            Mixing_method::AndersonExtrapolation_ompunnested_mp(Nd, this->aar_control.mixing_history, result, x_old,
                                                 f_old, X, F, T(this->aar_control.anderson_beta), comm, pool_fast, pool_cap);
            #pragma omp parallel
            {
                if (this->aar_control.residual_method == 0) {
                    #pragma omp barrier
                    this->res_exarr_mpi_package.fill_domain_par_ex_arr(result, local_vertices, ex_result, ex_vertices);
                    #pragma omp barrier
                    Stencil_method::calc_laplacian(ex_result, ex_vertices, this->res_stencil, local_vertices, r, local_vertices,
                                                rhs, (T)0.0, rhs, (T)1.0);
                }
                #pragma omp barrier
                // r_2norm = r.vector_square_sum();
                T r_2norm_2 = Linalg::vector_norm_square_sum(r, Nd, comm);
                #pragma omp single
                r_2norm = std::sqrt(r_2norm_2);
            }
        } else {
            #pragma omp parallel
            {
                /***********************
                 *  Richardson update  *
                 ***********************/
                // result.data = f_old.data * (T)this->aar_control.omega + x_old.data
                Linalg::scalar_product_general(result, f_old, (T)this->aar_control.richardson_omega, Nd, x_old);

                // update residual r = b - Ax
                if (this->aar_control.residual_method == 0) {
                    #pragma omp barrier
                    this->res_exarr_mpi_package.fill_domain_par_ex_arr(result, local_vertices, ex_result, ex_vertices);
                    #pragma omp barrier
                    Stencil_method::calc_laplacian(ex_result, ex_vertices, this->res_stencil, local_vertices, r, local_vertices,
                                                rhs, (T)0.0, rhs, (T)1.0);
                    #pragma omp barrier
                    T r_2norm_2 = Linalg::vector_norm_square_sum(r, Nd, comm);
                    #pragma omp single
                    r_2norm = std::sqrt(r_2norm_2);
                }
            }
        }
        // if (rank == 0) {
        //     std::string s = "_" + std::to_string(iter);
        //     Linalg::print_vectorX(r, Nd, "r" + s);
        //     Linalg::print_vectorX(x_old, Nd, "x_old" + s);
        //     Linalg::print_vectorX(f, Nd, "f" + s);
        //     Linalg::print_vectorX(f_old, Nd, "f_old" + s);
        //     Linalg::print_vectorX(X, Nd * this->aar_control.mixing_history, "X" + s);
        //     Linalg::print_vectorX(F, Nd * this->aar_control.mixing_history, "F" + s);
        //     Linalg::print_vectorX(ex_result, ex_vertices.get_size(), "ex_result" + s);
        // }
        iter++;
    }

    if (rank == 0) {
        this->print_runtime_result(iter, r_2norm);
    }

    return;
}

template<typename T> 
void Aar<T>::print_runtime_result(const uint iter, const T norm2) {
    std::cout << YELLOW << "AAR :: " << RESET <<std::endl;
    if (this->aar_control.precondition_method == 0) {
        std::cout << YELLOW << "JACOBI PRECONDTION "; 
    } else if (this->aar_control.precondition_method == 1) {
        std::cout << YELLOW << "DST PRECONDTION "; 
    } else {
        std::cout << YELLOW << "MG PRECONDTION "; 
    }
    std::cout << "with omega = " << std::setprecision(4) << std::fixed << this->aar_control.richardson_omega
              << ", beta = " << std::setprecision(4) << std::fixed << this->aar_control.anderson_beta
              << ", tolerance = " << std::setprecision(4) << std::fixed << this->aar_control.tolerance
              << "." << RESET << std::endl;
    if (iter == aar_control.max_iter) {
        std::cout << RED <<"Not converge with iter = "
                  << iter << ", 2norm = "
                  << std::setprecision(6) << std::scientific << norm2
                  << "." << RESET << std::endl;
    } else {
        std::cout << YELLOW << "Converge with iter = "
                  << iter << ", 2norm = "
                  << std::setprecision(6) << std::scientific << norm2
                  << "." << RESET << std::endl;
    }
    return;
}

template<typename T> void Aar<T>::init() {
    return;
}

template<typename T>
template<typename T2>
void Aar<T>::init(const Aar<T2>& aar) {
    this->aar_control = aar.aar_control;
    //res
    this->res_stencil.init(aar.res_stencil);
    this->res_exarr_mpi_package.init(aar.res_exarr_mpi_package);
    //pre
    this->pre_stencil.init(aar.pre_stencil);
    if (this->aar_control.precondition_method == 1) {
        this->dst_pre_data.init(aar.dst_pre_data);
    } else if (this->aar_control.precondition_method == 2) {
        printf("[ERROR] NOT IMPLEMENTED YET\n");
        exit(-1);
    }
    return;
}
template void Aar<float>::init(const Aar<float>& aar);
template void Aar<double>::init(const Aar<double>& aar);
template void Aar<float>::init(const Aar<double>& aar);
template void Aar<double>::init(const Aar<float>& aar);

template<typename T> void Aar<T>::init_res_method0(const Stencil<T>& stencil,
                                                   const Exarr_3D_mpi_package& exarr_mpi_package,
                                                   const double& ratio,
                                                   const double& shfit) {
    this->res_stencil = stencil.coeffs_scale(ratio, 2);
    this->res_stencil.shift_D2_coeffs(shfit);
    // this->res_exarr_mpi_package = exarr_mpi_package;
    this->res_exarr_mpi_package.init(exarr_mpi_package);
    return;
}

template<typename T> void Aar<T>::init_pre_method0(const Stencil<T>& stencil,
                                                   const double& ratio,
                                                   const double& shfit) {
    this->pre_stencil = stencil.coeffs_scale(ratio, 2);
    this->pre_stencil.shift_D2_coeffs(shfit);
    return;
}

template<typename T> void Aar<T>::init_pre_dst() {
    this->dst_pre_data.init(this->domain_vertice.get_3D_local_vertices(), this->pre_stencil);
    return;
}

template<typename T> void Aar<T>::des_pre_dst() {
    this->dst_pre_data.destructor();
    return;
}

template<typename T> void Aar<T>::init_pre_mg(const Mesh_control& mesh_control) {
    #ifdef USE_SSTRUCTMG
    #define F(x) ((x)>=0?(x):-(x))
    #define idx_t int
    const idx_t ndim = 3;
    const idx_t num_diag = 37;
    const idx_t radius = 6;
    // Grid setup
    idx_t glb_dims[3], my_ilower[3], my_iupper[3];
    glb_dims[0] = domain_vertice.shared_vertices.get_ke()-domain_vertice.shared_vertices.get_ks()+1;  // reverse
    glb_dims[1] = domain_vertice.shared_vertices.get_je()-domain_vertice.shared_vertices.get_js()+1;
    glb_dims[2] = domain_vertice.shared_vertices.get_ie()-domain_vertice.shared_vertices.get_is()+1;
    my_ilower[0] = domain_vertice.local_vertices.get_ks();  // outer
    my_iupper[0] = domain_vertice.local_vertices.get_ke();  // (include)
    my_ilower[1] = domain_vertice.local_vertices.get_js();
    my_iupper[1] = domain_vertice.local_vertices.get_je();
    my_ilower[2] = domain_vertice.local_vertices.get_is();  // inner
    my_iupper[2] = domain_vertice.local_vertices.get_ie();
    bool periodic[3];
    for (idx_t d = 0; d < ndim; ++d) periodic[d] = mesh_control.is_periodic[2-d];  // reverse

    // Stencil setup
    T star_3d37_vals[num_diag];  // Laplace. construct stencil weight by the way
    // i.e., 1/4pi * Laplace
    std::vector<T> D2_coeffs_i0(this->pre_stencil.get_D2_coeffs_z(), this->pre_stencil.get_D2_coeffs_z()+radius+1);  // (outer)
    std::vector<T> D2_coeffs_i1(this->pre_stencil.get_D2_coeffs_y(), this->pre_stencil.get_D2_coeffs_y()+radius+1);
    std::vector<T> D2_coeffs_i2(this->pre_stencil.get_D2_coeffs_x(), this->pre_stencil.get_D2_coeffs_x()+radius+1);  // (inner)

    assert(F(D2_coeffs_i0[0])+F(D2_coeffs_i1[0]+F(D2_coeffs_i2[0])) > 1e-4);
    star_3d37_vals[num_diag >> 1] = D2_coeffs_i0[0] + D2_coeffs_i1[0] + D2_coeffs_i2[0];  // center
    D2_coeffs_i0[0] = D2_coeffs_i1[0] = D2_coeffs_i2[0] = 0.0;
    // assert(F(this->pre_stencil.D2_coeffs_x[0])+F(this->pre_stencil.D2_coeffs_y[0])+F(this->pre_stencil.D2_coeffs_z[0]) > 1e-4);
    T precond_stencil_value[7] = {0, 0, 0, 0, 0, 0, 0};
    for (idx_t i0 = -radius, vcnt = 0; i0 <= radius; ++i0) {
        for (idx_t i1 = -radius; i1 <= radius; ++i1) {
            for (idx_t i2 = -radius; i2 <= radius; ++i2) {
                if ((i0 != 0) + (i1 != 0) + (i2 != 0) != 1) continue;  // star stencil
                T x = D2_coeffs_i0[F(i0)] + D2_coeffs_i1[F(i1)] + D2_coeffs_i2[F(i2)];
                star_3d37_vals[vcnt] = x;


                // 计算半径的平方 k^2 = dx^2 + dy^2 + dz^2
                int k2 = i0*i0 + i1*i1 + i2*i2;

                // 利用高阶泰勒展开系数公式： c_low = sum(k^2 * c_high_k)
                if (i0 < 0) {
                    precond_stencil_value[0] += k2 * x;
                } else if (i1 < 0) {
                    precond_stencil_value[1] += k2 * x;
                } else if (i2 < 0) {
                    precond_stencil_value[2] += k2 * x;
                } else if (i2 > 0) {
                    precond_stencil_value[4] += k2 * x;
                } else if (i1 > 0) {
                    precond_stencil_value[5] += k2 * x;
                } else if (i0 > 0) {
                    precond_stencil_value[6] += k2 * x;
                }


                vcnt++;
                if (vcnt == (num_diag >> 1)) vcnt++;  // skip center
            }
        }
    }
    // 计算中心点的值：保证行和为0
    precond_stencil_value[3] = -(precond_stencil_value[0] + precond_stencil_value[1] + 
                        precond_stencil_value[2] + precond_stencil_value[4] + 
                        precond_stencil_value[5] + precond_stencil_value[6]);

    TEST_CONFIG config;
    config.use_in_memory_mg_config = true;
    config.print_level = 0;


        // config.rtol = 1e-6;
        // config.max_iter = 200;
        // config.config_mg_file = config_mg_file;
        // config.print_level = 2;
        // config.its_name = "GMRES";


    #ifdef USE_LOCAL_MG
    glb_dims[0] = my_iupper[0]-my_ilower[0]+1;
    glb_dims[1] = my_iupper[1]-my_ilower[1]+1;
    glb_dims[2] = my_iupper[2]-my_ilower[2]+1;
    periodic[0] = periodic[1] = periodic[2] = false;
    my_ilower[0] = my_ilower[1] = my_ilower[2] = 0;
    my_iupper[0] = glb_dims[0]-1; my_iupper[1] = glb_dims[1]-1; my_iupper[2] = glb_dims[2]-1;
    #endif

    int num_proc = domain_vertice.get_comm_size();
    std::vector<idx_t> glb_begs(3 * num_proc);
    std::vector<idx_t> glb_ends(3 * num_proc);
    bool fine_grid_all_active = true;
    #ifdef USE_LOCAL_MG
    {
        int ptr = 0;
        Vertices_3D vert = domain_vertice.generate_local_vertices();
        glb_begs[ptr*3+0] = 0;
        glb_begs[ptr*3+1] = 0;
        glb_begs[ptr*3+2] = 0;
        glb_ends[ptr*3+0] = vert.get_nk();
        glb_ends[ptr*3+1] = vert.get_nj();
        glb_ends[ptr*3+2] = vert.get_ni();
    }
    #else
    for (int i = 0, ptr = 0; i < num_proc; ++i) {
        Vertices_3D vert = domain_vertice.generate_local_vertices(i);
        if (vert.get_ks() > vert.get_ke() ||
            vert.get_js() > vert.get_je() ||
            vert.get_is() > vert.get_ie()) {
            fine_grid_all_active = false;
            continue;
        }
        glb_begs[ptr*3+0] = vert.get_ks();
        glb_begs[ptr*3+1] = vert.get_js();
        glb_begs[ptr*3+2] = vert.get_is();
        glb_ends[ptr*3+0] = vert.get_ke()+1;
        glb_ends[ptr*3+1] = vert.get_je()+1;
        glb_ends[ptr*3+2] = vert.get_ie()+1;
        ++ptr;
    }
    #endif

    MPI_Comm mg_comm = domain_vertice.get_mpi_comm();
    #ifdef USE_LOCAL_MG
    mg_comm = MPI_COMM_SELF;
    #endif

    #ifdef USE_MG_3d7
    sstruct_mg.init(mg_comm, glb_dims, periodic, glb_begs.data(), glb_ends.data(), my_ilower, my_iupper, star_3d37_vals, precond_stencil_value, config, true, fine_grid_all_active);
    #else
    sstruct_mg.init(mg_comm, glb_dims, periodic, glb_begs.data(), glb_ends.data(), my_ilower, my_iupper, star_3d37_vals, nullptr, config, true, fine_grid_all_active);
    #endif

    // #ifdef USE_MG_3d7
    // sstruct_mg.init(mg_comm, glb_dims, periodic, my_ilower, my_iupper, star_3d37_vals, precond_stencil_value, config, false);
    // #else
    // sstruct_mg.init(mg_comm, glb_dims, periodic, my_ilower, my_iupper, star_3d37_vals, nullptr, config, false);
    // #endif

    #undef F
    #undef idx_t
    #else
    assert(false && "MG preconditioner not implemented yet");
    (void) mesh_control;  // avoid unused parameter warning
    #endif

    return;
}

template<typename T>
void Aar<T>::destructor() {
    this->res_stencil.destructor();
    this->res_exarr_mpi_package.destructor();
    this->pre_stencil.destructor();
    if (this->aar_control.precondition_method == 1) {
        this->dst_pre_data.destructor();
    }  // TODO: MG destructor
    return;
}

template<typename T>
void Aar<T>::show() const {
    this->aar_control.show();
    return;
}

template class Aar<float>;
template class Aar<double>;
