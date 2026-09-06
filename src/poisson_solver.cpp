#include "poisson_solver.h"

#ifdef USE_HYPRE
// fake functions to cheat with compiler
HYPRE_Int HYPRE_SStructVectorSetBoxValues(HYPRE_SStructVector vector, HYPRE_Int part, HYPRE_Int *ilower, HYPRE_Int *iupper, HYPRE_Int var, float* values) {
    (void) vector;
    (void) part;
    (void) ilower;
    (void) iupper;
    (void) var;
    (void) values;
    printf("HYPREERROR: FLOAT NOT SUPPORTED YET\n");
    return -1;
}
HYPRE_Int HYPRE_SStructMatrixSetBoxValues(HYPRE_SStructMatrix matrix, HYPRE_Int part, HYPRE_Int *ilower, HYPRE_Int *iupper, HYPRE_Int var, HYPRE_Int nentries, HYPRE_Int *entries, float *values) {
    (void) matrix;
    (void) part;
    (void) ilower;
    (void) iupper;
    (void) var;
    (void) nentries;
    (void) entries;
    (void) values;
    printf("HYPREERROR: FLOAT NOT SUPPORTED YET\n");
    return -1;
}
HYPRE_Int HYPRE_SStructVectorGetBoxValues(HYPRE_SStructVector vector, HYPRE_Int part, HYPRE_Int *ilower, HYPRE_Int *iupper, HYPRE_Int var, float *values) {
    (void) vector;
    (void) part;
    (void) ilower;
    (void) iupper;
    (void) var;
    (void) values;
    printf("HYPREERROR: FLOAT NOT SUPPORTED YET\n");
    return -1;
}
HYPRE_Int HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(HYPRE_Solver solver, float *norm) {
    (void) solver;
    (void) norm;
    printf("HYPREERROR: FLOAT NOT SUPPORTED YET\n");
    return -1;
}
#endif


#include <sys/resource.h>
void print_mem_usage(std::string note_str="") {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    
    // Linux 下 ru_maxrss 的单位是 KB
    long max_rss = usage.ru_maxrss; 

    long global_max = 0;
    long global_sum = 0;

    // 使用 MPI 聚合所有进程的峰值
    MPI_Reduce(&max_rss, &global_max, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&max_rss, &global_sum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "========== Memory Report (Peak) of " << note_str << " ==========" << std::endl;
        std::cout << "Max Single Process Peak: " << global_max / 1024.0 << " MB" << std::endl;
        std::cout << "Global Peak Sum: " << global_sum / 1024.0 << " MB" << std::endl;
        std::cout << "==========================================" << std::endl;
    }
}

template<typename T>
Poisson_solver<T>::Poisson_solver(const Poisson_solver_control& poisson_solver_control,
                                  const Stencil<T>& stencil,
                                  const Mesh_control& mesh_control,
                                  const Domain_parallel_vertices_3D& domain_vertices,
                                  const Exarr_3D_mpi_package& exarr_mpi_package)
                  : poisson_solver_control(poisson_solver_control),
                    stencil(stencil),
                    mesh_control(mesh_control),
                    domain_vertices(domain_vertices),
                    exarr_mpi_package(exarr_mpi_package),
                    aar(this->domain_vertices) {}

template<typename T>
Poisson_solver<T>::~Poisson_solver() {
    #ifdef USE_HYPRE
    if (sizeof(T) == 8) {  // currently only support double
        if (poisson_solver_control.method == 1) {
            if (my_pid == 0) printf("HYPREDEBUG: HYPRE_Finalize\n"), fflush(stdout);
            HYPRE_Finalize();
        }
    }  // double
    #endif
}

template<typename T>
Array_3D<T>& Poisson_solver<T>::cal_electrostatic_potential(const std::vector<Array_3D<T>>& electron_densities,
                                                            const Array_3D<T>& pseudo_charge_density) {
    const uint nspin = electron_densities.size();
    if (nspin == 1) {
        return this->cal_electrostatic_potential(electron_densities[0], pseudo_charge_density);
    } else if (nspin == 2) {
        Array_3D<T> electron_density(electron_densities[0] + electron_densities[1]);
        return this->cal_electrostatic_potential(electron_density, pseudo_charge_density);
    } else {
        assert(!"The number of spin is not supported.");
        return this->electrostatic_potential;
    }
}

template<typename T>
Array_3D<T>& Poisson_solver<T>::cal_electrostatic_potential_mp(
        T const* const* const electron_densities,
        T const* const pseudo_charge_density,
        Memory_pool<T, Fast_memory>& pool_fast,
        Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    // const uint nspin = electron_densities.size();
    constexpr uint nspin = 1;
    if (nspin == 1) {
        return this->cal_electrostatic_potential_mp(electron_densities[0], pseudo_charge_density, pool_fast, pool_cap);
    } else if (nspin == 2) {
        // Array_3D<T> electron_density(electron_densities[0] + electron_densities[1]);
        const uint nd = this->domain_vertices.get_3D_local_vertices().get_size();
        T* electron_density = pool_fast.allocate(nd);
        #pragma omp parallel
        Linalg::hadamard_plus_general(electron_density, electron_densities[0], electron_densities[1], nd);
        return this->cal_electrostatic_potential_mp(electron_density, pseudo_charge_density, pool_fast, pool_cap);
    } else {
        assert(!"The number of spin is not supported.");
        return this->electrostatic_potential;
    }
}

template<typename T>
Array_3D<T>& Poisson_solver<T>::cal_electrostatic_potential(const Array_3D<T>& electron_density,
                                                            const Array_3D<T>& pseudo_charge_density) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->poisson_solver_control.method == 0) { //AAR or DST
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        // this->aar_backup(electron_density, pseudo_charge_density);
        // this->aar.run(this->electrostatic_potential, electron_density + pseudo_charge_density, this->domain_vertices.comm);
        Array_3D<T> temp = electron_density + pseudo_charge_density;
        if (!this->mesh_control.is_periodic[0]
         && !this->mesh_control.is_periodic[1]
         && !this->mesh_control.is_periodic[2]) {
            Array_3D<T> temp2(temp.length);
            Poisson_solver_method::MultipoleExpansion_phi(temp.data, temp2.data, this->stencil, this->domain_vertices, this->mesh_control);
            Linalg::hadamard_minus_general(temp.data, temp2.data, temp.length);
        } else if ((this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && !this->mesh_control.is_periodic[2])
                || (this->mesh_control.is_periodic[0] && !this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2])
                || (!this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2])) {
            Array_3D<T> temp2(temp.length);
            Poisson_solver_method::PartrialDipole_surface(temp.data, temp2.data, T(0), this->stencil, this->domain_vertices, this->mesh_control);
            Linalg::hadamard_minus_general(temp.data, temp2.data, temp.length);
        } else if ((this->mesh_control.is_periodic[0] && !this->mesh_control.is_periodic[1] && !this->mesh_control.is_periodic[2])
                || (!this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && !this->mesh_control.is_periodic[2])
                || (!this->mesh_control.is_periodic[0] && !this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2])) {
            Array_3D<T> temp2(temp.length);
            Poisson_solver_method::PartrialDipole_wire(temp.data, temp2.data, this->stencil, this->domain_vertices, this->mesh_control);
            Linalg::hadamard_minus_general(temp.data, temp2.data, temp.length);
        }

        this->aar.run(this->electrostatic_potential.data, temp.data,
                      this->domain_vertices.get_3D_local_vertices(), this->domain_vertices.comm);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (this->domain_vertices.get_comm_rank() == 0) std::cout << "AAR took "
                                                                  << Tools::time_cost(begin, end) << "."<< std::endl;
    } else if (this->poisson_solver_control.method == 1) {  // CG + BoomerAMG
                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        Array_3D<T> temp = electron_density + pseudo_charge_density;
        if (!this->mesh_control.is_periodic[0]
         && !this->mesh_control.is_periodic[1]
         && !this->mesh_control.is_periodic[2]) {
            Array_3D<T> temp2(temp.length);
            Poisson_solver_method::MultipoleExpansion_phi(temp.data, temp2.data, this->stencil, this->domain_vertices, this->mesh_control);
            Linalg::hadamard_minus_general(temp.data, temp2.data, temp.length);
        } else if ((this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && !this->mesh_control.is_periodic[2])
                || (this->mesh_control.is_periodic[0] && !this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2])
                || (!this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2])) {
            Array_3D<T> temp2(temp.length);
            Poisson_solver_method::PartrialDipole_surface(temp.data, temp2.data, T(0), this->stencil, this->domain_vertices, this->mesh_control);
            Linalg::hadamard_minus_general(temp.data, temp2.data, temp.length);
        } else if ((this->mesh_control.is_periodic[0] && !this->mesh_control.is_periodic[1] && !this->mesh_control.is_periodic[2])
                || (!this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && !this->mesh_control.is_periodic[2])
                || (!this->mesh_control.is_periodic[0] && !this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2])) {
            Array_3D<T> temp2(temp.length);
            Poisson_solver_method::PartrialDipole_wire(temp.data, temp2.data, this->stencil, this->domain_vertices, this->mesh_control);
            Linalg::hadamard_minus_general(temp.data, temp2.data, temp.length);
        }

        // this->aar.run(this->electrostatic_potential.data, temp.data,
        //               this->domain_vertices.get_3D_local_vertices(), this->domain_vertices.comm);

        #ifdef USE_HYPRE
        if (sizeof(T) == 8) {  // currently only support double
        {  // hypre solve (CG + BoomerAMG)

        // temp *= -1;  // -b (since -Laplace)
        #define data_t T
        #define idx_t long long int
        // 建立向量
        {// 填充向量数据
            HYPRE_SStructVectorSetBoxValues(x, my_part, hypre_ilower, hypre_iupper, 0, this->electrostatic_potential.data);
            // HYPRE_SStructVectorSetConstantValues(x, 0.0);
            HYPRE_SStructVectorSetBoxValues(b, my_part, hypre_ilower, hypre_iupper, 0, temp.data);
            HYPRE_SStructVectorSetConstantValues(y, 0.0);
        }
        HYPRE_SStructVectorAssemble(b);
        HYPRE_SStructVectorAssemble(y);// 对y也要创建
        HYPRE_SStructVectorAssemble(x);
        // HYPRE_SStructMatrixGetObject(A, (void **) &par_A);
        HYPRE_SStructVectorGetObject(b, (void **) &par_b);
        HYPRE_SStructVectorGetObject(x, (void **) &par_x);
        HYPRE_SStructVectorGetObject(y, (void **) &par_y);


        // #define HYPRE_DEBUG
        #ifdef HYPRE_DEBUG
        {
            data_t b_dot, x_dot, Ab_dot, Ax_dot;
            idx_t ret;
            // 做spmv检验一下数据传对没有
            ret = HYPRE_ParVectorInnerProd(par_b, par_b, &b_dot); assert(!ret);
            ret = HYPRE_ParVectorInnerProd(par_x, par_x, &x_dot); assert(!ret);
            HYPRE_ParCSRMatrixMatvec(1.0, par_A, par_b, 0.0, par_y);
            ret = HYPRE_ParVectorInnerProd(par_y, par_y, &Ab_dot); assert(!ret);
            HYPRE_ParCSRMatrixMatvec(1.0, par_A, par_x, 0.0, par_y);
            ret = HYPRE_ParVectorInnerProd(par_y, par_y, &Ax_dot); assert(!ret);
            if (my_pid == 0) {
                printf("(  b,   b) = %.20e\n",  (double)b_dot);
                printf("(  x,   x) = %.20e\n",  (double)x_dot);
                printf("(A*b, A*b) = %.20e\n", (double)Ab_dot);
                printf("(A*x, A*x) = %.20e\n", (double)Ax_dot);
            }
            {  // spmv test
                idx_t warm_cnt = 100, test_cnt = 100;
                double tt_spmv = 0.0;
                for (idx_t w = 0; w < warm_cnt; w++)
                    HYPRE_ParCSRMatrixMatvec(1.0, par_A, par_x, 0.0, par_y);

                my_barrier(); tt_spmv = MPI_Wtime();
                for (idx_t w = 0; w < test_cnt; w++)
                    HYPRE_ParCSRMatrixMatvec(1.0, par_A, par_x, 0.0, par_y);
                my_barrier(); tt_spmv = MPI_Wtime() - tt_spmv;
                tt_spmv /= test_cnt;

                double struct_mem = ((double)glb_dims[0]) * glb_dims[1] * glb_dims[2] * (num_diag + 2) * sizeof(data_t);
                struct_mem /= (1024.0 * 1024.0 * 1024.0);// GB
                if (my_pid == 0)
                    printf("[Profile] ParCSR SpMV time: %.6f s, Bandwidth: %.6f GB/s\n", tt_spmv, struct_mem / tt_spmv);
            }
        }
        #endif
        

        idx_t num_iterations = 0;
        data_t t_setup = 0.0, t_solve = 0.0, final_res_norm = 0.0;
        t_solve -= MPI_Wtime();
        HYPRE_ParCSRPCGSolve(par_solver, par_A, par_b, par_x);
        t_solve += MPI_Wtime();
        HYPRE_ParCSRPCGGetNumIterations( par_solver, &num_iterations );
        HYPRE_ParCSRPCGGetFinalRelativeResidualNorm( par_solver, &final_res_norm );
        HYPRE_SStructVectorGather(b);
        HYPRE_SStructVectorGather(x);
        // std::vector<data_t> x_data(electrostatic_potential.get_length());
        HYPRE_SStructVectorGetBoxValues(x, 0, hypre_ilower, hypre_iupper, 0, electrostatic_potential.data);
        // for (unsigned i = 0; i < x_data.size(); ++i) electrostatic_potential[i] = -x_data[i];  // -x (since -Laplace)

        #ifdef HYPRE_DEBUG
        {  // print timing
            const int TEST_CNT = 1;
            TEST_RECORD records[TEST_CNT];
            const int test = 0;
            records[test].iter = num_iterations;
            records[test].setup = t_setup;
            records[test].solve = t_solve;
            if (my_pid == 0) {
                const TEST_RECORD & best = records[0], & worst = records[TEST_CNT - 1];
                TEST_RECORD avg;
                for (idx_t i = 0; i < TEST_CNT; i++) {
                    avg.setup += records[i].setup;
                    avg.solve += records[i].solve;
                    avg.prec  += records[i].prec;
                    avg.iter  += records[i].iter;
                }
                avg.setup /= TEST_CNT;
                avg.solve /= TEST_CNT;
                avg.prec  /= TEST_CNT;
                avg.iter  /= TEST_CNT;
                data_t min_tot = best .setup + best .solve;
                data_t max_tot = worst.setup + worst.solve;
                data_t avg_tot = avg  .setup + avg  .solve;
                printf("\n  Summary of %lld tests:\n", TEST_CNT);
                printf("     Setup time  Solve time  #Iter  Total time\n");
                printf("Min    %.6f    %.6f  %5d  %.6f\n", best .setup, best .solve, best .iter, min_tot);
                printf("Avg    %.6f    %.6f  %5d  %.6f\n", avg  .setup, avg  .solve, avg  .iter, avg_tot);
                printf("Max    %.6f    %.6f  %5d  %.6f\n", worst.setup, worst.solve, worst.iter, max_tot);    
            }
        }
        {  // 计算真实残差
            data_t r_nrm2 = 0.0, b_nrm2 = 0.0;
            check_residual(par_precond, A, x, b, y, r_nrm2, b_nrm2);
            if (my_pid == 0) {
                printf("\033[1;35mtrue ||r|| = %20.16e ||b|| = %20.16e ||r||/||b||= %20.16e\033[0m\n", r_nrm2, b_nrm2, r_nrm2/b_nrm2);
                // printf("\033[1;35mtrue ||r|| = %20.16e ||r||/||b||= %20.16e\033[0m\n", r_nrm2, r_nrm2/b_nrm2);
                printf("Iterations = %lld\n", num_iterations);
                printf("Time cost %.5f %.5f %.5f %lld\n", (double)t_setup, (double)t_solve, (double)(t_setup + t_solve), num_iterations);
                printf("Final Relative Residual Norm = %e\n", (double)final_res_norm);
                printf("\n");
            }
        }
        #else
            if (my_pid == 0) {
                puts("");
                printf("Iterations = %lld\n", num_iterations);
                printf("Time cost (of rank 0)\nsetup, solve, total, iter: %.5f %.5f %.5f %lld\n", (double)t_setup, (double)t_solve, (double)(t_setup + t_solve), num_iterations);
                printf("Final Relative Residual Norm = %e\n", (double)final_res_norm);
                printf("\n");
            }
        #endif


        #undef data_t
        #undef idx_t
        }  // hypre solver (CG + BoomerAMG)
        }  // double
        #endif


        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (this->domain_vertices.get_comm_rank() == 0) std::cout << "CG took "
                                                                  << Tools::time_cost(begin, end) << "."<< std::endl;
    } else if (this->poisson_solver_control.method == 2 || this->poisson_solver_control.method == 3) {  // CG / GMRES + SStructMG
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        Array_3D<T> temp = electron_density + pseudo_charge_density;
        if (!this->mesh_control.is_periodic[0]
         && !this->mesh_control.is_periodic[1]
         && !this->mesh_control.is_periodic[2]) {
            Array_3D<T> temp2(temp.length);
            Poisson_solver_method::MultipoleExpansion_phi(temp.data, temp2.data, this->stencil, this->domain_vertices, this->mesh_control);
            Linalg::hadamard_minus_general(temp.data, temp2.data, temp.length);
        } else if ((this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && !this->mesh_control.is_periodic[2])
                || (this->mesh_control.is_periodic[0] && !this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2])
                || (!this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2])) {
            Array_3D<T> temp2(temp.length);
            Poisson_solver_method::PartrialDipole_surface(temp.data, temp2.data, T(0), this->stencil, this->domain_vertices, this->mesh_control);
            Linalg::hadamard_minus_general(temp.data, temp2.data, temp.length);
        } else if ((this->mesh_control.is_periodic[0] && !this->mesh_control.is_periodic[1] && !this->mesh_control.is_periodic[2])
                || (!this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && !this->mesh_control.is_periodic[2])
                || (!this->mesh_control.is_periodic[0] && !this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2])) {
            Array_3D<T> temp2(temp.length);
            Poisson_solver_method::PartrialDipole_wire(temp.data, temp2.data, this->stencil, this->domain_vertices, this->mesh_control);
            Linalg::hadamard_minus_general(temp.data, temp2.data, temp.length);
        }

        #ifdef USE_SSTRUCTMG
        // Note: Don't let temp to be -temp, or the answer will be -answer compared with AAR (????? for unknown reason ?????)
        // temp *= -1;  // -b (since -Laplace)
        int iter = sstruct_mg.solve(temp.data, this->electrostatic_potential.data);  // (-A)x=(-b)
        if (this->domain_vertices.get_comm_rank() == 0) printf("SStructMG took %d iterations\n", iter);
        sstruct_mg.show_breakdown_and_reset();
        sstruct_mg.release_work_vectors();
        #endif
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (this->domain_vertices.get_comm_rank() == 0) std::cout << "SStructMG took "
                                                                  << Tools::time_cost(begin, end) << "."<< std::endl;
    } else {
        puts("[ERROR] NOT IMPLEMENTED YET");
        exit(-1);
    }
    if (this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2]) {
        T int_electrostatic_potential = this->electrostatic_potential.vector_sum();
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        MPI_Allreduce(MPI_IN_PLACE, &int_electrostatic_potential, 1, mpi_datatype, MPI_SUM, this->domain_vertices.comm);
        this->electrostatic_potential -= (int_electrostatic_potential/(T) this->domain_vertices.get_3D_shared_vertices().get_size());
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // double x_dot_val = Linalg::vector_2norm(electrostatic_potential.data, electrostatic_potential.get_length(), domain_vertices.get_mpi_comm());
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The cal_electrostatic_potential took " << Tools::time_cost(begin, end) << "." << std::endl;
        // printf("POISSON_SOLVER DEBUG: sqrt(electrostatic_potential, electrostatic_potential) = %.10e\n", x_dot_val);
        printf("POISSON_SOLVER DEBUG: electrostatic_potential, proc 0, x[0-4]: %.10e, %.10e, %.10e, %.10e, %.10e\n",
            (double)electrostatic_potential.data[0], (double)electrostatic_potential.data[1], (double)electrostatic_potential.data[2], (double)electrostatic_potential.data[3], (double)electrostatic_potential.data[4]);
    }
    print_mem_usage("after solve");
    return this->electrostatic_potential;
}

template<typename T>
Array_3D<T>& Poisson_solver<T>::cal_electrostatic_potential_mp(T const* const electron_density,
                                                            T const* const pseudo_charge_density,
                                                            Memory_pool<T, Fast_memory>& pool_fast,
                                                            Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->poisson_solver_control.method == 0) { //AAR or DST
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        // this->aar_backup(electron_density, pseudo_charge_density);
        // this->aar.run(this->electrostatic_potential, electron_density + pseudo_charge_density, this->domain_vertices.comm);
        // Array_3D<T> temp = electron_density + pseudo_charge_density;
        const uint nd = this->domain_vertices.get_3D_local_vertices().get_size();
        T* temp = pool_fast.allocate(nd);
        Linalg::hadamard_plus_general(temp, electron_density, pseudo_charge_density, nd);
        // if (!this->mesh_control.is_periodic[0]
        //  && !this->mesh_control.is_periodic[1]
        //  && !this->mesh_control.is_periodic[2]) {
        //     Array_3D<T> temp2(temp.length);
        //     Poisson_solver_method::MultipoleExpansion_phi(temp.data, temp2.data, this->stencil, this->domain_vertices, this->mesh_control);
        //     Linalg::hadamard_minus_general(temp.data, temp2.data, temp.length);
        // } else if ((this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && !this->mesh_control.is_periodic[2])
        //         || (this->mesh_control.is_periodic[0] && !this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2])
        //         || (!this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2])) {
        //     Array_3D<T> temp2(temp.length);
        //     Poisson_solver_method::PartrialDipole_surface(temp.data, temp2.data, T(0), this->stencil, this->domain_vertices, this->mesh_control);
        //     Linalg::hadamard_minus_general(temp.data, temp2.data, temp.length);
        // } else if ((this->mesh_control.is_periodic[0] && !this->mesh_control.is_periodic[1] && !this->mesh_control.is_periodic[2])
        //         || (!this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && !this->mesh_control.is_periodic[2])
        //         || (!this->mesh_control.is_periodic[0] && !this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2])) {
        //     Array_3D<T> temp2(temp.length);
        //     Poisson_solver_method::PartrialDipole_wire(temp.data, temp2.data, this->stencil, this->domain_vertices, this->mesh_control);
        //     Linalg::hadamard_minus_general(temp.data, temp2.data, temp.length);
        // }

        this->aar.run_ompunnested_mp(this->electrostatic_potential.data, temp,
                      this->domain_vertices.get_3D_local_vertices(),
                      this->domain_vertices.comm, pool_fast, pool_cap);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (this->domain_vertices.get_comm_rank() == 0) std::cout << "AAR took "
                                                                  << Tools::time_cost(begin, end) << "."<< std::endl;
    } else if (this->poisson_solver_control.method == 1) {  // CG + BoomerAMG
                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        // Array_3D<T> temp = electron_density + pseudo_charge_density;
        const uint nd = this->domain_vertices.get_3D_local_vertices().get_size();
        T* temp = pool_fast.allocate(nd);
        Linalg::hadamard_plus_general(temp, electron_density, pseudo_charge_density, nd);
        // if (!this->mesh_control.is_periodic[0]
        //  && !this->mesh_control.is_periodic[1]
        //  && !this->mesh_control.is_periodic[2]) {
        //     Array_3D<T> temp2(temp.length);
        //     Poisson_solver_method::MultipoleExpansion_phi(temp.data, temp2.data, this->stencil, this->domain_vertices, this->mesh_control);
        //     Linalg::hadamard_minus_general(temp.data, temp2.data, temp.length);
        // } else if ((this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && !this->mesh_control.is_periodic[2])
        //         || (this->mesh_control.is_periodic[0] && !this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2])
        //         || (!this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2])) {
        //     Array_3D<T> temp2(temp.length);
        //     Poisson_solver_method::PartrialDipole_surface(temp.data, temp2.data, T(0), this->stencil, this->domain_vertices, this->mesh_control);
        //     Linalg::hadamard_minus_general(temp.data, temp2.data, temp.length);
        // } else if ((this->mesh_control.is_periodic[0] && !this->mesh_control.is_periodic[1] && !this->mesh_control.is_periodic[2])
        //         || (!this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && !this->mesh_control.is_periodic[2])
        //         || (!this->mesh_control.is_periodic[0] && !this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2])) {
        //     Array_3D<T> temp2(temp.length);
        //     Poisson_solver_method::PartrialDipole_wire(temp.data, temp2.data, this->stencil, this->domain_vertices, this->mesh_control);
        //     Linalg::hadamard_minus_general(temp.data, temp2.data, temp.length);
        // }

        // this->aar.run(this->electrostatic_potential.data, temp.data,
        //               this->domain_vertices.get_3D_local_vertices(), this->domain_vertices.comm);

        #ifdef USE_HYPRE
        if (sizeof(T) == 8) {  // currently only support double
        {  // hypre solve (CG + BoomerAMG)

        // temp *= -1;  // -b (since -Laplace)
        #define data_t T
        #define idx_t long long int
        // 建立向量
        {// 填充向量数据
            HYPRE_SStructVectorSetBoxValues(x, my_part, hypre_ilower, hypre_iupper, 0, this->electrostatic_potential.data);
            // HYPRE_SStructVectorSetConstantValues(x, 0.0);
            HYPRE_SStructVectorSetBoxValues(b, my_part, hypre_ilower, hypre_iupper, 0, temp);
            HYPRE_SStructVectorSetConstantValues(y, 0.0);
        }
        HYPRE_SStructVectorAssemble(b);
        HYPRE_SStructVectorAssemble(y);// 对y也要创建
        HYPRE_SStructVectorAssemble(x);
        // HYPRE_SStructMatrixGetObject(A, (void **) &par_A);
        HYPRE_SStructVectorGetObject(b, (void **) &par_b);
        HYPRE_SStructVectorGetObject(x, (void **) &par_x);
        HYPRE_SStructVectorGetObject(y, (void **) &par_y);


        // #define HYPRE_DEBUG
        #ifdef HYPRE_DEBUG
        {
            data_t b_dot, x_dot, Ab_dot, Ax_dot;
            idx_t ret;
            // 做spmv检验一下数据传对没有
            ret = HYPRE_ParVectorInnerProd(par_b, par_b, &b_dot); assert(!ret);
            ret = HYPRE_ParVectorInnerProd(par_x, par_x, &x_dot); assert(!ret);
            HYPRE_ParCSRMatrixMatvec(1.0, par_A, par_b, 0.0, par_y);
            ret = HYPRE_ParVectorInnerProd(par_y, par_y, &Ab_dot); assert(!ret);
            HYPRE_ParCSRMatrixMatvec(1.0, par_A, par_x, 0.0, par_y);
            ret = HYPRE_ParVectorInnerProd(par_y, par_y, &Ax_dot); assert(!ret);
            if (my_pid == 0) {
                printf("(  b,   b) = %.20e\n",  (double)b_dot);
                printf("(  x,   x) = %.20e\n",  (double)x_dot);
                printf("(A*b, A*b) = %.20e\n", (double)Ab_dot);
                printf("(A*x, A*x) = %.20e\n", (double)Ax_dot);
            }
            {  // spmv test
                idx_t warm_cnt = 100, test_cnt = 100;
                double tt_spmv = 0.0;
                for (idx_t w = 0; w < warm_cnt; w++)
                    HYPRE_ParCSRMatrixMatvec(1.0, par_A, par_x, 0.0, par_y);

                my_barrier(); tt_spmv = MPI_Wtime();
                for (idx_t w = 0; w < test_cnt; w++)
                    HYPRE_ParCSRMatrixMatvec(1.0, par_A, par_x, 0.0, par_y);
                my_barrier(); tt_spmv = MPI_Wtime() - tt_spmv;
                tt_spmv /= test_cnt;

                double struct_mem = ((double)glb_dims[0]) * glb_dims[1] * glb_dims[2] * (num_diag + 2) * sizeof(data_t);
                struct_mem /= (1024.0 * 1024.0 * 1024.0);// GB
                if (my_pid == 0)
                    printf("[Profile] ParCSR SpMV time: %.6f s, Bandwidth: %.6f GB/s\n", tt_spmv, struct_mem / tt_spmv);
            }
        }
        #endif
        

        idx_t num_iterations = 0;
        data_t t_setup = 0.0, t_solve = 0.0, final_res_norm = 0.0;
        t_solve -= MPI_Wtime();
        HYPRE_ParCSRPCGSolve(par_solver, par_A, par_b, par_x);
        t_solve += MPI_Wtime();
        HYPRE_ParCSRPCGGetNumIterations( par_solver, &num_iterations );
        HYPRE_ParCSRPCGGetFinalRelativeResidualNorm( par_solver, &final_res_norm );
        HYPRE_SStructVectorGather(b);
        HYPRE_SStructVectorGather(x);
        // std::vector<data_t> x_data(electrostatic_potential.get_length());
        HYPRE_SStructVectorGetBoxValues(x, 0, hypre_ilower, hypre_iupper, 0, electrostatic_potential.data);
        // for (unsigned i = 0; i < x_data.size(); ++i) electrostatic_potential[i] = -x_data[i];  // -x (since -Laplace)

        #ifdef HYPRE_DEBUG
        {  // print timing
            const int TEST_CNT = 1;
            TEST_RECORD records[TEST_CNT];
            const int test = 0;
            records[test].iter = num_iterations;
            records[test].setup = t_setup;
            records[test].solve = t_solve;
            if (my_pid == 0) {
                const TEST_RECORD & best = records[0], & worst = records[TEST_CNT - 1];
                TEST_RECORD avg;
                for (idx_t i = 0; i < TEST_CNT; i++) {
                    avg.setup += records[i].setup;
                    avg.solve += records[i].solve;
                    avg.prec  += records[i].prec;
                    avg.iter  += records[i].iter;
                }
                avg.setup /= TEST_CNT;
                avg.solve /= TEST_CNT;
                avg.prec  /= TEST_CNT;
                avg.iter  /= TEST_CNT;
                data_t min_tot = best .setup + best .solve;
                data_t max_tot = worst.setup + worst.solve;
                data_t avg_tot = avg  .setup + avg  .solve;
                printf("\n  Summary of %lld tests:\n", TEST_CNT);
                printf("     Setup time  Solve time  #Iter  Total time\n");
                printf("Min    %.6f    %.6f  %5d  %.6f\n", best .setup, best .solve, best .iter, min_tot);
                printf("Avg    %.6f    %.6f  %5d  %.6f\n", avg  .setup, avg  .solve, avg  .iter, avg_tot);
                printf("Max    %.6f    %.6f  %5d  %.6f\n", worst.setup, worst.solve, worst.iter, max_tot);    
            }
        }
        {  // 计算真实残差
            data_t r_nrm2 = 0.0, b_nrm2 = 0.0;
            check_residual(par_precond, A, x, b, y, r_nrm2, b_nrm2);
            if (my_pid == 0) {
                printf("\033[1;35mtrue ||r|| = %20.16e ||b|| = %20.16e ||r||/||b||= %20.16e\033[0m\n", r_nrm2, b_nrm2, r_nrm2/b_nrm2);
                // printf("\033[1;35mtrue ||r|| = %20.16e ||r||/||b||= %20.16e\033[0m\n", r_nrm2, r_nrm2/b_nrm2);
                printf("Iterations = %lld\n", num_iterations);
                printf("Time cost %.5f %.5f %.5f %lld\n", (double)t_setup, (double)t_solve, (double)(t_setup + t_solve), num_iterations);
                printf("Final Relative Residual Norm = %e\n", (double)final_res_norm);
                printf("\n");
            }
        }
        #else
            if (my_pid == 0) {
                puts("");
                printf("Iterations = %lld\n", num_iterations);
                printf("Time cost (of rank 0)\nsetup, solve, total, iter: %.5f %.5f %.5f %lld\n", (double)t_setup, (double)t_solve, (double)(t_setup + t_solve), num_iterations);
                printf("Final Relative Residual Norm = %e\n", (double)final_res_norm);
                printf("\n");
            }
        #endif


        #undef data_t
        #undef idx_t
        }  // hypre solver (CG + BoomerAMG)
        }  // double
        #endif


        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (this->domain_vertices.get_comm_rank() == 0) std::cout << "CG took "
                                                                  << Tools::time_cost(begin, end) << "."<< std::endl;
    } else if (this->poisson_solver_control.method == 2 || this->poisson_solver_control.method == 3) {  // CG / GMRES + SStructMG
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        // Array_3D<T> temp = electron_density + pseudo_charge_density;
        const uint nd = this->domain_vertices.get_3D_local_vertices().get_size();
        T* temp = pool_fast.allocate(nd);
        Linalg::hadamard_plus_general(temp, electron_density, pseudo_charge_density, nd);
        // if (!this->mesh_control.is_periodic[0]
        //  && !this->mesh_control.is_periodic[1]
        //  && !this->mesh_control.is_periodic[2]) {
        //     Array_3D<T> temp2(temp.length);
        //     Poisson_solver_method::MultipoleExpansion_phi(temp.data, temp2.data, this->stencil, this->domain_vertices, this->mesh_control);
        //     Linalg::hadamard_minus_general(temp.data, temp2.data, temp.length);
        // } else if ((this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && !this->mesh_control.is_periodic[2])
        //         || (this->mesh_control.is_periodic[0] && !this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2])
        //         || (!this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2])) {
        //     Array_3D<T> temp2(temp.length);
        //     Poisson_solver_method::PartrialDipole_surface(temp.data, temp2.data, T(0), this->stencil, this->domain_vertices, this->mesh_control);
        //     Linalg::hadamard_minus_general(temp.data, temp2.data, temp.length);
        // } else if ((this->mesh_control.is_periodic[0] && !this->mesh_control.is_periodic[1] && !this->mesh_control.is_periodic[2])
        //         || (!this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && !this->mesh_control.is_periodic[2])
        //         || (!this->mesh_control.is_periodic[0] && !this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2])) {
        //     Array_3D<T> temp2(temp.length);
        //     Poisson_solver_method::PartrialDipole_wire(temp.data, temp2.data, this->stencil, this->domain_vertices, this->mesh_control);
        //     Linalg::hadamard_minus_general(temp.data, temp2.data, temp.length);
        // }

        #ifdef USE_SSTRUCTMG
        // Note: Don't let temp to be -temp, or the answer will be -answer compared with AAR (????? for unknown reason ?????)
        // temp *= -1;  // -b (since -Laplace)
        int iter = sstruct_mg.solve(temp, this->electrostatic_potential.data);  // (-A)x=(-b)
        if (this->domain_vertices.get_comm_rank() == 0) printf("SStructMG took %d iterations\n", iter);
        sstruct_mg.show_breakdown_and_reset();
        sstruct_mg.release_work_vectors();
        #endif
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (this->domain_vertices.get_comm_rank() == 0) std::cout << "SStructMG took "
                                                                  << Tools::time_cost(begin, end) << "."<< std::endl;
    } else {
        puts("[ERROR] NOT IMPLEMENTED YET");
        exit(-1);
    }
    if (this->mesh_control.is_periodic[0] && this->mesh_control.is_periodic[1] && this->mesh_control.is_periodic[2]) {
        T int_electrostatic_potential = this->electrostatic_potential.vector_sum();
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        MPI_Allreduce(MPI_IN_PLACE, &int_electrostatic_potential, 1, mpi_datatype, MPI_SUM, this->domain_vertices.comm);
        this->electrostatic_potential -= (int_electrostatic_potential/(T) this->domain_vertices.get_3D_shared_vertices().get_size());
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // double x_dot_val = Linalg::vector_2norm(electrostatic_potential.data, electrostatic_potential.get_length(), domain_vertices.get_mpi_comm());
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The cal_electrostatic_potential took " << Tools::time_cost(begin, end) << "." << std::endl;
        // printf("POISSON_SOLVER DEBUG: sqrt(electrostatic_potential, electrostatic_potential) = %.10e\n", x_dot_val);
        printf("POISSON_SOLVER DEBUG: electrostatic_potential, proc 0, x[0-4]: %.10e, %.10e, %.10e, %.10e, %.10e\n",
            (double)electrostatic_potential.data[0], (double)electrostatic_potential.data[1], (double)electrostatic_potential.data[2], (double)electrostatic_potential.data[3], (double)electrostatic_potential.data[4]);
    }
    print_mem_usage("after solve");
    return this->electrostatic_potential;
}

template<typename T>
void Poisson_solver<T>::aar_backup(const Array_3D<T>& electron_density, const Array_3D<T>& pseudo_charge_density) {
    MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
    double tol = this->poisson_solver_control.tolerance;
    double r_2norm;
    int Nd = (int)this->domain_vertices.local_vertices.get_size();

    Array_3D<T> rhs = electron_density + pseudo_charge_density;
    // rhs *= (4.0 * M_PI);

    Array_3D<T> r(this->domain_vertices.local_vertices);            // residual vector, r = b - Ax
    Array_3D<T> x_old(this->domain_vertices.local_vertices);
    Array_3D<T> f(this->domain_vertices.local_vertices, 0);            // preconditioned residual vector, f = inv(M) * r
    Array_3D<T> f_old(this->domain_vertices.local_vertices);
    assert(r.data != nullptr && x_old.data != nullptr && f.data != nullptr && f_old.data != nullptr);

    Array_4D<T> X(Vertices_4D(
                  this->domain_vertices.local_vertices.is, this->domain_vertices.local_vertices.get_ie(), 
                  this->domain_vertices.local_vertices.js, this->domain_vertices.local_vertices.get_je(), 
                  this->domain_vertices.local_vertices.ks, this->domain_vertices.local_vertices.get_ke(),
                  0, this->aar.aar_control.mixing_history - 1), 0);
    Array_4D<T> F(X.get_vertices(), 0);

    x_old = this->electrostatic_potential;

    T rhs_2norm = rhs.vector_norm_square_sum();
    MPI_Allreduce(MPI_IN_PLACE, &rhs_2norm, 1, mpi_datatype, MPI_SUM, this->domain_vertices.comm);
    rhs_2norm = std::sqrt(rhs_2norm);

    // Stencil<T> stencil_tem = this->stencil;
    Stencil<T> stencil_tem = this->stencil.coeffs_scale(0.25 * M_1_PI, 2);
    Vertices_3D ex_vertice = this->domain_vertices.local_vertices.generate_ex_vertices(stencil_tem.FDn);
    Array_3D<T> ex_electrostatic_potential(ex_vertice, 0);

    this->exarr_mpi_package.fill_domain_par_ex_arr(this->electrostatic_potential, ex_electrostatic_potential);
    // r = ex_electrostatic_potential.calc_laplacian(this->domain_vertices.local_vertices, stencil_tem, rhs, (T)0.0, rhs, 1.0);
    Stencil_method::calc_laplacian(ex_electrostatic_potential.data, ex_vertice, stencil_tem, this->domain_vertices.local_vertices,
                                   r.data, this->domain_vertices.local_vertices, rhs.data, (T)0.0, rhs.data, (T)1.0);
    tol *= rhs_2norm;
    r_2norm = tol + 1.0;
    uint iter = 0;
    while (r_2norm > tol && iter < this->poisson_solver_control.max_iter) {
        // *** calculate preconditioned residual f *** //
        Stencil_method::jacobi_preconditioner(stencil_tem, Nd, (T)0.0, r.data, f.data);
        // *** store residual & iteration history *** //
        if (iter > 0) {
            int i_hist = (iter - 1) % this->aar.aar_control.mixing_history;
            //X.data + i_hist * Nd = this->electrostatic_potential.data - x_old.data;
            //F.data + i_hist * Nd = f.data - f_old.data;
            Linalg::hadamard_minus_general(X.data + i_hist * Nd, this->electrostatic_potential.data, x_old.data, Nd);
            Linalg::hadamard_minus_general(F.data + i_hist * Nd, f.data, f_old.data, Nd);
        }

        x_old = this->electrostatic_potential;
        f_old = f;

        if((iter + 1) % this->aar.aar_control.anderson_frequency == 0) {
            /***********************************
             *  Anderson extrapolation update  *
             ***********************************/
            Mixing_method::AndersonExtrapolation(Nd, this->aar.aar_control.mixing_history, this->electrostatic_potential.data, x_old.data,
                                                 f_old.data, X.data, F.data, T(this->aar.aar_control.anderson_beta), this->domain_vertices.comm);

            this->exarr_mpi_package.fill_domain_par_ex_arr(this->electrostatic_potential, ex_electrostatic_potential);
            // r = ex_electrostatic_potential.calc_laplacian(this->domain_vertices.local_vertices, stencil_tem, rhs, (T)0.0, rhs, 1.0);
            Stencil_method::calc_laplacian(ex_electrostatic_potential.data, ex_vertice, stencil_tem, this->domain_vertices.local_vertices,
                                            r.data, this->domain_vertices.local_vertices, rhs.data, (T)0.0, rhs.data, (T)1.0);

            r_2norm = r.vector_norm_square_sum();
            MPI_Allreduce(MPI_IN_PLACE, &r_2norm, 1, mpi_datatype, MPI_SUM, this->domain_vertices.comm);
            r_2norm = std::sqrt(r_2norm);

        } else {
            /***********************
             *  Richardson update  *
             ***********************/
            // this->electrostatic_potential.data = f.data * (T)this->aar_control.omega + x_old.data
            this->electrostatic_potential.scalar_product_general(f, (T)this->aar.aar_control.richardson_omega, x_old);

            // update residual r = b - Ax
            this->exarr_mpi_package.fill_domain_par_ex_arr(this->electrostatic_potential, ex_electrostatic_potential);
            // r = ex_electrostatic_potential.calc_laplacian(this->domain_vertices.local_vertices, stencil_tem, rhs, (T)0.0, rhs, 1.0);
            Stencil_method::calc_laplacian(ex_electrostatic_potential.data, ex_vertice, stencil_tem, this->domain_vertices.local_vertices,
                                   r.data, this->domain_vertices.local_vertices, rhs.data, (T)0.0, rhs.data, (T)1.0);
        }
        iter++;
    }
    if (this->domain_vertices.get_comm_rank() == 0) std::cout << "Poisson AAR solver took iter: " << iter << std::endl;
    return;
}

template<typename T>
void Poisson_solver<T>::init() {
    print_mem_usage("before init");
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    //basic
    // method control
    if (this->poisson_solver_control.method == 0) { //AAR
        // this->aar_control.init();
        this->aar.aar_control.set_precondition_method(this->poisson_solver_control.precondition_method);
        this->aar.init();
        this->aar.init_res_method0(this->stencil, this->exarr_mpi_package, 0.25 * M_1_PI, 0.0);
        this->aar.init_pre_method0(this->stencil, 0.25 * M_1_PI, 0.0);
        this->aar.aar_control.set_max_iter(this->poisson_solver_control.max_iter);
        if (Linalg::get_type_id<T>() == 2) {
            this->aar.aar_control.set_tolerance(this->poisson_solver_control.tolerance);
        } else if (Linalg::get_type_id<T>() == 1) {
            this->aar.aar_control.set_tolerance(this->poisson_solver_control.tolerance * 1e3);
        } else {
            assert(Linalg::get_type_id<T>() == 1
                || Linalg::get_type_id<T>() == 2);
        }
        if (this->poisson_solver_control.precondition_method == 0) { // jacobi
        } else if (this->poisson_solver_control.precondition_method == 1) { //DST
            this->aar.init_pre_dst();
        } else if (this->poisson_solver_control.precondition_method == 2) { // MG
            this->aar.init_pre_mg(this->mesh_control);
        } else {
            assert(this->poisson_solver_control.precondition_method == 0
                || this->poisson_solver_control.precondition_method == 1
                || this->poisson_solver_control.precondition_method == 2);
        }
    } else if (this->poisson_solver_control.method == 1) {  // CG + BoomerAMG
        
        #ifdef USE_HYPRE
        assert(this->poisson_solver_control.precondition_method  ==  2);
        if (sizeof(T) == 8) {  // currently only support double
        #define idx_t long long int
        #define data_t T
        #define F(x) ((x)>=0?(x):-(x))
        
        // Hypre 初始化
        HYPRE_Init();
        my_pid = domain_vertices.get_comm_rank();
        int ret;
        std::string case_name = "POISSON";
        std::string config_mg_file = "./config/amg.poisson.json";
        HYPRE_TEST_CONFIG config(case_name, config_mg_file);
        config.rtol = poisson_solver_control.tolerance;
        config.max_iter = poisson_solver_control.max_iter;

        // Grid setup
        glb_dims[0] = domain_vertices.shared_vertices.get_ke()-domain_vertices.shared_vertices.get_ks()+1;
        glb_dims[1] = domain_vertices.shared_vertices.get_je()-domain_vertices.shared_vertices.get_js()+1;
        glb_dims[2] = domain_vertices.shared_vertices.get_ie()-domain_vertices.shared_vertices.get_is()+1;
        // TODO: some process may have zero grid
        hypre_ilower[0] = domain_vertices.local_vertices.get_is();  // inner
        hypre_iupper[0] = domain_vertices.local_vertices.get_ie();  // (include)
        hypre_ilower[1] = domain_vertices.local_vertices.get_js();
        hypre_iupper[1] = domain_vertices.local_vertices.get_je();
        hypre_ilower[2] = domain_vertices.local_vertices.get_ks();  // outer
        hypre_iupper[2] = domain_vertices.local_vertices.get_ke();
        // for (int d = 0; d < ndim; ++d) {
        //     printf("HYPREDEBUG: hypre_i[%d] = {%d, %d}, dim: %d\n", d, hypre_ilower[d], hypre_iupper[d], glb_dims[2-d]); fflush(stdout);
        // }
        idx_t n_my_elements = 1;
        for (idx_t d = 0; d < ndim; ++d) n_my_elements *= (hypre_iupper[d]-hypre_ilower[d]+1);

        ret = HYPRE_SStructGridCreate(domain_vertices.get_mpi_comm(), ndim, nparts, &ssgrid); assert(!ret);
        ret = HYPRE_SStructGridSetExtents(ssgrid, my_part, hypre_ilower, hypre_iupper); assert(!ret);
        HYPRE_SStructVariable vtypes[nvars];
        for (idx_t v = 0; v < nvars; v++)
            vtypes[v] = HYPRE_SSTRUCT_VARIABLE_CELL;
        for (idx_t part = 0; part < nparts; part++) {// 每个part都要执行定义变量的操作
            ret = HYPRE_SStructGridSetVariables(ssgrid, part, sizeof(vtypes)/sizeof(HYPRE_SStructVariable), vtypes);  assert(!ret);
            idx_t periodic[3] = {0, 0, 0};
            for (idx_t d = 0; d < ndim; ++d)
                if (mesh_control.is_periodic[d]) periodic[d] = glb_dims[2-d];  // reverse
            ret = HYPRE_SStructGridSetPeriodic(ssgrid, part, periodic); assert(!ret);
            if (my_pid == 0) printf("HYPRE: periodic[3] = (inner){%lld %lld %lld}(outer)\n", periodic[0], periodic[1], periodic[2]), fflush(stdout);
        }
        ret = HYPRE_SStructGridAssemble(ssgrid);  assert(!ret); // a collective call finalizing the grid assembly.

        // Stencil setup
        idx_t offsets[num_diag][ndim];  // {inner, mid, outer}
        data_t star_3d37_vals[num_diag];  // Laplace. construct stencil weight by the way
        for (idx_t i0 = -radius, ocnt = 0; i0 <= radius; ++i0) {
            for (idx_t i1 = -radius; i1 <= radius; ++i1) {
                for (idx_t i2 = -radius; i2 <= radius; ++i2) {
                    if ((i0 != 0) + (i1 != 0) + (i2 != 0) > 1) continue;  // star stencil
                    offsets[ocnt][0] = i2;
                    offsets[ocnt][1] = i1;
                    offsets[ocnt][2] = i0;
                    ++ocnt;
                }
            }
        }
        Stencil<data_t> scaled_stencil = stencil.coeffs_scale(-1.0 / M_PI / 4.0, 2);  // stencil (D2) / -4pi   ===> (-A)
        // i.e., -1/4pi * Laplace
        std::vector<data_t> D2_coeffs_i0(scaled_stencil.get_D2_coeffs_z(), scaled_stencil.get_D2_coeffs_z()+radius+1);  // (outer)
        std::vector<data_t> D2_coeffs_i1(scaled_stencil.get_D2_coeffs_y(), scaled_stencil.get_D2_coeffs_y()+radius+1);
        std::vector<data_t> D2_coeffs_i2(scaled_stencil.get_D2_coeffs_x(), scaled_stencil.get_D2_coeffs_x()+radius+1);  // (inner)
        assert(F(D2_coeffs_i0[0])+F(D2_coeffs_i1[0]+F(D2_coeffs_i2[0])) > 1e-4);
        star_3d37_vals[num_diag >> 1] = D2_coeffs_i0[0] + D2_coeffs_i1[0] + D2_coeffs_i2[0];  // center
        D2_coeffs_i0[0] = D2_coeffs_i1[0] = D2_coeffs_i2[0] = 0.0;
        // assert(F(scaled_stencil.D2_coeffs_x[0])+F(scaled_stencil.D2_coeffs_y[0])+F(scaled_stencil.D2_coeffs_z[0]) > 1e-4);
        for (idx_t i0 = -radius, vcnt = 0; i0 <= radius; ++i0) {
            for (idx_t i1 = -radius; i1 <= radius; ++i1) {
                for (idx_t i2 = -radius; i2 <= radius; ++i2) {
                    if ((i0 != 0) + (i1 != 0) + (i2 != 0) != 1) continue;  // star stencil
                    data_t x = D2_coeffs_i0[F(i0)] + D2_coeffs_i1[F(i1)] + D2_coeffs_i2[F(i2)];
                    star_3d37_vals[vcnt++] = x;
                    if (vcnt == (num_diag >> 1)) ++vcnt;  // skip center
                }
            }
        }
        ret = HYPRE_SStructStencilCreate(ndim, num_diag, &stencils[0]);  assert(!ret);
        for (idx_t e = 0; e < num_diag; e++) {
            // printf("HYPREDEBUG: Stencil Set Entry: %d: offsets = {%d %d %d}. (value: %.6f)\n", e, offsets[e][0], offsets[e][1], offsets[e][2], star_3d37_vals[e]), fflush(stdout);
            ret = HYPRE_SStructStencilSetEntry(stencils[0], e, offsets[e], 0);  assert(!ret);// 只有0号变量一种
        }

        // Graph setup
        ret = HYPRE_SStructGraphCreate(domain_vertices.get_mpi_comm(), ssgrid, &ssgraph);  assert(!ret);
        ret = HYPRE_SStructGraphSetObjectType(ssgraph, obj_type); assert(!ret);
        for (idx_t part = 0; part < nparts; part++) {// 每个part都要执行定义模板的操作
            for (HYPRE_Int s = 0; s < nvars; s++)
                ret = HYPRE_SStructGraphSetStencil(ssgraph, part, s, stencils[s]), assert(!ret);
        }
        ret = HYPRE_SStructGraphAssemble(ssgraph); assert(!ret);

        // if (my_pid == 0) printf("HYPREDEBUG: before vectors construction\n"), fflush(stdout);
        // 建立向量
        ret = HYPRE_SStructVectorCreate(domain_vertices.get_mpi_comm(), ssgrid, &b);  assert(!ret); // Create an empty vector object
        ret = HYPRE_SStructVectorCreate(domain_vertices.get_mpi_comm(), ssgrid, &x); assert(!ret);
        ret = HYPRE_SStructVectorCreate(domain_vertices.get_mpi_comm(), ssgrid, &y); assert(!ret);
        ret = HYPRE_SStructVectorSetObjectType(b, obj_type); assert(!ret);// Set the object type for the vectors to be the same as was already set for the matrix
        ret = HYPRE_SStructVectorSetObjectType(x, obj_type); assert(!ret);
        ret = HYPRE_SStructVectorSetObjectType(y, obj_type); assert(!ret);
        ret = HYPRE_SStructVectorInitialize(b); assert(!ret);// Indicate that the vector coefficients are ready to be set
        ret = HYPRE_SStructVectorInitialize(x); assert(!ret);
        ret = HYPRE_SStructVectorInitialize(y); assert(!ret);

        // 建立矩阵
        ret = HYPRE_SStructMatrixCreate(domain_vertices.get_mpi_comm(), ssgraph, &A); assert(!ret);// Create an empty matrix object
        ret = HYPRE_SStructMatrixSetObjectType(A, obj_type); assert(!ret);
        ret = HYPRE_SStructMatrixInitialize(A);  assert(!ret);// Indicate that the matrix coefficients are ready to be set 
        idx_t stencil_indices[num_diag];
        for (idx_t j = 0; j < num_diag; j++)
            stencil_indices[j] = j;
        std::vector<data_t> A_buf(n_my_elements * num_diag);
        for (int i0 = hypre_ilower[2], local_cnt = 0; i0 <= hypre_iupper[2]; ++i0) 
        for (int i1 = hypre_ilower[1]; i1 <= hypre_iupper[1]; ++i1) 
        for (int i2 = hypre_ilower[0]; i2 <= hypre_iupper[0]; ++i2) {
            for (int d = 0; d < num_diag; ++d) {
                double v = star_3d37_vals[d];  // - 1/4pi Laplace
                if (!mesh_control.is_periodic[2]) {
                    int tar_i0 = i0 + offsets[d][2];  // reverse
                    if (tar_i0 < domain_vertices.shared_vertices.get_ks() || tar_i0 > domain_vertices.shared_vertices.get_ke()) v = 0.0;  // Dirichlet
                }
                if (!mesh_control.is_periodic[1]) {
                    int tar_i1 = i1 + offsets[d][1];  // reverse
                    if (tar_i1 < domain_vertices.shared_vertices.get_js() || tar_i1 > domain_vertices.shared_vertices.get_je()) v = 0.0;  // Dirichlet
                }
                if (!mesh_control.is_periodic[0]) {
                    int tar_i2 = i2 + offsets[d][0];  // reverse
                    if (tar_i2 < domain_vertices.shared_vertices.get_is() || tar_i2 > domain_vertices.shared_vertices.get_ie()) v = 0.0;  // Dirichlet
                }
                A_buf[local_cnt++] = v;
            }
        }
        ret = HYPRE_SStructMatrixSetBoxValues(A, my_part, hypre_ilower, hypre_iupper, 0, num_diag, stencil_indices, A_buf.data());  assert(!ret);
        ret = HYPRE_SStructMatrixAssemble(A); assert(!ret);// a collective call finalizing the matrix assembly.

        buildup_solver(par_solver, par_precond, config);

        // if (my_pid == 0) puts("HYPREDEBUG: after buildup_solver");

        ret = HYPRE_SStructMatrixGetObject(A, (void **) &par_A);   assert(!ret);
        // create empty vectors
        ret = HYPRE_SStructVectorSetConstantValues(b, 0.0);  assert(!ret);
        ret = HYPRE_SStructVectorSetConstantValues(x, 0.0);  assert(!ret);
        ret = HYPRE_SStructVectorAssemble(b);  assert(!ret);
        ret = HYPRE_SStructVectorAssemble(x);  assert(!ret);
        ret = HYPRE_SStructVectorGetObject(b, (void **) &par_b);  assert(!ret);
        ret = HYPRE_SStructVectorGetObject(x, (void **) &par_x);  assert(!ret);

        // if (my_pid == 0) puts("HYPREDEBUG: before ParCSRPCGSetup"), fflush(stdout);

        ret = HYPRE_ParCSRPCGSetup(par_solver, par_A, par_b, par_x);  assert(!ret);

        // if (my_pid == 0) puts("HYPREDEBUG: ParCSRPCGSetup finished"), fflush(stdout);


        #undef idx_t
        #undef data_t
        #undef F
        }  // double
        #else
        if (this->domain_vertices.get_comm_rank() == 0) printf("HYPRE not found. Will skip it when solving...\n");
        #endif
    } else if (this->poisson_solver_control.method == 2 || this->poisson_solver_control.method == 3) {  // CG(2) / GMRES(3) + SStructMG
        #ifdef USE_SSTRUCTMG
        #define idx_t int
        assert(this->poisson_solver_control.precondition_method  ==  2);
        #define F(x) ((x)>=0?(x):-(x))
        const idx_t ndim = 3;
        const idx_t num_diag = 37;
        const idx_t radius = 6;
        // Grid setup
        idx_t glb_dims[3], my_ilower[3], my_iupper[3];
        glb_dims[0] = domain_vertices.shared_vertices.get_ke()-domain_vertices.shared_vertices.get_ks()+1;  // reverse
        glb_dims[1] = domain_vertices.shared_vertices.get_je()-domain_vertices.shared_vertices.get_js()+1;
        glb_dims[2] = domain_vertices.shared_vertices.get_ie()-domain_vertices.shared_vertices.get_is()+1;
        my_ilower[0] = domain_vertices.local_vertices.get_ks();  // outer
        my_iupper[0] = domain_vertices.local_vertices.get_ke();  // (include)
        my_ilower[1] = domain_vertices.local_vertices.get_js();
        my_iupper[1] = domain_vertices.local_vertices.get_je();
        my_ilower[2] = domain_vertices.local_vertices.get_is();  // inner
        my_iupper[2] = domain_vertices.local_vertices.get_ie();
        bool periodic[3];
        for (idx_t d = 0; d < ndim; ++d) periodic[d] = mesh_control.is_periodic[2-d];  // reverse

        // Stencil setup
        T star_3d37_vals[num_diag];  // Laplace. construct stencil weight by the way
        Stencil<T> scaled_stencil = stencil.coeffs_scale(-1.0 / M_PI / 4.0, 2);  // stencil (D2) / -4pi   ===> (-A)
        // i.e., -1/4pi * Laplace
        std::vector<T> D2_coeffs_i0(scaled_stencil.get_D2_coeffs_z(), scaled_stencil.get_D2_coeffs_z()+radius+1);  // (outer)
        std::vector<T> D2_coeffs_i1(scaled_stencil.get_D2_coeffs_y(), scaled_stencil.get_D2_coeffs_y()+radius+1);
        std::vector<T> D2_coeffs_i2(scaled_stencil.get_D2_coeffs_x(), scaled_stencil.get_D2_coeffs_x()+radius+1);  // (inner)
        assert(F(D2_coeffs_i0[0])+F(D2_coeffs_i1[0]+F(D2_coeffs_i2[0])) > 1e-4);
        star_3d37_vals[num_diag >> 1] = D2_coeffs_i0[0] + D2_coeffs_i1[0] + D2_coeffs_i2[0];  // center
        D2_coeffs_i0[0] = D2_coeffs_i1[0] = D2_coeffs_i2[0] = 0.0;
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
                    if (vcnt == (num_diag >> 1)) ++vcnt;  // skip center
                }
            }
        }
        // 计算中心点的值：保证行和为0
        precond_stencil_value[3] = -(precond_stencil_value[0] + precond_stencil_value[1] + 
                            precond_stencil_value[2] + precond_stencil_value[4] + 
                            precond_stencil_value[5] + precond_stencil_value[6]);

        TEST_CONFIG config;
        config.rtol = poisson_solver_control.tolerance;
        config.max_iter = poisson_solver_control.max_iter;
        config.use_in_memory_mg_config = true;
        config.print_level = 1;
        config.its_name = this->poisson_solver_control.method == 2 ? "CG" : "GMRES";

        int num_proc = domain_vertices.get_comm_size();
        std::vector<idx_t> glb_begs(3 * num_proc);
        std::vector<idx_t> glb_ends(3 * num_proc);
        bool fine_grid_all_active = true;
        for (int i = 0, ptr = 0; i < num_proc; ++i) {
            Vertices_3D vert = domain_vertices.generate_local_vertices(i);
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
        
        MPI_Comm mg_comm = domain_vertices.get_mpi_comm();
        #ifdef USE_MG_3d7
        sstruct_mg.init(mg_comm, glb_dims, periodic, glb_begs.data(), glb_ends.data(), my_ilower, my_iupper, star_3d37_vals, precond_stencil_value, config, false, fine_grid_all_active);
        #else
        sstruct_mg.init(mg_comm, glb_dims, periodic, glb_begs.data(), glb_ends.data(), my_ilower, my_iupper, star_3d37_vals, nullptr, config, false, fine_grid_all_active);
        #endif

        MPI_Barrier(domain_vertices.get_mpi_comm());  // to seperate setup and solver more clearly
        #undef F
        #undef idx_t
        #else
        if (this->domain_vertices.get_comm_rank() == 0) printf("SStructMG not found. Will skip it when solving...\n");
        #endif
    } else {
        assert(!"ERROR:: POISSON_SOLVER should be AAR, DST or MG!");
    }

    // // domain_vertices
    // Vertices_3D shared_vertices(mesh_control.nx, mesh_control.ny, mesh_control.nz);
    // if (poisson_solver_control.comm_ni != 0) this->domain_vertices.set_comm_ni(poisson_solver_control.comm_ni, mesh_control.nx);
    // if (poisson_solver_control.comm_nj != 0) this->domain_vertices.set_comm_nj(poisson_solver_control.comm_nj, mesh_control.ny);
    // if (poisson_solver_control.comm_nk != 0) this->domain_vertices.set_comm_nk(poisson_solver_control.comm_nk, mesh_control.nz);
    // this->domain_vertices.init(shared_vertices, comm, poisson_solver_control.comm_np_max);

    //stencil
    // if (geometry.cell_type <= 2) {
    //     stencil.set_order(this->fd_order);
    //     stencil.set_D2_coeffs(mesh_control.delta_x, mesh_control.delta_y, mesh_control.delta_z);
    // } else {
    //     assert(!"ERROR:: only Orthorhombi is supported~");
    // }

    // electrostatic_potential
    this->electrostatic_potential.reconstructor(domain_vertices.local_vertices);
    if (this->poisson_solver_control.is_rand_fixed) {
        // this->electrostatic_potential.domain_vertices_rand(domain_vertices);
        Parallel_vertices::domain_vertices_rand<T>(this->electrostatic_potential, this->domain_vertices, -1.0, 1.0);
    } else {
        this->electrostatic_potential.seededrand(this->domain_vertices.get_comm_rank() * 100 + 1, T(-1.0), T(1.0));
    }

    // mpi package init
    // int FDn[3] = {this->stencil.FDn, this->stencil.FDn, this->stencil.FDn};
    // this->exarr_mpi_package.init(this->domain_vertices, poisson_solver_control.is_periodic, FDn);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The Poisson_solver init took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    print_mem_usage("after init");
    return;
}

template<typename T>
template<typename T2>
void Poisson_solver<T>::init(const Poisson_solver<T2>& poisson_solver) {
    if (this->poisson_solver_control.method == 1) {  // CG+BoomerAMG
        puts("HYPREERROR: NOT IMPLEMENTED YET");
        exit(-1);
    } else if (this->poisson_solver_control.method == 2 || this->poisson_solver_control.method == 3) {  // CG/GMRES+SSTRUCTMG
        puts("SSTRUCTMG ERROR: NOT IMPLEMENTED YET");
        exit(-1);
    }

    this->electrostatic_potential.deepcopy(
        std::move(poisson_solver.electrostatic_potential.as_type(this->electrostatic_potential.data)));
    if (this->poisson_solver_control.method == 0) {
        this->aar.init(poisson_solver.aar);
        if (Linalg::get_type_id<T>() == 2) {
            this->aar.aar_control.set_tolerance(this->poisson_solver_control.tolerance);
        } else if (Linalg::get_type_id<T>() == 1) {
            this->aar.aar_control.set_tolerance(this->poisson_solver_control.tolerance * 1e3);
        } else {
            assert(Linalg::get_type_id<T>() == 1
                || Linalg::get_type_id<T>() == 2);
        }
    } else {
        assert(this->poisson_solver_control.method == 0);
    }
    return;
}
template void Poisson_solver<float>::init(const Poisson_solver<float>& poisson_solver);
template void Poisson_solver<double>::init(const Poisson_solver<double>& poisson_solver);
template void Poisson_solver<float>::init(const Poisson_solver<double>& poisson_solver);
template void Poisson_solver<double>::init(const Poisson_solver<float>& poisson_solver);

template<typename T>
void Poisson_solver<T>::destructor() {
    this->electrostatic_potential.destructor();
    if (this->poisson_solver_control.method == 0) {
        this->aar.destructor();
    }
    if (sizeof(T) == 8) {  // currently only support double
    if (poisson_solver_control.method == 1){  // hypre_destruct
    #ifdef USE_HYPRE
        if (my_pid == 0) printf("HYPREDEBUG: destroy\n"), fflush(stdout);
        destroy_solver(par_solver, par_precond);
        HYPRE_SStructMatrixDestroy(A);
        HYPRE_SStructVectorDestroy(b); HYPRE_SStructVectorDestroy(x); HYPRE_SStructVectorDestroy(y);
        HYPRE_SStructGraphDestroy(ssgraph);
        for (HYPRE_Int s = 0; s < nvars; s++) 
            HYPRE_SStructStencilDestroy(stencils[s]);
        HYPRE_SStructGridDestroy(ssgrid);
    #endif
    } else if (poisson_solver_control.method == 2 || poisson_solver_control.method == 3) {
    #ifdef USE_SSTRUCTMG
        sstruct_mg.destroy();
    #endif
    }
    }  // double

    return;
}


template<typename T>
void Poisson_solver<T>::show() const {
    this->poisson_solver_control.show();
    // this->aar_control.show();
    if (this->poisson_solver_control.method == 0) {
        this->aar.show();
    } else {
        assert(this->poisson_solver_control.method == 0);
    }
    this->domain_vertices.show();
    this->stencil.show();
    this->electrostatic_potential.show();
    return;
}

template class Poisson_solver<float>;
template class Poisson_solver<double>;

namespace Poisson_solver_method {

/**
 * @brief   Perform multipole expansion to find boundary condition for the poisson equation
 *                                      -D2 phi(x) = f.
 *          It is required that f decays to zero on the boundary.
 *
 *          Note that this is only done in "phi-domain".
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/electrostatics.c#L1712
 */
template<typename T>
void MultipoleExpansion_phi(T const* const& f, T* const& d_cor,
                            const Stencil<T>& stencil, const Domain_parallel_vertices_3D& domain_vertices,
                            const Mesh_control& mesh_control) {
#define d_cor(i,j,k) d_cor[(k)*DMnx*DMny+(j)*DMnx+(i)]
#define phi(i,j,k) phi[(k)*nx_phi*ny_phi+(j)*nx_phi+(i)]

    assert(domain_vertices.is_Col_Maj && stencil.cell_type <= 2);
    MPI_Comm dmcomm_phi;
    int color = domain_vertices.is_active ? 1 : 0;
    MPI_Comm_split(domain_vertices.get_mpi_comm(), color, 1, &(dmcomm_phi));
    if (color == 0) {
        MPI_Comm_free(&(dmcomm_phi));
        dmcomm_phi = MPI_COMM_NULL;
        return;
    }
    const int LMAX = 6;
    const uint FDn = stencil.order / 2;

    const Vertices_3D& local_vertices = domain_vertices.get_3D_local_vertices();
    const Vertices_3D& shared_vertices = domain_vertices.get_3D_shared_vertices();
    const uint DMnx = local_vertices.get_ni();
    const uint DMny = local_vertices.get_nj();
    const uint DMnz = local_vertices.get_nk();
    const uint DMnd = local_vertices.get_size();
    const int is_local = local_vertices.get_is();
    const int js_local = local_vertices.get_js();
    const int ks_local = local_vertices.get_ks();
    const int ie_local = local_vertices.get_ie();
    const int je_local = local_vertices.get_je();
    const int ke_local = local_vertices.get_ke();
    const uint Nx = shared_vertices.get_ni();
    const uint Ny = shared_vertices.get_nj();
    const uint Nz = shared_vertices.get_nk();
    const double dx = mesh_control.delta_x;
    const double dy = mesh_control.delta_y;
    const double dz = mesh_control.delta_z;
    const double dv = mesh_control.delta_V;
    T Lx = mesh_control.is_periodic[0] == 1 ? Nx * dx : (Nx - 1) * mesh_control.delta_x;
    T Ly = mesh_control.is_periodic[1] == 1 ? Ny * dy : (Ny - 1) * mesh_control.delta_y;
    T Lz = mesh_control.is_periodic[2] == 1 ? Nz * dz : (Nz - 1) * mesh_control.delta_z;

    /* find multipole moments Qlm */
    T* r_pos_x = new T [DMnd];
    T* r_pos_y = new T [DMnd];
    T* r_pos_z = new T [DMnd];
    T* r_pos_r = new T [DMnd];

    // find distance between the center of the domain and finite-difference grids
    T x;
    T y;
    T z;
    T r;
    T x2;
    T y2;
    T z2;
    uint count = 0;
    for (uint k = 0; k < DMnz; k++) {
        int k_global = k + ks_local; // global coord
        z = k_global * dz - Lz/2.0;
        z2 = z * z;
        for (uint j = 0; j < DMny; j++) {
            int j_global = j + js_local; // global coord
            y = j_global * dy - Ly/2.0;
            y2 = y * y;
            for (uint i = 0; i < DMnx; i++) {
                int i_global = i + is_local; // global coord
                x = i_global * dx - Lx/2.0;
                x2 = x * x;
                r_pos_x[count] = x;
                r_pos_y[count] = y;
                r_pos_z[count] = z;
                r_pos_r[count] = std::sqrt(x2 + y2 + z2);
                count++;
            }
        }
    }

    T* Ylm = new T [DMnd];
    int Q_len = (LMAX+1)*(LMAX+1);
    T* Qlm = new T [Q_len] ();
    T* r_pow_l = new T [DMnd];
    Linalg::set_value_general(r_pow_l, T(1.0), DMnd);
    int index = 0;
    for (int l = 0; l <= LMAX; l++) {
        // find r^l
        if (l) {
            for (uint i = 0; i < DMnd; i++) r_pow_l[i] *= r_pos_r[i];
        }
        for (int m = -l; m <= l; m++) {
            // RealSphericalHarmonic(DMnd, r_pos_x, r_pos_y, r_pos_z, r_pos_r, l, m, Ylm);
            Tools::RealSphericalHarmonic(Ylm, l, m, DMnd, r_pos_x, r_pos_y, r_pos_z, r_pos_r);
            Qlm[index] = T(0.0);
            for (uint i = 0; i < DMnd; i++)
                Qlm[index] += r_pow_l[i] * f[i] * Ylm[i];
            Qlm[index] *= dv;
            index++;
        }
    }
    delete [] r_pos_x;
    delete [] r_pos_y;
    delete [] r_pos_z;
    delete [] r_pos_r;
    delete [] Ylm;
    delete [] r_pow_l;

    // do allreduce to sum over all phi process
    MPI_Allreduce(MPI_IN_PLACE, Qlm, Q_len, Linalg::get_mpi_datatype<T>(), MPI_SUM, dmcomm_phi);

	/* find "charge correction" (boudary correction) */
    // define the “correction domain” which contributes to the charge correction. i.e. 0 to FDn-1 and
    // nx-FDn nx-1 in each direction.
    int DMCorVert[6][6];
    DMCorVert[0][0]=0;      DMCorVert[0][1]=FDn-1; DMCorVert[0][2]=0;       DMCorVert[0][3]=Ny-1;  DMCorVert[0][4]=0;       DMCorVert[0][5]=Nz-1;
    DMCorVert[1][0]=Nx-FDn; DMCorVert[1][1]=Nx-1;  DMCorVert[1][2]=0;       DMCorVert[1][3]=Ny-1;  DMCorVert[1][4]=0;       DMCorVert[1][5]=Nz-1;
    DMCorVert[2][0]=0;      DMCorVert[2][1]=Nx-1;  DMCorVert[2][2]=0;       DMCorVert[2][3]=FDn-1; DMCorVert[2][4]=0;       DMCorVert[2][5]=Nz-1;
    DMCorVert[3][0]=0;      DMCorVert[3][1]=Nx-1;  DMCorVert[3][2]=Ny-FDn;  DMCorVert[3][3]=Ny-1;  DMCorVert[3][4]=0;       DMCorVert[3][5]=Nz-1;
    DMCorVert[4][0]=0;      DMCorVert[4][1]=Nx-1;  DMCorVert[4][2]=0;       DMCorVert[4][3]=Ny-1;  DMCorVert[4][4]=0;       DMCorVert[4][5]=FDn-1;
    DMCorVert[5][0]=0;      DMCorVert[5][1]=Nx-1;  DMCorVert[5][2]=0;       DMCorVert[5][3]=Ny-1;  DMCorVert[5][4]=Nz-FDn;  DMCorVert[5][5]=Nz-1;

//     for (i = 0; i < DMnd; i++) d_cor[i] = 0.0; // init correction to 0
    Linalg::set_value_general(d_cor, T(0.0), DMnd);

    T const* D2_stencil_coeffs_x = stencil.get_D2_coeffs_x();
    T const* D2_stencil_coeffs_y = stencil.get_D2_coeffs_y();
    T const* D2_stencil_coeffs_z = stencil.get_D2_coeffs_z();

    // find correction contribution from each side
    for (int nbr_i = 0; nbr_i < 6; nbr_i++) {
        const int is = std::max(DMCorVert[nbr_i][0], is_local);
        const int ie = std::min(DMCorVert[nbr_i][1], ie_local);
        const int js = std::max(DMCorVert[nbr_i][2], js_local);
        const int je = std::min(DMCorVert[nbr_i][3], je_local);
        const int ks = std::max(DMCorVert[nbr_i][4], ks_local);
        const int ke = std::min(DMCorVert[nbr_i][5], ke_local);
        const int nx_cor = ie - is + 1;
        const int ny_cor = je - js + 1;
        const int nz_cor = ke - ks + 1;
        const int nd_cor = nx_cor * ny_cor * nz_cor;
        if (nd_cor <= 0) continue;

        // find the region of phi that have contribution to the correction domain
        int is_phi = is; int ie_phi = ie;
        int js_phi = js; int je_phi = je;
        int ks_phi = ks; int ke_phi = ke;
        switch (nbr_i) {
            case 0:
                is_phi = is - FDn; ie_phi = -1; break;
            case 1:
                is_phi = Nx; ie_phi = ie + FDn; break;
            case 2:
                js_phi = js - FDn; je_phi = -1; break;
            case 3:
                js_phi = Ny; je_phi = je + FDn; break;
            case 4:
                ks_phi = ks - FDn; ke_phi = -1; break;
            case 5:
                ks_phi = Nz; ke_phi = ke + FDn; break;
        }
        int nx_phi = ie_phi - is_phi + 1;
        int ny_phi = je_phi - js_phi + 1;
        int nz_phi = ke_phi - ks_phi + 1;
        int nd_phi = nx_phi * ny_phi * nz_phi;
        // calculate electrostatic potential "phi" inside
        T* phi = new T [nd_phi] ();
        T* Ylm = new T [nd_phi];
        T* r_pos_x = new T [nd_phi];
        T* r_pos_y = new T [nd_phi];
        T* r_pos_z = new T [nd_phi];
        T* r_pos_r = new T [nd_phi];
        T* r_pow_l = new T [nd_phi];
        int count = 0;
        for (int k = 0; k < nz_phi; k++) {
            z = (k + ks_phi) * dz - Lz*0.5;
            for (int j = 0; j < ny_phi; j++) {
                y = (j + js_phi) * dy - Ly*0.5;
                for (int i = 0; i < nx_phi; i++) {
                    x = (i + is_phi) * dx - Lx*0.5;
                    r = std::sqrt(x * x + y * y + z * z);
                    r_pos_x[count] = x;
                    r_pos_y[count] = y;
                    r_pos_z[count] = z;
                    r_pos_r[count] = r;
                    count++;
                }
            }
        }
        Linalg::set_value_general(r_pow_l, T(1.0), nd_phi);
        int index = 0;
        for (int l = 0; l <= LMAX; l++) {
            // find r^(l+1)
            Linalg::hadamard_product_general(r_pow_l, r_pos_r, nd_phi);
            for (int m = -l; m <= l; m++) {
                Tools::RealSphericalHarmonic(Ylm, l, m, nd_phi, r_pos_x, r_pos_y, r_pos_z, r_pos_r);
                for (int i = 0; i < nd_phi; i++)
                    phi[i] += 1.0 / ((2*l+1) * r_pow_l[i]) * Ylm[i] * Qlm[index];
                index++;
            }
        }
        delete [] Ylm;
        delete [] r_pos_x;
        delete [] r_pos_y;
        delete [] r_pos_z;
        delete [] r_pos_r;
        delete [] r_pow_l;

        // calculate the correction "d_cor"
        for (int k = ks; k <= ke; k++) {
            int k_phi = k - ks_phi;
            int k_DM = k - ks_local;
            for (int j = js; j <= je; j++) {
                int j_phi = j - js_phi;
                int j_DM = j - js_local;
                for (int i = is; i <= ie; i++) {
                    int i_phi = i - is_phi;
                    int i_DM = i - is_local;
                    for (int p = 1; p <= (int)FDn; p++) {
                        switch (nbr_i) {
                            case 0:
                                if ((i-p) < 0)
                                    d_cor(i_DM,j_DM,k_DM) -= D2_stencil_coeffs_x[p] * phi(i_phi-p,j_phi,k_phi);
                                break;
                            case 1:
                                if ((i+p) >= (int)Nx)
                                    d_cor(i_DM,j_DM,k_DM) -= D2_stencil_coeffs_x[p] * phi(i_phi+p,j_phi,k_phi);
                                break;
                            case 2:
                                if ((j-p) < 0)
                                    d_cor(i_DM,j_DM,k_DM) -= D2_stencil_coeffs_y[p] * phi(i_phi,j_phi-p,k_phi);
                                break;
                            case 3:
                                if ((j+p) >= (int)Ny)
                                    d_cor(i_DM,j_DM,k_DM) -= D2_stencil_coeffs_y[p] * phi(i_phi,j_phi+p,k_phi);
                                break;
                            case 4:
                                if ((k-p) < 0)
                                    d_cor(i_DM,j_DM,k_DM) -= D2_stencil_coeffs_z[p] * phi(i_phi,j_phi,k_phi-p);
                                break;
                            case 5:
                                if ((k+p) >= (int)Nz)
                                    d_cor(i_DM,j_DM,k_DM) -= D2_stencil_coeffs_z[p] * phi(i_phi,j_phi,k_phi+p);
                                break;
                        }
                    }
                }
            }
        }
        delete [] phi;
    }
    delete [] Qlm;
#undef d_cor
#undef phi
    MPI_Comm_free(&(dmcomm_phi));
    dmcomm_phi = MPI_COMM_NULL;
    return;
}
template void MultipoleExpansion_phi<float>(float const* const& f, float* const& d_cor,
                            const Stencil<float>& stencil, const Domain_parallel_vertices_3D& domain_vertices,
                            const Mesh_control& mesh_control);
template void MultipoleExpansion_phi<double>(double const* const& f, double* const& d_cor,
                            const Stencil<double>& stencil, const Domain_parallel_vertices_3D& domain_vertices,
                            const Mesh_control& mesh_control);

/**
 * @brief   Use partial dipole to correct boundary condition for the poisson equation
 *                                      -D2 phi(x) = f.
 *          with periodic BCs in 2 directions, and Dirichlet BC in the other direction (surface).
 *          So that when discretized in finite difference with Dirichlet BC, the equation will be
 *                                  - DiscreteLaplacian phi = f - d.
 *          It is required that f decays to zero on the Dirichlet boundary.
 *
 *          Note that this is only done in "phi-domain".
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/electrostatics.c#L1957
 */
template<typename T>
void PartrialDipole_surface(T const* const& f, T* const& d_cor, const T& NetCharge,
                            const Stencil<T>& stencil, const Domain_parallel_vertices_3D& domain_vertices,
                            const Mesh_control& mesh_control) {
#define d_cor(i,j,k) d_cor[(k)*DMnx*DMny+(j)*DMnx+(i)]
#define f(i,j,k) f[(k)*DMnx*DMny+(j)*DMnx+(i)]
#define phi(i,j,k) phi[(k)*nx_phi*ny_phi+(j)*nx_phi+(i)]
    assert(domain_vertices.is_Col_Maj && stencil.cell_type <= 2);
    MPI_Comm dmcomm_phi;
    int color = domain_vertices.is_active ? 1 : 0;
    MPI_Comm_split(domain_vertices.get_mpi_comm(), color, 1, &(dmcomm_phi));
    if (color == 0) {
        MPI_Comm_free(&(dmcomm_phi));
        dmcomm_phi = MPI_COMM_NULL;
        return;
    }

    assert(stencil.cell_type <= 2);
    const uint FDn = stencil.order / 2;

    const Vertices_3D& local_vertices = domain_vertices.get_3D_local_vertices();
    const Vertices_3D& shared_vertices = domain_vertices.get_3D_shared_vertices();
    const uint DMnx = local_vertices.get_ni();
    const uint DMny = local_vertices.get_nj();
    const uint DMnz = local_vertices.get_nk();
    const uint DMnd = local_vertices.get_size();
    const int is_local = local_vertices.get_is();
    const int js_local = local_vertices.get_js();
    const int ks_local = local_vertices.get_ks();
    const int ie_local = local_vertices.get_ie();
    const int je_local = local_vertices.get_je();
    const int ke_local = local_vertices.get_ke();
    const uint Nx = shared_vertices.get_ni();
    const uint Ny = shared_vertices.get_nj();
    const uint Nz = shared_vertices.get_nk();
    const double dx = mesh_control.delta_x;
    const double dy = mesh_control.delta_y;
    const double dz = mesh_control.delta_z;
    T Lx = mesh_control.is_periodic[0] == 1 ? Nx * dx : (Nx - 1) * mesh_control.delta_x;
    T Ly = mesh_control.is_periodic[1] == 1 ? Ny * dy : (Ny - 1) * mesh_control.delta_y;
    T Lz = mesh_control.is_periodic[2] == 1 ? Nz * dz : (Nz - 1) * mesh_control.delta_z;
    // define the “correction domain” which contributes to the charge correction, 
	// i.e. 0 to FDn-1 and, nx-FDn nx-1 in each direction.
	int DMCorVert[6][6];
	DMCorVert[0][0]=0;      DMCorVert[0][1]=FDn-1; DMCorVert[0][2]=0;       DMCorVert[0][3]=Ny-1;  DMCorVert[0][4]=0;       DMCorVert[0][5]=Nz-1;
	DMCorVert[1][0]=Nx-FDn; DMCorVert[1][1]=Nx-1;  DMCorVert[1][2]=0;       DMCorVert[1][3]=Ny-1;  DMCorVert[1][4]=0;       DMCorVert[1][5]=Nz-1;  
	DMCorVert[2][0]=0;      DMCorVert[2][1]=Nx-1;  DMCorVert[2][2]=0;       DMCorVert[2][3]=FDn-1; DMCorVert[2][4]=0;       DMCorVert[2][5]=Nz-1;
	DMCorVert[3][0]=0;      DMCorVert[3][1]=Nx-1;  DMCorVert[3][2]=Ny-FDn;  DMCorVert[3][3]=Ny-1;  DMCorVert[3][4]=0;       DMCorVert[3][5]=Nz-1;
	DMCorVert[4][0]=0;      DMCorVert[4][1]=Nx-1;  DMCorVert[4][2]=0;       DMCorVert[4][3]=Ny-1;  DMCorVert[4][4]=0;       DMCorVert[4][5]=FDn-1;
	DMCorVert[5][0]=0;      DMCorVert[5][1]=Nx-1;  DMCorVert[5][2]=0;       DMCorVert[5][3]=Ny-1;  DMCorVert[5][4]=Nz-FDn;  DMCorVert[5][5]=Nz-1;

    // init correction to 0
	Linalg::set_value_general(d_cor, T(0), DMnd);
    int gridsizes[3], DMsizes[3];
	gridsizes[0] = Nx;
	gridsizes[1] = Ny;
	gridsizes[2] = Nz;
	DMsizes[0] = DMnx;
	DMsizes[1] = DMny;
	DMsizes[2] = DMnz;
    DMsizes[0] = DMnx;
	DMsizes[1] = DMny;
	DMsizes[2] = DMnz;
    T cellsizes[3], meshsizes[3];
    cellsizes[0] = Lx;
    cellsizes[1] = Ly;
    cellsizes[2] = Lz;
    meshsizes[0] = dx;
    meshsizes[1] = dy;
    meshsizes[2] = dz;
    // find Dirichlet direction and call it the Z direction
	int dir_Z = (mesh_control.is_periodic[0] == 0) ? 0 : (mesh_control.is_periodic[1] == 0 ? 1 : 2);
	int dir_X = (dir_Z + 1) % 3;
	int dir_Y = (dir_X + 1) % 3;

    // once we find the direction, we assume that direction is the Z
	// direction, the other two directions are then called X, Y
	int NX = gridsizes[dir_X]; 
	int NY = gridsizes[dir_Y]; 
	// int NZ = gridsizes[dir_Z];
	int NXY = NX * NY;
	// int DMnX = DMsizes[dir_X];
	// int DMnY = DMsizes[dir_Y];
	int DMnZ = DMsizes[dir_Z];
	T LX = cellsizes[dir_X];  
	T LY = cellsizes[dir_Y];  
	T dZ = meshsizes[dir_Z];
	T A_XY = LX * LY; // area of (X,Y) surface, neglecting the Jacobian

	// Find rho_av = int (rho + b) dXdY / int (1) dXdY locally
	T *rho_av = new T [DMnZ] ();
    // first find sum
	int ind_orig[3], k_new;
	for (uint k = 0; k < DMnz; k++) {
		for (uint j = 0; j < DMny; j++) {
			for (uint i = 0; i < DMnx; i++) {
				ind_orig[0] = i;
				ind_orig[1] = j;
				ind_orig[2] = k;
				k_new = ind_orig[dir_Z];
				rho_av[k_new] += f(i,j,k); // note here f = 4*pi* (rho + b)
			}
		}
	}

    // find average, note here we assume mesh is uniform, otherwise
	// use rho_av = int (rho + b) dXdY / int (1) dXdY
	for (int k = 0; k < DMnZ; k++) {
		rho_av[k] /= (NXY*4*M_PI);
	}

    // Create sub-comm slices of the Cartesian topology
	// int remain_dims[3]; // which dimensions to keep
	// remain_dims[dir_X] = 1;
	// remain_dims[dir_Y] = 1;
	// remain_dims[dir_Z] = 0;
	MPI_Comm XY_comm;
	// MPI_Cart_sub(dmcomm_phi, remain_dims, &XY_comm);
    if (mesh_control.is_periodic[0] == 0) { // i is Z
        color = domain_vertices.get_active_comm_i();
    } else if (mesh_control.is_periodic[1] == 0) { // j is Z
        color = domain_vertices.get_active_comm_j();
    } else if (mesh_control.is_periodic[2] == 0) { // k is Z
        color = domain_vertices.get_active_comm_k();
    } else {
        assert(mesh_control.is_periodic[0] == 0 || mesh_control.is_periodic[1] == 0 || mesh_control.is_periodic[2] == 0);
    }
    MPI_Comm_split(dmcomm_phi, color, 1, &(XY_comm));

    // sum over processors in the sub-slices in the X-Y plane
	MPI_Allreduce(MPI_IN_PLACE, rho_av, DMnZ, Linalg::get_mpi_datatype<T>(),
		MPI_SUM, XY_comm);
    //** evaluate P(0) = int_0^{LZ} rho_av(Z)*Z dZ **//
	// find P0 locally
	T P0 = 0.0;
    int z_is = dir_Z == 0
             ? is_local
             : dir_Z == 1
             ? js_local
             : ks_local;
	for (int k = 0; k < DMnZ; k++) {
		T Z = (k + z_is) * dZ;
		P0 += rho_av[k] * Z * dZ;
	}

    // create sub-communicators in the Z direction
	// remain_dims[dir_X] = 0;
	// remain_dims[dir_Y] = 0;
	// remain_dims[dir_Z] = 1;
	MPI_Comm Z_comm;
	// MPI_Cart_sub(dmcomm_phi, remain_dims, &Z_comm);
    if (mesh_control.is_periodic[0] == 0) { // i is Z
        color = domain_vertices.get_active_comm_j() + domain_vertices.get_active_comm_k() * domain_vertices.get_active_comm_nj();
    } else if (mesh_control.is_periodic[1] == 0) { // j is Z
        color = domain_vertices.get_active_comm_i() + domain_vertices.get_active_comm_k() * domain_vertices.get_active_comm_ni();
    } else if (mesh_control.is_periodic[2] == 0) { // k is Z
        color = domain_vertices.get_active_comm_i() + domain_vertices.get_active_comm_j() * domain_vertices.get_active_comm_ni();
    } else {
        assert(mesh_control.is_periodic[0] == 0 || mesh_control.is_periodic[1] == 0 || mesh_control.is_periodic[2] == 0);
    }
    MPI_Comm_split(dmcomm_phi, color, 1, &(Z_comm));

    // sum over processors in the Z direction
	MPI_Allreduce(MPI_IN_PLACE, &P0, 1, Linalg::get_mpi_datatype<T>(),
                    MPI_SUM, Z_comm);

    MPI_Comm_free(&XY_comm);
	MPI_Comm_free(&Z_comm);

    int nbr_BCs[6];
	nbr_BCs[0] = nbr_BCs[1] = mesh_control.is_periodic[0] ? 0 : 1;
	nbr_BCs[2] = nbr_BCs[3] = mesh_control.is_periodic[1] ? 0 : 1;
	nbr_BCs[4] = nbr_BCs[5] = mesh_control.is_periodic[2] ? 0 : 1;

    T const* D2_stencil_coeffs_x = stencil.get_D2_coeffs_x();
    T const* D2_stencil_coeffs_y = stencil.get_D2_coeffs_y();
    T const* D2_stencil_coeffs_z = stencil.get_D2_coeffs_z();

    // find correction contribution from each side (in Dirichlet BC side)
	for (int nbr_i = 0; nbr_i < 6; nbr_i++) {
		// skip if BC is periodic in this side
		if (nbr_BCs[nbr_i] == 0) continue; 

		int is = std::max(DMCorVert[nbr_i][0], is_local);
		int ie = std::min(DMCorVert[nbr_i][1], ie_local);
		int js = std::max(DMCorVert[nbr_i][2], js_local);
		int je = std::min(DMCorVert[nbr_i][3], je_local);
		int ks = std::max(DMCorVert[nbr_i][4], ks_local);
		int ke = std::min(DMCorVert[nbr_i][5], ke_local);
		int nx_cor = ie - is + 1;
		int ny_cor = je - js + 1;
		int nz_cor = ke - ks + 1;
		int nd_cor = nx_cor * ny_cor * nz_cor;
		if (nd_cor <= 0) continue;

		// find the region of phi that have contribution to the correction domain
		int is_phi = is, ie_phi = ie;
		int js_phi = js, je_phi = je;
		int ks_phi = ks, ke_phi = ke;
		switch (nbr_i) {
			case 0:
				is_phi = is - FDn; ie_phi = -1; break;
			case 1:
				is_phi = Nx; ie_phi = ie + FDn; break;
			case 2:
				js_phi = js - FDn; je_phi = -1; break;
			case 3:
				js_phi = Ny; je_phi = je + FDn; break;
			case 4:
				ks_phi = ks - FDn; ke_phi = -1; break;
			case 5:
				ks_phi = Nz; ke_phi = ke + FDn; break; 
		}

		int nx_phi = ie_phi - is_phi + 1;
		int ny_phi = je_phi - js_phi + 1;
		int nz_phi = ke_phi - ks_phi + 1;
		int nd_phi = nx_phi * ny_phi * nz_phi;

		// calculate electrostatic potential "phi" inside
        T *phi = new T [nd_phi]();

		int count = 0;
		for (int k = 0; k < nz_phi; k++) {
			T z = (k + ks_phi) * dz; 
			for (int j = 0; j < ny_phi; j++) {
				T y = (j + js_phi) * dy;
				for (int i = 0; i < nx_phi; i++) {
					T x = (i + is_phi) * dx;
					T coords[3] = {x,y,z};
					T Z = coords[dir_Z];
					// phi = phi_av = 2*pi*int_0^{Z_cell} rho_av(Z') |Z-Z'| dZ'
					// note: int_0^{Z_cell} rho_av(Z') dZ' = NetCharge/A_XY
					T phi_av = -2.0 * M_PI * (NetCharge/A_XY*Z - P0);
					if (Z <= 0.0) phi_av *= -1.0; 
					phi[count] = phi_av;
					count++;
				}
			}
		} 

		// calculate the correction "d_cor"
		for (int k = ks; k <= ke; k++) {
			int k_phi = k - ks_phi, k_DM = k - ks_local;
			for (int j = js; j <= je; j++) {
				int j_phi = j - js_phi, j_DM = j - js_local;
				for (int i = is; i <= ie; i++) {
					int i_phi = i - is_phi, i_DM = i - is_local;
					for (int p = 1; p <= (int)FDn; p++) {
						switch (nbr_i) {
							case 0:
								if ((i-p) < 0) 
									d_cor(i_DM,j_DM,k_DM) -= D2_stencil_coeffs_x[p] * phi(i_phi-p,j_phi,k_phi);
								break;
							case 1:
								if ((i+p) >= (int)Nx) 
									d_cor(i_DM,j_DM,k_DM) -= D2_stencil_coeffs_x[p] * phi(i_phi+p,j_phi,k_phi);
								break;
							case 2:
								if ((j-p) < 0) 
									d_cor(i_DM,j_DM,k_DM) -= D2_stencil_coeffs_y[p] * phi(i_phi,j_phi-p,k_phi);
								break;
							case 3:
								if ((j+p) >= (int)Ny) 
									d_cor(i_DM,j_DM,k_DM) -= D2_stencil_coeffs_y[p] * phi(i_phi,j_phi+p,k_phi);
								break;
							case 4:
								if ((k-p) < 0) 
									d_cor(i_DM,j_DM,k_DM) -= D2_stencil_coeffs_z[p] * phi(i_phi,j_phi,k_phi-p);
								break;
							case 5:
								if ((k+p) >= (int)Nz) 
									d_cor(i_DM,j_DM,k_DM) -= D2_stencil_coeffs_z[p] * phi(i_phi,j_phi,k_phi+p);
								break;
						}
					}
				}
			}
		}
        delete [] phi;
	}
    delete [] rho_av;

    MPI_Comm_free(&(dmcomm_phi));
    dmcomm_phi = MPI_COMM_NULL;
#undef d_cor
#undef f
#undef phi
    return;
}
template void PartrialDipole_surface<float>(float const* const& f, float* const& d_cor, const float& NetCharge,
                            const Stencil<float>& stencil, const Domain_parallel_vertices_3D& domain_vertices,
                            const Mesh_control& mesh_control);
template void PartrialDipole_surface<double>(double const* const& f, double* const& d_cor, const double& NetCharge,
                            const Stencil<double>& stencil, const Domain_parallel_vertices_3D& domain_vertices,
                            const Mesh_control& mesh_control);

/**
 * @brief   Use partial dipole to correct boundary condition for the poisson equation
 *                                      -D2 phi(x) = f.
 *          with periodic BCs in 1 directions, and Dirichlet BC in the other two directions (wire).
 *          So that when discretized in finite difference with Dirichlet BC, the equation will be
 *                                  - DiscreteLaplacian phi = f - d.
 *          It is required that f decays to zero on the Dirichlet boundary.
 *
 *          Note that this is only done in "phi-domain".
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/electrostatics.c#L2206
 */
template<typename T>
void PartrialDipole_wire(T const* const& f, T* const& d_cor,
                            const Stencil<T>& stencil, const Domain_parallel_vertices_3D& domain_vertices,
                            const Mesh_control& mesh_control) {
#define d_cor(i,j,k) d_cor[(k)*DMnx*DMny+(j)*DMnx+(i)]
#define f(i,j,k) f[(k)*DMnx*DMny+(j)*DMnx+(i)]
#define phi(i,j,k) phi[(k)*nx_phi*ny_phi+(j)*nx_phi+(i)]

    assert(domain_vertices.is_Col_Maj && stencil.cell_type <= 2);
    MPI_Comm dmcomm_phi;
    int color = domain_vertices.is_active ? 1 : 0;
    MPI_Comm_split(domain_vertices.get_mpi_comm(), color, 1, &(dmcomm_phi));
    if (color == 0) {
        MPI_Comm_free(&(dmcomm_phi));
        dmcomm_phi = MPI_COMM_NULL;
        return;
    }
    assert(stencil.cell_type <= 2);
    const uint FDn = stencil.order / 2;

    const Vertices_3D& local_vertices = domain_vertices.get_3D_local_vertices();
    const Vertices_3D& shared_vertices = domain_vertices.get_3D_shared_vertices();
    const uint DMnx = local_vertices.get_ni();
    const uint DMny = local_vertices.get_nj();
    const uint DMnz = local_vertices.get_nk();
    const uint DMnd = local_vertices.get_size();
    const int is_local = local_vertices.get_is();
    const int js_local = local_vertices.get_js();
    const int ks_local = local_vertices.get_ks();
    const int ie_local = local_vertices.get_ie();
    const int je_local = local_vertices.get_je();
    const int ke_local = local_vertices.get_ke();
    const uint Nx = shared_vertices.get_ni();
    const uint Ny = shared_vertices.get_nj();
    const uint Nz = shared_vertices.get_nk();
    const double dx = mesh_control.delta_x;
    const double dy = mesh_control.delta_y;
    const double dz = mesh_control.delta_z;
    // T Lx = mesh_control.is_periodic[0] == 1 ? Nx * dx : (Nx - 1) * mesh_control.delta_x;
    // T Ly = mesh_control.is_periodic[1] == 1 ? Ny * dy : (Ny - 1) * mesh_control.delta_y;
    // T Lz = mesh_control.is_periodic[2] == 1 ? Nz * dz : (Nz - 1) * mesh_control.delta_z;
    // define the “correction domain” which contributes to the charge correction,
	// i.e. 0 to FDn-1 and, nx-FDn nx-1 in each direction.
	// define the “correction domain” which contributes to the charge correction. i.e. 0 to FDn-1 and
	// nx-FDn nx-1 in each direction.
	int DMCorVert[6][6];
	DMCorVert[0][0]=0;      DMCorVert[0][1]=FDn-1; DMCorVert[0][2]=0;       DMCorVert[0][3]=Ny-1;  DMCorVert[0][4]=0;       DMCorVert[0][5]=Nz-1;
	DMCorVert[1][0]=Nx-FDn; DMCorVert[1][1]=Nx-1;  DMCorVert[1][2]=0;       DMCorVert[1][3]=Ny-1;  DMCorVert[1][4]=0;       DMCorVert[1][5]=Nz-1;
	DMCorVert[2][0]=0;      DMCorVert[2][1]=Nx-1;  DMCorVert[2][2]=0;       DMCorVert[2][3]=FDn-1; DMCorVert[2][4]=0;       DMCorVert[2][5]=Nz-1;
	DMCorVert[3][0]=0;      DMCorVert[3][1]=Nx-1;  DMCorVert[3][2]=Ny-FDn;  DMCorVert[3][3]=Ny-1;  DMCorVert[3][4]=0;       DMCorVert[3][5]=Nz-1;
	DMCorVert[4][0]=0;      DMCorVert[4][1]=Nx-1;  DMCorVert[4][2]=0;       DMCorVert[4][3]=Ny-1;  DMCorVert[4][4]=0;       DMCorVert[4][5]=FDn-1;
	DMCorVert[5][0]=0;      DMCorVert[5][1]=Nx-1;  DMCorVert[5][2]=0;       DMCorVert[5][3]=Ny-1;  DMCorVert[5][4]=Nz-FDn;  DMCorVert[5][5]=Nz-1;

    // init correction to 0
	Linalg::set_value_general(d_cor, T(0), DMnd);

    //** Find rho_av = int (rho + b) dZ **//
	int gridsizes[3], DMsizes[3];
	gridsizes[0] = Nx;
	gridsizes[1] = Ny;
	gridsizes[2] = Nz;
	DMsizes[0] = DMnx;
	DMsizes[1] = DMny;
	DMsizes[2] = DMnz;
    T meshsizes[3];
    meshsizes[0] = dx;
    meshsizes[1] = dy;
    meshsizes[2] = dz;

    // find Dirichlet direction and call it the Z direction
	int dir_Z = (mesh_control.is_periodic[0] == 1) ? 0 : (mesh_control.is_periodic[1] == 1 ? 1 : 2);
	int dir_X = (dir_Z + 1) % 3;
	int dir_Y = (dir_X + 1) % 3;

    int nbr_BCs[6];
	nbr_BCs[0] = nbr_BCs[1] = mesh_control.is_periodic[0] ? 0 : 1;
	nbr_BCs[2] = nbr_BCs[3] = mesh_control.is_periodic[1] ? 0 : 1;
	nbr_BCs[4] = nbr_BCs[5] = mesh_control.is_periodic[2] ? 0 : 1;

    // once we find the direction, we assume that direction is the Z
	// direction, the other two directions are then called X, Y
	int NX = gridsizes[dir_X]; 
	int NY = gridsizes[dir_Y]; 
	int NZ = gridsizes[dir_Z];
	// int NXY = NX * NY;
	int DMnX = DMsizes[dir_X];
	int DMnY = DMsizes[dir_Y];
	// int DMnZ = DMsizes[dir_Z];
	// double LX = cellsizes[dir_X];  
	// double LY = cellsizes[dir_Y];  
	// double LZ = cellsizes[dir_Z];
	T dX = meshsizes[dir_X]; 
	T dY = meshsizes[dir_Y]; 
	// double dZ = meshsizes[dir_Z];
	// double A_XY = LX * LY; // area of (X,Y) surface, neglecting the Jacobian

	// Find rho_av = int (rho + b) dXdY / int (1) dXdY locally
	T *rho_av = new T [DMnX*DMnY] ();

    // first find sum
	int ind_orig[3], i_new, j_new;
	for (uint k = 0; k < DMnz; k++) {
		for (uint j = 0; j < DMny; j++) {
			for (uint i = 0; i < DMnx; i++) {
				ind_orig[0] = i;
				ind_orig[1] = j;
				ind_orig[2] = k;
				i_new = ind_orig[dir_X];
				j_new = ind_orig[dir_Y];
				rho_av[j_new*DMnX+i_new] += f(i,j,k); // note here f = 4*pi* (rho + b)
			}
		}
	}

    // find average, note here we assume mesh is uniform, otherwise
	// use rho_av = int (rho + b) dZ / int (1) dZ	
	for (int i = 0; i < DMnX*DMnY; i++) {
		rho_av[i] /= (NZ*4*M_PI);
	}

    // Create sub-comm slices of the Cartesian topology
	// int remain_dims[3]; // which dimensions to keep
	// remain_dims[dir_X] = 1;
	// remain_dims[dir_Y] = 1;
	// remain_dims[dir_Z] = 0;
	MPI_Comm XY_comm;
	// MPI_Cart_sub(pSPARC->dmcomm_phi, remain_dims, &XY_comm);
    if (mesh_control.is_periodic[0] == 1) { // i is Z
        color = domain_vertices.get_active_comm_i();
    } else if (mesh_control.is_periodic[1] == 1) { // j is Z
        color = domain_vertices.get_active_comm_j();
    } else if (mesh_control.is_periodic[2] == 1) { // k is Z
        color = domain_vertices.get_active_comm_k();
    } else {
        assert(mesh_control.is_periodic[0] == 0 || mesh_control.is_periodic[1] == 0 || mesh_control.is_periodic[2] == 0);
    }
    MPI_Comm_split(dmcomm_phi, color, 1, &(XY_comm));

    // create sub-communicators in the Z direction
	// remain_dims[dir_X] = 0;
	// remain_dims[dir_Y] = 0;
	// remain_dims[dir_Z] = 1;
	MPI_Comm Z_comm;
	// MPI_Cart_sub(pSPARC->dmcomm_phi, remain_dims, &Z_comm);
    // MPI_Cart_sub(dmcomm_phi, remain_dims, &Z_comm);
    if (mesh_control.is_periodic[0] == 1) { // i is Z
        color = domain_vertices.get_active_comm_j() + domain_vertices.get_active_comm_k() * domain_vertices.get_active_comm_nj();
    } else if (mesh_control.is_periodic[1] == 1) { // j is Z
        color = domain_vertices.get_active_comm_i() + domain_vertices.get_active_comm_k() * domain_vertices.get_active_comm_ni();
    } else if (mesh_control.is_periodic[2] == 1) { // k is Z
        color = domain_vertices.get_active_comm_i() + domain_vertices.get_active_comm_j() * domain_vertices.get_active_comm_ni();
    } else {
        assert(mesh_control.is_periodic[0] == 0 || mesh_control.is_periodic[1] == 0 || mesh_control.is_periodic[2] == 0);
    }
    MPI_Comm_split(dmcomm_phi, color, 1, &(Z_comm));

    // sum over processors in the sub-slices in the Z direction
	MPI_Allreduce(MPI_IN_PLACE, rho_av, DMnX*DMnY, Linalg::get_mpi_datatype<T>(),
		MPI_SUM, Z_comm);

    //** evaluate V_av(X,Y) = int rho_av(X,Y)*ln((X-X')^2+(Y-Y')^2) dX'dY' **//
	T *V_av =  new T [2 * FDn * (NX+NY)] ();
	T *V_av_nbr[6];
	V_av_nbr[dir_X*2]   = &V_av[0];
	V_av_nbr[dir_X*2+1] = &V_av[FDn*NY];
	V_av_nbr[dir_Y*2]   = &V_av[FDn*2*NY];
	V_av_nbr[dir_Y*2+1] = &V_av[FDn*(2*NY+NX)];

    int DMVert_phi[6][6];
	for (int nbr_i = 0; nbr_i < 6; nbr_i++) {
		DMVert_phi[nbr_i][0] = DMCorVert[nbr_i][0];
		DMVert_phi[nbr_i][1] = DMCorVert[nbr_i][1];
		DMVert_phi[nbr_i][2] = DMCorVert[nbr_i][2];
		DMVert_phi[nbr_i][3] = DMCorVert[nbr_i][3];
		DMVert_phi[nbr_i][4] = DMCorVert[nbr_i][4];
		DMVert_phi[nbr_i][5] = DMCorVert[nbr_i][5];
		switch (nbr_i) {
			case 0:
				DMVert_phi[nbr_i][0] -= FDn; DMVert_phi[nbr_i][1] -= FDn; break;
			case 1:
				DMVert_phi[nbr_i][0] += FDn; DMVert_phi[nbr_i][1] += FDn; break;
			case 2:
				DMVert_phi[nbr_i][2] -= FDn; DMVert_phi[nbr_i][3] -= FDn; break;
			case 3:
				DMVert_phi[nbr_i][2] += FDn; DMVert_phi[nbr_i][3] += FDn; break;
			case 4:
				DMVert_phi[nbr_i][4] -= FDn; DMVert_phi[nbr_i][5] -= FDn; break;
			case 5:
				DMVert_phi[nbr_i][4] += FDn; DMVert_phi[nbr_i][5] += FDn; break; 
		}
	}

    // find V_av locally
    // int z_is = dir_Z == 0
    //          ? is_local
    //          : dir_Z == 1
    //          ? js_local
    //          : ks_local;
    int y_is = dir_Z == 0
             ? ks_local
             : dir_Z == 1
             ? is_local
             : js_local;
    int x_is = dir_Z == 0
             ? js_local
             : dir_Z == 1
             ? ks_local
             : is_local;
	for (int jp = 0; jp < DMnY; jp++) {
		T Yp = (jp + y_is) * dY;
		for (int ip = 0; ip < DMnX; ip++) {
			T Xp = (ip + x_is) * dX;
			T rho_av_xp_yp = rho_av[jp*DMnX+ip];
			for (int nbr_i = 0; nbr_i < 6; nbr_i++) {
				if (nbr_BCs[nbr_i] == 0) continue;
				int Is_phi_full = DMVert_phi[nbr_i][dir_X*2];
				int Ie_phi_full = DMVert_phi[nbr_i][dir_X*2+1];
				int Js_phi_full = DMVert_phi[nbr_i][dir_Y*2];
				int Je_phi_full = DMVert_phi[nbr_i][dir_Y*2+1];
				int nX_phi_full = Ie_phi_full - Is_phi_full + 1;
				// int nY_phi = je_phi - js_phi + 1;
				for (int j = Js_phi_full; j <= Je_phi_full; j++) {
					T Y = j * dY;
					for (int i = Is_phi_full; i <= Ie_phi_full; i++) {
						T X = i * dX;
						T r2 = (X-Xp)*(X-Xp) + (Y-Yp)*(Y-Yp);
						V_av_nbr[nbr_i][(j- Js_phi_full)*nX_phi_full+(i-Is_phi_full)] -= rho_av_xp_yp * std::log(r2) * (dX*dY); 
					}
				}
			}
		}
	}

    // sum over processors in the sub-slices in the X-Y plane
	MPI_Allreduce(MPI_IN_PLACE, V_av, 2*FDn*(NX+NY), Linalg::get_mpi_datatype<T>(),
		MPI_SUM, XY_comm);

	MPI_Comm_free(&XY_comm);
	MPI_Comm_free(&Z_comm);

    T const* D2_stencil_coeffs_x = stencil.get_D2_coeffs_x();
    T const* D2_stencil_coeffs_y = stencil.get_D2_coeffs_y();
    T const* D2_stencil_coeffs_z = stencil.get_D2_coeffs_z();

    // find correction contribution from each side (in Dirichlet BC side)
	for (int nbr_i = 0; nbr_i < 6; nbr_i++) {
		// skip if BC is periodic in this side
		if (nbr_BCs[nbr_i] == 0) continue; 

		int is = std::max(DMCorVert[nbr_i][0], is_local);
		int ie = std::min(DMCorVert[nbr_i][1], ie_local);
		int js = std::max(DMCorVert[nbr_i][2], js_local);
		int je = std::min(DMCorVert[nbr_i][3], je_local);
		int ks = std::max(DMCorVert[nbr_i][4], ks_local);
		int ke = std::min(DMCorVert[nbr_i][5], ke_local);
		int nx_cor = ie - is + 1;
		int ny_cor = je - js + 1;
		int nz_cor = ke - ks + 1;
		int nd_cor = nx_cor * ny_cor * nz_cor;
		if (nd_cor <= 0) continue;

		// find the region of phi that have contribution to the correction domain
		int is_phi = is, ie_phi = ie;
		int js_phi = js, je_phi = je;
		int ks_phi = ks, ke_phi = ke;
		switch (nbr_i) {
			case 0:
				is_phi = is - FDn; ie_phi = -1; break;
			case 1:
				is_phi = Nx; ie_phi = ie + FDn; break;
			case 2:
				js_phi = js - FDn; je_phi = -1; break;
			case 3:
				js_phi = Ny; je_phi = je + FDn; break;
			case 4:
				ks_phi = ks - FDn; ke_phi = -1; break;
			case 5:
				ks_phi = Nz; ke_phi = ke + FDn; break; 
		}

		int nx_phi = ie_phi - is_phi + 1;
		int ny_phi = je_phi - js_phi + 1;
		int nz_phi = ke_phi - ks_phi + 1;
		int nd_phi = nx_phi * ny_phi * nz_phi;

		// calculate electrostatic potential "phi" inside
		T *phi = new T [nd_phi] ();
		int nX_phi_full = DMVert_phi[nbr_i][dir_X*2+1] - DMVert_phi[nbr_i][dir_X*2] + 1;

		int count = 0;
		for (int k = ks_phi; k <= ke_phi; k++) {
			for (int j = js_phi; j <= je_phi; j++) {
				for (int i = is_phi; i <= ie_phi; i++) {
					int inds[3] = {i,j,k};
					int I_phi_full = inds[dir_X] - DMVert_phi[nbr_i][dir_X*2];
					int J_phi_full = inds[dir_Y] - DMVert_phi[nbr_i][dir_Y*2];
					phi[count] = V_av_nbr[nbr_i][J_phi_full*nX_phi_full+I_phi_full];
					count++;
				}
			}
		}

		// calculate the correction "d_cor"
		for (int k = ks; k <= ke; k++) {
			int k_phi = k - ks_phi, k_DM = k - ks_local;
			for (int j = js; j <= je; j++) {
				int j_phi = j - js_phi, j_DM = j - js_local;
				for (int i = is; i <= ie; i++) {
					int i_phi = i - is_phi, i_DM = i - is_local;
					for (int p = 1; p <= (int)FDn; p++) {
						switch (nbr_i) {
							case 0:
								if ((i-p) < 0) 
									d_cor(i_DM,j_DM,k_DM) -= D2_stencil_coeffs_x[p] * phi(i_phi-p,j_phi,k_phi);
								break;
							case 1:
								if ((i+p) >= (int)Nx) 
									d_cor(i_DM,j_DM,k_DM) -= D2_stencil_coeffs_x[p] * phi(i_phi+p,j_phi,k_phi);
								break;
							case 2:
								if ((j-p) < 0) 
									d_cor(i_DM,j_DM,k_DM) -= D2_stencil_coeffs_y[p] * phi(i_phi,j_phi-p,k_phi);
								break;
							case 3:
								if ((j+p) >= (int)Ny) 
									d_cor(i_DM,j_DM,k_DM) -= D2_stencil_coeffs_y[p] * phi(i_phi,j_phi+p,k_phi);
								break;
							case 4:
								if ((k-p) < 0) 
									d_cor(i_DM,j_DM,k_DM) -= D2_stencil_coeffs_z[p] * phi(i_phi,j_phi,k_phi-p);
								break;
							case 5:
								if ((k+p) >= (int)Nz) 
									d_cor(i_DM,j_DM,k_DM) -= D2_stencil_coeffs_z[p] * phi(i_phi,j_phi,k_phi+p);
								break;
						}
					}
				}
			}
		}
		delete [] phi;
	}
    delete [] rho_av;
    delete [] V_av;

    MPI_Comm_free(&(dmcomm_phi));
    dmcomm_phi = MPI_COMM_NULL;
#undef d_cor
#undef f
#undef phi
    return;
}
template void PartrialDipole_wire<float>(float const* const& f, float* const& d_cor,
                            const Stencil<float>& stencil, const Domain_parallel_vertices_3D& domain_vertices,
                            const Mesh_control& mesh_control);
template void PartrialDipole_wire<double>(double const* const& f, double* const& d_cor,
                            const Stencil<double>& stencil, const Domain_parallel_vertices_3D& domain_vertices,
                            const Mesh_control& mesh_control);

}
