#ifdef USE_HYPRE
#ifndef __WRAPPER_HYPRE_H__
#define __WRAPPER_HYPRE_H__

// #ifndef TEST_CNT
// #define TEST_CNT 1
// #endif

// #include "common.h"
#include <string>
#include <algorithm>
class TEST_RECORD {
public:
    double setup = 0.0, solve = 0.0, prec = 0.0;
    int iter = 0;
    bool operator < (const TEST_RECORD & rec) {
        double my_tot = setup + solve;
        double his_tot = rec.setup + rec.solve;
        return my_tot < his_tot;
    }
};

struct HYPRE_TEST_CONFIG
{
    std::string case_name = "null";
    std::string config_mg_file = "";
    int max_iter = 300;
    double atol = 0.0;// 绝对残差
    double rtol = 1.0e-7;// 相对残差
    int iter_print_level = 2;
    int prec_print_level = 1;
    int restart_len = 10;
    int num_diag = -1;
    HYPRE_TEST_CONFIG(const std::string _case_name, const std::string _config_mg_file);
    void Print() const {
        puts("=========HYPRE_TEST_CONFIG========");
        printf("case_name = %s\n", case_name.c_str());
        printf("config_mg_file = %s\n", config_mg_file.c_str());
        printf("max_iter = %d\n", max_iter);
        printf("atol = %.6e\n", atol);
        printf("rtol = %.6e\n", rtol);
        printf("iter_print_level = %d\n", iter_print_level);
        printf("prec_print_level = %d\n", prec_print_level);
        printf("restart_len = %d\n", restart_len);
        printf("num_diag = %d\n", num_diag);
        puts("=========HYPRE_TEST_CONFIG========");
    }
};

template<typename idx_t>
class PartDescriptor
{
public:
    idx_t id;// 全局块序号
    idx_t ilower[3], iupper[3];
    idx_t nelems;
};


#include "HYPRE_sstruct_ls.h"
#include "_hypre_sstruct_ls.h"

// extern HYPRE_Solver par_solver, par_precond;
// extern HYPRE_SStructSolver solver, precond;

void my_barrier();

void buildup_solver(HYPRE_Solver &par_solver, HYPRE_Solver &par_precond, const HYPRE_TEST_CONFIG & config_file);
void destroy_solver(HYPRE_Solver &par_solver, HYPRE_Solver &par_precond);
// void setup_and_solve(HYPRE_Solver &par_solver, HYPRE_Solver &par_precond, void* A, void* b, void* x,
//     HYPRE_Real & t_setup, HYPRE_Real & t_solve, HYPRE_Real & final_res_norm, HYPRE_Int & num_iterations);

void check_residual(const HYPRE_SStructMatrix ss_A, const HYPRE_SStructVector ss_x,
    const HYPRE_SStructVector ss_b, HYPRE_SStructVector ss_y, HYPRE_Real & r_nrm2, HYPRE_Real & b_nrm2);
// void check_residual(std::string prec_name,
//     void* A, void* x, void* b, void* y, HYPRE_Real & r_nrm2, HYPRE_Real & b_nrm2);

void stat_amg_pattern(HYPRE_Solver &par_precond);

#endif
#endif