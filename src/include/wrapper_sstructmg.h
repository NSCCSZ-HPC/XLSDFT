#ifdef USE_SSTRUCTMG
#ifndef __WRAPPER_SSTRUCTMG_H__
#define __WRAPPER_SSTRUCTMG_H__

#include "SStructMG.hpp"
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>
#include <string>
#include <filesystem>

void SStructMG_generate_config_file(const int N0, const int N1, const int N2, std::string file_name);

// Set use_in_memory_mg_config for the built-in 3-D Cell_3d8 hierarchy.  The
// JSON path remains the default for compatibility with custom configurations.
struct TEST_CONFIG
{
    int max_iter = 300;
    double atol = 0.0;// 绝对残差
    double rtol = 1.0e-7;// 相对残差
    int print_level = 1;
    int restart_len = 10;
    std::string its_name = "CG";  // "CG" or "GMRES"
    std::string config_mg_file;
    bool use_in_memory_mg_config = false;
};

// Structured multigrid wrapper.
template<typename idx_t, typename ksp_t, typename pc_data_t, typename pc_calc_t>
class SStructMG_wrapper {
    static constexpr int my_nblks = 1;
    static constexpr int num_diag = 37;
    static constexpr int radius = 6;
public:
    const __int128_t _mask = stencil_mask_star3d6r;
    const __int128_t _mask3d7 = stencil_mask_3d7;
    const int *stencil_offset = stencil_offset_star3d6r;
    const int *stencil3d7_offset = stencil_offset_3d7;

    bool is_initialized = false;
    bool as_precond = false;  // set true for using MG as a preconditioner; otherwise GMRES+MG
    bool owns_comm = false;
    idx_t box_beg[3], box_end[3];  // box_end: my_iupper+1
    MPI_Comm comm = MPI_COMM_NULL;
    SStructGrid<idx_t> *ssgrid = nullptr;
    par_SstructMatrix<idx_t, ksp_t, ksp_t, 1> *mat_A = nullptr;
    par_SstructMatrix<idx_t, ksp_t, ksp_t, 1> *precond_mat_A = nullptr;
    par_SstructVector<idx_t, ksp_t, 1> *vec_x = nullptr, *vec_b = nullptr;
    par_SstructVector<idx_t, pc_calc_t, 1> * pc_buf_x = nullptr, * pc_buf_b = nullptr;  // only used when pc_calc_t != ksp_t
    IterativeSolver<idx_t, pc_data_t, pc_calc_t, ksp_t, 1> * solver = nullptr;
    // MultiGrid is a Solver<idx_t, calc_t, setup_t, dof> in the refactored API.
    // Its calculation precision is pc_calc_t and its setup/operator precision is ksp_t.
    Solver         <idx_t, pc_calc_t, ksp_t            , 1> * precond = nullptr;
    SStructMG_wrapper(){
    }  // do nothing
    ~SStructMG_wrapper();
    /// @brief Initialization of grid and matrix, setup solver and precond. Can be called only once.
    /// @param comm_ MPI comm consist of all procs that have cells in the grid. (If don't have cells, let my_iupper < my_ilower)
    /// @param glb_dim Global extents of 3 dimensions of the grid. 3 dimensions ordered by (outer, middle, inner), i.e., in C-style. The same as the followings.
    /// @param is_period Periodic boundaries or not in each dimension.
    /// @param config_file Path to the config file.
    /// @param glb_begs 3*num_proc numbers. begs[proc0][0], begs[proc0][1], begs[proc0][2], begs[proc1][0], ...
    /// @param glb_ends (exclusive) 3*num_proc numbers. ends[proc0][0], ends[proc0][1], ends[proc0][2], ends[proc1][0], ...
    /// @param my_ilower Lowest coordination among the cells this proc handles. (including)
    /// @param my_iupper Highest coordination among the cells this proc handles. (including)
    /// @param stencil_value Values of the stencil used by solver. The order should be consistent to "stencil_offset".
    /// @param precond_stencil3d7_value Values of the 3d7 stencil used by preconditioner. If it is nullptr, use stencil_value and star3d6r offset in preconditioner.
    /// @param as_precond_ Set true for using MG as a preconditioner; otherwise GMRES+MG.
    /// @param fine_grid_all_active True only when every rank in comm_ owns a nonempty finest-level box. The wrapper then borrows comm_ directly; false takes the safe split path.
    void init(MPI_Comm comm_, const idx_t *glb_dim, const bool *is_period,
              const idx_t *glb_begs, const idx_t *glb_ends,
              const idx_t *my_ilower, const idx_t *my_iupper,
              const ksp_t *stencil_value, const ksp_t *precond_stencil3d7_value,
              const TEST_CONFIG config, const bool as_precond_,
              const bool fine_grid_all_active = false);

    /// @brief solve the equation "A * x = coeff * rhs"
    /// @param rhs_data Pointer to rhs data.
    /// @param x_data x0 data and result.
    /// @return iterations
    idx_t solve(const ksp_t *rhs_data, ksp_t *x_data);

    /// @brief Show the breakdowns of solver and precond. And then reset them.
    void show_breakdown_and_reset();

    /// @brief Release memory of work vectors. The "solve" next time will automatically alloc memory for them.
    void release_work_vectors();
    void destroy();
private:
    int mg_num_levels = 0;
    void buildup_MG(const TEST_CONFIG & config_file);
    void buildup_solver(std::string solver_name, std::string prec_name, const TEST_CONFIG & config_file);
};

#endif  // WRAPPER_SSTRUCTMG_H
#endif  // USE_SSTRUCTMG
