#ifdef USE_HYPRE
#include "wrapper_hypre.h"
#include "json/json.h"
#include <fstream>
#include <filesystem>

// HYPRE_Solver par_solver, par_precond;
// HYPRE_StructSolver s_solver, s_precond;
// HYPRE_SStructSolver solver, precond;

// HYPRE_TEST_CONFIG
HYPRE_TEST_CONFIG::HYPRE_TEST_CONFIG(const std::string _case_name, const std::string _config_mg_file) 
{
    case_name = _case_name;
    config_mg_file = _config_mg_file;
    {  // check if config file exists
        std::string path = config_mg_file;
        bool config_exists = std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
        assert(config_exists);
    }

    if (case_name == "DEMO07") {
        rtol = 1.0e-9;
        num_diag = 7;
    }
    else if (case_name == "DEMO27") {
        rtol = 1.0e-9;
        num_diag = 27;
    }
    else if (case_name == "LASER") {
        rtol = 1.0e-9;
        num_diag = 7;
    }
    else if (case_name == "GRAPES") {
        atol = 1.0e-6;
        rtol = 0;
        num_diag = 19;
    } else if (case_name == "POISSON" || case_name == "periodPOISSON") {
        rtol = 1.0e-7;
        num_diag = 37;
        // num_diag = 31;  // star-3D31
        // num_diag = 25;  // star-3D25
        // num_diag = 19;  // star-3D19
        // num_diag = 7;  // star-3D7
        iter_print_level = 0;
        prec_print_level = 0;
    } else {
        printf("Unknown case name %s\n", case_name.c_str());
        MPI_Abort(MPI_COMM_WORLD, -2026);
    }
}

void buildup_solver(HYPRE_Solver &par_solver, HYPRE_Solver &par_precond, const HYPRE_TEST_CONFIG & config_file)
{
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    HYPRE_Int major, minor, patch, single;
    char * version;
    HYPRE_VersionNumber(&major, &minor, &patch, &single);
    HYPRE_Version(&version);
    if (my_pid == 0) config_file.Print();
    if (my_pid == 0) printf("Major %lld Minor %lld Patch %lld single %lld :: %s\n", 
        major, minor, patch, single, version);


    if (my_pid == 0) printf("using ParCSR PCG\n");
    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &par_solver);
    HYPRE_ParCSRPCGSetMaxIter(par_solver, config_file.max_iter);
    HYPRE_ParCSRPCGSetTol(par_solver, config_file.rtol);
    HYPRE_ParCSRPCGSetAbsoluteTol(par_solver, config_file.atol);
    HYPRE_ParCSRPCGSetTwoNorm(par_solver, 1 );
    // HYPRE_ParCSRPCGSetRelChange(par_solver, 0 );
    HYPRE_ParCSRPCGSetPrintLevel(par_solver, config_file.iter_print_level);
    HYPRE_ParCSRPCGSetLogging(par_solver, 1);

    std::ifstream f(config_file.config_mg_file);
    if (my_pid == 0) printf("Reading config from %s\n", config_file.config_mg_file.c_str());
    Json::Reader reader;
    Json::Value data;
    reader.parse(f, data);

    // 设置预条件子

    if (my_pid == 0) printf("using ParCSR BoomerAMG\n");
    HYPRE_BoomerAMGCreate(&par_precond);
    HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
    HYPRE_BoomerAMGSetTol(par_precond, 0.0);
    HYPRE_BoomerAMGSetPrintLevel(par_precond, config_file.prec_print_level);
    HYPRE_BoomerAMGSetLogging(par_precond, 0);        
    // HYPRE_BoomerAMGSetMaxLevels(par_precond, 4);  // 设置最大层数
    HYPRE_BoomerAMGSetMaxLevels(par_precond, 25);

    if (data.isMember("CoarsenType")) HYPRE_BoomerAMGSetCoarsenType(par_precond, data["CoarsenType"].asInt());
    else HYPRE_BoomerAMGSetCoarsenType(par_precond, 10);   // TODO

    if (data.isMember("StrengthThreshold")) HYPRE_BoomerAMGSetStrongThreshold(par_precond, data["StrengthThreshold"].asDouble()); // default: 0.25
    if (data.isMember("InterpType")       ) HYPRE_BoomerAMGSetInterpType   (par_precond, data["InterpType"].asInt()); /* 3, 15, 6, 14, 18 */
    
    if (data.isMember("AggressiveLevels") ) HYPRE_BoomerAMGSetAggNumLevels (par_precond, data["AggressiveLevels"].asInt()); // >= 0  // TODO: 0, 1, 2
    if (data.isMember("AggInterpType")    ) HYPRE_BoomerAMGSetAggInterpType(par_precond, data["AggInterpType"].asInt()); /* 4, 5, 6, 7, 8 */
    if (data.isMember("AggPMaxElmts")    ) HYPRE_BoomerAMGSetAggPMaxElmts (par_precond, data["AggPMaxElmts"].asInt()); // default: 0
    if (data.isMember("AggP12MaxElmts")  ) HYPRE_BoomerAMGSetAggP12MaxElmts(par_precond, data["AggP12MaxElmts"].asInt()); // default: 0

    if (data.isMember("TruncFactor")      ) HYPRE_BoomerAMGSetTruncFactor(par_precond, data["TruncFactor"].asDouble()); // default: 0
    if (data.isMember("MaxRowSum"        ) ) HYPRE_BoomerAMGSetMaxRowSum(par_precond, data["MaxRowSum"].asDouble()); // default: 0.9
    if (data.isMember("PMaxElmts")       ) HYPRE_BoomerAMGSetPMaxElmts(par_precond, data["PMaxElmts"].asInt()); // default: 4

    if (data.isMember("RelaxType")        ) HYPRE_BoomerAMGSetRelaxType(par_precond, data["RelaxType"].asInt()); /* 3, 4, 6, 7, 18, 11, 12 */  // TODO
    if (data.isMember("RelaxWt")         ) HYPRE_BoomerAMGSetRelaxWt(par_precond, data["RelaxWt"].asDouble()); // default: 1.0  // TODO: 0.7-1.0 0.05
    
    // if (data.isMember("CycleType")        ) HYPRE_BoomerAMGSetCycleType(par_precond, data["CycleType"].asInt());
    // if (data.isMember("FCycle")           ) HYPRE_BoomerAMGSetFCycle(par_precond, 1);
    // if (data.isMember("PreSweeps")        ) HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, data["PreSweeps"].asInt(), 1);
    // if (data.isMember("PostSweeps")       ) HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, data["PostSweeps"].asInt(), 2);
    // if (data.isMember("CoarseSweeps")     ) HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, data["CoarseSweeps"].asInt(), 3);
    
    // if (data.isMember("SmoothType")       ) {
    //     HYPRE_BoomerAMGSetSmoothType     (par_precond, data["SmoothType"].asInt());
    //     if (data.isMember("SmoothNumLevels")) HYPRE_BoomerAMGSetSmoothNumLevels(par_precond, data["SmoothNumLevels"].asInt());
    //     if (data.isMember("ILUType"))         HYPRE_BoomerAMGSetILUType(par_precond, data["ILUType"].asInt());// HYPRE_ILUSetType()
    //     if (data.isMember("ILUk-levels"))     HYPRE_BoomerAMGSetILULevel(par_precond, data["ILUk-levels"].asInt());
    //     if (data.isMember("ILUOrdering"))     HYPRE_BoomerAMGSetILULocalReordering(par_precond, data["ILUOrdering"].asInt());
    // }

    // ↓ CPU
    // HYPRE_BoomerAMGSetModuleRAP2(par_precond, 1);// modularized option for computing the Galerkin product RAP

    HYPRE_ParCSRPCGSetPrecond  (par_solver, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, par_precond);
}

void my_barrier() {
#ifdef USE_CUDA
    cudaDeviceSynchronize();
#endif
    MPI_Barrier(MPI_COMM_WORLD);
}


void destroy_solver(HYPRE_Solver &par_solver, HYPRE_Solver &par_precond)
{
    HYPRE_BoomerAMGDestroy(par_precond);
    HYPRE_ParCSRPCGDestroy(par_solver);
}

void check_residual(
    const HYPRE_SStructMatrix ss_A, const HYPRE_SStructVector ss_x,
    const HYPRE_SStructVector ss_b, HYPRE_SStructVector ss_y, HYPRE_Real & r_nrm2, HYPRE_Real & b_nrm2)
{
    HYPRE_SStructVectorCopy(ss_b, ss_y);// y = b
    // HYPRE_SStructVectorSetConstantValues(ss_y, 0.0);  // y = 0
    HYPRE_SStructMatrixMatvec(-1.0, ss_A, ss_x, 1.0, ss_y);// y += -A*x
    HYPRE_SStructInnerProd(ss_b, ss_b, &b_nrm2); b_nrm2 = sqrt(b_nrm2);
    HYPRE_SStructInnerProd(ss_y, ss_y, &r_nrm2); r_nrm2 = sqrt(r_nrm2);
}

void stat_amg_pattern(HYPRE_Solver &par_precond)
{

    hypre_ParAMGData * amg_data = (hypre_ParAMGData*) par_precond;

    
    const HYPRE_Int num_levels = amg_data->num_levels;
    // if (glb_pid == 0) printf("Num levels %d\n", num_levels);
    for (int ilev = 0; ilev < num_levels; ilev ++) {
        const hypre_ParCSRMatrix * par_A = amg_data->A_array[ilev];
        // if (glb_pid == 0) printf("%d nrows %d\n", ilev, par_A->global_num_rows);

        const hypre_ParCSRCommPkg * comm_pkg = par_A->comm_pkg;
        // printf("lev %d Proc %d num_recv %d num_send %d\n", ilev, glb_pid, comm_pkg->num_recvs, comm_pkg->num_sends);

        const MPI_Comm lev_comm = comm_pkg->comm;
        int lev_pid, lev_nprocs;
        MPI_Comm_rank(lev_comm, & lev_pid);
        MPI_Comm_size(lev_comm, & lev_nprocs);

        // if (glb_pid == 0) printf("lev %d num_Procs %d\n", ilev, lev_nprocs);

        FILE * fp = nullptr;
        if (lev_pid == 0) { fp = fopen(("amg.L"+std::to_string(ilev)).c_str(), "w"); fclose(fp); }

        for (int p = 0; p < lev_nprocs; p++) {
            if (p == lev_pid) {
                fp = fopen(("amg.L"+std::to_string(ilev)).c_str(), "a");// attach

                // for (int is = 0; is < comm_pkg->num_sends; is++) {
                //     const int dst_pid = comm_pkg->send_procs[is];
                //     const int num = comm_pkg->send_map_starts[is + 1] - comm_pkg->send_map_starts[is];
                //     printf("lev %d Proc %d send  to  %d num %d\n", ilev, glb_pid, dst_pid, num);
                // }

                for (int ir = 0; ir < comm_pkg->num_recvs; ir++) {
                    const int src_pid = comm_pkg->recv_procs[ir];
                    const int num = comm_pkg->recv_vec_starts[ir + 1] - comm_pkg->recv_vec_starts[ir];
                    // printf("lev %d Proc %d recv from %d num %d\n", ilev, glb_pid, src_pid, num);
                    fprintf(fp, "%d %d %d\n", lev_pid, src_pid, num);
                }
                fclose(fp);
                fflush(stdout);
            }
            MPI_Barrier(lev_comm);
        }// process print sequentially
    }// lev loop
}
#endif