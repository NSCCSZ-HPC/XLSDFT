#include <chrono>
#include <ctime>
#include <iostream>
#include <mpi.h>

#include "args.h"
#include "preparation.h"
#include "scf.h"

int main(int argc, char *argv[]) {
    #ifdef USE_OPENMP
    int thead_level_provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thead_level_provided);
    #else
    MPI_Init(&argc, &argv);
    #endif //USE_OPENMP

    MPI_Barrier(MPI_COMM_WORLD);
    // start timer
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        /* Print start time of the program */
        std::time_t start_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << "Program start time: " << std::ctime(&start_time);
    }
    
    Args args(argc, argv);
    Preparation preparation(args.filename);

    if (preparation.control.flow_control.scf_flag == 1) {

        const MPI_Comm comm = MPI_COMM_WORLD;
        constexpr size_t GB = 1024 * 1024 * 1024 / sizeof(double);
        Memory_pool<double, Fast_memory> pool_fast(38 * GB / 10);
        Memory_pool<double, Capacity_memory> pool_cap(10 * GB / 10);
        // Scf scf(preparation, MPI_COMM_WORLD);
        Scf scf(preparation.control, preparation.geometry, preparation.psp8_files, comm);
        scf.init();
        // scf.run();
        scf.run_mp(pool_fast, pool_cap);

        Array_3D<double> density_temp;
        if (preparation.if_input_file2_exists) {
            density_temp.reconstructor(scf.ddensity_solver.domain_vertices.get_3D_local_vertices());
            Linalg::set_value_general(density_temp.data, scf.ddensity_solver.electron_densities[0].data, density_temp.length);
        }
        scf.destructor();

        if (preparation.if_input_file2_exists) {
            Scf scf2(preparation.control2, preparation.geometry, preparation.psp8_files, comm);
            scf2.init();
            scf2.set_electron_density(density_temp);
            scf2.run_mp(pool_fast, pool_cap);
            scf2.destructor();
        }

    } else {
        assert(!"ONLY SCF IS SUPPORTED");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    // end timer
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (rank == 0) {
        std::cout << "The program took " << Tools::time_cost(begin, end) << "." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
