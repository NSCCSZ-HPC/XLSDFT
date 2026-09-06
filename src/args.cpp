#include "args.h"

Args::Args() {}

Args::Args(const int& argc, char* argv[]) {
    this->parse(argc, argv);
    if (this->info == EXIT_FAILURE) {
        MPI_Finalize();
        exit(this->info);
    }
}

Args::~Args() {}

void Args::print_usage() const {
    printf("\n");
    printf("USAGE:\n");
    printf("    mpirun -np <nproc> {XLSROOT}/bin/xlsdft -name <filename> [-h] [-n] [-c] [-a]\n");
    printf("\n");
    printf("    {XLSROOT} is the location of the XLSDFT folder\n");
    printf("\n");
    printf("REQUIRED ARGUMENT:\n");
    printf("    -name <filename>\n");
    printf("           The filename shared by .inpt file and .ion\n");
    printf("           file (without extension)\n");
    printf("\n");
    printf("OPTIONS: \n");
    printf("    -h, --help\n");
    printf("           Display help (from command line).\n");
    printf("    -n <number of Nodes>\n");
    printf("    -c <number of CPUs per node>\n");
    printf("    -a <number of Accelerators (e.g., GPUs) per node>\n");
    printf("\n");
    printf("EXAMPLE:\n");
    printf("\n");
    printf("    mpirun -np 8 {XLSROOT}/bin/xlsdft -name test\n");
    printf("\n");
    printf("    The example command runs xlsdft with 8 cores, with input file named\n");
    printf("    test.inpt, and ion file named test.ion.\n");
    printf("\n");
    printf("\n");
    return;
}

void Args::printVersion() const {
    std::cout << "XLSDFT Version: 0.0.1\n";
    return;
}

void Args::printShortUsage() const {
    std::cout << "Usage: [mpirun -np <nproc>] " << this->program_name << " -name <filename> [-n <nnodes>] [-c <ncpus>] [-a <nacc>]\n";
    std::cout << "Try '"<< this->program_name << " --help' for more information.\n";
    return;
}

void Args::printUsage() const {
    std::cout << "Usage: [mpirun -np <nproc>] " << this->program_name << " -name <filename> [-n <nnodes>] [-c <ncpus>] [-a <nacc>]\n";
    std::cout << "Options:\n";
    std::cout << "  -name <filename>      Specify the filename (required)\n";
    std::cout << "  -h, --help            Display this help message\n";
    std::cout << "  -v, -V, --version     Display version information\n";
    std::cout << "  -n <nnodes>           Specify the number of Nodes\n";
    std::cout << "  -c <ncpus>            Specify the number of CPUs per node\n";
    std::cout << "  -a <nacc>             Specify the number of Accelerators (e.g., GPUs) per node\n";
    std::cout << "Example:\n";
    std::cout << "  The following example runs xlsdft with 8 processes, with input file named\n";
    std::cout << "  test.inpt, and ion file named test.ion.\n";
    std::cout << "\n";
    std::cout << "  $ mpirun -np 8 " << this->program_name << " -name test\n";
    return;
}

void Args::parse(const int& argc, char* argv[]) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Save program name
    this->program_name = argv[0];

    // Parse command line arguments starting from index 1
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            if (rank == 0) this->printUsage();
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "-V") == 0 || strcmp(argv[i], "--version") == 0) {
            if (rank == 0) this->printVersion();
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        } else if (strcmp(argv[i], "-name") == 0 && i + 1 < argc) {
            this->filename = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            this->nnodes = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            this->ncpus = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-a") == 0 && i + 1 < argc) {
            this->nacc = std::atoi(argv[++i]);
        } else {
            // Unknown option
            if (rank == 0) {
                std::cerr << "Error: Unknown option '" << argv[i] << "'.\n";
                this->printShortUsage();
            }
            this->info = EXIT_FAILURE;
            return;
        }
    }

    // Check if filename is provided
    if (filename.empty()) {
        if (rank == 0) {
            std::cerr << "Error: Filename (-name) is required.\n";
            this->printShortUsage();
        }
        this->info = EXIT_FAILURE;
        return;
    }

    // Output parsed values
    // this->nnodes = Tools::get_node_number();
    // Tools::print_rankfile();
    if (rank == 0) {
        this->read_runtime_environment();
        std::cout << "Program name: " << this->program_name << std::endl;
        std::cout << "Filename: " << this->filename << std::endl;
        std::cout << "Number of nodes: " << this->nnodes << std::endl;
        std::cout << "Number of CPUs: " << this->ncpus << std::endl;
        std::cout << "Number of threads per core: " << this->nthreads << std::endl;
        std::cout << "Number of accelerators per node: " << this->nacc << std::endl;
    }

    this->info =  EXIT_SUCCESS;
    return;
}

void Args::read_runtime_environment() {
    MPI_Comm_size(MPI_COMM_WORLD, &(this->ncpus));
    #ifdef USE_OPENMP
    #pragma omp parallel
    #pragma omp single nowait
    this->nthreads = omp_get_num_threads();
    #endif
    return;
}
