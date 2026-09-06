#ifndef _ARGS_H_
#define _ARGS_H_
#include <iostream>
#include <mpi.h>
#include <cstring>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#include "tools.h"

class Args
{
public:
    std::string program_name;
    std::string filename;
    
    int info;
    int nnodes = 1;
    int ncpus = 1;
    int nthreads = 1;
    int nacc = 0;
    Args();
    Args(const int& argc, char* argv[]);
    ~Args();
    void print_usage() const;
    void printVersion() const;
    void printShortUsage() const;
    void printUsage() const;
    void parse(const int& argc, char* argv[]);
    void read_runtime_environment();
};

#endif //_ARGS_H_
