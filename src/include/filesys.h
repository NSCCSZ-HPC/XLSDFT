#ifndef _FILESYS_H_
#define _FILESYS_H_

#include <iostream>
#include <unordered_map>
#include <vector>
#include <iomanip>
#include <cassert>
#include <mpi.h>
#include <sstream>
#include "tools.h"

#ifdef BCAST_CHECK
#include "linalg.h"
#include <chrono>
#include <thread>
#endif

#ifndef M_1_PI
#define M_1_PI		0.31830988618379067154	/* 1/pi */
#endif

class Path : public std::string {
public:
    Path();
    Path(const char* str);
    Path(const std::string str);
    template<typename T, typename... Args>
    Path(T t, Args... args);
    bool is_absolute() const;
    Path& operator/=(const Path& p1);
    Path operator/(const Path& p1);
    std::string simplifyPath(std::string path);
};

template<typename T, typename... Args>
Path::Path(T t, Args... args) 
{
    *this = Path(t) / Path(args...);
}

class Input_file {
public:
    std::unordered_map<std::string, std::string> map =
    {
        // {"BC", "P P P"},
        // {"EXCHANGE_CORRELATION", "LDA_PZ"},
        // {"ELEC_TEMP_TYPE", "Fermi-Dirac"},
        // {"SMEARING", "0.001"},
        // {"MESH_SPACING", "0.3"},
        // {"FD_ORDER", "12"},
        // {"TOL_SCF", "1e-5"},
    };
    Input_file();
    Input_file(const char* fname);
    Input_file(const Path& fname);
    std::string get_value(const std::string& key) const;
    void read(const Path& fname);
    void show() const;
    void show(const std::string& fname) const;
    void print(std::ostream& output) const;
};

class Ion_file {
public:
    std::vector<std::string> atom_types;
    std::vector<std::size_t> atom_numbers;
    std::vector<std::string> pseudo_pot_files;
    std::vector<bool> is_abs_coord;
    std::vector<std::vector<double>> atom_coords;
    std::vector<std::vector<double>> atom_spins;
    Ion_file();
    Ion_file(const Path& fname);
    Ion_file(const char* fname);
    void read(const Path& fname);
    void show() const;
    void show(const std::string& fname) const;
    void print(std::ostream& output) const;
};

class Atomic_orbital {
public:
    int n = -1;
    int l = -1;
    std::vector<double> chi;
    std::vector<double> chi_r;
    std::vector<double> chi_rD;
    Atomic_orbital();
    void set_label(const std::string& label);
    void after_read(const std::vector<double>& r_grid);
    void print(std::ostream& output = std::cout) const;
};

class Upf_file {
public:
    int n_chi = -1;
    int n_grid = -1;
    std::vector<double> r_grid;
    std::vector<Atomic_orbital> atomic_orbitals;
    Upf_file();
    void read(const Path& fname);
    int count_string(std::ifstream &file, const std::string str);
    void read_PP_R(std::ifstream &file);
    void read_PP_CHI(std::ifstream &file, const int& i_chi);
    void after_read();
    void print(std::ostream& output = std::cout) const;
    MPI_Datatype register_mpi_type_1st() const;
    void recv_preparation();
    MPI_Datatype register_mpi_type_2nd() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
};

class Psp8_file {
public:
    std::string title;
    std::vector<double> r_core;
    double zatom;
    double zion;        // Znucl
    int pspd;
    int pspcod;
    int pspxc;
    int lmax;
    int lloc;
    int mmax;
    int r2well;
    double rchrg;
    double fchrg;
    double qchrg;
    uint nproj[5] = {0, 0, 0, 0, 0};
    int extension_switch = -1;
    std::vector<std::vector<double>> ekb;  //gamma
    std::vector<std::vector<double>> rgrid;  //radial grid mesh point
    std::vector<std::vector<std::vector<double>>> kbk_projector;  //BKB projector
    std::vector<double> rgrid_local_potential;       // local potential for lloc
    std::vector<double> local_potential;       // local potential for lloc
    std::vector<double> rgrid_charge;  //radial grid mesh point
    std::vector<std::vector<double>> r_charge;  //
    std::vector<double> rgrid_density;  //radial grid mesh point
    std::vector<std::vector<double>> r_density;  //isolated atom electron density
    //after read data
    std::vector<double> local_potential_R;       // local potential * rgrid[lloc] for lloc
    std::vector<double> local_potential_RD;       // Spliner d(local potential * rgrid[lloc])/d(rgrid[lloc]) for lloc
    std::vector<double> r_density_4pi;       // r_density[1] / 4.0/pi
    std::vector<double> r_density_4piD;       // Spliner d(r_density_4pi)/d(rgrid[lloc])
    std::vector<std::vector<std::vector<double>>> kbk_projector_R;  //BKB projector / R
    std::vector<std::vector<std::vector<double>>> kbk_projector_RD;  //BKB d(kbk_projector_R)/d(rgrid[lloc])
    std::vector<double> rho_c_table;  //r_charge[0]/ (4 Pi)
    std::vector<double> rho_c_tableD;  //d(rho_c_table)/d(rgrid[lloc])
    bool is_rgrid_uniform[5] = {false, false, false, false, false};
    bool is_rgrid_local_potential_uniform = false;
    bool is_rgrid_charge_uniform = false;
    bool is_rgrid_density_uniform = false;

    double charge_cut_x = 0.0;
    double charge_cut_y = 0.0;
    double charge_cut_z = 0.0;

    //for PDOS calculation
    Upf_file upf_file;
    bool if_has_upf_file = false; 

    Psp8_file();
    Psp8_file(const Path& fname);
    Psp8_file(const char* fname);
    void read(const Path& fname);
    void read_basic(std::ifstream& input_file);
    void read_nonlocal(std::ifstream& input_file, const int& l);
    void read_local(std::ifstream& input_file);
    void read_charge(std::ifstream& input_file);
    void read_density(std::ifstream& input_file);
    void after_read();
    void show() const;
    void show(const std::string& fname) const;
    void print(std::ostream& output = std::cout) const;
    void print_basic(std::ostream& output) const;
    void print_nonlocal(std::ostream& output, const int& l) const;
    void print_local(std::ostream& output) const;
    void print_charge(std::ostream& output) const;
    void print_density(std::ostream& output) const;
    MPI_Datatype register_mpi_type_1st() const;
    void recv_preparation();
    int generate_ncol() const;
    MPI_Datatype register_mpi_type_2nd() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    #ifdef BCAST_CHECK
    void bcast_check(const MPI_Comm& comm = MPI_COMM_WORLD) const;
    void bcast_print() const;
    #endif
};

namespace Filesys {
    std::string get_effective_string(const std::string& str_in, const char& C);
    std::string get_effective_string(const std::string& str_in);
    std::string toupper(const std::string& str_in);
}

#endif //_FILESYS_H_
