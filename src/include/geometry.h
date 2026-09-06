#ifndef _GEOMETRY_H_
#define _GEOMETRY_H_

#include <iostream>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <sstream>
#include <climits>
#include <mpi.h>
#include <cstddef>
#include "filesys.h"
#include "atom.h"

#ifndef M_PI
# define M_PI		3.14159265358979323846	/* pi */
#endif
#ifndef M_PI_2
# define M_PI_2		1.57079632679489661923	/* pi/2 */
#endif

/**
 * @brief 
 * 
 * @ref https://web.archive.org/web/20081004101125/http://www.ccdc.cam.ac.uk/support/documentation/mercury_csd/portable/mercury_portable-4-70.html
 * 
 */
class Geometry
{
public:
    uint cell_type = 2;     // lattice type
    double origin[3] = {0.0, 0.0, 0.0};
    double v1[3];           // lattice vector
    double v2[3];           // lattice vector
    double v3[3];           // lattice vector
    double a;               // length of edges vector v1
    double b;               // length of edges vector v2
    double c;               // length of edges vector v3
    double alpha = M_PI_2;           // angle between b and c
    double beta = M_PI_2;            // angle between a and c
    double gamma = M_PI_2;           // angle between a and b
    double volume;
    uint natom = 0;
    uint natom_type = 0;
    std::vector<Atom> atoms;
    Geometry();
    Geometry(const Geometry& other);
    Geometry(Geometry&& other);
    ~Geometry();
    Geometry& operator=(const Geometry& other);
    Geometry& operator=(Geometry&& other);
    void set_cell_type(const uint& cell_type);
    void set_v1(const double& v1_0);
    void set_v1(const double* v1);
    void set_v1(const double& v1_0, const double& v1_1, const double& v1_2);
    void set_v2(const double& v2_1);
    void set_v2(const double* v2);
    void set_v2(const double& v2_0, const double& v2_1, const double& v2_2);
    void set_v3(const double& v3_2);
    void set_v3(const double* v3);
    void set_v3(const double& v3_0, const double& v3_1, const double& v3_2);
    void set_a(const double& a);
    void set_b(const double& b);
    void set_c(const double& c);
    void set_edge_lengths(const double* lengths);
    void set_edge_lengths(const double& a, const double& b, const double& c);
    void set_alpha(const double& alpha);
    void set_beta(const double& beta);
    void set_gamma(const double& gamma);
    void set_angles(const double& alpha, const double& beta, const double& gamma);
    void sync_from_vectors();
    void sync_from_angles();
    void atom_fractional_coordinates(double* const& frac_xs, double* const& frac_ys, double* const& frac_zs) const;
    // Geometry generate_extended_geometry(const Vertices_3D& my_vertices, const Vertices_3D& other_vertices,
    //                                     const double& dx, const double& dy, const double& dz,
    //                                     bool const* const& is_periodic, double const* const& buffers) const;
    Geometry generate_extended_geometry(const Vertices_3D& my_vertices, const Vertices_3D& other_vertices,
                                        const double& dx, const double& dy, const double& dz,
                                        bool const* const& is_periodic, bool const* const& if_one_more_grid,
                                        const std::vector<Psp8_file>& psp8_files) const;
    void init(Input_file& input_file, const Ion_file& ion_file, const bool& is_map_into_cell = true);
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm& comm = MPI_COMM_WORLD, const int& root = 0);
    void show() const;
};

#endif //_GEOMETRY_H_