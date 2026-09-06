#ifndef _ATOM_H_
#define _ATOM_H_

#include <iostream>
#include <cassert>
#include <climits>
#include "mpi.h"
#include <cmath>
#include "filesys.h"
#include "vertices.h"
#include "arr.h"
#include "tools.h"
#include "stencil.h"
#include "parallel_vertices.h"
#include "nloc_projector.h"

#ifndef M_PI
# define M_PI		3.14159265358979323846	/* pi */
#endif
#ifndef M_PI_2
# define M_PI_2		1.57079632679489661923	/* pi/2 */
#endif
#ifndef M_1_PI
#define M_1_PI		0.31830988618379067154	/* 1/pi */
#endif

#ifdef LOW_MEMORY
using Atom_real = float;
#define MPI_ATOM_REAL MPI_FLOAT
#else
using Atom_real = double;
#define MPI_ATOM_REAL MPI_DOUBLE
#endif

class Atom
{
public:
    Atom_real x;
    Atom_real y;
    Atom_real z;
    uint type = UINT_MAX;
    uint index = UINT_MAX;
    // bool is_spin = false;
    // Atom_real atom_spin[3] = {0.0, 0.0, 0.0};
    Atom();
    Atom(const Atom& atom);
    Atom(Atom&& atom);
    Atom(const Atom_real x, const Atom_real y, const Atom_real z, const uint type);
    Atom(const Atom_real x, const Atom_real y, const Atom_real z, const uint type, const uint index);
    ~Atom();
    Atom& operator=(const Atom& other);
    Atom& operator=(Atom&& other);
    Vertices_3D generate_rc_vertices(const uint cell_type, const double dx, const double dy, const double dz,
                                     const double r_x, const double r_y, const double r_z) const;
    template<typename T>
    Array_3D<T> generate_R_array(const uint cell_type, const Vertices_3D& vertices,
                                 const double dx, const double dy, const double dz) const;
    template<typename T>
    Array_3D<T> generate_relative_x_array(const uint cell_type, const Vertices_3D& vertices,
                                          const double dx) const;
    template<typename T>
    Array_3D<T> generate_relative_y_array(const uint cell_type, const Vertices_3D& vertices,
                                          const double dy) const;
    template<typename T>
    Array_3D<T> generate_relative_z_array(const uint cell_type, const Vertices_3D& vertices,
                                          const double dz) const;
    template<typename T>
    void generate_xyz_r_array(Array_3D<T>& ref_x, Array_3D<T>& ref_y, Array_3D<T>& ref_z, Array_3D<T>& R,
                              const uint cell_type, const Vertices_3D& vertices,
                              const double dx, const double dy, const double dz) const;
    template<typename T>
    void generate_xyz_r_array(T* const ref_x, T* const ref_y, T* const ref_z, T* const R,
                              const uint cell_type, const Vertices_3D& vertices,
                              const double dx, const double dy, const double dz) const;
    template<typename T>
    Nloc_projector<T> generate_nloc_projector_chi(const uint cell_type, const Psp8_file& psp8_file,
                                                  const Vertices_3D& vertices, const Vertices_3D& local_vertices,
                                                  const double dx, const double dy, const double dz,
                                                  const int atom_index, const int cell_shift_x,
                                                  const int cell_shift_y, const int cell_shift_z,
                                                  const bool is_in_domain) const;

    template<typename T>
    Array_3D<T> generate_pseudo_charge_density(const uint cell_type, const Vertices_3D& vertices,
                                               const Psp8_file& psp8_file, const Stencil<T>& stencil,
                                               const double dx, const double dy, const double dz) const;
    template<typename T>
    Array_3D<T> generate_pseudo_charge_density_ref(const uint cell_type, const Vertices_3D& vertices,
                                                   const Psp8_file& psp8_file, const Stencil<T>& stencil,
                                                   const double dx, const double dy, const double dz) const;
    template<typename T>
    T generate_pseudo_charge_density_related(Array_3D<T>& pseudo_charge_density,
                                             Array_3D<T>& pseudo_charge_density_ref,
                                             Array_3D<T>& pseudo_charge_density_potiential_correction,
                                             const uint cell_type, const Vertices_3D& vertices,
                                             const Psp8_file& psp8_file, const Stencil<T>& stencil,
                                             const double dx, const double dy, const double dz) const;
    template<typename T>
    void generate_pseudo_charge_density_local_force_related(Array_3D<T>& psoducharge_density,
                                                            Array_3D<T>& psoducharge_density_ref,
                                                            Array_3D<T>& Dpotiential_correction_x,
                                                            Array_3D<T>& Dpotiential_correction_y,
                                                            Array_3D<T>& Dpotiential_correction_z,
                                                            const uint cell_type, const Vertices_3D& vertices,
                                                            const Psp8_file& psp8_file, const Stencil<T>& stencil,
                                                            const double dx, const double dy, const double dz) const;

    template<typename T>
    Array_3D<T> generate_initial_electron_density_array(const uint cell_type, const Vertices_3D& vertices,
                                                        const Psp8_file& psp8_file,
                                                        const double dx, const double dy, const double dz) const;
    template<typename T>
    Array_3D<T> generate_initial_electron_density_core_array(const uint cell_type, const Vertices_3D& vertices,
                                                        const Psp8_file& psp8_file,
                                                        const double dx, const double dy, const double dz) const;
    template<typename T>
    Array_4D<T> generate_atomic_orbitals_nlm_array(const uint cell_type, const uint n_atomic_orbital_Ynlm,
                                                    const Vertices_3D& vertices, const Upf_file& upf_file, const T cutoff,
                                                    const double dx, const double dy, const double dz) const;
    uint generate_valid_images(std::vector<Atom>& atoms, const uint cell_type, 
                               const Domain_parallel_vertices_3D& domain_vertices, const bool* is_periodic,
                               const double dx, const double dy, const double dz,
                               const double lx, const double ly, const double lz,
                               const double r_x, const double r_y, const double r_z) const;
    uint generate_valid_images(std::vector<Atom>& atoms, const uint cell_type, 
                               const Vertices_3D& shared_vertices, const Vertices_3D& local_vertices,
                               const bool* is_periodic,
                               const double dx, const double dy, const double dz,
                               const double lx, const double ly, const double lz,
                               const double r_x, const double r_y, const double r_z) const;
    MPI_Datatype register_mpi_type() const;
    void deregister_mpi_type(MPI_Datatype type) const;
    void bcast(const MPI_Comm comm = MPI_COMM_WORLD, const int root = 0);
    void show() const;
};

namespace Atom_method {
    template<typename T>
    void print_pdos(std::ostream &file, const Atom& atom, const uint nspin,
                    const Array_2D<T>& pdos_atom, const Array_0D<T>& eigen_values,
                    const Psp8_file& psp8_file, const T chemical_potential,
                    const bool if_print_header);
    void generate_valid_images(const std::vector<Atom>& atoms, std::vector<Atom>& image_atoms, const uint cell_type,
                               const Domain_parallel_vertices_3D& domain_vertices, const bool* is_periodic,
                               const double dx, const double dy, const double dz,
                               const double lx, const double ly, const double lz,
                               const std::vector<Psp8_file>& psp8_files, const uint mode = 0);
    template<typename T>
    void Calculate_Pseudopot_Ref(T const* const R,  const int len, const double ref_cut,
                                 const double zion, T* const Vref);
}

#endif //_ATOM_H_