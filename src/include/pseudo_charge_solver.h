#ifndef _PSEUDO_CHARGE_H_
#define _PSEUDO_CHARGE_H_

#include "geometry.h"
#include "control.h"
#include "arr.h"
#include "mixing.h"
#include "parallel_vertices.h"

template<typename T>
class Pseudo_charge_solver
{
public:
    const Geometry& geometry;
    const Mesh_control& mesh_control;
    const std::vector<Psp8_file>& psp8_files;
    const Stencil<T>& stencil;
    const Domain_parallel_vertices_3D& domain_vertices;
    T pseudo_charge = (T)0.0;
    T self_and_correction_energy = (T)0.0;
    Array_3D<T> pseudo_charge_density;
    Array_3D<T> pseudo_charge_density_ref;
    Array_3D<T> pseudo_charge_density_potiential_correction;
    Pseudo_charge_solver(const Geometry& geometry, const Mesh_control& mesh_control,
                         const std::vector<Psp8_file>& psp8_files, const Stencil<T>& stencil,
                         const Domain_parallel_vertices_3D& domain_vertices);
    ~Pseudo_charge_solver();
    Array_3D<T>& generate_pseudo_charge_density();
    Array_3D<T>& generate_pseudo_charge_density_ref();
    void generate_pseudo_charge_density_related();
    T& generate_pseudo_charge();
    T& cal_self_and_correction_energy();
    void init();
    template<typename T2> void init(const Pseudo_charge_solver<T2>& pseudo_charge_solver);
    void destructor();
    void show() const;
};

#endif //_PSEUDO_CHARGE_H_
