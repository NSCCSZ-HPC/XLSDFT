#ifndef _EFFECTIVE_POTENTIAL_NLOC_H_
#define _EFFECTIVE_POTENTIAL_NLOC_H_

#include "geometry.h"
#include "control.h"

template<typename T>
class Effective_potential_nloc
{
public:
    const Geometry& geometry;
    const Mesh_control& mesh_control;
    const std::vector<Psp8_file>& psp8_files;
    const Domain_parallel_vertices_3D& domain_vertices;
    std::vector<Nloc_projector<T>> nloc_projectors;
    std::vector<uint> offsets;
    Effective_potential_nloc(const Geometry& geometry,
                             const Mesh_control& mesh_control,
                             const std::vector<Psp8_file>& psp8_files,
                             const Domain_parallel_vertices_3D& domain_vertices);
    ~Effective_potential_nloc();
    void generate_nloc_projectors();
    void generate_nloc_projectors_mp(Memory_pool<T, Fast_memory>& pool_fast,
                                    Memory_pool<T, Capacity_memory>& pool_cap);
    void init(const bool print_flag = true);
    void init_mp(const bool print_flag,
              Memory_pool<T, Fast_memory>& pool_fast,
              Memory_pool<T, Capacity_memory>& pool_cap);
    template<typename T2> void init(const Effective_potential_nloc<T2>& effective_potential_nloc);
    void destructor();
    void destructor_mp();
    void show(std::ostream& output = std::cout) const;
};

#endif //_EFFECTIVE_POTENTIAL_NLOC_H_