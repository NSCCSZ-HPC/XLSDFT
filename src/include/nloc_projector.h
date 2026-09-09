#ifndef _NLOC_PROJECTOR_H_
#define _NLOC_PROJECTOR_H_

#include <vector>
#include "arr.h"
#include "filesys.h"
// #include "geometry.h"
// #include "control.h"

template<typename T>
class Nloc_projector
{
public:
    const Psp8_file& psp8_file;
    uint ncol = 0;
    uint nrow = 0;
    int atom_index = -1;
    bool is_real = true;                //not a image as a mark 
    bool is_in_domain = true;                //not a image as a mark 
    int cell_shift_x = 0;
    int cell_shift_y = 0;
    int cell_shift_z = 0;
    Array_2D<T> chi;
    Array_0D<T> gamma;
    std::vector<uint> index;
#ifdef USE_HBM
    uint index_hbm_length = 0;
    uint* index_hbm = nullptr;
#endif
    Nloc_projector(const Psp8_file& psp8_file);
    Nloc_projector(const Nloc_projector<T>& nloc_projector);
    template<typename T2> Nloc_projector(const Nloc_projector<T2>& nloc_projector);
    Nloc_projector(Nloc_projector<T>&& nloc_projector);
    ~Nloc_projector();
    void generate_ncol();
    void generate_basic();
    void set_nrow(const uint& nrow);
    void resize();
    Nloc_projector<T>& operator=(const Nloc_projector<T>& other);
    template<typename T2> Nloc_projector<T>& operator=(const Nloc_projector<T2>& other);
    void copy_mp(const Nloc_projector<T>& other,
                Memory_pool<T, Fast_memory>& pool_fast,
                Memory_pool<T, Capacity_memory>& pool_cap);
    void init();
    template<typename T2> void init(const Nloc_projector<T2>& other);
    void destructor();
    void destructor_mp();
    void show(std::ostream& output = std::cout) const;
    uint const* index_data() const;
};

namespace Nloc_projector_method {
    template<typename T> Nloc_projector<T> merge(const std::vector<Nloc_projector<T>>& nloc_projectors, const Psp8_file& psp8_file);
    // template<typename T> void nloc_projector_product_vectors(Array_1D<T>& result, const Array_3D<T>& vectors,
    //                                                          const Nloc_projector<T> nloc_projector);
    template<typename T> void nloc_projector_product_vectors(T* const& result, T const* const& vectors,
                                                             const Nloc_projector<T>& nloc_projector, const T& dv);
    // template<typename T> void nloc_projector_product_vectors(Array_2D<T>& result, const Array_4D<T>& vectors,
    //                                                          const Nloc_projector<T> nloc_projector);
    template<typename T> void nloc_projector_product_vectors(T* const& result, T const* const& vectors, const Vertices_4D& vertices,
                                                             const Nloc_projector<T>& nloc_projector, const T& dv);
    template<typename T> void nloc_projector_product_vectors_sequential(T* const result, T const* const vectors, const Vertices_4D& vertices,
                                                             const Nloc_projector<T>& nloc_projector, const T dv);
    template<typename T> void nloc_projector_product_vectors_sequential_with_buffer(T* const result, T const* const vectors, const Vertices_4D& vertices,
                                                             const Nloc_projector<T>& nloc_projector, const T dv, T* const buffer);
}

#endif //_NLOC_PROJECTOR_H_