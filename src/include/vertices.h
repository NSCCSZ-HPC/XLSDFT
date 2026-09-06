#ifndef _VERTICES_H_
#define _VERTICES_H_

#include <iostream>
#include <climits>
#include <vector>
#include "linalg.h"

class Vertices_1D
{
public:
    int64_t is = 0;
    uint64_t ni = 0;
    Vertices_1D();
    Vertices_1D(const uint64_t ni);
    Vertices_1D(const int64_t is, const int64_t ie);
    Vertices_1D(const Vertices_1D& other);
    ~Vertices_1D();
    void swap(Vertices_1D& other);
    void set_vertices(const int64_t is, const int64_t ie);
    void set_vertices(const Vertices_1D& other);
    void set_i_vertices(const int64_t is, const int64_t ie);
    void set_ni(const uint64_t ni);
    Vertices_1D& operator=(const Vertices_1D& other);
    bool operator==(const Vertices_1D& other) const;
    bool operator!=(const Vertices_1D& other) const;
    int64_t get_is() const;
    int64_t get_ie() const;
    uint64_t get_ni() const;
    Vertices_1D get_vertices() const;
    Vertices_1D& get_vertices();
    bool contain_point(const int64_t i) const;
    bool contain_i(const int64_t i) const;
    bool is_index_legal(const uint64_t index) const;
    uint64_t get_size() const;
    int64_t get_i(const uint64_t index) const;
    int64_t get_i_nocheck(const uint64_t index) const;
    uint64_t get_index(const int64_t i) const;
    uint64_t get_index_DBC(const int64_t i) const;
    uint64_t get_index_PBC(const int64_t i) const;
    uint64_t get_index_nocheck(const int64_t i) const;
    int64_t map_i_into_vertices(const int64_t i) const;
    bool is_ex_vertices(const Vertices_1D& other) const;
    bool is_sub_vertices(const Vertices_1D& other) const;
    bool is_overlaped(const Vertices_1D& other) const;
    std::vector<Vertices_1D> split(const uint64_t npart_i) const;
    Vertices_1D get_overlap_vertices(const Vertices_1D& other) const;
    Vertices_1D get_overlap_vertices(const Vertices_1D& other, const int64_t i_shift) const;
    Vertices_1D get_super_vertices(const Vertices_1D& other) const;
    Vertices_1D get_super_vertices(const Vertices_1D& other, const int64_t i_shift) const;
    template<typename T>
    Vertices_1D generate_ex_vertices(const T nnode) const;
    template<typename T>
    Vertices_1D generate_ex_vertices(const T* nnode) const;
    void show() const;
};

class Vertices_2D : public Vertices_1D
{
public:
    int64_t js = 0;
    uint64_t nj = 0;
    Vertices_2D();
    Vertices_2D(const Vertices_1D& vertices_1d, const uint64_t nj);
    Vertices_2D(const Vertices_1D& vertices_1d, const int64_t js, const int64_t je);
    Vertices_2D(const uint64_t ni, const uint64_t nj);
    Vertices_2D(const int64_t is, const int64_t ie, const int64_t js, const int64_t je);
    Vertices_2D(const Vertices_2D& other);
    ~Vertices_2D();
    void swap(Vertices_2D& other);
    void set_vertices(const int64_t is, const int64_t ie, const int64_t js, const int64_t je);
    void set_vertices(const Vertices_2D& other);
    // // void set_i_vertices(const int64_t is, const int64_t ie);
    void set_j_vertices(const int64_t js, const int64_t je);
    void set_nj(const uint64_t nj);
    Vertices_2D& operator=(const Vertices_2D& other);
    bool operator==(const Vertices_2D& other) const;
    bool operator!=(const Vertices_2D& other) const;
    int64_t get_js() const;
    int64_t get_je() const;
    uint64_t get_nj() const;
    Vertices_2D get_vertices() const;
    Vertices_2D& get_vertices();
    bool contain_point(const int64_t i, const int64_t j) const;
    bool contain_i(const int64_t i) const;
    bool contain_j(const int64_t j) const;
    bool is_index_legal(const uint64_t index) const;
    uint64_t get_size() const;
    int64_t get_i(const uint64_t index) const;
    int64_t get_i_nocheck(const uint64_t index) const;
    int64_t get_j(const uint64_t index) const;
    int64_t get_j_nocheck(const uint64_t index) const;
    uint64_t get_index(const int64_t i, const int64_t j) const;
    uint64_t get_index_DBC(const int64_t i, const int64_t j) const;
    uint64_t get_index_PBC(const int64_t i, const int64_t j) const;
    uint64_t get_index_nocheck(const int64_t i, const int64_t j) const;
    int64_t map_i_into_vertices(const int64_t i) const;
    int64_t map_j_into_vertices(const int64_t j) const;
    bool is_ex_vertices(const Vertices_2D& other) const;
    bool is_sub_vertices(const Vertices_2D& other) const;
    bool is_overlaped(const Vertices_2D& other) const;
    std::vector<Vertices_2D> split(const uint64_t npart_i, const uint64_t npart_j) const;
    Vertices_2D get_overlap_vertices(const Vertices_2D& other) const;
    Vertices_2D get_overlap_vertices(const Vertices_2D& other, const int64_t i_shift, const int64_t j_shift) const;
    Vertices_2D get_super_vertices(const Vertices_2D& other) const;
    Vertices_2D get_super_vertices(const Vertices_2D& other, const int64_t i_shift, const int64_t j_shift) const;
    template<typename T>
    Vertices_2D generate_ex_vertices(const T nnode) const;
    template<typename T>
    Vertices_2D generate_ex_vertices(const T* nnode) const;
    void show() const;
};

class Vertices_3D : public Vertices_2D
{
public:
    int64_t ks = 0;
    uint64_t nk= 0;
    Vertices_3D();
    Vertices_3D(const Vertices_2D& vertices_2d, const uint64_t nk);
    Vertices_3D(const Vertices_2D& vertices_2d, const int64_t ks, const int64_t ke);
    Vertices_3D(const uint64_t ni, const uint64_t nj, const uint64_t nk);
    Vertices_3D(const int64_t is, const int64_t ie, const int64_t js, const int64_t je, const int64_t ks, const int64_t ke);
    Vertices_3D(const Vertices_3D& other);
    ~Vertices_3D();
    void swap(Vertices_3D& other);
    void set_vertices(const int64_t is, const int64_t ie, const int64_t js, const int64_t je, const int64_t ks, const int64_t ke);
    void set_vertices(const Vertices_3D& other);
//     // void set_i_vertices(const int64_t is, const int64_t ie);
//     // void set_j_vertices(const int64_t js, const int64_t je);
    void set_k_vertices(const int64_t ks, const int64_t ke);
    void set_nk(const uint64_t nk);
    Vertices_3D& operator=(const Vertices_3D& other);
    bool operator==(const Vertices_3D& other) const;
    bool operator!=(const Vertices_3D& other) const;
    int64_t get_ks() const;
    int64_t get_ke() const;
    uint64_t get_nk() const;
    Vertices_3D get_vertices() const;
    Vertices_3D& get_vertices();
    bool contain_point(const int64_t i, const int64_t j, const int64_t k) const;
    bool contain_i(const int64_t i) const;
    bool contain_j(const int64_t j) const;
    bool contain_k(const int64_t k) const;
    bool is_index_legal(const uint64_t index) const;
    uint64_t get_size() const;
    int64_t get_i(const uint64_t index) const;
    int64_t get_i_nocheck(const uint64_t index) const;
    int64_t get_j(const uint64_t index) const;
    int64_t get_j_nocheck(const uint64_t index) const;
    int64_t get_k(const uint64_t index) const;
    int64_t get_k_nocheck(const uint64_t index) const;
    uint64_t get_index(const int64_t i, const int64_t j, const int64_t k) const;
    uint64_t get_index_DBC(const int64_t i, const int64_t j, const int64_t k) const;
    uint64_t get_index_PBC(const int64_t i, const int64_t j, const int64_t k) const;
    uint64_t get_index_nocheck(const int64_t i, const int64_t j, const int64_t k) const;
    int64_t map_i_into_vertices(const int64_t i) const;
    int64_t map_j_into_vertices(const int64_t j) const;
    int64_t map_k_into_vertices(const int64_t k) const;
    bool is_ex_vertices(const Vertices_3D& other) const;
    bool is_sub_vertices(const Vertices_3D& other) const;
    bool is_overlaped(const Vertices_3D& other) const;
    std::vector<Vertices_3D> split(const uint64_t npart_i, const uint64_t npart_j, const uint64_t npart_k) const;
    Vertices_3D get_shifed_vertices(const int64_t i_shift, const int64_t j_shift, const int64_t k_shift) const;
    Vertices_3D get_overlap_vertices(const Vertices_3D& other) const;
    Vertices_3D get_overlap_vertices(const Vertices_3D& other, const int64_t i_shift, const int64_t j_shift, const int64_t k_shift) const;
    Vertices_3D get_super_vertices(const Vertices_3D& other) const;
    Vertices_3D get_super_vertices(const Vertices_3D& other, const int64_t i_shift, const int64_t j_shift, const int64_t k_shift) const;
    template<typename T>
    Vertices_3D generate_ex_vertices(const T nnode) const;
    template<typename T>
    Vertices_3D generate_ex_vertices(const T* nnode) const;
    void show() const;
};

class Vertices_4D : public Vertices_3D
{
public:
    int64_t bs = 0;
    uint64_t nb = 0;
    Vertices_4D();
    Vertices_4D(const Vertices_3D& vertices_3d, const uint64_t nb);
    Vertices_4D(const Vertices_3D& vertices_3d, const int64_t bs, const int64_t be);
    Vertices_4D(const uint64_t ni, const uint64_t nj, const uint64_t nk, const uint64_t nb);
    Vertices_4D(const int64_t is, const int64_t ie, const int64_t js, const int64_t je,
                const int64_t ks, const int64_t ke, const int64_t bs, const int64_t be);
    Vertices_4D(const Vertices_4D& other);
    ~Vertices_4D();
    void swap(Vertices_4D& other);
    void set_vertices(const int64_t is, const int64_t ie, const int64_t js, const int64_t je,
                      const int64_t ks, const int64_t ke, const int64_t bs, const int64_t be);
    void set_vertices(const Vertices_4D& other);
    // void set_i_vertices(const int64_t is, const int64_t ie);
    // void set_j_vertices(const int64_t js, const int64_t je);
    // void set_k_vertices(const int64_t ks, const int64_t ke);
    void set_b_vertices(const int64_t bs, const int64_t be);
    void set_nb(const uint64_t nb);
    Vertices_4D& operator=(const Vertices_4D& other);
    bool operator==(const Vertices_4D& other) const;
    bool operator!=(const Vertices_4D& other) const;
    int64_t get_bs() const;
    int64_t get_be() const;
    uint64_t get_nb() const;
    Vertices_4D get_vertices() const;
    Vertices_4D& get_vertices();
    bool contain_point(const int64_t i, const int64_t j, const int64_t k, const int64_t b) const;
    bool contain_i(const int64_t i) const;
    bool contain_j(const int64_t j) const;
    bool contain_k(const int64_t k) const;
    bool contain_b(const int64_t b) const;
    bool is_index_legal(const uint64_t index) const;
    uint64_t get_size() const;
    int64_t get_i(const uint64_t index) const;
    int64_t get_i_nocheck(const uint64_t index) const;
    int64_t get_j(const uint64_t index) const;
    int64_t get_j_nocheck(const uint64_t index) const;
    int64_t get_k(const uint64_t index) const;
    int64_t get_k_nocheck(const uint64_t index) const;
    int64_t get_b(const uint64_t index) const;
    int64_t get_b_nocheck(const uint64_t index) const;
    uint64_t get_index(const int64_t i, const int64_t j, const int64_t k, const int64_t b) const;
    uint64_t get_index_DBC(const int64_t i, const int64_t j, const int64_t k, const int64_t b) const;
    uint64_t get_index_PBC(const int64_t i, const int64_t j, const int64_t k, const int64_t b) const;
    uint64_t get_index_nocheck(const int64_t i, const int64_t j, const int64_t k, const int64_t b) const;
    uint64_t get_index_nocheck_rowmaj(const int64_t i, const int64_t j, const int64_t k, const int64_t b) const;
    int64_t map_i_into_vertices(const int64_t i) const;
    int64_t map_j_into_vertices(const int64_t j) const;
    int64_t map_k_into_vertices(const int64_t k) const;
    int64_t map_b_into_vertices(const int64_t d) const;
    bool is_ex_vertices(const Vertices_4D& other) const;
    bool is_sub_vertices(const Vertices_4D& other) const;
    bool is_overlaped(const Vertices_4D& other) const;
    std::vector<Vertices_4D> split(const uint64_t npart_i, const uint64_t npart_j, const uint64_t npart_k, const uint64_t npart_b) const;
    Vertices_4D get_overlap_vertices(const Vertices_4D& other) const;
    Vertices_4D get_overlap_vertices(const Vertices_4D& other, const int64_t i_shift, const int64_t j_shift, const int64_t k_shift, const int64_t b_shift) const;
    Vertices_4D get_super_vertices(const Vertices_4D& other) const;
    Vertices_4D get_super_vertices(const Vertices_4D& other, const int64_t i_shift, const int64_t j_shift, const int64_t k_shift, const int64_t b_shift) const;
    template<typename T>
    Vertices_4D generate_ex_vertices(const T nnode) const;
    template<typename T>
    Vertices_4D generate_ex_vertices(const T* nnode) const;
    void show() const;
};

#endif //_VERTICES_H_

namespace Vertices_method {
    template<typename T> void fill_vector(T const* const my_data, const Vertices_3D& my_vertices, T* const other_data,
                                          const Vertices_3D& my_fill_region, const uint64_t offset);
    template<typename T> void fill_vector(T const* const my_data, const Vertices_4D& my_vertices, T* const other_data,
                                          const Vertices_4D& my_fill_region, const uint64_t offset);
    template<typename T> void be_filled_vector(T* const my_data, const Vertices_2D& my_vertices,
                                               T const* const other_data, const Vertices_2D& other_vertices,
                                               const Vertices_2D& my_filled_region, const uint64_t offset);
    template<typename T> void be_filled_vector(T* const my_data, const Vertices_3D& my_vertices,
                                               T const* const other_data, const Vertices_3D& other_vertices,
                                               const Vertices_3D& my_filled_region, const uint64_t offset);
    template<typename T> void be_filled_vector(T* const my_data, const Vertices_4D& my_vertices,
                                               T const* const other_data, const Vertices_4D& other_vertices,
                                               const Vertices_4D& my_filled_region, const uint64_t offset);
    template<typename T> void fill_region(T const* const my_data, const Vertices_1D& my_vertices, 
                                          T* const other_data, const Vertices_1D& other_vertices,
                                          const Vertices_1D& region);
    template<typename T> void fill_region(T const* const my_data, const Vertices_2D& my_vertices, 
                                          T* const other_data, const Vertices_2D& other_vertices,
                                          const Vertices_2D& region);
    template<typename T> void fill_region(T const* const my_data, const Vertices_3D& my_vertices, 
                                          T* const other_data, const Vertices_3D& other_vertices,
                                          const Vertices_3D& region);
    template<typename T> void fill_region(T const* const my_data, const Vertices_4D& my_vertices, 
                                          T* const other_data, const Vertices_4D& other_vertices,
                                          const Vertices_4D& region);
    template<typename T> void accumulate_overlap(T const* const& my_data, const Vertices_3D& my_vertices, 
                                                 T* const& other_data, const Vertices_3D& other_vertices,
                                                 const Vertices_3D& region);
    template<typename T> void accumulate_overlap(T const* const& my_data, const Vertices_4D& my_vertices, 
                                                 T* const& other_data, const Vertices_4D& other_vertices,
                                                 const Vertices_4D& region);
}
