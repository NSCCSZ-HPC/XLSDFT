#ifndef _ARR_H_
#define _ARR_H_

#include <iostream>
#include <climits>
#include <cassert>
#include <vector>
#include "linalg.h"
#include "vertices.h"
#include "stencil.h"
#include "memory_pool.h"
#include <mpi.h>
#include <new> //c++17 aligned new

#ifdef USE_HUGEPAGE_HBM
/**
 * 向上对齐到 align 的倍数（align 必须为 2 的幂）
 */
inline std::size_t align_size(std::size_t align, std::size_t sz) noexcept {
    return (sz + align - 1) & ~(align - 1);
}

/**
 * 分配绑定到指定 NUMA 节点的 HBM 内存（支持普通页或 2MB 大页）
 * @param size      [in,out] 请求的字节数，返回实际分配的字节数（对齐后，可能因大页不足而缩小）
 * @param hbm_node  目标 NUMA 节点 ID（通常为 HBM 所在节点）
 * @param use_hugepage  true 使用大页 (2MB)，false 使用普通页 (sysconf(_SC_PAGESIZE))
 * @return          分配的内存指针，失败时抛出异常
 */
void* hbm_alloc(std::size_t& size, unsigned long hbm_node, bool use_hugepage);

/**
 * 释放由 hbm_alloc 分配的内存
 * @param ptr         内存指针
 * @param size        当初分配时返回的实际字节数（必须与 hbm_alloc 修改后的 size 一致）
 * @param use_hugepage 必须与分配时使用的标志相同
 */
void hbm_free(void* ptr, std::size_t size, bool use_hugepage);
#endif

template<typename T>
class Array_0D
{
public:
    bool is_Col_Maj = true;
    uint64_t length;
    T* data = nullptr;
    Array_0D();
    Array_0D(const uint64_t length);
    Array_0D(const uint64_t length, const uint);
    Array_0D(const Array_0D& other);
    Array_0D(Array_0D&& other);
    virtual ~Array_0D();
    void destructor();
    void constructor(const uint64_t length);
    void constructor(const uint64_t length, const uint);
    void reconstructor(const uint64_t length);
    void reconstructor(const uint64_t length, const uint);
    #ifdef USE_HBM
    #ifdef USE_HUGEPAGE_HBM
        void constructor_hbm(const uint64_t length);
        void destructor_hbm();
        void reconstructor_hbm(const uint64_t length);
        void constructor_hp_hbm(const uint64_t length);
        void destructor_hp_hbm();
        void reconstructor_hp_hbm(const uint64_t length);
    #else
        void constructor_hbm(const uint64_t length);
        void destructor_hbm();
        void reconstructor_hbm(const uint64_t length);
    #endif
    #endif
    void swap(Array_0D<T>& other);
    template<typename T2> Array_0D<T2> as_type(const T2*) const;
    // template<typename T2> void convert_type(const Array_0D<T2>& array_A);
    void set_is_Col_Maj(const bool is_Col_Maj);
    bool get_is_Col_Maj() const;
    // virtual bool get_is_Col_Maj() const {
    //     assert(0 && "Array_0D dont have the leading dimension.");
    //     return 3;
    // }
    virtual uint get_leading_dimension() const {
        assert(0 && "Array_0D dont have the leading dimension.");
        return 3;
    }
    virtual uint get_second_dimension() const {
        assert(0 && "Array_0D dont have the second dimension.");
        return 3;
    }
    Array_0D& operator=(const Array_0D& other);
    Array_0D& operator=(Array_0D&& other);
    Array_0D& operator=(const T num);
    Array_0D& deepcopy(const Array_0D& other);
    Array_0D& deepcopy(Array_0D&& other);
    Array_0D& operator+=(const Array_0D& other);
    Array_0D& operator-=(const Array_0D& other);
    Array_0D& operator*=(const Array_0D& other);
    Array_0D& operator/=(const Array_0D& other);
    Array_0D& operator+=(const T num);
    Array_0D& operator-=(const T num);
    Array_0D& operator*=(const T num);
    Array_0D& operator/=(const T num);
    Array_0D operator+(const Array_0D& other) const;
    Array_0D operator-(const Array_0D& other) const;
    Array_0D operator*(const Array_0D& other) const;
    Array_0D operator/(const Array_0D& other) const;
    Array_0D operator+(const T num);
    Array_0D operator-(const T num);
    Array_0D operator*(const T num);
    Array_0D operator/(const T num);
    T& operator[](const int index);
    T operator[](const int index) const;
    uint get_length() const;
    T*& get_data();
    T* get_data() const;
    std::size_t get_bytes() const;
    int get_type_id() const;
    bool is_type_double() const;
    bool is_type_float() const;
    bool is_type_int() const;
    Array_0D sub_arr(const int index_s, const int index_e);
    void show() const;
    void print() const;
    void printX() const;
    void print_Col_Maj() const;
    void print_Row_Maj() const;
    void print(const std::string& fname) const;
    void printX(const std::string& fname) const;
    void print_Col_Maj(const std::string& fname) const;
    void print_Row_Maj(const std::string& fname) const;
    virtual void print_h(std::ostream& output = std::cout) const {this->print_Col_Maj_h(output);return;}
    void printX_h(std::ostream& output = std::cout) const;
    virtual void print_Col_Maj_h(std::ostream& output = std::cout) const {
        for (uint index = 0; index < this->length; index++) {
            output << std::right << std::setw(14) << std::setprecision(6) << std::scientific << std::uppercase << this->data[index];
            if (index % 6 == 5)
                output << std::endl;
        }
        output << std::endl;
        return;
    }
    virtual void print_Row_Maj_h(std::ostream& output = std::cout) const {this->print_Col_Maj_h(output);return;}

//Vertor functions
    void seededrand(const int seed, const T rand_min = -0.5, const T rand_max = 0.5);
    T vector_sum() const;
    T vector_norm_square_sum(const MPI_Comm comm = MPI_COMM_NULL) const;
    T vector_2norm(const MPI_Comm comm = MPI_COMM_NULL) const;
    void hadamard_plus_general(const Array_0D& array_A);
    void hadamard_minus_general(const Array_0D& array_A);
    void hadamard_product_general(const Array_0D& array_A);
    void hadamard_divide_general(const Array_0D& array_A);
    void hadamard_product_general(const Array_0D& array_A, T const alpha);
    void hadamard_product_general(const Array_0D& array_A, T const alpha, T const beta);
    void hadamard_product_general(const Array_0D& array_A, T const alpha, T const beta, T const gamma);
    void hadamard_plus_general(const Array_0D& array_A, const Array_0D& array_B);
    void hadamard_minus_general(const Array_0D& array_A, const Array_0D& array_B);
    void hadamard_product_general(const Array_0D& array_A, const Array_0D& array_B);
    void hadamard_divide_general(const Array_0D& array_A, const Array_0D& array_B);
    void hadamard_product_general(const Array_0D& array_A, const Array_0D& array_B, T const alpha);
    void hadamard_product_general(const Array_0D& array_A, const Array_0D& array_B, T const alpha, T const beta);
    void hadamard_product_general(const Array_0D& array_A, const Array_0D& array_B, T const alpha, T const beta, T const gamma);
    void accumulate_hadamard_plus_general(const Array_0D& array_A, const Array_0D& array_B);
    void accumulate_hadamard_minus_general(const Array_0D& array_A, const Array_0D& array_B);
    void accumulate_hadamard_product_general(const Array_0D& array_A, const Array_0D& array_B);
    void accumulate_hadamard_divide_general(const Array_0D& array_A, const Array_0D& array_B);
    void accumulate_hadamard_product_general(const Array_0D& array_A, const Array_0D& array_B, T const alpha);
    void accumulate_hadamard_product_general(const Array_0D& array_A, const Array_0D& array_B, T const alpha, T const beta);
    void accumulate_hadamard_product_general(const Array_0D& array_A, const Array_0D& array_B, T const alpha, T const beta, T const gamma);
    void scalar_plus_general(const T num);
    void scalar_minus_general(const T num);
    void scalar_product_general(const T num);
    void scalar_divide_general(const T num);
    void scalar_product_general(const T num, const Array_0D& array_B);
    void scalar_product_general(const T num, const Array_0D& array_B, T const alpha);
    void scalar_product_general(const T num, const Array_0D& array_B, T const alpha, T const gamma);
    void scalar_plus_general(const Array_0D& array_A, const T num);
    void scalar_minus_general(const Array_0D& array_A, const T num);
    void scalar_product_general(const Array_0D& array_A, const T num);
    void scalar_divide_general(const Array_0D& array_A, const T num);
    void scalar_product_general(const Array_0D& array_A, const T num, const Array_0D& array_B);
    void scalar_product_general(const Array_0D& array_A, const T num, const Array_0D& array_B, T const alpha);
    void scalar_product_general(const Array_0D& array_A, const T num, const Array_0D& array_B, T const alpha, T const gamma);
    void accumulate_scalar_plus_general(const Array_0D& array_A, const T num);
    void accumulate_scalar_minus_general(const Array_0D& array_A, const T num);
    void accumulate_scalar_product_general(const Array_0D& array_A, const T num);
    void accumulate_scalar_divide_general(const Array_0D& array_A, const T num);
    void accumulate_scalar_product_general(const Array_0D& array_A, const T num, const Array_0D& array_B);
    void accumulate_scalar_product_general(const Array_0D& array_A, const T num, const Array_0D& array_B, T const alpha);
    void accumulate_scalar_product_general(const Array_0D& array_A, const T num, const Array_0D& array_B, T const alpha, T const gamma);
};

template<typename T>
class Array_1D : public Array_0D<T>, public Vertices_1D
{
public:
    Array_1D();
    Array_1D(const uint64_t length);
    Array_1D(const uint64_t length, const uint);
    Array_1D(const Vertices_1D& other);
    Array_1D(const Vertices_1D& other, const uint);
    Array_1D(const Array_1D& other);
    Array_1D(Array_1D&& other);
    ~Array_1D() override;
    void destructor();
    void constructor(const uint64_t length);
    void constructor(const Vertices_1D& other);
    void constructor(const Vertices_1D& other, const uint);
    void reconstructor(const uint64_t length);
    void reconstructor(const Vertices_1D& other);
    void reconstructor(const Vertices_1D& other, const uint);
    #ifdef USE_HBM
    #ifdef USE_HUGEPAGE_HBM
        void constructor_hbm(const uint64_t length);
        void constructor_hbm(const Vertices_1D& other);
        void destructor_hbm();
        void reconstructor_hbm(const uint64_t length);
        void reconstructor_hbm(const Vertices_1D& other);
        void constructor_hp_hbm(const uint64_t length);
        void constructor_hp_hbm(const Vertices_1D& other);
        void destructor_hp_hbm();
        void reconstructor_hp_hbm(const uint64_t length);
        void reconstructor_hp_hbm(const Vertices_1D& other);
    #else
        void constructor_hbm(const uint64_t length);
        void constructor_hbm(const Vertices_1D& other);
        void destructor_hbm();
        void reconstructor_hbm(const uint64_t length);
        void reconstructor_hbm(const Vertices_1D& other);
    #endif
    #endif
    void swap(Array_1D<T>& other);
    template<typename T2> Array_1D<T2> as_type(const T2*) const;
    uint get_leading_dimension() const override;
    uint get_second_dimension() const override;
    Array_1D& operator=(const Array_1D& other);
    Array_1D& operator=(Array_1D&& other);
    Array_1D& operator=(const T num);
    Array_1D& deepcopy(const Array_1D& other);
    Array_1D& deepcopy(Array_1D&& other);
    Array_1D& operator+=(const Array_1D& other);
    Array_1D& operator-=(const Array_1D& other);
    Array_1D& operator*=(const Array_1D& other);
    Array_1D& operator/=(const Array_1D& other);
    Array_1D& operator+=(const T num);
    Array_1D& operator-=(const T num);
    Array_1D& operator*=(const T num);
    Array_1D& operator/=(const T num);
    Array_1D operator+(const Array_1D& other) const;
    Array_1D operator-(const Array_1D& other) const;
    Array_1D operator*(const Array_1D& other) const;
    Array_1D operator/(const Array_1D& other) const;
    Array_1D operator+(const T num) const;
    Array_1D operator-(const T num) const;
    Array_1D operator*(const T num) const;
    Array_1D operator/(const T num) const;
    Array_1D sub_arr(const Vertices_1D& other) const;
    Array_1D& sub_arr(Array_1D& other) const;
    Array_1D& fill_overlap(Array_1D& other) const;
    Array_1D& fill_overlap(Array_1D& other, const Vertices_1D& temp) const;
    Array_1D& be_filled_overlap(const Array_1D& other);
    Array_1D& be_filled_overlap(const Array_1D& other, const Vertices_1D& temp);
    void show() const;
    void print_h(std::ostream& output = std::cout) const override;
    void print_Col_Maj_h(std::ostream& output = std::cout) const override;
    void print_Row_Maj_h(std::ostream& output = std::cout) const override;
};

template<typename T>
class Array_2D : public Array_0D<T>, public Vertices_2D
{
public:
    Array_2D();
    Array_2D(const uint64_t length);
    Array_2D(const uint64_t length, const uint);
    Array_2D(const Vertices_2D& other);
    Array_2D(const Vertices_2D& other, const uint);
    Array_2D(const Array_2D& other);
    Array_2D(Array_2D&& other);
    ~Array_2D() override;
    void destructor();
    void constructor(const uint64_t length);
    void constructor(const uint64_t ni, const uint64_t nj);
    void constructor(const Vertices_2D& other);
    void constructor(const Vertices_2D& other, const uint);
    void reconstructor(const uint64_t length);
    void reconstructor(const uint64_t ni, const uint64_t nj);
    void reconstructor(const Vertices_2D& other);
    void reconstructor(const Vertices_2D& other, const uint);
    #ifdef USE_HBM
    #ifdef USE_HUGEPAGE_HBM
        void constructor_hbm(const uint64_t length);
        void constructor_hbm(const uint64_t ni, const uint64_t nj);
        void constructor_hbm(const Vertices_2D& other);
        void destructor_hbm();
        void reconstructor_hbm(const uint64_t length);
        void reconstructor_hbm(const uint64_t ni, const uint64_t nj);
        void reconstructor_hbm(const Vertices_2D& other);
        void constructor_hp_hbm(const uint64_t length);
        void constructor_hp_hbm(const uint64_t ni, const uint64_t nj);
        void constructor_hp_hbm(const Vertices_2D& other);
        void destructor_hp_hbm();
        void reconstructor_hp_hbm(const uint64_t length);
        void reconstructor_hp_hbm(const uint64_t ni, const uint64_t nj);
        void reconstructor_hp_hbm(const Vertices_2D& other);
    #else
        void constructor_hbm(const uint64_t length);
        void constructor_hbm(const uint64_t ni, const uint64_t nj);
        void constructor_hbm(const Vertices_2D& other);
        void destructor_hbm();
        void reconstructor_hbm(const uint64_t length);
        void reconstructor_hbm(const uint64_t ni, const uint64_t nj);
        void reconstructor_hbm(const Vertices_2D& other);
    #endif
    #endif
    void swap(Array_2D<T>& other);
    template<typename T2> Array_2D<T2> as_type(const T2*) const;
    uint get_leading_dimension() const override;
    uint get_second_dimension() const override;
    Array_2D& operator=(const Array_2D& other);
    Array_2D& operator=(Array_2D&& other);
    Array_2D& operator=(const T num);
    Array_2D& deepcopy(const Array_2D& other);
    Array_2D& deepcopy(Array_2D&& other);
    Array_2D& operator+=(const Array_2D& other);
    Array_2D& operator-=(const Array_2D& other);
    Array_2D& operator*=(const Array_2D& other);
    Array_2D& operator/=(const Array_2D& other);
    Array_2D& operator+=(const T num);
    Array_2D& operator-=(const T num);
    Array_2D& operator*=(const T num);
    Array_2D& operator/=(const T num);
    Array_2D operator+(const Array_2D& other) const;
    Array_2D operator-(const Array_2D& other) const;
    Array_2D operator*(const Array_2D& other) const;
    Array_2D operator/(const Array_2D& other) const;
    Array_2D operator+(const T num) const;
    Array_2D operator-(const T num) const;
    Array_2D operator*(const T num) const;
    Array_2D operator/(const T num) const;
    Array_2D sub_arr(const Vertices_2D& other) const;
    Array_2D& sub_arr(Array_2D& other) const;
    Array_2D& fill_overlap(Array_2D& other) const;
    Array_2D& fill_overlap(Array_2D& other, const Vertices_2D& temp) const;
    Array_2D& be_filled_overlap(const Array_2D& other);
    Array_2D& be_filled_overlap(const Array_2D& other, const Vertices_2D& temp);
    void show() const;
    void print_h(std::ostream& output = std::cout) const override;
    void print_Col_Maj_h(std::ostream& output = std::cout) const override;
    void print_Row_Maj_h(std::ostream& output = std::cout) const override;
};

template<typename T>
class Array_3D : public Array_0D<T>, public Vertices_3D
{
public:
    Array_3D();
    Array_3D(const uint64_t length);
    Array_3D(const uint64_t length, const uint);
    Array_3D(const Vertices_3D& other);
    Array_3D(const Vertices_3D& other, const uint);
    Array_3D(const Array_3D& other);
    Array_3D(Array_3D&& other);
    ~Array_3D() override;
    void destructor();
    void constructor(const uint64_t length);
    void constructor(const uint64_t ni, const uint64_t nj, const uint64_t nk);
    void constructor(const Vertices_3D& other);
    void constructor(const Vertices_3D& other, const uint);
    void reconstructor(const uint64_t length);
    void reconstructor(const uint64_t ni, const uint64_t nj, const uint64_t nk);
    void reconstructor(const Vertices_3D& other);
    void reconstructor(const Vertices_3D& other, const uint);
    #ifdef USE_HBM
    #ifdef USE_HUGEPAGE_HBM
        void constructor_hbm(const uint64_t length);
        void constructor_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk);
        void constructor_hbm(const Vertices_3D& other);
        void destructor_hbm();
        void reconstructor_hbm(const uint64_t length);
        void reconstructor_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk);
        void reconstructor_hbm(const Vertices_3D& other);
        void constructor_hp_hbm(const uint64_t length);
        void constructor_hp_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk);
        void constructor_hp_hbm(const Vertices_3D& other);
        void destructor_hp_hbm();
        void reconstructor_hp_hbm(const uint64_t length);
        void reconstructor_hp_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk);
        void reconstructor_hp_hbm(const Vertices_3D& other);
    #else
        void constructor_hbm(const uint64_t length);
        void constructor_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk);
        void constructor_hbm(const Vertices_3D& other);
        void destructor_hbm();
        void reconstructor_hbm(const uint64_t length);
        void reconstructor_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk);
        void reconstructor_hbm(const Vertices_3D& other);
    #endif
    #endif
    void swap(Array_3D<T>& other);
    template<typename T2> Array_3D<T2> as_type(const T2*) const;
    uint get_leading_dimension() const override;
    uint get_second_dimension() const override;
    Array_3D& operator=(const Array_3D& other);
    Array_3D& operator=(Array_3D&& other);
    Array_3D& operator=(const T num);
    Array_3D& deepcopy(const Array_3D& other);
    Array_3D& deepcopy(Array_3D&& other);
    Array_3D& operator+=(const Array_3D& other);
    Array_3D& operator-=(const Array_3D& other);
    Array_3D& operator*=(const Array_3D& other);
    Array_3D& operator/=(const Array_3D& other);
    Array_3D& operator+=(const T num);
    Array_3D& operator-=(const T num);
    Array_3D& operator*=(const T num);
    Array_3D& operator/=(const T num);
    Array_3D operator+(const Array_3D& other) const;
    Array_3D operator-(const Array_3D& other) const;
    Array_3D operator*(const Array_3D& other) const;
    Array_3D operator/(const Array_3D& other) const;
    Array_3D operator+(const T num) const;
    Array_3D operator-(const T num) const;
    Array_3D operator*(const T num) const;
    Array_3D operator/(const T num) const;
    Array_3D sub_arr(const Vertices_3D& other) const;
    Array_3D& sub_arr(Array_3D& other) const;
    Array_3D& fill_overlap(Array_3D& other) const;
    Array_3D& fill_overlap(Array_3D& other, const Vertices_3D& temp) const;
    Array_3D& accumulate_overlap(Array_3D& other) const;
    Array_3D& accumulate_overlap(Array_3D& other, const Vertices_3D& temp) const;
    Array_0D<T>& fill_vector(Array_0D<T>& other, const Vertices_3D& temp, const uint& offset) const;
    Array_3D& be_filled_overlap(const Array_3D& other);
    Array_3D& be_filled_overlap(const Array_3D& other, const Vertices_3D& temp);
    Array_3D& be_accumulated_overlap(const Array_3D& other);
    Array_3D& be_accumulated_overlap(const Array_3D& other, const Vertices_3D& temp);
    Array_3D& be_filled_vector(const Array_0D<T>& other, const Vertices_3D& temp, uint offset);
    Array_3D& be_filled_vector(const Array_0D<T>& other, const Vertices_3D& s_vertices,
                               const Vertices_3D& t_vertices, const uint& offset);
    void show() const;
    void print_h(std::ostream& output = std::cout) const override;
    void print_Col_Maj_h(std::ostream& output = std::cout) const override;
    void print_Row_Maj_h(std::ostream& output = std::cout) const override;

//FD functions
    // Array_3D calc_gradient(const Vertices_3D& temp, const Stencil<T>& stencil, char direction) const;
    // Array_3D calc_gradient(const Array_3D& other, const Stencil<T>& stencil, char direction) const;
    // Array_3D calc_gradient(const Array_3D& other, const Stencil<T>& stencil, char direction, const T& beta) const;
    // Array_3D calc_laplacian(const Vertices_3D& temp, const Stencil<T>& stencil) const;
    // Array_3D calc_laplacian(const Vertices_3D& temp, const Stencil<T>& stencil, const Array_3D<T>& arr_A, const T& alpha) const;
    // Array_3D calc_laplacian(const Vertices_3D& temp, const Stencil<T>& stencil, const Array_3D<T>& arr_A, const T& alpha,
    //                         const Array_3D<T>& arr_B, const T& beta) const;
    // Array_3D calc_laplacian(const Vertices_3D& temp, const Stencil<T>& stencil, const Array_3D<T>& arr_A, const T& alpha,
    //                         const Array_3D<T>& arr_B, const T& beta, const T& gamma) const;
};

template<typename T>
class Array_4D : public Array_0D<T>, public Vertices_4D
{
public:
    Array_4D();
    Array_4D(const uint64_t length);
    Array_4D(const uint64_t length, const uint);
    Array_4D(const Vertices_4D& other);
    Array_4D(const Vertices_4D& other, const uint);
    Array_4D(const Array_4D& other);
    Array_4D(Array_4D&& other);
    ~Array_4D() override;
    void destructor();
    void constructor(const uint64_t length);
    void constructor(const uint64_t ni, const uint64_t nj, const uint64_t nk, const uint64_t nb);
    void constructor(const Vertices_4D& other);
    void constructor(const Vertices_4D& other, const uint);
    void reconstructor(const uint64_t length);
    void reconstructor(const uint64_t ni, const uint64_t nj, const uint64_t nk, const uint64_t nb);
    void reconstructor(const Vertices_4D& other);
    void reconstructor(const Vertices_4D& other, const uint);
    #ifdef USE_HBM
    #ifdef USE_HUGEPAGE_HBM
        void constructor_hbm(const uint64_t length);
        void constructor_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk, const uint64_t nb);
        void constructor_hbm(const Vertices_4D& other);
        void destructor_hbm();
        void reconstructor_hbm(const uint64_t length);
        void reconstructor_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk, const uint64_t nb);
        void reconstructor_hbm(const Vertices_4D& other);
        void constructor_hp_hbm(const uint64_t length);
        void constructor_hp_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk, const uint64_t nb);
        void constructor_hp_hbm(const Vertices_4D& other);
        void destructor_hp_hbm();
        void reconstructor_hp_hbm(const uint64_t length);
        void reconstructor_hp_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk, const uint64_t nb);
        void reconstructor_hp_hbm(const Vertices_4D& other);
    #else
        void constructor_hbm(const uint64_t length);
        void constructor_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk, const uint64_t nb);
        void constructor_hbm(const Vertices_4D& other);
        void destructor_hbm();
        void reconstructor_hbm(const uint64_t length);
        void reconstructor_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk, const uint64_t nb);
        void reconstructor_hbm(const Vertices_4D& other);
    #endif
    #endif
    void swap(Array_4D<T>& other);
    template<typename T2> Array_4D<T2> as_type(const T2*) const;
    uint get_leading_dimension() const override;
    uint get_second_dimension() const override;
    Array_4D& operator=(const Array_4D& other);
    Array_4D& operator=(Array_4D&& other);
    Array_4D& operator=(const T num);
    Array_4D& deepcopy(const Array_4D& other);
    Array_4D& deepcopy(Array_4D&& other);
    Array_4D& operator+=(const Array_4D& other);
    Array_4D& operator-=(const Array_4D& other);
    Array_4D& operator*=(const Array_4D& other);
    Array_4D& operator/=(const Array_4D& other);
    Array_4D& operator+=(const T num);
    Array_4D& operator-=(const T num);
    Array_4D& operator*=(const T num);
    Array_4D& operator/=(const T num);
    Array_4D operator+(const Array_4D& other) const;
    Array_4D operator-(const Array_4D& other) const;
    Array_4D operator*(const Array_4D& other) const;
    Array_4D operator/(const Array_4D& other) const;
    Array_4D operator+(const T num) const;
    Array_4D operator-(const T num) const;
    Array_4D operator*(const T num) const;
    Array_4D operator/(const T num) const;
    Array_4D sub_arr(const Vertices_4D& other) const;
    Array_4D& sub_arr(Array_4D& other) const;
    Array_4D& fill_overlap(Array_4D& other) const;
    Array_4D& fill_overlap(Array_4D& other, const Vertices_4D& temp) const;
    Array_0D<T>& fill_vector(Array_0D<T>& other, const Vertices_4D& temp, const uint offset) const;
    Array_4D& be_filled_overlap(const Array_4D& other);
    Array_4D& be_filled_overlap(const Array_4D& other, const Vertices_4D& temp);
    Array_4D& be_filled_vector(const Array_0D<T>& other, const Vertices_4D& temp, uint offset);
    Array_4D& be_filled_vector(const Array_0D<T>& other, const Vertices_4D& s_vertices,
                               const Vertices_4D& t_vertices, const uint offset);
    void show() const;
    void print_h(std::ostream& output = std::cout) const override;
    void print_Col_Maj_h(std::ostream& output = std::cout) const override;
    void print_Row_Maj_h(std::ostream& output = std::cout) const override;

//FD functions
    // Array_4D calc_gradient(const Vertices_4D& temp, const Stencil<T>& stencil, char direction) const;
    // Array_4D calc_gradient(const Array_4D& other, const Stencil<T>& stencil, char direction) const;
    // Array_4D calc_gradient(const Array_4D& other, const Stencil<T>& stencil, char direction, const T& beta) const;
    // Array_4D calc_laplacian(const Vertices_4D& temp, const Stencil<T>& stencil) const;
    // Array_4D calc_laplacian(const Vertices_4D& temp, const Stencil<T>& stencil, const Array_4D<T>& arr_A, const T& alpha) const;
    // Array_4D calc_laplacian(const Vertices_4D& temp, const Stencil<T>& stencil, const Array_4D<T>& arr_A, const T& alpha,
    //                         const Array_4D<T>& arr_B, const T& beta) const;
    // Array_4D calc_laplacian(const Vertices_4D& temp, const Stencil<T>& stencil, const Array_4D<T>& arr_A, const T& alpha,
    //                         const Array_4D<T>& arr_B, const T& beta, const T& gamma) const;
};

#endif //_ARR_H_
