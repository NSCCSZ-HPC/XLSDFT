#ifndef _PARALLEL_VERTICES_H_
#define _PARALLEL_VERTICES_H_

#include <iostream>
#include <math.h>
#include <array>
#include <vector>
#include "mpi.h"
#include <cassert>

#include "vertices.h"
#include "arr.h"

class Domain_parallel_vertices_3D
{
public:
    bool is_active = true;
    bool is_Col_Maj = true;
    MPI_Comm comm = MPI_COMM_NULL;
    Vertices_3D shared_vertices;
    Vertices_3D local_vertices;
    uint chunk_size_i = UINT_MAX;
    uint chunk_size_j = UINT_MAX;
    uint chunk_size_k = UINT_MAX;
    Domain_parallel_vertices_3D();
    Domain_parallel_vertices_3D(const Domain_parallel_vertices_3D& other);
    virtual ~Domain_parallel_vertices_3D();
    Domain_parallel_vertices_3D& deepcopy(const Domain_parallel_vertices_3D& other);
    // void set_mpi_comm(const MPI_Comm& comm);
    void deepcopy_mpi_comm(const MPI_Comm& comm);
    MPI_Comm get_mpi_comm() const;
    virtual MPI_Comm get_domain_3d_comm() const {
        return this->comm;
    }
    virtual MPI_Comm get_band_comm() const {
        return this->comm;
    }
    int get_comm_rank() const;
    int get_domain_3d_comm_rank() const;
    int get_comm_size() const;
    int get_domain_3d_comm_size() const;
    bool get_is_Col_Maj() const;
    void set_is_active(const bool& is_active);
    void set_is_Col_Maj(const bool& is_active);
    void set_chunk_size_i(const uint& chunk_size_i);
    void set_chunk_size_j(const uint& chunk_size_j);
    void set_chunk_size_k(const uint& chunk_size_k);
    void set_comm_ni(const uint& comm_ni);
    void set_comm_ni(const uint& comm_ni, const uint& shared_ni);
    void set_comm_nj(const uint& comm_nj);
    void set_comm_nj(const uint& comm_nj, const uint& shared_nj);
    void set_comm_nk(const uint& comm_nk);
    void set_comm_nk(const uint& comm_nk, const uint& shared_nk);
    bool get_is_active() const;
    virtual const Vertices_3D& get_3D_shared_vertices() const {return this->shared_vertices;}
    virtual const Vertices_3D& get_3D_local_vertices() const {return this->local_vertices;}
    uint get_chunk_size_i() const;
    uint get_chunk_size_j() const;
    uint get_chunk_size_k() const;
    uint get_last_block_size_i() const;
    uint get_last_block_size_j() const;
    uint get_last_block_size_k() const;
    uint get_i_left_block_n(const uint& nodes, const bool& is_periodic) const;
    uint get_j_left_block_n(const uint& nodes, const bool& is_periodic) const;
    uint get_k_left_block_n(const uint& nodes, const bool& is_periodic) const;
    uint get_i_right_block_n(const uint& nodes, const bool& is_periodic) const;
    uint get_j_right_block_n(const uint& nodes, const bool& is_periodic) const;
    uint get_k_right_block_n(const uint& nodes, const bool& is_periodic) const;
    void generate_chunk_sizes(const uint& max_size = 0);
    void set_local_vertices();
    void set_local_vertices(const Vertices_3D& local_vertices);
    void set_shared_vertices(const Vertices_3D& shared_vertices);
    uint get_active_comm_ni() const;
    uint get_active_comm_nj() const;
    uint get_active_comm_nk() const;
    uint get_active_comm_size() const;
    // uint get_active_comm_i() const;
    // uint get_active_comm_j() const;
    // uint get_active_comm_k() const;
    virtual uint get_active_comm_i() const {
        return this->Domain_parallel_vertices_3D::get_active_comm_i(this->get_domain_3d_comm_rank());
    }
    virtual uint get_active_comm_j() const {
        return this->Domain_parallel_vertices_3D::get_active_comm_j(this->get_domain_3d_comm_rank());
    }
    virtual uint get_active_comm_k() const {
        return this->Domain_parallel_vertices_3D::get_active_comm_k(this->get_domain_3d_comm_rank());
    }
    virtual uint get_active_comm_b() const {
        return this->Domain_parallel_vertices_3D::get_active_comm_b(this->get_domain_3d_comm_rank());
    }
    virtual uint get_active_comm_i(const uint& rank) const {
        return this->is_Col_Maj ? rank % this->get_active_comm_ni()
                                : rank / (this->get_active_comm_nk() * this->get_active_comm_nj());
    }
    virtual uint get_active_comm_j(const uint& rank) const {
        return this->is_Col_Maj ? rank % (this->get_active_comm_ni() * this->get_active_comm_nj()) / this->get_active_comm_ni()
                                : rank % (this->get_active_comm_nk() * this->get_active_comm_nj()) / this->get_active_comm_nk();
    }
    virtual uint get_active_comm_k(const uint& rank) const {
        return this->is_Col_Maj ? rank / (this->get_active_comm_ni() * this->get_active_comm_nj())
                                : rank % this->get_active_comm_nk();
    };
    virtual uint get_active_comm_b(const uint&) const {
        return 0;
    };
    uint get_active_comm_rank() const;
    uint get_active_comm_rank(const uint& i, const uint& j, const uint& k) const;
    virtual void get_comm_local_vertices_list(Vertices_3D* local_vertices_list) const {
        for (int i = 0; i < this->get_comm_size(); i++) {
            (local_vertices_list++)->set_vertices(this->generate_local_vertices((uint)i));
        }
        return;
    }
    Vertices_3D generate_local_vertices() const;
    Vertices_3D generate_local_vertices(const uint& rank) const;
    int64_t generate_ith_vertices(Vertices_3D& other, const int64_t ith, const uint64_t nodes = INT64_MAX) const;
    int64_t generate_jth_vertices(Vertices_3D& other, const int64_t jth, const uint64_t nodes = INT64_MAX) const;
    int64_t generate_kth_vertices(Vertices_3D& other, const int64_t kth, const uint64_t nodes = INT64_MAX) const;
    std::vector<int64_t> generate_ijkth_vertices(Vertices_3D& other, const int64_t ith, const int64_t jth, const int64_t kth,
                                             int64_t const* const& nodeses) const;
    void generate_ith_need_vertices(Vertices_3D& result, const int& ith, const uint& nodes = INT_MAX) const;
    void generate_jth_need_vertices(Vertices_3D& result, const int& jth, const uint& nodes = INT_MAX) const;
    void generate_kth_need_vertices(Vertices_3D& result, const int& kth, const uint& nodes = INT_MAX) const;
    void generate_ijkth_need_vertices(Vertices_3D& result, const int& ith, const int& jth, const int& kth,
                                      int const* const& nodeses) const;
    uint generate_domain_counts(uint* domain_counts, const bool* is_periodic, const int* FDn) const;
    uint generate_domain_counts_full(uint* domain_counts, const bool* is_periodic, const int* FDn) const;
    void data_transfer_prepare(const Domain_parallel_vertices_3D& other_domain_vertices,
                               Vertices_3D* overlap_vertices_list,
                               int* nnode_list) const;
    void ex_arr_receive_data_transfer_prepare(const int* nodeses, const uint* domain_counts,
                                              Vertices_3D* stencil_vertices_list,
                                              Vertices_3D* super_stencil_vertices_list,
                                              int* stencil_vertices_offset,
                                              int* nnode_list) const;
    void ex_arr_receive_data_transfer_prepare_full(const int* nodeses, const uint* domain_counts,
                                                   Vertices_3D* stencil_vertices_list,
                                                   Vertices_3D* super_stencil_vertices_list,
                                                   int* stencil_vertices_offset,
                                                   int* nnode_list) const;
    void ex_arr_send_data_transfer_prepare(const int* nodeses, const uint* domain_counts,
                                           Vertices_3D* overlap_vertices_list,
                                           int* nnode_list) const;
    void ex_arr_send_data_transfer_prepare_full(const int* nodeses, const uint* domain_counts,
                                                Vertices_3D* overlap_vertices_list,
                                                int* nnode_list) const;
    void init(const Vertices_3D& shared_vertices, const MPI_Comm& comm, const uint& max_size = 0);
    void init(const Domain_parallel_vertices_3D& domain_parallel_vertices_3D);
    void destructor();
    void show() const;
};

class Domain_parallel_vertices_4D : public Domain_parallel_vertices_3D
{
public:
    MPI_Comm domain_3d_comm = MPI_COMM_NULL;
    MPI_Comm band_comm = MPI_COMM_NULL;
    Vertices_4D shared_vertices;
    Vertices_4D local_vertices;
    uint chunk_size_b = UINT_MAX;
    Domain_parallel_vertices_4D();
    Domain_parallel_vertices_4D(const Domain_parallel_vertices_4D& other);
    ~Domain_parallel_vertices_4D() override;
    Domain_parallel_vertices_4D& deepcopy(const Domain_parallel_vertices_4D& other);
    MPI_Comm get_domain_3d_comm() const override;
    MPI_Comm get_band_comm() const override;
    MPI_Comm get_domain_4d_comm() const;
    int get_domain_4d_comm_size() const;
    int get_domain_4d_comm_rank() const;
    void set_chunk_size_b(const uint& chunk_size_b);
    void set_comm_nb(const uint& comm_nb);
    void set_comm_nb(const uint& comm_nb, const uint& shared_nb);
    const Vertices_3D& get_3D_shared_vertices() const override;
    const Vertices_3D& get_3D_local_vertices() const override;
    const Vertices_4D& get_4D_shared_vertices() const;
    const Vertices_4D& get_4D_local_vertices() const;
    uint get_chunk_size_b() const;
    uint get_last_block_size_b() const;
    uint get_b_left_block_n(const uint& nodes, const bool& is_periodic) const;
    uint get_b_right_block_n(const uint& nodes, const bool& is_periodic) const;
    void generate_chunk_sizes(const uint& max_size = 0);
    void set_local_vertices();
    void set_local_vertices(const Vertices_4D& local_vertices);
    void set_shared_vertices(const Vertices_4D& shared_vertices);
    uint get_active_comm_nb() const;
    uint get_active_comm_size() const;
    uint get_active_comm_i() const override;
    uint get_active_comm_j() const override;
    uint get_active_comm_k() const override;
    uint get_active_comm_b() const override;
    uint get_active_comm_i(const uint& rank) const override;
    uint get_active_comm_j(const uint& rank) const override;
    uint get_active_comm_k(const uint& rank) const override;
    uint get_active_comm_b(const uint& rank) const override;
    uint get_active_comm_rank() const;
    uint get_active_comm_rank(const uint& i, const uint& j, const uint& k, const uint& b) const;
    void get_comm_local_vertices_list(Vertices_3D* local_vertices_list) const override;
    void get_comm_local_vertices_list(Vertices_4D* local_vertices_list) const;
    Vertices_4D generate_local_vertices() const;
    Vertices_4D generate_local_vertices(const uint& rank) const;
    int64_t generate_ith_vertices(Vertices_4D& other, const int64_t ith, const uint64_t nodes = INT64_MAX) const;
    int64_t generate_jth_vertices(Vertices_4D& other, const int64_t jth, const uint64_t nodes = INT64_MAX) const;
    int64_t generate_kth_vertices(Vertices_4D& other, const int64_t kth, const uint64_t nodes = INT64_MAX) const;
    int64_t generate_bth_vertices(Vertices_4D& other, const int64_t bth, const uint64_t nodes = INT64_MAX) const;
    void generate_ith_need_vertices(Vertices_4D& result, const int& ith, const uint& nodes = INT_MAX) const;
    void generate_jth_need_vertices(Vertices_4D& result, const int& jth, const uint& nodes = INT_MAX) const;
    void generate_kth_need_vertices(Vertices_4D& result, const int& kth, const uint& nodes = INT_MAX) const;
    void generate_bth_need_vertices(Vertices_4D& result, const int& bth, const uint& nodes = INT_MAX) const;
    void data_transfer_prepare(const Domain_parallel_vertices_4D& other_domain_vertices,
                               Vertices_4D* overlap_vertices_list,
                               int* nnode_list) const;
    uint generate_domain_counts(uint* domain_counts, const bool* is_periodic, const int* FDn) const;
    void ex_arr_receive_data_transfer_prepare(const int* FDn, const uint* domain_counts,
                                              Vertices_4D* stencil_vertices_list,
                                              Vertices_4D* super_stencil_vertices_list,
                                              int* stencil_vertices_offset,
                                              int* nnode_list) const;
    void ex_arr_send_data_transfer_prepare(const int* FDn, const uint* domain_counts,
                                           Vertices_4D* overlap_vertices_list,
                                           int* nnode_list) const;
    void init(const Vertices_4D& shared_vertices, const MPI_Comm& comm, const uint& max_size = 0);
    void init(const Domain_parallel_vertices_4D& domain_parallel_vertices_4D);
    void destructor();
    void show() const;
};

class Exarr_3D_mpi_package
{
public:
    const Domain_parallel_vertices_3D& domain_vertices;
    MPI_Comm comm = MPI_COMM_NULL;
    uint domain_counts_sum = 0;
    bool need_comm = false;
    //send mpi data
    std::vector<Vertices_3D> send_overlap_vertices_list;
    std::vector<int> send_nnode_list;
    std::vector<int> send_offsets;
    //receive mpi data
    std::vector<Vertices_3D> receive_stencil_vertices_list;
    std::vector<Vertices_3D> receive_super_stencil_vertices_list;
    std::vector<int> receive_stencil_vertices_offset;
    std::vector<int> receive_nnode_list;
    std::vector<int> receive_offsets;
    Exarr_3D_mpi_package(const Domain_parallel_vertices_3D& domain_vertices);
    ~Exarr_3D_mpi_package();
    template<typename T> void fill_domain_par_ex_arr(const Array_3D<T>& send_arr, Array_3D<T>& recv_arr) const;
    template<typename T> void fill_domain_par_ex_arr(T const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                     T* const recv_arr, const Vertices_3D& recv_arr_vertices) const;
    template<typename T> void fill_domain_par_ex_arr_sequential(T const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                     T* const recv_arr, const Vertices_3D& recv_arr_vertices) const;
    void init(const bool* is_periodic, const int* FDn);
    void init_full(const bool* is_periodic, const int* FDn);
    void init(const Exarr_3D_mpi_package& exarr_3D_mpi_package);
    void destructor();
    void show() const;
};

class Exarr_4D_mpi_package
{
public:
    const Domain_parallel_vertices_4D& domain_vertices;
    MPI_Comm comm = MPI_COMM_NULL;
    uint domain_counts_sum = 0;
    bool need_comm = false;
    //send mpi data
    std::vector<Vertices_4D> send_overlap_vertices_list;
    std::vector<int> send_nnode_list;
    std::vector<int> send_offsets;
    //receive mpi data
    std::vector<Vertices_4D> receive_stencil_vertices_list;
    std::vector<Vertices_4D> receive_super_stencil_vertices_list;
    std::vector<int> receive_stencil_vertices_offset;
    std::vector<int> receive_nnode_list;
    std::vector<int> receive_offsets;
    Exarr_4D_mpi_package(const Domain_parallel_vertices_4D& domain_vertices);
    ~Exarr_4D_mpi_package();
    template<typename T> void fill_domain_par_ex_arr(const Array_4D<T>& send_arr, Array_4D<T>& recv_arr) const;
    template<typename T> void fill_domain_par_ex_arr(T const* const& send_arr, Vertices_4D send_arr_vertices,
                                                     T* const& recv_arr, Vertices_4D recv_arr_vertices) const;
    void init(const bool* is_periodic, const int* FDn);
    void init(const Exarr_4D_mpi_package& exarr_4D_mpi_package);
    void destructor();
    void show() const;
};

class Domain_3D_to_3D_mpi_package
{
public:
    const Domain_parallel_vertices_3D& my_domain_vertices;
    const Domain_parallel_vertices_3D& other_domain_vertices;
    MPI_Comm comm = MPI_COMM_NULL;
    bool need_comm = false;
    //send mpi data
    std::vector<Vertices_3D> send_overlap_vertices_list;
    std::vector<int> send_nnode_list;
    std::vector<int> send_offsets;
    //receive mpi data
    std::vector<Vertices_3D> receive_overlap_vertices_list;
    std::vector<int> receive_nnode_list;
    std::vector<int> receive_offsets;
    Domain_3D_to_3D_mpi_package(const Domain_parallel_vertices_3D& my_domain_vertices,
                                const Domain_parallel_vertices_3D& other_domain_vertices);
    ~Domain_3D_to_3D_mpi_package();
    template<typename T> void send_data(const Array_3D<T>& send_arr, Array_3D<T>& recv_arr) const;
    template<typename T> void send_data(T const* const send_arr, const Vertices_3D& send_arr_vertices,
                                        T* const recv_arr, const Vertices_3D& recv_arr_vertices) const;
    template<typename T> void send_data_mp(T const* const send_arr, const Vertices_3D& send_arr_vertices,
                                        T* const recv_arr, const Vertices_3D& recv_arr_vertices,
                                        Memory_pool<T, Fast_memory>& pool_fast,
                                        Memory_pool<T, Capacity_memory>& pool_cap) const;
    template<typename T> void recv_data(Array_3D<T>& recv_arr, const Array_3D<T>& send_arr) const;
    void init();
    void init(const Domain_3D_to_3D_mpi_package& domain_3D_to_3D_mpi_package);
    void destructor();
    void show() const;
};

class Domain_3D_to_4D_mpi_package : public Domain_3D_to_3D_mpi_package
{
public:
    MPI_Comm band_comm = MPI_COMM_NULL;
    bool need_band_comm = false;
    Domain_3D_to_4D_mpi_package(const Domain_parallel_vertices_3D& my_domain_vertices,
                                const Domain_parallel_vertices_4D& other_domain_vertices);
    ~Domain_3D_to_4D_mpi_package();
    template<typename T> void send_data(const Array_3D<T>& send_arr, Array_3D<T>& recv_arr) const;
    template<typename T> void send_data(T const* const send_arr, const Vertices_3D& send_arr_vertices,
                                        T* const recv_arr, const Vertices_3D& recv_arr_vertices) const;
    template<typename T> void send_data_mp(T const* const send_arr, const Vertices_3D& send_arr_vertices,
                                        T* const recv_arr, const Vertices_3D& recv_arr_vertices,
                                        Memory_pool<T, Fast_memory>& pool_fast,
                                        Memory_pool<T, Capacity_memory>& pool_cap) const;
    template<typename T> void recv_data(Array_3D<T>& recv_arr, const Array_3D<T>& send_arr) const;
    void init();
    void init(const Domain_3D_to_4D_mpi_package& domain_3D_to_4D_mpi_package);
    void destructor();
    void show() const;
};

class Domain_4D_to_4D_mpi_package
{
public:
    const Domain_parallel_vertices_4D& my_domain_vertices;
    const Domain_parallel_vertices_4D& other_domain_vertices;
    MPI_Comm comm = MPI_COMM_NULL;
    bool need_comm = false;
    //send mpi data
    std::vector<Vertices_4D> send_overlap_vertices_list;
    std::vector<int> send_nnode_list;
    std::vector<int> send_offsets;
    //receive mpi data
    std::vector<Vertices_4D> receive_overlap_vertices_list;
    std::vector<int> receive_nnode_list;
    std::vector<int> receive_offsets;
    Domain_4D_to_4D_mpi_package(const Domain_parallel_vertices_4D& my_domain_vertices,
                                const Domain_parallel_vertices_4D& other_domain_vertices);
    ~Domain_4D_to_4D_mpi_package();
    template<typename T> void send_data(const Array_4D<T>& send_arr, Array_4D<T>& recv_arr) const;
    template<typename T> void send_data(T const* const& send_arr, const Vertices_4D& send_arr_vertices,
                                        T* const& recv_arr, const Vertices_4D& recv_arr_vertices) const;
    template<typename T> void recv_data(Array_4D<T>& recv_arr, const Array_4D<T>& send_arr) const;
    template<typename T> void recv_data(T* const& recv_arr, const Vertices_4D& recv_arr_vertices,
                                        T const* const& send_arr, const Vertices_4D& send_arr_vertices) const;
    void init();
    void init(const Domain_4D_to_4D_mpi_package& domain_4D_to_4D_mpi_package);
    void destructor();
    void show() const;
};

namespace Parallel_vertices {
    uint64_t cal_optimize_chunksize(const uint64_t ni, const uint64_t ideal_chuck, const uint64_t unit_pool_size);
    uint64_t cal_chunksize_1D(const uint64_t nunit, const uint64_t ni);
    std::array<uint64_t,3> cal_chunk_sizes_3D(const uint64_t nunit, const uint64_t ni, const uint64_t nj, const uint64_t nk);
    std::array<uint64_t,2> cal_chunk_sizes_2D(const uint64_t nunit, const uint64_t ni, const uint64_t nj);
    int64_t generate_nth_vertices(int64_t* is, int64_t* ie, const int64_t ith, const uint64_t comm_i, const uint64_t comm_ni,
                                    const uint64_t chunk_size, const uint64_t last_block_size, const uint64_t nodes = INT64_MAX);
    uint64_t generate_left_domain_n(const uint64_t comm_i, const uint64_t comm_ni, const uint64_t chunk_size,
                                const uint64_t last_block_size, const uint64_t nodes, const bool is_periodic);
    uint64_t generate_right_domain_n(const uint64_t comm_i, const uint64_t comm_ni, const uint64_t chunk_size,
                                 const uint64_t last_block_size, const uint64_t nodes, const bool is_periodic);
    template<typename T> void domain_vertices_rand(T* const arr, const Domain_parallel_vertices_3D& domain_vertices,
                                                   const T rand_min = -0.5, const T rand_max = 0.5);
    template<typename T> void domain_vertices_rand(Array_3D<T>& arr, const Domain_parallel_vertices_3D& domain_vertices,
                                                   const T& rand_min = -0.5, const T& rand_max = 0.5);
    template<typename T> void domain_vertices_rand(Array_4D<T>& arr, const Domain_parallel_vertices_4D& domain_vertices,
                                                   const T& rand_min = -0.5, const T& rand_max = 0.5);
    bool need_comm(int const* const& receive_nnode_list, const MPI_Comm& comm);
    void Cblacs_gridmap_subcomm(int* icontxt, const int ldup, const int nprow, const int npcol,
                                const bool& is_active, const MPI_Comm subcomm);
    void dims_divide_skbd(const int Nspin, const int Nk, const int Ns, const int *gridsizes, const int np, 
                          int *nps, int *npk, int *npb, int *npd, int minsize, int isfock);
    void SPARC_Dims_create(int nproc, int ndims, int *gridsizes, int minsize, int *dims, int *ierr);
    std::array<uint,4> cal_comm_ns_4D_by_sparc(const Vertices_4D& shared_vertices,
                                                   const MPI_Comm& comm, const uint& max_size = 0);
    std::array<uint,3> cal_comm_ns_3D_by_sparc(const Vertices_3D& shared_vertices,
                                                   const MPI_Comm& comm, const uint& max_size = 0);
    bool is_comm_equal(const MPI_Comm& comm1, const MPI_Comm& comm2);
    template<typename T>
    void cal_gradient_d3(T const* const& arr, const Vertices_3D& vertices,
                         const Stencil<T>& stencil, const Exarr_3D_mpi_package& exarr_mpi_package,
                         T* const& darr_x, T* const& darr_y, T* const& darr_z);
}

#endif //_PARALLEL_VERTICES_H_
