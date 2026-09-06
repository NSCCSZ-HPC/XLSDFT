#include "parallel_vertices.h"

Domain_parallel_vertices_3D::Domain_parallel_vertices_3D() {}

Domain_parallel_vertices_3D::Domain_parallel_vertices_3D(const Domain_parallel_vertices_3D& other) {
    this->deepcopy(other);
}

Domain_parallel_vertices_3D::~Domain_parallel_vertices_3D() {
    this->destructor();
}

Domain_parallel_vertices_3D& Domain_parallel_vertices_3D::deepcopy(const Domain_parallel_vertices_3D& other) {
    this->is_active = other.is_active;
    this->is_Col_Maj = other.is_Col_Maj;
    if (other.comm != MPI_COMM_NULL) {
        this->deepcopy_mpi_comm(other.comm);
    } else {
        this->comm = MPI_COMM_NULL;
    }
    this->shared_vertices = other.shared_vertices;
    this->local_vertices = other.local_vertices;
    this->chunk_size_i = other.chunk_size_i;
    this->chunk_size_j = other.chunk_size_j;
    this->chunk_size_k = other.chunk_size_k;
    return *this;
}

// void Domain_parallel_vertices_3D::set_mpi_comm(const MPI_Comm& comm) {
//     this->comm = comm;
//     return;
// }

void Domain_parallel_vertices_3D::deepcopy_mpi_comm(const MPI_Comm& comm) {
    if (comm != MPI_COMM_SELF && comm != MPI_COMM_WORLD && comm != MPI_COMM_NULL) {
        MPI_Comm_split(comm, 1, 1, &(this->comm));
    } else {
        this->comm = comm;
    }
    return;
}

MPI_Comm Domain_parallel_vertices_3D::get_mpi_comm() const {
    return this->comm;
}

int Domain_parallel_vertices_3D::get_comm_rank() const {
    int rank;
    MPI_Comm_rank(this->comm, &rank);
    return rank;
}

int Domain_parallel_vertices_3D::get_domain_3d_comm_rank() const {
    int rank;
    MPI_Comm_rank(this->get_domain_3d_comm(), &rank);
    return rank;
}

int Domain_parallel_vertices_3D::get_comm_size() const  {
    int size;
    MPI_Comm_size(this->comm, &size);
    return size;
}

int Domain_parallel_vertices_3D::get_domain_3d_comm_size() const {
    int size;
    MPI_Comm_size(this->get_domain_3d_comm(), &size);
    return size;
}

bool Domain_parallel_vertices_3D::get_is_Col_Maj() const {
    return this->is_Col_Maj;
}

void Domain_parallel_vertices_3D::set_is_active(const bool& is_active) {
    this->is_active = is_active;
    return;
}

void Domain_parallel_vertices_3D::set_is_Col_Maj(const bool& is_Col_Maj) {
    this->is_Col_Maj = is_Col_Maj;
    return;
}

void Domain_parallel_vertices_3D::set_chunk_size_i(const uint& chunk_size_i) {
    this->chunk_size_i = chunk_size_i;
    return;
}

void Domain_parallel_vertices_3D::set_chunk_size_j(const uint& chunk_size_j) {
    this->chunk_size_j = chunk_size_j;
    return;
}

void Domain_parallel_vertices_3D::set_chunk_size_k(const uint& chunk_size_k) {
    this->chunk_size_k = chunk_size_k;
    return;
}

void Domain_parallel_vertices_3D::set_comm_ni(const uint& comm_ni) {
    this->set_comm_ni(comm_ni, this->get_3D_shared_vertices().ni);
    return;
}

void Domain_parallel_vertices_3D::set_comm_ni(const uint& comm_ni, const uint& shared_ni) {
    this->set_chunk_size_i(Parallel_vertices::cal_chunksize_1D(comm_ni, shared_ni));
    return;
}

void Domain_parallel_vertices_3D::set_comm_nj(const uint& comm_nj) {
    this->set_comm_nj(comm_nj, this->get_3D_shared_vertices().nj);
    return;
}

void Domain_parallel_vertices_3D::set_comm_nj(const uint& comm_nj, const uint& shared_nj) {
    this->set_chunk_size_j(Parallel_vertices::cal_chunksize_1D(comm_nj, shared_nj));
    return;
}

void Domain_parallel_vertices_3D::set_comm_nk(const uint& comm_nk) {
    this->set_comm_nk(comm_nk, this->get_3D_shared_vertices().nk);
    return;
}

void Domain_parallel_vertices_3D::set_comm_nk(const uint& comm_nk, const uint& shared_nk) {
    this->set_chunk_size_k(Parallel_vertices::cal_chunksize_1D(comm_nk, shared_nk));
    return;
}

bool Domain_parallel_vertices_3D::get_is_active() const {
    return this->is_active;
}

uint Domain_parallel_vertices_3D::get_chunk_size_i() const {
    return this->chunk_size_i;
}

uint Domain_parallel_vertices_3D::get_chunk_size_j() const {
    return this->chunk_size_j;
}

uint Domain_parallel_vertices_3D::get_chunk_size_k() const {
    return this->chunk_size_k;
}

uint Domain_parallel_vertices_3D::get_last_block_size_i() const {
    return this->get_3D_shared_vertices().get_ni() % this->get_chunk_size_i() == 0
         ? this->get_chunk_size_i()
         : this->get_3D_shared_vertices().get_ni() % this->get_chunk_size_i();
}

uint Domain_parallel_vertices_3D::get_last_block_size_j() const {
    return this->get_3D_shared_vertices().get_nj() % this->get_chunk_size_j() == 0
         ? this->get_chunk_size_j()
         : this->get_3D_shared_vertices().get_nj() % this->get_chunk_size_j();
}

uint Domain_parallel_vertices_3D::get_last_block_size_k() const {
    return this->get_3D_shared_vertices().get_nk() % this->get_chunk_size_k() == 0
         ? this->get_chunk_size_k()
         : this->get_3D_shared_vertices().get_nk() % this->get_chunk_size_k();
}

uint Domain_parallel_vertices_3D::get_i_left_block_n(const uint& nodes, const bool& is_periodic) const {
    return Parallel_vertices::generate_left_domain_n(this->get_active_comm_i(), this->get_active_comm_ni(), this->get_chunk_size_i(),
                                this->get_last_block_size_i(), nodes, is_periodic);
}

uint Domain_parallel_vertices_3D::get_j_left_block_n(const uint& nodes, const bool& is_periodic) const {
    return Parallel_vertices::generate_left_domain_n(this->get_active_comm_j(), this->get_active_comm_nj(), this->get_chunk_size_j(),
                                this->get_last_block_size_j(), nodes, is_periodic);
}

uint Domain_parallel_vertices_3D::get_k_left_block_n(const uint& nodes, const bool& is_periodic) const {
    return Parallel_vertices::generate_left_domain_n(this->get_active_comm_k(), this->get_active_comm_nk(), this->get_chunk_size_k(),
                                this->get_last_block_size_k(), nodes, is_periodic);
}

uint Domain_parallel_vertices_3D::get_i_right_block_n(const uint& nodes, const bool& is_periodic) const {
    return Parallel_vertices::generate_right_domain_n(this->get_active_comm_i(), this->get_active_comm_ni(), this->get_chunk_size_i(),
                                this->get_last_block_size_i(), nodes, is_periodic);
}

uint Domain_parallel_vertices_3D::get_j_right_block_n(const uint& nodes, const bool& is_periodic) const {
    return Parallel_vertices::generate_right_domain_n(this->get_active_comm_j(), this->get_active_comm_nj(), this->get_chunk_size_j(),
                                this->get_last_block_size_j(), nodes, is_periodic);
}

uint Domain_parallel_vertices_3D::get_k_right_block_n(const uint& nodes, const bool& is_periodic) const {
    return Parallel_vertices::generate_right_domain_n(this->get_active_comm_k(), this->get_active_comm_nk(), this->get_chunk_size_k(),
                                this->get_last_block_size_k(), nodes, is_periodic);
}

void Domain_parallel_vertices_3D::generate_chunk_sizes(const uint& max_size) {
    uint comm_size = max_size == 0
                     ? this->get_comm_size()
                     : max_size < (uint)this->get_comm_size()
                     ? max_size
                     : (uint)this->get_comm_size();
    if (this->get_chunk_size_i() == UINT_MAX
     && this->get_chunk_size_j() == UINT_MAX
     && this->get_chunk_size_k() == UINT_MAX) {
        // std::array<uint,3> chunk_sizes_array = Parallel_vertices::cal_chunk_sizes_3D(comm_size,
        //                                                                              this->get_3D_shared_vertices().ni,
        //                                                                              this->get_3D_shared_vertices().nj,
        //                                                                              this->get_3D_shared_vertices().nk);
        // this->set_chunk_size_i(chunk_sizes_array[0]);
        // this->set_chunk_size_j(chunk_sizes_array[1]);
        // this->set_chunk_size_k(chunk_sizes_array[2]);
        std::array<uint,3> comm_ns =
                            Parallel_vertices::cal_comm_ns_3D_by_sparc(this->get_3D_shared_vertices(),
                                                                       this->comm, comm_size);
        this->set_comm_ni(comm_ns[0]);
        this->set_comm_nj(comm_ns[1]);
        this->set_comm_nk(comm_ns[2]);
    } else if ((this->chunk_size_i == UINT_MAX && this->chunk_size_j == UINT_MAX)
            || (this->chunk_size_i == UINT_MAX && this->chunk_size_k == UINT_MAX)
            || (this->chunk_size_j == UINT_MAX && this->chunk_size_k == UINT_MAX)) {
        uint shared_vertices_n_1 = 0;
        uint shared_vertices_n_2 = 0;
        uint comm_size_plane = 0;
        if (this->chunk_size_i != UINT_MAX) {
            shared_vertices_n_1 = this->get_3D_shared_vertices().nj;
            shared_vertices_n_2 = this->get_3D_shared_vertices().nk;
            uint comm_ni = (this->get_3D_shared_vertices().ni - 1 + this->chunk_size_i) / this->chunk_size_i;
            comm_size_plane = comm_size/comm_ni;
        } else if (this->chunk_size_j != UINT_MAX) {
            shared_vertices_n_1 = this->get_3D_shared_vertices().ni;
            shared_vertices_n_2 = this->get_3D_shared_vertices().nk;
            uint comm_nj = (this->get_3D_shared_vertices().nj - 1 + this->chunk_size_j) / this->chunk_size_j;
            comm_size_plane = comm_size/comm_nj;
        } else if (this->chunk_size_k != UINT_MAX) {
            shared_vertices_n_1 = this->get_3D_shared_vertices().ni;
            shared_vertices_n_2 = this->get_3D_shared_vertices().nj;
            uint comm_nk = (this->get_3D_shared_vertices().nk - 1 + this->chunk_size_k) / this->chunk_size_k;
            comm_size_plane = comm_size/comm_nk;
        }
        std::array<uint64_t,2> chunk_sizes_array = Parallel_vertices::cal_chunk_sizes_2D(comm_size_plane,
                                                                                     shared_vertices_n_1,
                                                                                     shared_vertices_n_2);
        if (this->chunk_size_i != UINT_MAX) {
            this->set_chunk_size_j(chunk_sizes_array[0]);
            this->set_chunk_size_k(chunk_sizes_array[1]);
        } else if (this->chunk_size_j != UINT_MAX) {
            this->set_chunk_size_i(chunk_sizes_array[0]);
            this->set_chunk_size_k(chunk_sizes_array[1]);
        } else if (this->chunk_size_k != UINT_MAX) {
            this->set_chunk_size_i(chunk_sizes_array[0]);
            this->set_chunk_size_j(chunk_sizes_array[1]);
        }
    } else if (this->chunk_size_i == UINT_MAX || this->chunk_size_j == UINT_MAX || this->chunk_size_k == UINT_MAX) {
        uint shared_vertices_n = 0;
        uint comm_size_dir = 0;
        if (this->chunk_size_i == UINT_MAX) {
            shared_vertices_n = this->get_3D_shared_vertices().ni;
            uint comm_nj = (this->get_3D_shared_vertices().nj - 1 + this->chunk_size_j) / this->chunk_size_j;
            uint comm_nk = (this->get_3D_shared_vertices().nk - 1 + this->chunk_size_k) / this->chunk_size_k;
            comm_size_dir = comm_size/comm_nj/comm_nk;
        } else if (this->chunk_size_j == UINT_MAX) {
            shared_vertices_n = this->get_3D_shared_vertices().nj;
            uint comm_ni = (this->get_3D_shared_vertices().ni - 1 + this->chunk_size_i) / this->chunk_size_i;
            uint comm_nk = (this->get_3D_shared_vertices().nk - 1 + this->chunk_size_k) / this->chunk_size_k;
            comm_size_dir = comm_size/comm_ni/comm_nk;
        } else if (this->chunk_size_k == UINT_MAX) {
            shared_vertices_n = this->get_3D_shared_vertices().nk;
            uint comm_ni = (this->get_3D_shared_vertices().ni - 1 + this->chunk_size_i) / this->chunk_size_i;
            uint comm_nj = (this->get_3D_shared_vertices().nj - 1 + this->chunk_size_j) / this->chunk_size_j;
            comm_size_dir = comm_size/comm_ni/comm_nj;
        }
        uint chunk_size = Parallel_vertices::cal_chunksize_1D(comm_size_dir, shared_vertices_n);
        if (this->chunk_size_i == UINT_MAX) {
            this->set_chunk_size_i(chunk_size);
        } else if (this->chunk_size_j == UINT_MAX) {
            this->set_chunk_size_j(chunk_size);
        } else if (this->chunk_size_k == UINT_MAX) {
            this->set_chunk_size_k(chunk_size);
        }
    }
    if (this->get_active_comm_size() > comm_size) {
        std::cout << "Domain_parallel_vertices_3D, this->get_active_comm_size() = " << this->get_active_comm_size() << std::endl;
        std::cout << "Domain_parallel_vertices_3D, comm_size = " << comm_size << std::endl;
        assert(this->get_active_comm_size() <= comm_size);
    }
    this->set_is_active(this->get_comm_rank() < (int)this->get_active_comm_size() ?
                        true : false);
    this->set_local_vertices();
    return;
}

void Domain_parallel_vertices_3D::set_local_vertices() {
    this->set_local_vertices(this->generate_local_vertices());
    return;
}

void Domain_parallel_vertices_3D::set_local_vertices(const Vertices_3D& local_vertices) {
    this->local_vertices.set_vertices(local_vertices);
    return;
}

void Domain_parallel_vertices_3D::set_shared_vertices(const Vertices_3D& shared_vertices) {
    this->shared_vertices.set_vertices(shared_vertices);
    return;
}

uint Domain_parallel_vertices_3D::get_active_comm_ni() const {
    return (this->get_3D_shared_vertices().ni - 1 + this->get_chunk_size_i())/this->get_chunk_size_i();
}

uint Domain_parallel_vertices_3D::get_active_comm_nj() const {
    return (this->get_3D_shared_vertices().nj - 1 + this->get_chunk_size_j())/this->get_chunk_size_j();
}

uint Domain_parallel_vertices_3D::get_active_comm_nk() const {
    return (this->get_3D_shared_vertices().nk - 1 + this->get_chunk_size_k())/this->get_chunk_size_k();
}

uint Domain_parallel_vertices_3D::get_active_comm_size() const {
    return this->get_active_comm_ni() * this->get_active_comm_nj() * this->get_active_comm_nk();
}

// uint Domain_parallel_vertices_3D::get_active_comm_i() const {
//     return this->Domain_parallel_vertices_3D::get_active_comm_i(this->get_domain_3d_comm_rank());
// }

// uint Domain_parallel_vertices_3D::get_active_comm_j() const {
//     return this->Domain_parallel_vertices_3D::get_active_comm_j(this->get_domain_3d_comm_rank());
// }

// uint Domain_parallel_vertices_3D::get_active_comm_k() const {
//     return this->Domain_parallel_vertices_3D::get_active_comm_k(this->get_domain_3d_comm_rank());
// }

// uint Domain_parallel_vertices_3D::get_active_comm_i(const uint& rank) const {
//     return rank % this->get_active_comm_ni();
// }

// uint Domain_parallel_vertices_3D::get_active_comm_j(const uint& rank) const {
//     return rank % (this->get_active_comm_ni() * this->get_active_comm_nj()) / this->get_active_comm_ni();
// }

// uint Domain_parallel_vertices_3D::get_active_comm_k(const uint& rank) const {
//     return rank / (this->get_active_comm_ni() * this->get_active_comm_nj());
// }

uint Domain_parallel_vertices_3D::get_active_comm_rank() const {
    return this->get_active_comm_rank(this->get_active_comm_i(),
                                      this->get_active_comm_j(),
                                      this->get_active_comm_k());
}

uint Domain_parallel_vertices_3D::get_active_comm_rank(const uint& i, const uint& j, const uint& k) const {
    if (!this->is_active) {
        return INT_MAX;
    }
    return this->is_Col_Maj ? i + (j + k * this->get_active_comm_nj()) * this->get_active_comm_ni()
                            : k + (j + i * this->get_active_comm_nj()) * this->get_active_comm_nk();
}

// void Domain_parallel_vertices_3D::get_comm_local_vertices_list(Vertices_3D* local_vertices_list) const {
//     for (int i = 0; i < this->get_comm_size(); i++) {
//         (local_vertices_list++)->set_vertices(this->generate_local_vertices((uint)i));
//     }
//     return;
// }

Vertices_3D Domain_parallel_vertices_3D::generate_local_vertices() const {
    return Domain_parallel_vertices_3D::generate_local_vertices(this->get_comm_rank());
}

Vertices_3D Domain_parallel_vertices_3D::generate_local_vertices(const uint& rank) const {
    Vertices_3D result;
    if (rank < this->get_active_comm_size()) {
        int local_is = (int)(this->get_active_comm_i(rank) * this->chunk_size_i) + this->get_3D_shared_vertices().is;
        int local_ie = local_is + (int)this->chunk_size_i - 1 <= get_3D_shared_vertices().get_ie() ?
                       local_is + (int)this->chunk_size_i - 1 : get_3D_shared_vertices().get_ie();
        int local_js = (int)(this->get_active_comm_j(rank) * this->chunk_size_j) + this->get_3D_shared_vertices().js;
        int local_je = local_js + (int)this->chunk_size_j - 1 <= get_3D_shared_vertices().get_je() ?
                       local_js + (int)this->chunk_size_j - 1 : get_3D_shared_vertices().get_je();
        int local_ks = (int)(this->get_active_comm_k(rank) * this->chunk_size_k) + this->get_3D_shared_vertices().ks;
        int local_ke = local_ks + (int)this->chunk_size_k - 1 <= get_3D_shared_vertices().get_ke() ?
                       local_ks + (int)this->chunk_size_k - 1 : get_3D_shared_vertices().get_ke();
        result.set_vertices(local_is, local_ie, local_js, local_je, local_ks, local_ke);
    } else {
        result.set_vertices(0, -1, 0, -1, 0, -1);
    }
    return result;
}

int64_t Domain_parallel_vertices_3D::generate_ith_vertices(Vertices_3D& other, const int64_t ith, const uint64_t nodes) const {
    int64_t is;
    int64_t ie;
    int64_t index = Parallel_vertices::generate_nth_vertices(&is, &ie, ith, this->get_active_comm_i(), this->get_active_comm_ni(),
                                    this->chunk_size_i, this->get_last_block_size_i(), nodes);
    other.set_vertices(is + this->get_3D_shared_vertices().is, ie + this->get_3D_shared_vertices().is,
                       this->get_3D_local_vertices().js, this->get_3D_local_vertices().get_je(),
                       this->get_3D_local_vertices().ks, this->get_3D_local_vertices().get_ke());
    return index;
}

int64_t Domain_parallel_vertices_3D::generate_jth_vertices(Vertices_3D& other, const int64_t jth, const uint64_t nodes) const {
    int64_t js;
    int64_t je;
    int64_t index = Parallel_vertices::generate_nth_vertices(&js, &je, jth, this->get_active_comm_j(), this->get_active_comm_nj(),
                                    this->chunk_size_j, this->get_last_block_size_j(), nodes);
    other.set_vertices(this->get_3D_local_vertices().is, this->get_3D_local_vertices().get_ie(),
                       js + this->get_3D_shared_vertices().js, je + this->get_3D_shared_vertices().js,
                       this->get_3D_local_vertices().ks, this->get_3D_local_vertices().get_ke());
    return index;
}

int64_t Domain_parallel_vertices_3D::generate_kth_vertices(Vertices_3D& other, const int64_t kth, const uint64_t nodes) const {
    int64_t ks;
    int64_t ke;
    int64_t index = Parallel_vertices::generate_nth_vertices(&ks, &ke, kth, this->get_active_comm_k(), this->get_active_comm_nk(),
                                    this->chunk_size_k, this->get_last_block_size_k(), nodes);
    other.set_vertices(this->get_3D_local_vertices().is, this->get_3D_local_vertices().get_ie(),
                       this->get_3D_local_vertices().js, this->get_3D_local_vertices().get_je(),
                       ks + this->get_3D_shared_vertices().ks, ke + this->get_3D_shared_vertices().ks);
    return index;
}

std::vector<int64_t> Domain_parallel_vertices_3D::generate_ijkth_vertices(Vertices_3D& other, const int64_t ith, const int64_t jth,
                                                         const int64_t kth, int64_t const* const& nodeses) const {
    std::vector<int64_t> indexes(3, 0);
    int64_t is;
    int64_t ie;
    indexes[0] = Parallel_vertices::generate_nth_vertices(&is, &ie, ith, this->get_active_comm_i(), this->get_active_comm_ni(),
                                    this->chunk_size_i, this->get_last_block_size_i(), nodeses[0]);
    int64_t js;
    int64_t je;
    indexes[1] = Parallel_vertices::generate_nth_vertices(&js, &je, jth, this->get_active_comm_j(), this->get_active_comm_nj(),
                                    this->chunk_size_j, this->get_last_block_size_j(), nodeses[1]);
    int64_t ks;
    int64_t ke;
    indexes[2] = Parallel_vertices::generate_nth_vertices(&ks, &ke, kth, this->get_active_comm_k(), this->get_active_comm_nk(),
                                    this->chunk_size_k, this->get_last_block_size_k(), nodeses[2]);
    other.set_vertices(is + this->get_3D_shared_vertices().is, ie + this->get_3D_shared_vertices().is,
                       js + this->get_3D_shared_vertices().js, je + this->get_3D_shared_vertices().js,
                       ks + this->get_3D_shared_vertices().ks, ke + this->get_3D_shared_vertices().ks);
    
    return indexes;
}

void Domain_parallel_vertices_3D::generate_ith_need_vertices(Vertices_3D& result, const int& ith, const uint& nodes) const {
    if (ith == 0) {
        result.set_vertices(this->get_3D_local_vertices());
        return;
    } else if (ith < 0) {
        Vertices_3D other;
        this->generate_ith_vertices(other, ith);
        int shifted_ie = other.get_ie() + nodes;
        if (shifted_ie < this->get_3D_local_vertices().is) {
            result.set_vertices(0, -1, 0, -1, 0, -1);
            return;
        } else if (shifted_ie >= this->get_3D_local_vertices().is && shifted_ie <= this->get_3D_local_vertices().get_ie()) {
            result.set_vertices(this->get_3D_local_vertices().is, shifted_ie,
                                  this->get_3D_local_vertices().js, this->get_3D_local_vertices().get_je(),
                                  this->get_3D_local_vertices().ks, this->get_3D_local_vertices().get_ke());
            return;
        } else {
            result.set_vertices(this->get_3D_local_vertices());
            return;
        }
    } else { //ith > 0
        Vertices_3D other;
        this->generate_ith_vertices(other, ith);
        int shifted_is = other.is - nodes;
        if (shifted_is > this->get_3D_local_vertices().get_ie()) {
            result.set_vertices(0, -1, 0, -1, 0, -1);
            return;
        } else if (shifted_is <= this->get_3D_local_vertices().get_ie() && shifted_is >= this->get_3D_local_vertices().is) {
            result.set_vertices(shifted_is, this->get_3D_local_vertices().get_ie(),
                                  this->get_3D_local_vertices().js, get_3D_local_vertices().get_je(),
                                  this->get_3D_local_vertices().ks, get_3D_local_vertices().get_ke());
            return;
        } else {
            result.set_vertices(this->get_3D_local_vertices());
            return;
        }
    }
}

void Domain_parallel_vertices_3D::generate_jth_need_vertices(Vertices_3D& result, const int& jth, const uint& nodes) const {
    if (jth == 0) {
        result.set_vertices(this->get_3D_local_vertices());
        return;
    } else if (jth < 0) {
        Vertices_3D other;
        this->generate_jth_vertices(other, jth);
        int shifted_je = other.get_je() + nodes;
        if (shifted_je < this->get_3D_local_vertices().js) {
            result.set_vertices(0, -1, 0, -1, 0, -1);
            return;
        } else if (shifted_je >= this->get_3D_local_vertices().js && shifted_je <= this->get_3D_local_vertices().get_je()) {
            result.set_vertices(this->get_3D_local_vertices().is, this->get_3D_local_vertices().get_ie(),
                                  this->get_3D_local_vertices().js, shifted_je,
                                  this->get_3D_local_vertices().ks, this->get_3D_local_vertices().get_ke());
            return;
        } else {
            result.set_vertices(this->get_3D_local_vertices());
            return;
        }
    } else { //jth > 0
        Vertices_3D other;
        this->generate_jth_vertices(other, jth);
        int shifted_js = other.js - nodes;
        if (shifted_js > this->get_3D_local_vertices().get_je()) {
            result.set_vertices(0, -1, 0, -1, 0, -1);
            return;
        } else if (shifted_js <= this->get_3D_local_vertices().get_je() && shifted_js >= this->get_3D_local_vertices().js) {
            result.set_vertices(this->get_3D_local_vertices().is, this->get_3D_local_vertices().get_ie(),
                                  shifted_js, this->get_3D_local_vertices().get_je(),
                                  this->get_3D_local_vertices().ks,this->get_3D_local_vertices().get_ke());
            return;
        } else {
            result.set_vertices(this->get_3D_local_vertices());
            return;
        }
    }
}

void Domain_parallel_vertices_3D::generate_kth_need_vertices(Vertices_3D& result, const int& kth, const uint& nodes) const {
    if (kth == 0) {
        result.set_vertices(this->get_3D_local_vertices());
        return;
    } else if (kth < 0) {
        Vertices_3D other;
        this->generate_kth_vertices(other, kth);
        int shifted_ke = other.get_ke() + nodes;
        if (shifted_ke < this->get_3D_local_vertices().ks) {
            result.set_vertices(0, -1, 0, -1, 0, -1);
            return;
        } else if (shifted_ke >= this->get_3D_local_vertices().ks && shifted_ke <= this->get_3D_local_vertices().get_ke()) {
            result.set_vertices(this->get_3D_local_vertices().is, this->get_3D_local_vertices().get_ie(),
                                this->get_3D_local_vertices().js, this->get_3D_local_vertices().get_je(),
                                this->get_3D_local_vertices().ks, shifted_ke);
            return;
        } else {
            result.set_vertices(this->get_3D_local_vertices());
            return;
        }
    } else { //kth > 0
        Vertices_3D other;
        this->generate_kth_vertices(other, kth);
        int shifted_ks = other.ks - nodes;
        if (shifted_ks > this->get_3D_local_vertices().get_ke()) {
            result.set_vertices(0, -1, 0, -1, 0, -1);
            return;
        } else if (shifted_ks <= this->get_3D_local_vertices().get_ke() && shifted_ks >= this->get_3D_local_vertices().ks) {
            result.set_vertices(this->get_3D_local_vertices().is, this->get_3D_local_vertices().get_ie(),
                                this->get_3D_local_vertices().js, this->get_3D_local_vertices().get_je(),
                                shifted_ks, this->get_3D_local_vertices().get_ke());
            return;
        } else {
            result.set_vertices(this->get_3D_local_vertices());
            return;
        }
    }
}

void Domain_parallel_vertices_3D::generate_ijkth_need_vertices(Vertices_3D& result, const int& ith, const int& jth,
                                                               const int& kth, int const* const& nodeses) const {
    Vertices_3D other;
    int64_t nodeses_int64[3] = {nodeses[0], nodeses[1], nodeses[2]};
    this->generate_ijkth_vertices(other, ith, jth, kth, nodeses_int64);
    int shifted_ie = other.get_ie() + nodeses_int64[0];
    int shifted_is = other.is - nodeses_int64[0];
    int shifted_je = other.get_je() + nodeses_int64[1];
    int shifted_js = other.js - nodeses_int64[1];
    int shifted_ke = other.get_ke() + nodeses_int64[2];
    int shifted_ks = other.ks - nodeses_int64[2];
    if ((ith < 0 && shifted_ie < this->get_3D_local_vertices().is)
     || (ith > 0 && shifted_is > this->get_3D_local_vertices().get_ie())
     || (jth < 0 && shifted_je < this->get_3D_local_vertices().js)
     || (jth > 0 && shifted_js > this->get_3D_local_vertices().get_je())
     || (kth < 0 && shifted_ke < this->get_3D_local_vertices().ks)
     || (kth > 0 && shifted_ks > this->get_3D_local_vertices().get_ke())) {
        result.set_vertices(0, -1, 0, -1, 0, -1);
    } else {
        int is = ith > 0 && shifted_is <= this->get_3D_local_vertices().get_ie() && shifted_is >= this->get_3D_local_vertices().is
               ? shifted_is : this->get_3D_local_vertices().is;
        int ie = ith < 0 && shifted_ie >= this->get_3D_local_vertices().is && shifted_ie <= this->get_3D_local_vertices().get_ie()
               ? shifted_ie : this->get_3D_local_vertices().get_ie();
        int js = jth > 0 && shifted_js <= this->get_3D_local_vertices().get_je() && shifted_js >= this->get_3D_local_vertices().js
               ? shifted_js : this->get_3D_local_vertices().js;
        int je = jth < 0 && shifted_je >= this->get_3D_local_vertices().js && shifted_je <= this->get_3D_local_vertices().get_je()
               ? shifted_je : this->get_3D_local_vertices().get_je();
        int ks = kth > 0 && shifted_ks <= this->get_3D_local_vertices().get_ke() && shifted_ks >= this->get_3D_local_vertices().ks
               ? shifted_ks : this->get_3D_local_vertices().ks;
        int ke = kth < 0 && shifted_ke >= this->get_3D_local_vertices().ks && shifted_ke <= this->get_3D_local_vertices().get_ke()
               ? shifted_ke : this->get_3D_local_vertices().get_ke();
        result.set_vertices(is, ie, js, je, ks, ke);
    }
    return;
}

uint Domain_parallel_vertices_3D::generate_domain_counts(uint* domain_counts, const bool* is_periodic, const int* FDn) const {
    int comm_rank = this->get_domain_3d_comm_rank();
    if (comm_rank < (int)this->get_active_comm_size()) {
        domain_counts[0] = this->get_i_left_block_n(FDn[0], is_periodic[0]);
        domain_counts[1] = this->get_i_right_block_n(FDn[0], is_periodic[0]);
        domain_counts[2] = this->get_j_left_block_n(FDn[1], is_periodic[1]);
        domain_counts[3] = this->get_j_right_block_n(FDn[1], is_periodic[1]);
        domain_counts[4] = this->get_k_left_block_n(FDn[2], is_periodic[2]);
        domain_counts[5] = this->get_k_right_block_n(FDn[2], is_periodic[2]);
    } else {
        domain_counts[0] = 0;
        domain_counts[1] = 0;
        domain_counts[2] = 0;
        domain_counts[3] = 0;
        domain_counts[4] = 0;
        domain_counts[5] = 0;
    }
    return domain_counts[0] + domain_counts[1] + domain_counts[2] + domain_counts[3] + domain_counts[4] + domain_counts[5];
}

uint Domain_parallel_vertices_3D::generate_domain_counts_full(uint* domain_counts, const bool* is_periodic, const int* FDn) const {
    int comm_rank = this->get_domain_3d_comm_rank();
    if (comm_rank < (int)this->get_active_comm_size()) {
        domain_counts[0] = this->get_i_left_block_n(FDn[0], is_periodic[0]);
        domain_counts[1] = this->get_i_right_block_n(FDn[0], is_periodic[0]);
        domain_counts[2] = this->get_j_left_block_n(FDn[1], is_periodic[1]);
        domain_counts[3] = this->get_j_right_block_n(FDn[1], is_periodic[1]);
        domain_counts[4] = this->get_k_left_block_n(FDn[2], is_periodic[2]);
        domain_counts[5] = this->get_k_right_block_n(FDn[2], is_periodic[2]);
    } else {
        domain_counts[0] = 0;
        domain_counts[1] = 0;
        domain_counts[2] = 0;
        domain_counts[3] = 0;
        domain_counts[4] = 0;
        domain_counts[5] = 0;
    }
    return (domain_counts[0] + domain_counts[1] + 1)
         * (domain_counts[2] + domain_counts[3] + 1)
         * (domain_counts[4] + domain_counts[5] + 1)
         - 1;
}

void Domain_parallel_vertices_3D::data_transfer_prepare(//const Domain_parallel_vertices_3D& my_domain_vertices,
                                                        const Domain_parallel_vertices_3D& other_domain_vertices, 
                                                        Vertices_3D* overlap_vertices_list,
                                                        int* nnode_list) const {
    int comm_size = other_domain_vertices.get_comm_size();
    std::vector<Vertices_3D> other_local_vertices_list(comm_size, Vertices_3D(0, 0, 0));
    other_domain_vertices.get_comm_local_vertices_list(other_local_vertices_list.data());
    for (uint i = 0; i < (uint)comm_size; i++) {
        if (!(other_domain_vertices.get_active_comm_b(i) == 0)) continue;
        overlap_vertices_list[i] = this->get_3D_local_vertices().get_overlap_vertices(other_local_vertices_list[i]);
        nnode_list[i] = overlap_vertices_list[i].get_size();
    }
    return;
}

void Domain_parallel_vertices_3D::ex_arr_receive_data_transfer_prepare(const int* nodeses,
                                                                       const uint* domain_counts,
                                                                       Vertices_3D* stencil_vertices_list,
                                                                       Vertices_3D* super_stencil_vertices_list,
                                                                    //    int* stencil_vertices_block_shift,
                                                                       int* stencil_vertices_offset,
                                                                    //    Vertices_3D* overlap_vertices_list,
                                                                       int* nnode_list) const {
    if (!this->is_active) return;
    int comm_size = this->get_domain_3d_comm_size();
    int domain_counts_sum = domain_counts[0] + domain_counts[1]
                          + domain_counts[2] + domain_counts[3]
                          + domain_counts[4] + domain_counts[5];
    std::vector<int> stencil_vertices_block_shift(domain_counts_sum, 0);
    std::vector<Vertices_3D> overlap_vertices_list(comm_size, Vertices_3D(0, 0, 0));
    std::vector<int> offsets(comm_size + 1, 0);

    uint count = 0;
    for (uint ith = 1; ith < domain_counts[0] + 1; ith++) {
        stencil_vertices_block_shift[count] = this->generate_ith_vertices(stencil_vertices_list[count], -(int)ith, nodeses[0]);
        ++count;
    }
    for (uint ith = 1; ith < domain_counts[1] + 1; ith++) {
        stencil_vertices_block_shift[count] = this->generate_ith_vertices(stencil_vertices_list[count], (int)ith, nodeses[0]);
        ++count;
    }
    for (uint jth = 1; jth < domain_counts[2] + 1; jth++) {
        stencil_vertices_block_shift[count] = this->generate_jth_vertices(stencil_vertices_list[count], -(int)jth, nodeses[1]);
        ++count;
    }
    for (uint jth = 1; jth < domain_counts[3] + 1; jth++) {
        stencil_vertices_block_shift[count] = this->generate_jth_vertices(stencil_vertices_list[count], (int)jth, nodeses[1]);
        ++count;
    }
    for (uint kth = 1; kth < domain_counts[4] + 1; kth++) {
        stencil_vertices_block_shift[count] = this->generate_kth_vertices(stencil_vertices_list[count], -(int)kth, nodeses[2]);
        ++count;
    }
    for (uint kth = 1; kth < domain_counts[5] + 1; kth++) {
        stencil_vertices_block_shift[count] = this->generate_kth_vertices(stencil_vertices_list[count], (int)kth, nodeses[2]);
        ++count;
    }
    
    count = 0;
    for (uint ith = 1; ith < domain_counts[0] + 1; ith++) {
        int other_comm_i = (int)this->get_active_comm_i() - (int)(ith % (int)this->get_active_comm_ni());
        if (other_comm_i < 0) other_comm_i = other_comm_i + (int)this->get_active_comm_ni();
        uint other_index = this->get_active_comm_rank(other_comm_i,
                                                      this->get_active_comm_j(),
                                                      this->get_active_comm_k());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count],
                                                            -stencil_vertices_block_shift[count] * (int)this->get_3D_shared_vertices().ni,
                                                            0, 0);
        ++count;
    }
    for (uint ith = 1; ith < domain_counts[1] + 1; ith++) {
        int other_comm_i = (int)this->get_active_comm_i() + (int)(ith % (int)this->get_active_comm_ni());
        if (other_comm_i > (int)this->get_active_comm_ni() - 1) other_comm_i = other_comm_i - (int)this->get_active_comm_ni();
        uint other_index = this->get_active_comm_rank(other_comm_i,
                                                      this->get_active_comm_j(),
                                                      this->get_active_comm_k());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count],
                                                            -stencil_vertices_block_shift[count] * (int)this->get_3D_shared_vertices().ni,
                                                            0, 0);
        ++count;
    }
    for (uint jth = 1; jth < domain_counts[2] + 1; jth++) {
        int other_comm_j = (int)this->get_active_comm_j() - (int)(jth % (int)this->get_active_comm_nj());
        if (other_comm_j < 0) other_comm_j = other_comm_j + (int)this->get_active_comm_nj();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(),
                                                      other_comm_j,
                                                      this->get_active_comm_k());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count], 0,
                                                            -stencil_vertices_block_shift[count] * (int)this->get_3D_shared_vertices().nj,
                                                            0);
        ++count;
    }
    for (uint jth = 1; jth < domain_counts[3] + 1; jth++) {
        int other_comm_j = (int)this->get_active_comm_j() + (int)(jth % (int)this->get_active_comm_nj());
        if (other_comm_j > (int)this->get_active_comm_nj() - 1) other_comm_j = other_comm_j - (int)this->get_active_comm_nj();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(),
                                                      other_comm_j,
                                                      this->get_active_comm_k());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count], 0,
                                                            -stencil_vertices_block_shift[count] * (int)this->get_3D_shared_vertices().nj,
                                                            0);
        ++count;
    }
    for (uint kth = 1; kth < domain_counts[4] + 1; kth++) {
        int other_comm_k = (int)this->get_active_comm_k() - (int)(kth % (int)this->get_active_comm_nk());
        if (other_comm_k < 0) other_comm_k = other_comm_k + (int)this->get_active_comm_nk();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(),
                                                      this->get_active_comm_j(),
                                                      other_comm_k);
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count], 0, 0,
                                                            -stencil_vertices_block_shift[count] * (int)this->get_3D_shared_vertices().nk);
        ++count;
    }
    for (uint kth = 1; kth < domain_counts[5] + 1; kth++) {
        int other_comm_k = (int)this->get_active_comm_k() + (int)(kth % (int)this->get_active_comm_nk());
        if (other_comm_k > (int)this->get_active_comm_nk() - 1) other_comm_k = other_comm_k - (int)this->get_active_comm_nk();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(),
                                                      this->get_active_comm_j(),
                                                      other_comm_k);
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count], 0, 0,
                                                            -stencil_vertices_block_shift[count] * (int)this->get_3D_shared_vertices().nk);
        ++count;
    }

    for (uint i = 0; i < (uint)comm_size; i++)
    {
        nnode_list[i] = overlap_vertices_list[i].get_size();
        offsets[i + 1] = nnode_list[i] + offsets[i];
    }

    count = 0;
    for (uint ith = 1; ith < domain_counts[0] + 1; ith++) {
        int other_comm_i = (int)this->get_active_comm_i() - (int)(ith % (int)this->get_active_comm_ni());
        if (other_comm_i < 0) other_comm_i = other_comm_i + (int)this->get_active_comm_ni();
        uint other_index = this->get_active_comm_rank(other_comm_i,
                                                      this->get_active_comm_j(),
                                                      this->get_active_comm_k());

        int is = stencil_vertices_list[count].is - stencil_vertices_block_shift[count] * (int)this->get_3D_shared_vertices().ni;
        int js = stencil_vertices_list[count].js;
        int ks = stencil_vertices_list[count].ks;
        stencil_vertices_offset[count] = overlap_vertices_list[other_index].get_index_nocheck(is, js, ks)
                                       + offsets[other_index];
        super_stencil_vertices_list[count] = overlap_vertices_list[other_index];
        ++count;
    }
    for (uint ith = 1; ith < domain_counts[1] + 1; ith++) {
        int other_comm_i = (int)this->get_active_comm_i() + (int)(ith % (int)this->get_active_comm_ni());
        if (other_comm_i > (int)this->get_active_comm_ni() - 1) other_comm_i = other_comm_i - (int)this->get_active_comm_ni();
        uint other_index = this->get_active_comm_rank(other_comm_i,
                                                      this->get_active_comm_j(),
                                                      this->get_active_comm_k());

        int is = stencil_vertices_list[count].is - stencil_vertices_block_shift[count] * (int)this->get_3D_shared_vertices().ni;
        int js = stencil_vertices_list[count].js;
        int ks = stencil_vertices_list[count].ks;
        stencil_vertices_offset[count] = overlap_vertices_list[other_index].get_index_nocheck(is, js, ks)
                                       + offsets[other_index];
        super_stencil_vertices_list[count] = overlap_vertices_list[other_index];
        ++count;
    }
    for (uint jth = 1; jth < domain_counts[2] + 1; jth++) {
        int other_comm_j = (int)this->get_active_comm_j() - (int)(jth % (int)this->get_active_comm_nj());
        if (other_comm_j < 0) other_comm_j = other_comm_j + (int)this->get_active_comm_nj();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(),
                                                      other_comm_j,
                                                      this->get_active_comm_k());

        int is = stencil_vertices_list[count].is;
        int js = stencil_vertices_list[count].js - stencil_vertices_block_shift[count] * (int)this->get_3D_shared_vertices().nj;
        int ks = stencil_vertices_list[count].ks;
        stencil_vertices_offset[count] = overlap_vertices_list[other_index].get_index_nocheck(is, js, ks)
                                       + offsets[other_index];
        super_stencil_vertices_list[count] = overlap_vertices_list[other_index];
        ++count;
    }
    for (uint jth = 1; jth < domain_counts[3] + 1; jth++) {
        int other_comm_j = (int)this->get_active_comm_j() + (int)(jth % (int)this->get_active_comm_nj());
        if (other_comm_j > (int)this->get_active_comm_nj() - 1) other_comm_j = other_comm_j - (int)this->get_active_comm_nj();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(),
                                                      other_comm_j,
                                                      this->get_active_comm_k());

        int is = stencil_vertices_list[count].is;
        int js = stencil_vertices_list[count].js - stencil_vertices_block_shift[count] * (int)this->get_3D_shared_vertices().nj;
        int ks = stencil_vertices_list[count].ks;
        stencil_vertices_offset[count] = overlap_vertices_list[other_index].get_index_nocheck(is, js, ks)
                                       + offsets[other_index];
        super_stencil_vertices_list[count] = overlap_vertices_list[other_index];
        ++count;
    }
    for (uint kth = 1; kth < domain_counts[4] + 1; kth++) {
        int other_comm_k = (int)this->get_active_comm_k() - (int)(kth % (int)this->get_active_comm_nk());
        if (other_comm_k < 0) other_comm_k = other_comm_k + (int)this->get_active_comm_nk();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(),
                                                      this->get_active_comm_j(),
                                                      other_comm_k);
        int is = stencil_vertices_list[count].is;
        int js = stencil_vertices_list[count].js;
        int ks = stencil_vertices_list[count].ks - stencil_vertices_block_shift[count] * (int)this->get_3D_shared_vertices().nk;
        stencil_vertices_offset[count] = overlap_vertices_list[other_index].get_index_nocheck(is, js, ks)
                                       + offsets[other_index];
        super_stencil_vertices_list[count] = overlap_vertices_list[other_index];
        ++count;
    }
    for (uint kth = 1; kth < domain_counts[5] + 1; kth++) {
        int other_comm_k = (int)this->get_active_comm_k() + (int)(kth % (int)this->get_active_comm_nk());
        if (other_comm_k > (int)this->get_active_comm_nk() - 1) other_comm_k = other_comm_k - (int)this->get_active_comm_nk();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(),
                                                      this->get_active_comm_j(),
                                                      other_comm_k);
        int is = stencil_vertices_list[count].is;
        int js = stencil_vertices_list[count].js;
        int ks = stencil_vertices_list[count].ks - stencil_vertices_block_shift[count] * (int)this->get_3D_shared_vertices().nk;
        stencil_vertices_offset[count] = overlap_vertices_list[other_index].get_index_nocheck(is, js, ks)
                                       + offsets[other_index];
        super_stencil_vertices_list[count] = overlap_vertices_list[other_index];
        ++count;
    }
    return;
}

void Domain_parallel_vertices_3D::ex_arr_receive_data_transfer_prepare_full(const int* nodeses,
                                                                       const uint* domain_counts,
                                                                       Vertices_3D* stencil_vertices_list,
                                                                       Vertices_3D* super_stencil_vertices_list,
                                                                    //    int* stencil_vertices_block_shift,
                                                                       int* stencil_vertices_offset,
                                                                    //    Vertices_3D* overlap_vertices_list,
                                                                       int* nnode_list) const {
    if (!this->is_active) return;
    int comm_size = this->get_domain_3d_comm_size();
    int domain_counts_sum = (domain_counts[0] + domain_counts[1] + 1)
                          * (domain_counts[2] + domain_counts[3] + 1)
                          * (domain_counts[4] + domain_counts[5] + 1)
                          - 1;
    std::vector<std::vector<int64_t>> stencil_vertices_block_shiftes(domain_counts_sum);
    std::vector<Vertices_3D> overlap_vertices_list(comm_size, Vertices_3D(0, 0, 0));
    std::vector<int> offsets(comm_size + 1, 0);
    int64_t nodeses_int64[3] = {nodeses[0], nodeses[1], nodeses[2]};

    uint count = 0;
    for (int kth = -(int)domain_counts[4]; kth <= (int)domain_counts[5]; ++kth) {
        for (int jth = -(int)domain_counts[2]; jth <= (int)domain_counts[3]; ++jth) {
            for (int ith = -(int)domain_counts[0]; ith <= (int)domain_counts[1]; ++ith) {
                if (ith == 0 && jth == 0 && kth == 0) continue;
                stencil_vertices_block_shiftes[count] = this->generate_ijkth_vertices(stencil_vertices_list[count], ith, jth, kth, nodeses_int64);
                int other_comm_i = (int)this->get_active_comm_i() + (int)(ith % (int)this->get_active_comm_ni());
                if (other_comm_i < 0) other_comm_i = other_comm_i + (int)this->get_active_comm_ni();
                if (other_comm_i > (int)this->get_active_comm_ni() - 1) other_comm_i = other_comm_i - (int)this->get_active_comm_ni();
                int other_comm_j = (int)this->get_active_comm_j() + (int)(jth % (int)this->get_active_comm_nj());
                if (other_comm_j < 0) other_comm_j = other_comm_j + (int)this->get_active_comm_nj();
                if (other_comm_j > (int)this->get_active_comm_nj() - 1) other_comm_j = other_comm_j - (int)this->get_active_comm_nj();
                int other_comm_k = (int)this->get_active_comm_k() + (int)(kth % (int)this->get_active_comm_nk());
                if (other_comm_k < 0) other_comm_k = other_comm_k + (int)this->get_active_comm_nk();
                if (other_comm_k > (int)this->get_active_comm_nk() - 1) other_comm_k = other_comm_k - (int)this->get_active_comm_nk();
                uint other_index = this->get_active_comm_rank(other_comm_i, other_comm_j, other_comm_k);
                overlap_vertices_list[other_index] =
                    overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count],
                                                            -stencil_vertices_block_shiftes[count][0]
                                                          * (int)this->get_3D_shared_vertices().ni,
                                                            -stencil_vertices_block_shiftes[count][1]
                                                          * (int)this->get_3D_shared_vertices().nj,
                                                            -stencil_vertices_block_shiftes[count][2]
                                                          * (int)this->get_3D_shared_vertices().nk);
                ++count;
            }
        }
    }

    for (uint i = 0; i < (uint)comm_size; i++) {
        nnode_list[i] = overlap_vertices_list[i].get_size();
        offsets[i + 1] = nnode_list[i] + offsets[i];
    }

    count = 0;
    for (int kth = -(int)domain_counts[4]; kth <= (int)domain_counts[5]; ++kth) {
        for (int jth = -(int)domain_counts[2]; jth <= (int)domain_counts[3]; ++jth) {
            for (int ith = -(int)domain_counts[0]; ith <= (int)domain_counts[1]; ++ith) {
                if (ith == 0 && jth == 0 && kth == 0) continue;
                int other_comm_i = (int)this->get_active_comm_i() + (int)(ith % (int)this->get_active_comm_ni());
                if (other_comm_i < 0) other_comm_i = other_comm_i + (int)this->get_active_comm_ni();
                else if (other_comm_i > (int)this->get_active_comm_ni() - 1) other_comm_i = other_comm_i - (int)this->get_active_comm_ni();
                int other_comm_j = (int)this->get_active_comm_j() + (int)(jth % (int)this->get_active_comm_nj());
                if (other_comm_j < 0) other_comm_j = other_comm_j + (int)this->get_active_comm_nj();
                else if (other_comm_j > (int)this->get_active_comm_nj() - 1) other_comm_j = other_comm_j - (int)this->get_active_comm_nj();
                int other_comm_k = (int)this->get_active_comm_k() + (int)(kth % (int)this->get_active_comm_nk());
                if (other_comm_k < 0) other_comm_k = other_comm_k + (int)this->get_active_comm_nk();
                else if (other_comm_k > (int)this->get_active_comm_nk() - 1) other_comm_k = other_comm_k - (int)this->get_active_comm_nk();
                uint other_index = this->get_active_comm_rank(other_comm_i, other_comm_j, other_comm_k);
                int is = stencil_vertices_list[count].is - stencil_vertices_block_shiftes[count][0] * (int)this->get_3D_shared_vertices().ni;
                int js = stencil_vertices_list[count].js - stencil_vertices_block_shiftes[count][1] * (int)this->get_3D_shared_vertices().nj;
                int ks = stencil_vertices_list[count].ks - stencil_vertices_block_shiftes[count][2] * (int)this->get_3D_shared_vertices().nk;
                stencil_vertices_offset[count] = overlap_vertices_list[other_index].get_index_nocheck(is, js, ks)
                                       + offsets[other_index];
                super_stencil_vertices_list[count] = overlap_vertices_list[other_index];
                ++count;
            }
        }
    }
    return;
}

void Domain_parallel_vertices_3D::ex_arr_send_data_transfer_prepare(const int* nodeses,
                                                                    const uint* domain_counts,
                                                                    Vertices_3D* overlap_vertices_list,
                                                                    int* nnode_list) const {
    if (!this->is_active) return;
    int comm_size = this->get_domain_3d_comm_size();
    int domain_counts_sum = domain_counts[0] + domain_counts[1]
                          + domain_counts[2] + domain_counts[3]
                          + domain_counts[4] + domain_counts[5];
    std::vector<Vertices_3D> stencil_vertices_list(domain_counts_sum, Vertices_3D(0, 0, 0));
    uint count = 0;
    for (uint ith = 1; ith < domain_counts[0] + 1; ith++) {
        this->generate_ith_need_vertices(stencil_vertices_list[count], -(int)ith, nodeses[0]);
        int other_comm_i = (int)this->get_active_comm_i() - (int)(ith % (int)this->get_active_comm_ni());
        if (other_comm_i < 0) other_comm_i = other_comm_i + (int)this->get_active_comm_ni();
        uint other_index = this->get_active_comm_rank(other_comm_i,
                                                      this->get_active_comm_j(),
                                                      this->get_active_comm_k());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count]);
        ++count;
    }
    for (uint ith = 1; ith < domain_counts[1] + 1; ith++) {
        this->generate_ith_need_vertices(stencil_vertices_list[count], (int)ith, nodeses[0]);
        int other_comm_i = (int)this->get_active_comm_i() + (int)(ith % (int)this->get_active_comm_ni());
        if (other_comm_i > (int)this->get_active_comm_ni() - 1) other_comm_i = other_comm_i - (int)this->get_active_comm_ni();
        uint other_index = this->get_active_comm_rank(other_comm_i,
                                                      this->get_active_comm_j(),
                                                      this->get_active_comm_k());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count]);
        ++count;
    }
    for (uint jth = 1; jth < domain_counts[2] + 1; jth++) {
        this->generate_jth_need_vertices(stencil_vertices_list[count], -(int)jth, nodeses[1]);
        int other_comm_j = (int)this->get_active_comm_j() - (int)(jth % (int)this->get_active_comm_nj());
        if (other_comm_j < 0) other_comm_j = other_comm_j + (int)this->get_active_comm_nj();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(),
                                                      other_comm_j,
                                                      this->get_active_comm_k());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count]);
        ++count;
    }
    for (uint jth = 1; jth < domain_counts[3] + 1; jth++) {
        this->generate_jth_need_vertices(stencil_vertices_list[count], (int)jth, nodeses[1]);
        int other_comm_j = (int)this->get_active_comm_j() + (int)(jth % (int)this->get_active_comm_nj());
        if (other_comm_j > (int)this->get_active_comm_nj() - 1) other_comm_j = other_comm_j - (int)this->get_active_comm_nj();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(),
                                                      other_comm_j,
                                                      this->get_active_comm_k());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count]);
        ++count;
    }
    for (uint kth = 1; kth < domain_counts[4] + 1; kth++) {
        this->generate_kth_need_vertices(stencil_vertices_list[count], -(int)kth, nodeses[2]);
        int other_comm_k = (int)this->get_active_comm_k() - (int)(kth % (int)this->get_active_comm_nk());
        if (other_comm_k < 0) other_comm_k = other_comm_k + (int)this->get_active_comm_nk();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(),
                                                      this->get_active_comm_j(),
                                                      other_comm_k);
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count]);
        ++count;
    }
    for (uint kth = 1; kth < domain_counts[5] + 1; kth++) {
        this->generate_kth_need_vertices(stencil_vertices_list[count], (int)kth, nodeses[2]);
        int other_comm_k = (int)this->get_active_comm_k() + (int)(kth % (int)this->get_active_comm_nk());
        if (other_comm_k > (int)this->get_active_comm_nk() - 1) other_comm_k = other_comm_k - (int)this->get_active_comm_nk();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(),
                                                      this->get_active_comm_j(),
                                                      other_comm_k);
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count]);
        ++count;
    }

    for (uint i = 0; i < (uint)comm_size; i++)
    {
        nnode_list[i] = overlap_vertices_list[i].get_size();
    }

    return;
}

void Domain_parallel_vertices_3D::ex_arr_send_data_transfer_prepare_full(const int* nodeses,
                                                                    const uint* domain_counts,
                                                                    Vertices_3D* overlap_vertices_list,
                                                                    int* nnode_list) const {
    if (!this->is_active) return;
    int comm_size = this->get_domain_3d_comm_size();
    int domain_counts_sum = (domain_counts[0] + domain_counts[1] + 1)
                          * (domain_counts[2] + domain_counts[3] + 1)
                          * (domain_counts[4] + domain_counts[5] + 1)
                          - 1;
    std::vector<Vertices_3D> stencil_vertices_list(domain_counts_sum, Vertices_3D(0, 0, 0));
    uint count = 0;
    for (int kth = -(int)domain_counts[4]; kth <= (int)domain_counts[5]; ++kth) {
        for (int jth = -(int)domain_counts[2]; jth <= (int)domain_counts[3]; ++jth) {
            for (int ith = -(int)domain_counts[0]; ith <= (int)domain_counts[1]; ++ith) {
                if (ith == 0 && jth == 0 && kth == 0) continue;
                this->generate_ijkth_need_vertices(stencil_vertices_list[count], ith, jth, kth, nodeses);
                int other_comm_i = (int)this->get_active_comm_i() + (int)(ith % (int)this->get_active_comm_ni());
                if (other_comm_i < 0) other_comm_i = other_comm_i + (int)this->get_active_comm_ni();
                else if (other_comm_i > (int)this->get_active_comm_ni() - 1) other_comm_i = other_comm_i - (int)this->get_active_comm_ni();
                int other_comm_j = (int)this->get_active_comm_j() + (int)(jth % (int)this->get_active_comm_nj());
                if (other_comm_j < 0) other_comm_j = other_comm_j + (int)this->get_active_comm_nj();
                else if (other_comm_j > (int)this->get_active_comm_nj() - 1) other_comm_j = other_comm_j - (int)this->get_active_comm_nj();
                int other_comm_k = (int)this->get_active_comm_k() + (int)(kth % (int)this->get_active_comm_nk());
                if (other_comm_k < 0) other_comm_k = other_comm_k + (int)this->get_active_comm_nk();
                else if (other_comm_k > (int)this->get_active_comm_nk() - 1) other_comm_k = other_comm_k - (int)this->get_active_comm_nk();
                uint other_index = this->get_active_comm_rank(other_comm_i, other_comm_j, other_comm_k);
                overlap_vertices_list[other_index] =
                    overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count]);
                ++count;
            }
        }
    }
    for (uint i = 0; i < (uint)comm_size; i++) {
        nnode_list[i] = overlap_vertices_list[i].get_size();
    }
    return;
}

void Domain_parallel_vertices_3D::init(const Vertices_3D& shared_vertices, const MPI_Comm& comm, const uint& max_size) {
    this->set_shared_vertices(shared_vertices);
    this->deepcopy_mpi_comm(comm);
    this->generate_chunk_sizes(max_size);
    return;
}

void Domain_parallel_vertices_3D::init(const Domain_parallel_vertices_3D& other) {
    this->deepcopy(other);
    return;
}

 void Domain_parallel_vertices_3D::destructor() {
    if (this->comm != MPI_COMM_SELF && this->comm != MPI_COMM_WORLD && this->comm != MPI_COMM_NULL) {
        MPI_Comm_free(&(this->comm));
    }
    this->comm = MPI_COMM_NULL;
    return;
 }

void Domain_parallel_vertices_3D::show() const {
    std::cout << "mpi rank = " << this->get_comm_rank() 
              << ", in size = " << this->get_comm_size() << std::endl;
    std::cout << "is_active = " << this->is_active << std::endl;
    std::cout << "size - this->get_active_comm_size() = "
              << this->get_comm_size() - this->get_active_comm_size() << std::endl;
    std::cout << "[get_active_comm_ni, nj, nk] = [" << this->get_active_comm_ni() << ", " 
                                                    << this->get_active_comm_nj() << ", " 
                                                    << this->get_active_comm_nk() << "] " << std::endl;
    std::cout << "[cart_coords i, j, k] = [" << this->get_active_comm_i() << ", " 
                                             << this->get_active_comm_j() << ", " 
                                             << this->get_active_comm_k() << "] " << std::endl;
    std::cout << "[chunk_size_i, j, k] = [" << this->chunk_size_i << ", "
                                            << this->chunk_size_j << ", "
                                            << this->chunk_size_k << "] " << std::endl;
    std::cout << "shared_vertices: " << std::endl;
    this->get_3D_shared_vertices().show();
    std::cout << "this->shared_vertices.get_size() = " << this->get_3D_shared_vertices().get_size() << std::endl;
    std::cout << "get_3D_local_vertices(): " << std::endl;
    this->get_3D_local_vertices().show();
    std::cout << "this->local_vertices.get_size() = " << this->get_3D_local_vertices().get_size() << std::endl;
    return;
}

Domain_parallel_vertices_4D::Domain_parallel_vertices_4D() {}

Domain_parallel_vertices_4D::Domain_parallel_vertices_4D(const Domain_parallel_vertices_4D& other)
                           : Domain_parallel_vertices_3D(other) {
    this->deepcopy(other);
    // this->domain_3d_comm = other.domain_3d_comm;
    // this->band_comm = other.band_comm;
    // this->shared_vertices = other.shared_vertices;
    // this->local_vertices = other.local_vertices;
    // this->chunk_size_b = other.chunk_size_b;
}

Domain_parallel_vertices_4D::~Domain_parallel_vertices_4D() {
    this->destructor();
}

Domain_parallel_vertices_4D& Domain_parallel_vertices_4D::deepcopy(const Domain_parallel_vertices_4D& other) {
    this->Domain_parallel_vertices_3D::deepcopy(other);
    this->shared_vertices = other.shared_vertices;
    this->local_vertices = other.local_vertices;
    this->chunk_size_b = other.chunk_size_b;
    if (other.domain_3d_comm != MPI_COMM_NULL) {
        MPI_Comm_split(other.domain_3d_comm, 1, 1, &(this->domain_3d_comm));
    } else {
        this->domain_3d_comm = MPI_COMM_NULL;
    }
    if (other.band_comm != MPI_COMM_NULL) {
        MPI_Comm_split(other.band_comm, 1, 1, &(this->band_comm));
    } else {
        this->band_comm = MPI_COMM_NULL;
    }
    // int band_comm_index = this->Domain_parallel_vertices_3D::get_active_comm_rank(this->get_active_comm_i(),
    //                                                                               this->get_active_comm_j(),
    //                                                                               this->get_active_comm_k());
    // int domain_3d_comm_index = this->get_active_comm_b();
    // MPI_Comm_split(this->comm, domain_3d_comm_index, domain_3d_comm_index, &(this->domain_3d_comm));
    // MPI_Comm_split(this->comm, band_comm_index, band_comm_index, &(this->band_comm));
    return *this;
}

MPI_Comm Domain_parallel_vertices_4D::get_domain_3d_comm() const {
    return this->domain_3d_comm;
}

MPI_Comm Domain_parallel_vertices_4D::get_band_comm() const {
    return this->band_comm;
}

MPI_Comm Domain_parallel_vertices_4D::get_domain_4d_comm() const {
    return this->comm;
}

int Domain_parallel_vertices_4D::get_domain_4d_comm_size() const {
    int size;
    MPI_Comm_size(this->get_domain_4d_comm(), &size);
    return size;
}

int Domain_parallel_vertices_4D::get_domain_4d_comm_rank() const {
    int rank;
    MPI_Comm_rank(this->get_domain_4d_comm(), &rank);
    return rank;
}

void Domain_parallel_vertices_4D::set_chunk_size_b(const uint& chunk_size_b) {
    this->chunk_size_b = chunk_size_b;
    return;
}

void Domain_parallel_vertices_4D::set_comm_nb(const uint& comm_nb) {
    this->set_comm_nb(comm_nb, this->get_4D_shared_vertices().nb);
    return;
}

void Domain_parallel_vertices_4D::set_comm_nb(const uint& comm_nb, const uint& shared_nb) {
    this->set_chunk_size_b(Parallel_vertices::cal_chunksize_1D(comm_nb, shared_nb));
    return;
}

const Vertices_3D& Domain_parallel_vertices_4D::get_3D_shared_vertices() const {
    return this->shared_vertices;
}

const Vertices_3D& Domain_parallel_vertices_4D::get_3D_local_vertices() const {
    return this->local_vertices;
}

const Vertices_4D& Domain_parallel_vertices_4D::get_4D_shared_vertices() const {
    return this->shared_vertices;
}

const Vertices_4D& Domain_parallel_vertices_4D::get_4D_local_vertices() const {
    return this->local_vertices;
}

uint Domain_parallel_vertices_4D::get_chunk_size_b() const {
    return this->chunk_size_b;
}

uint Domain_parallel_vertices_4D::get_last_block_size_b() const {
    return this->get_4D_shared_vertices().get_nb() % this->get_chunk_size_b() == 0
         ? this->get_chunk_size_b()
         : this->get_4D_shared_vertices().get_nb() % this->get_chunk_size_b();
}

uint Domain_parallel_vertices_4D::get_b_left_block_n(const uint& nodes, const bool& is_periodic) const {
    return Parallel_vertices::generate_left_domain_n(this->get_active_comm_b(), this->get_active_comm_nb(), this->get_chunk_size_b(),
                                this->get_last_block_size_b(), nodes, is_periodic);
}

uint Domain_parallel_vertices_4D::get_b_right_block_n(const uint& nodes, const bool& is_periodic) const {
    return Parallel_vertices::generate_right_domain_n(this->get_active_comm_b(), this->get_active_comm_nb(), this->get_chunk_size_b(),
                                this->get_last_block_size_b(), nodes, is_periodic);
}

void Domain_parallel_vertices_4D::generate_chunk_sizes(const uint& max_size) {
    if (this->get_chunk_size_i() == UINT_MAX
     && this->get_chunk_size_j() == UINT_MAX
     && this->get_chunk_size_k() == UINT_MAX
     && this->get_chunk_size_b() == UINT_MAX) {
        std::array<uint,4> comm_ns =
                            Parallel_vertices::cal_comm_ns_4D_by_sparc(this->shared_vertices,
                                                                       this->comm, max_size);
        this->set_comm_ni(comm_ns[0]);
        this->set_comm_nj(comm_ns[1]);
        this->set_comm_nk(comm_ns[2]);
        this->set_comm_nb(comm_ns[3]);
    } else {
        uint comm_size = max_size == 0
                        ? this->get_comm_size()
                        : max_size < (uint)this->get_comm_size()
                        ? max_size
                        : (uint)this->get_comm_size();
        if (this->get_chunk_size_b() == UINT_MAX) {
            this->set_chunk_size_b(Parallel_vertices::cal_chunksize_1D(comm_size, this->get_4D_shared_vertices().nb));
            // this->set_chunk_size_b(this->get_4D_shared_vertices().nb);
        }
        uint comm_size_block = comm_size / this->get_active_comm_nb();
        this->Domain_parallel_vertices_3D::generate_chunk_sizes(comm_size_block);
        assert(this->get_active_comm_size() <= comm_size);
    }
    this->set_is_active(this->get_comm_rank() < (int)this->get_active_comm_size() ?
                        true : false);
    this->set_local_vertices();
    return;
}

void Domain_parallel_vertices_4D::set_local_vertices() {
    this->set_local_vertices(this->generate_local_vertices());
    return;
}

void Domain_parallel_vertices_4D::set_local_vertices(const Vertices_4D& local_vertices) {
    this->local_vertices.set_vertices(local_vertices);
    return;
}

void Domain_parallel_vertices_4D::set_shared_vertices(const Vertices_4D& shared_vertices) {
    this->shared_vertices.set_vertices(shared_vertices);
    return;
}

uint Domain_parallel_vertices_4D::get_active_comm_nb() const {
    return (this->get_4D_shared_vertices().nb - 1 + this->get_chunk_size_b())/this->get_chunk_size_b();
}

uint Domain_parallel_vertices_4D::get_active_comm_size() const {
    return this->get_active_comm_ni() * this->get_active_comm_nj()
         * this->get_active_comm_nk() * this->get_active_comm_nb();
}

uint Domain_parallel_vertices_4D::get_active_comm_i() const {
    return this->get_active_comm_i(this->get_comm_rank());
}

uint Domain_parallel_vertices_4D::get_active_comm_j() const {
    return this->get_active_comm_j(this->get_comm_rank());
}

uint Domain_parallel_vertices_4D::get_active_comm_k() const {
    return this->get_active_comm_k(this->get_comm_rank());
}

uint Domain_parallel_vertices_4D::get_active_comm_b() const {
    return this->get_active_comm_b(this->get_comm_rank());
}

uint Domain_parallel_vertices_4D::get_active_comm_i(const uint& rank) const {
    return this->is_Col_Maj ? rank % this->get_active_comm_ni()
                            : rank / (this->get_active_comm_nb() * this->get_active_comm_nk() * this->get_active_comm_nj());
}

uint Domain_parallel_vertices_4D::get_active_comm_j(const uint& rank) const {
    return this->is_Col_Maj ? rank % (this->get_active_comm_ni() * this->get_active_comm_nj()) / this->get_active_comm_ni()
                            : rank % (this->get_active_comm_nb() * this->get_active_comm_nk() * this->get_active_comm_nj()) 
                                   / (this->get_active_comm_nb() * this->get_active_comm_nk());
}

uint Domain_parallel_vertices_4D::get_active_comm_k(const uint& rank) const {
    return this->is_Col_Maj ? rank % (this->get_active_comm_ni() * this->get_active_comm_nj() * this->get_active_comm_nk())
                                   / (this->get_active_comm_ni() * this->get_active_comm_nj())
                            : rank % (this->get_active_comm_nb() * this->get_active_comm_nk()) 
                                   / this->get_active_comm_nb();
}

uint Domain_parallel_vertices_4D::get_active_comm_b(const uint& rank) const {
    return this->is_Col_Maj ? rank / (this->get_active_comm_ni() * this->get_active_comm_nj() * this->get_active_comm_nk())
                            : rank % this->get_active_comm_nb();
}

uint Domain_parallel_vertices_4D::get_active_comm_rank() const {
    return this->get_active_comm_rank(this->get_active_comm_i(),
                                      this->get_active_comm_j(),
                                      this->get_active_comm_k(),
                                      this->get_active_comm_b());
}

uint Domain_parallel_vertices_4D::get_active_comm_rank(const uint& i, const uint& j, const uint& k, const uint& b) const {
    if (!this->is_active) {
        return INT_MAX;
    }
    return this->is_Col_Maj ? i + (j + (k + b * this->get_active_comm_nk()) * this->get_active_comm_nj()) * this->get_active_comm_ni()
                            : b + (k + (j + i * this->get_active_comm_nj()) * this->get_active_comm_nk()) * this->get_active_comm_nb();
}

void Domain_parallel_vertices_4D::get_comm_local_vertices_list(Vertices_3D* local_vertices_list) const {
    for (int i = 0; i < this->get_comm_size(); i++) {
        (local_vertices_list++)->set_vertices(this->generate_local_vertices((uint)i));
    }
    return;
}

void Domain_parallel_vertices_4D::get_comm_local_vertices_list(Vertices_4D* local_vertices_list) const {
    for (int i = 0; i < this->get_comm_size(); i++) {
        (local_vertices_list++)->set_vertices(this->generate_local_vertices((uint)i));
    }
    return;
}

Vertices_4D Domain_parallel_vertices_4D::generate_local_vertices() const {
    return Domain_parallel_vertices_4D::generate_local_vertices(this->get_comm_rank());
}

Vertices_4D Domain_parallel_vertices_4D::generate_local_vertices(const uint& rank) const {
    Vertices_4D result;
    if (rank < this->get_active_comm_size()) {
        int local_is = (int)(this->get_active_comm_i(rank) * this->chunk_size_i) + this->get_4D_shared_vertices().is;
        int local_ie = local_is + (int)this->chunk_size_i - 1 <= get_4D_shared_vertices().get_ie() ?
                       local_is + (int)this->chunk_size_i - 1 : get_4D_shared_vertices().get_ie();
        int local_js = (int)(this->get_active_comm_j(rank) * this->chunk_size_j) + this->get_4D_shared_vertices().js;
        int local_je = local_js + (int)this->chunk_size_j - 1 <= get_4D_shared_vertices().get_je() ?
                       local_js + (int)this->chunk_size_j - 1 : get_4D_shared_vertices().get_je();
        int local_ks = (int)(this->get_active_comm_k(rank) * this->chunk_size_k) + this->get_4D_shared_vertices().ks;
        int local_ke = local_ks + (int)this->chunk_size_k - 1 <= get_4D_shared_vertices().get_ke() ?
                       local_ks + (int)this->chunk_size_k - 1 : get_4D_shared_vertices().get_ke();
        int local_bs = (int)(this->get_active_comm_b(rank) * this->chunk_size_b) + this->get_4D_shared_vertices().bs;
        int local_be = local_bs + (int)this->chunk_size_b - 1 <= get_4D_shared_vertices().get_be() ?
                       local_bs + (int)this->chunk_size_b - 1 : get_4D_shared_vertices().get_be();
        result.set_vertices(local_is, local_ie, local_js, local_je, local_ks, local_ke, local_bs, local_be);
    } else {
        result.set_vertices(0, -1, 0, -1, 0, -1, 0, -1);
    }
    return result;
}

int64_t Domain_parallel_vertices_4D::generate_ith_vertices(Vertices_4D& other, const int64_t ith, const uint64_t nodes) const {
    int64_t is;
    int64_t ie;
    int64_t index = Parallel_vertices::generate_nth_vertices(&is, &ie, ith, this->get_active_comm_i(), this->get_active_comm_ni(),
                                    this->chunk_size_i, this->get_last_block_size_i(), nodes);
    other.set_vertices(is + this->get_4D_shared_vertices().is, ie + this->get_4D_shared_vertices().is,
                       this->get_4D_local_vertices().js, this->get_4D_local_vertices().get_je(),
                       this->get_4D_local_vertices().ks, this->get_4D_local_vertices().get_ke(),
                       this->get_4D_local_vertices().bs, this->get_4D_local_vertices().get_be());
    return index;
}

int64_t Domain_parallel_vertices_4D::generate_jth_vertices(Vertices_4D& other, const int64_t jth, const uint64_t nodes) const {
    int64_t js;
    int64_t je;
    int64_t index = Parallel_vertices::generate_nth_vertices(&js, &je, jth, this->get_active_comm_j(), this->get_active_comm_nj(),
                                    this->chunk_size_j, this->get_last_block_size_j(), nodes);
    other.set_vertices(this->get_4D_local_vertices().is, this->get_4D_local_vertices().get_ie(),
                       js + this->get_4D_shared_vertices().js, je + this->get_4D_shared_vertices().js,
                       this->get_4D_local_vertices().ks, this->get_4D_local_vertices().get_ke(),
                       this->get_4D_local_vertices().bs, this->get_4D_local_vertices().get_be());
    return index;
}

int64_t Domain_parallel_vertices_4D::generate_kth_vertices(Vertices_4D& other, const int64_t kth, const uint64_t nodes) const {
    int64_t ks;
    int64_t ke;
    int64_t index = Parallel_vertices::generate_nth_vertices(&ks, &ke, kth, this->get_active_comm_k(), this->get_active_comm_nk(),
                                    this->chunk_size_k, this->get_last_block_size_k(), nodes);
    other.set_vertices(this->get_4D_local_vertices().is, this->get_4D_local_vertices().get_ie(),
                       this->get_4D_local_vertices().js, this->get_4D_local_vertices().get_je(),
                       ks + this->get_4D_shared_vertices().ks, ke + this->get_4D_shared_vertices().ks,
                       this->get_4D_local_vertices().bs, this->get_4D_local_vertices().get_be());
    return index;
}

int64_t Domain_parallel_vertices_4D::generate_bth_vertices(Vertices_4D& other, const int64_t bth, const uint64_t nodes) const {
    int64_t bs;
    int64_t be;
    int64_t index = Parallel_vertices::generate_nth_vertices(&bs, &be, bth, this->get_active_comm_b(), this->get_active_comm_nb(),
                                    this->chunk_size_b, this->get_last_block_size_b(), nodes);
    other.set_vertices(this->get_4D_local_vertices().is, this->get_4D_local_vertices().get_ie(),
                       this->get_4D_local_vertices().js, this->get_4D_local_vertices().get_je(),
                       this->get_4D_local_vertices().ks, this->get_4D_local_vertices().get_ke(),
                       bs + this->get_4D_shared_vertices().bs, be + this->get_4D_shared_vertices().bs);
    return index;
}

void Domain_parallel_vertices_4D::generate_ith_need_vertices(Vertices_4D& result, const int& ith, const uint& nodes) const {
    if (ith == 0) {
        result.set_vertices(this->get_4D_local_vertices());
        return;
    } else if (ith < 0) {
        Vertices_4D other;
        this->generate_ith_vertices(other, ith);
        int shifted_ie = other.get_ie() + nodes;
        if (shifted_ie < this->get_4D_local_vertices().is) {
            result.set_vertices(0, -1, 0, -1, 0, -1, 0, -1);
            return;
        } else if (shifted_ie>=this->get_4D_local_vertices().is && shifted_ie <= this->get_4D_local_vertices().get_ie()) {
            result.set_vertices(this->get_4D_local_vertices().is, shifted_ie,
                                  this->get_4D_local_vertices().js, get_4D_local_vertices().get_je(),
                                  this->get_4D_local_vertices().ks, get_4D_local_vertices().get_ke(),
                                  this->get_4D_local_vertices().bs, get_4D_local_vertices().get_be());
            return;
        } else {
            result.set_vertices(this->get_4D_local_vertices());
            return;
        }
    } else { //ith > 0
        Vertices_4D other;
        this->generate_ith_vertices(other, ith);
        int shifted_is = other.is - nodes;
        if (shifted_is > this->get_4D_local_vertices().get_ie()) {
            result.set_vertices(0, -1, 0, -1, 0, -1, 0, -1);
            return;
        } else if (shifted_is <= this->get_4D_local_vertices().get_ie() && shifted_is >= this->get_4D_local_vertices().is) {
            result.set_vertices(shifted_is, get_4D_local_vertices().get_ie(),
                                  this->get_4D_local_vertices().js, get_4D_local_vertices().get_je(),
                                  this->get_4D_local_vertices().ks, get_4D_local_vertices().get_ke(),
                                  this->get_4D_local_vertices().bs, get_4D_local_vertices().get_be());
            return;
        } else {
            result.set_vertices(this->get_4D_local_vertices());
            return;
        }
    }
}

void Domain_parallel_vertices_4D::generate_jth_need_vertices(Vertices_4D& result, const int& jth, const uint& nodes) const {
    if (jth == 0) {
        result.set_vertices(this->get_4D_local_vertices());
        return;
    } else if (jth < 0) {
        Vertices_4D other;
        this->generate_jth_vertices(other, jth);
        int shifted_je = other.get_je() + nodes;
        if (shifted_je < this->get_4D_local_vertices().js) {
            result.set_vertices(0, -1, 0, -1, 0, -1, 0, -1);
            return;
        } else if (shifted_je>=this->get_4D_local_vertices().js && shifted_je <= this->get_4D_local_vertices().get_je()) {
            result.set_vertices(this->get_4D_local_vertices().is, get_4D_local_vertices().get_ie(),
                                  this->get_4D_local_vertices().js, shifted_je,
                                  this->get_4D_local_vertices().ks, get_4D_local_vertices().get_ke(),
                                  this->get_4D_local_vertices().bs, get_4D_local_vertices().get_be());
            return;
        } else {
            result.set_vertices(this->get_4D_local_vertices());
            return;
        }
    } else { //jth > 0
        Vertices_4D other;
        this->generate_jth_vertices(other, jth);
        int shifted_js = other.js - nodes;
        if (shifted_js > this->get_4D_local_vertices().get_je()) {
            result.set_vertices(0, -1, 0, -1, 0, -1, 0, -1);
            return;
        } else if (shifted_js <= this->get_4D_local_vertices().get_je() && shifted_js >= this->get_4D_local_vertices().js) {
            result.set_vertices(this->get_4D_local_vertices().is, get_4D_local_vertices().get_ie(),
                                  shifted_js, get_4D_local_vertices().get_je(),
                                  this->get_4D_local_vertices().ks, get_4D_local_vertices().get_ke(),
                                  this->get_4D_local_vertices().bs, get_4D_local_vertices().get_be());
            return;
        } else {
            result.set_vertices(this->get_4D_local_vertices());
            return;
        }
    }
}

void Domain_parallel_vertices_4D::generate_kth_need_vertices(Vertices_4D& result, const int& kth, const uint& nodes) const {
    if (kth == 0) {
        result.set_vertices(this->get_4D_local_vertices());
        return;
    } else if (kth < 0) {
        Vertices_4D other;
        this->generate_kth_vertices(other, kth);
        int shifted_ke = other.get_ke() + nodes;
        if (shifted_ke < this->get_4D_local_vertices().ks) {
            result.set_vertices(0, -1, 0, -1, 0, -1, 0, -1);
            return;
        } else if (shifted_ke >= this->get_4D_local_vertices().ks && shifted_ke <= this->get_4D_local_vertices().get_ke()) {
            result.set_vertices(this->get_4D_local_vertices().is, get_4D_local_vertices().get_ie(),
                                  this->get_4D_local_vertices().js, get_4D_local_vertices().get_je(),
                                  this->get_4D_local_vertices().ks, shifted_ke,
                                  this->get_4D_local_vertices().bs, get_4D_local_vertices().get_be());
            return;
        } else {
            result.set_vertices(this->get_4D_local_vertices());
            return;
        }
    } else { //jth > 0
        Vertices_4D other;
        this->generate_kth_vertices(other, kth);
        int shifted_ks = other.ks - nodes;
        if (shifted_ks > this->get_4D_local_vertices().get_ke()) {
            result.set_vertices(0, -1, 0, -1, 0, -1, 0, -1);
            return;
        } else if (shifted_ks <= this->get_4D_local_vertices().get_ke() && shifted_ks >= this->get_4D_local_vertices().ks) {
            result.set_vertices(this->get_4D_local_vertices().is, get_4D_local_vertices().get_ie(),
                                  this->get_4D_local_vertices().js, get_4D_local_vertices().get_je(),
                                  shifted_ks, get_4D_local_vertices().get_ke(),
                                  this->get_4D_local_vertices().bs, get_4D_local_vertices().get_be());
            return;
        } else {
            result.set_vertices(this->get_4D_local_vertices());
            return;
        }
    }
}

void Domain_parallel_vertices_4D::generate_bth_need_vertices(Vertices_4D& result, const int& bth, const uint& nodes) const {
    if (bth == 0) {
        result.set_vertices(this->get_4D_local_vertices());
        return;
    } else if (bth < 0) {
        Vertices_4D other;
        this->generate_bth_vertices(other, bth);
        int shifted_be = other.get_be() + nodes;
        if (shifted_be < this->get_4D_local_vertices().bs) {
            result.set_vertices(0, -1, 0, -1, 0, -1, 0, -1);
            return;
        } else if (shifted_be >= this->get_4D_local_vertices().bs && shifted_be <= this->get_4D_local_vertices().get_be()) {
            result.set_vertices(this->get_4D_local_vertices().is, get_4D_local_vertices().get_ie(),
                                  this->get_4D_local_vertices().js, get_4D_local_vertices().get_je(),
                                  this->get_4D_local_vertices().ks, get_4D_local_vertices().get_ke(),
                                  this->get_4D_local_vertices().bs, shifted_be);
            return;
        } else {
            result.set_vertices(this->get_4D_local_vertices());
            return;
        }
    } else { //jth > 0
        Vertices_4D other;
        this->generate_bth_vertices(other, bth);
        int shifted_bs = other.bs - nodes;
        if (shifted_bs > this->get_4D_local_vertices().get_be()) {
            result.set_vertices(0, -1, 0, -1, 0, -1, 0, -1);
            return;
        } else if (shifted_bs <= this->get_4D_local_vertices().get_be() && shifted_bs >= this->get_4D_local_vertices().bs) {
            result.set_vertices(this->get_4D_local_vertices().is, get_4D_local_vertices().get_ie(),
                                  this->get_4D_local_vertices().js, get_4D_local_vertices().get_je(),
                                  this->get_4D_local_vertices().ks, get_4D_local_vertices().get_ke(),
                                  shifted_bs, get_4D_local_vertices().get_be());
            return;
        } else {
            result.set_vertices(this->get_4D_local_vertices());
            return;
        }
    }
}

void Domain_parallel_vertices_4D::data_transfer_prepare(//const Domain_parallel_vertices_3D& my_domain_vertices,
                                                        const Domain_parallel_vertices_4D& other_domain_vertices, 
                                                        Vertices_4D* overlap_vertices_list,
                                                        int* nnode_list) const {
    int comm_size = this->get_comm_size();
    std::vector<Vertices_4D> other_local_vertices_list(comm_size, Vertices_4D(0, 0, 0, 0));
    other_domain_vertices.get_comm_local_vertices_list(other_local_vertices_list.data());
    for (uint i = 0; i < (uint)comm_size; i++)
    {
        overlap_vertices_list[i] = this->get_4D_local_vertices().get_overlap_vertices(other_local_vertices_list[i]);
        nnode_list[i] = overlap_vertices_list[i].get_size();
    }
    return;
}

uint Domain_parallel_vertices_4D::generate_domain_counts(uint* domain_counts, const bool* is_periodic, const int* FDn) const {
    int comm_rank = this->get_comm_rank();
    if (comm_rank < (int)this->get_active_comm_size()) {
        domain_counts[0] = this->get_i_left_block_n(FDn[0], is_periodic[0]);
        domain_counts[1] = this->get_i_right_block_n(FDn[0], is_periodic[0]);
        domain_counts[2] = this->get_j_left_block_n(FDn[1], is_periodic[1]);
        domain_counts[3] = this->get_j_right_block_n(FDn[1], is_periodic[1]);
        domain_counts[4] = this->get_k_left_block_n(FDn[2], is_periodic[2]);
        domain_counts[5] = this->get_k_right_block_n(FDn[2], is_periodic[2]);
        domain_counts[6] = this->get_b_left_block_n(FDn[3], is_periodic[3]);
        domain_counts[7] = this->get_b_right_block_n(FDn[3], is_periodic[3]);
    } else {
        domain_counts[0] = 0;
        domain_counts[1] = 0;
        domain_counts[2] = 0;
        domain_counts[3] = 0;
        domain_counts[4] = 0;
        domain_counts[5] = 0;
        domain_counts[6] = 0;
        domain_counts[7] = 0;
    }
    return domain_counts[0] + domain_counts[1] + domain_counts[2] + domain_counts[3]
         + domain_counts[4] + domain_counts[5] + domain_counts[6] + domain_counts[7];
}

void Domain_parallel_vertices_4D::ex_arr_receive_data_transfer_prepare(const int* FDn,
                                                                       const uint* domain_counts,
                                                                       Vertices_4D* stencil_vertices_list,
                                                                       Vertices_4D* super_stencil_vertices_list,
                                                                    //    int* stencil_vertices_block_shift,
                                                                       int* stencil_vertices_offset,
                                                                    //    Vertices_3D* overlap_vertices_list,
                                                                       int* nnode_list) const {
    int comm_size = this->get_domain_4d_comm_size();
    int domain_counts_sum = domain_counts[0] + domain_counts[1]
                          + domain_counts[2] + domain_counts[3]
                          + domain_counts[4] + domain_counts[5]
                          + domain_counts[6] + domain_counts[7];
    std::vector<int> stencil_vertices_block_shift(domain_counts_sum, 0);
    std::vector<Vertices_4D> overlap_vertices_list(comm_size, Vertices_4D(0, 0, 0, 0));
    std::vector<int> offsets(comm_size + 1, 0);

    uint count = 0;
    for (uint ith = 1; ith < domain_counts[0] + 1; ith++) {
        stencil_vertices_block_shift[count] = this->generate_ith_vertices(stencil_vertices_list[count], -(int)ith, FDn[0]);
        ++count;
    }
    for (uint ith = 1; ith < domain_counts[1] + 1; ith++) {
        stencil_vertices_block_shift[count] = this->generate_ith_vertices(stencil_vertices_list[count], (int)ith, FDn[0]);
        ++count;
    }
    for (uint jth = 1; jth < domain_counts[2] + 1; jth++) {
        stencil_vertices_block_shift[count] = this->generate_jth_vertices(stencil_vertices_list[count], -(int)jth, FDn[1]);
        ++count;
    }
    for (uint jth = 1; jth < domain_counts[3] + 1; jth++) {
        stencil_vertices_block_shift[count] = this->generate_jth_vertices(stencil_vertices_list[count], (int)jth, FDn[1]);
        ++count;
    }
    for (uint kth = 1; kth < domain_counts[4] + 1; kth++) {
        stencil_vertices_block_shift[count] = this->generate_kth_vertices(stencil_vertices_list[count], -(int)kth, FDn[2]);
        ++count;
    }
    for (uint kth = 1; kth < domain_counts[5] + 1; kth++) {
        stencil_vertices_block_shift[count] = this->generate_kth_vertices(stencil_vertices_list[count], (int)kth, FDn[2]);
        ++count;
    }
    for (uint bth = 1; bth < domain_counts[6] + 1; bth++) {
        stencil_vertices_block_shift[count] = this->generate_bth_vertices(stencil_vertices_list[count], -(int)bth, FDn[3]);
        ++count;
    }
    for (uint bth = 1; bth < domain_counts[7] + 1; bth++) {
        stencil_vertices_block_shift[count] = this->generate_bth_vertices(stencil_vertices_list[count], (int)bth, FDn[3]);
        ++count;
    }

    count = 0;
    for (uint ith = 1; ith < domain_counts[0] + 1; ith++) {
        int other_comm_i = (int)this->get_active_comm_i() - (int)(ith % (int)this->get_active_comm_ni());
        if (other_comm_i < 0) other_comm_i = other_comm_i + (int)this->get_active_comm_ni();
        uint other_index = this->get_active_comm_rank(other_comm_i, this->get_active_comm_j(),
                                                      this->get_active_comm_k(), this->get_active_comm_b());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count],
                                                            -stencil_vertices_block_shift[count] * (int)this->get_4D_shared_vertices().ni,
                                                            0, 0, 0);
        ++count;
    }
    for (uint ith = 1; ith < domain_counts[1] + 1; ith++) {
        int other_comm_i = (int)this->get_active_comm_i() + (int)(ith % (int)this->get_active_comm_ni());
        if (other_comm_i > (int)this->get_active_comm_ni() - 1) other_comm_i = other_comm_i - (int)this->get_active_comm_ni();
        uint other_index = this->get_active_comm_rank(other_comm_i, this->get_active_comm_j(),
                                                      this->get_active_comm_k(), this->get_active_comm_b());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count],
                                                            -stencil_vertices_block_shift[count] * (int)this->get_4D_shared_vertices().ni,
                                                            0, 0, 0);
        ++count;
    }
    for (uint jth = 1; jth < domain_counts[2] + 1; jth++) {
        int other_comm_j = (int)this->get_active_comm_j() - (int)(jth % (int)this->get_active_comm_nj());
        if (other_comm_j < 0) other_comm_j = other_comm_j + (int)this->get_active_comm_nj();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(), other_comm_j,
                                                      this->get_active_comm_k(), this->get_active_comm_b());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count], 0,
                                                            -stencil_vertices_block_shift[count] * (int)this->get_4D_shared_vertices().nj,
                                                            0, 0);
        ++count;
    }
    for (uint jth = 1; jth < domain_counts[3] + 1; jth++) {
        int other_comm_j = (int)this->get_active_comm_j() + (int)(jth % (int)this->get_active_comm_nj());
        if (other_comm_j > (int)this->get_active_comm_nj() - 1) other_comm_j = other_comm_j - (int)this->get_active_comm_nj();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(), other_comm_j,
                                                      this->get_active_comm_k(), this->get_active_comm_b());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count], 0,
                                                            -stencil_vertices_block_shift[count] * (int)this->get_4D_shared_vertices().nj,
                                                            0, 0);
        ++count;
    }
    for (uint kth = 1; kth < domain_counts[4] + 1; kth++) {
        int other_comm_k = (int)this->get_active_comm_k() - (int)(kth % (int)this->get_active_comm_nk());
        if (other_comm_k < 0) other_comm_k = other_comm_k + (int)this->get_active_comm_nk();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(), this->get_active_comm_j(),
                                                      other_comm_k, this->get_active_comm_b());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count], 0, 0,
                                                            -stencil_vertices_block_shift[count] * (int)this->get_4D_shared_vertices().nk,
                                                            0);
        ++count;
    }
    for (uint kth = 1; kth < domain_counts[5] + 1; kth++) {
        int other_comm_k = (int)this->get_active_comm_k() + (int)(kth % (int)this->get_active_comm_nk());
        if (other_comm_k > (int)this->get_active_comm_nk() - 1) other_comm_k = other_comm_k - (int)this->get_active_comm_nk();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(), this->get_active_comm_j(),
                                                      other_comm_k, this->get_active_comm_b());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count], 0, 0,
                                                            -stencil_vertices_block_shift[count] * (int)this->get_4D_shared_vertices().nk,
                                                            0);
        ++count;
    }
    for (uint bth = 1; bth < domain_counts[6] + 1; bth++) {
        int other_comm_b = (int)this->get_active_comm_b() - (int)(bth % (int)this->get_active_comm_nb());
        if (other_comm_b < 0) other_comm_b = other_comm_b + (int)this->get_active_comm_nb();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(), this->get_active_comm_j(),
                                                      this->get_active_comm_k(), other_comm_b);
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count], 0, 0, 0,
                                                            -stencil_vertices_block_shift[count] * (int)this->get_4D_shared_vertices().nb);
        ++count;
    }
    for (uint bth = 1; bth < domain_counts[7] + 1; bth++) {
        int other_comm_b = (int)this->get_active_comm_b() + (int)(bth % (int)this->get_active_comm_nb());
        if (other_comm_b > (int)this->get_active_comm_nb() - 1) other_comm_b = other_comm_b - (int)this->get_active_comm_nb();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(), this->get_active_comm_j(),
                                                      this->get_active_comm_k(), other_comm_b);
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count], 0, 0, 0, 
                                                            -stencil_vertices_block_shift[count] * (int)this->get_4D_shared_vertices().nb);
        ++count;
    }

    for (uint i = 0; i < (uint)comm_size; i++) {
        nnode_list[i] = overlap_vertices_list[i].get_size();
        offsets[i + 1] = nnode_list[i] + offsets[i];
    }

    count = 0;
    for (uint ith = 1; ith < domain_counts[0] + 1; ith++) {
        int other_comm_i = (int)this->get_active_comm_i() - (int)(ith % (int)this->get_active_comm_ni());
        if (other_comm_i < 0) other_comm_i = other_comm_i + (int)this->get_active_comm_ni();
        uint other_index = this->get_active_comm_rank(other_comm_i, this->get_active_comm_j(),
                                                      this->get_active_comm_k(), this->get_active_comm_b());

        int is = stencil_vertices_list[count].is - stencil_vertices_block_shift[count] * (int)this->get_4D_shared_vertices().ni;
        int js = stencil_vertices_list[count].js;
        int ks = stencil_vertices_list[count].ks;
        int bs = stencil_vertices_list[count].bs;
        stencil_vertices_offset[count] = overlap_vertices_list[other_index].get_index_nocheck(is, js, ks, bs)
                                       + offsets[other_index];
        super_stencil_vertices_list[count] = overlap_vertices_list[other_index];
        ++count;
    }
    for (uint ith = 1; ith < domain_counts[1] + 1; ith++) {
        int other_comm_i = (int)this->get_active_comm_i() + (int)(ith % (int)this->get_active_comm_ni());
        if (other_comm_i > (int)this->get_active_comm_ni() - 1) other_comm_i = other_comm_i - (int)this->get_active_comm_ni();
        uint other_index = this->get_active_comm_rank(other_comm_i, this->get_active_comm_j(),
                                                      this->get_active_comm_k(), this->get_active_comm_b());

        int is = stencil_vertices_list[count].is - stencil_vertices_block_shift[count] * (int)this->get_4D_shared_vertices().ni;
        int js = stencil_vertices_list[count].js;
        int ks = stencil_vertices_list[count].ks;
        int bs = stencil_vertices_list[count].bs;
        stencil_vertices_offset[count] = overlap_vertices_list[other_index].get_index_nocheck(is, js, ks, bs)
                                       + offsets[other_index];
        super_stencil_vertices_list[count] = overlap_vertices_list[other_index];
        ++count;
    }
    for (uint jth = 1; jth < domain_counts[2] + 1; jth++) {
        int other_comm_j = (int)this->get_active_comm_j() - (int)(jth % (int)this->get_active_comm_nj());
        if (other_comm_j < 0) other_comm_j = other_comm_j + (int)this->get_active_comm_nj();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(), other_comm_j,
                                                      this->get_active_comm_k(), this->get_active_comm_b());

        int is = stencil_vertices_list[count].is;
        int js = stencil_vertices_list[count].js - stencil_vertices_block_shift[count] * (int)this->get_4D_shared_vertices().nj;
        int ks = stencil_vertices_list[count].ks;
        int bs = stencil_vertices_list[count].bs;
        stencil_vertices_offset[count] = overlap_vertices_list[other_index].get_index_nocheck(is, js, ks, bs)
                                       + offsets[other_index];
        super_stencil_vertices_list[count] = overlap_vertices_list[other_index];
        ++count;
    }
    for (uint jth = 1; jth < domain_counts[3] + 1; jth++) {
        int other_comm_j = (int)this->get_active_comm_j() + (int)(jth % (int)this->get_active_comm_nj());
        if (other_comm_j > (int)this->get_active_comm_nj() - 1) other_comm_j = other_comm_j - (int)this->get_active_comm_nj();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(), other_comm_j,
                                                      this->get_active_comm_k(), this->get_active_comm_b());

        int is = stencil_vertices_list[count].is;
        int js = stencil_vertices_list[count].js - stencil_vertices_block_shift[count] * (int)this->get_4D_shared_vertices().nj;
        int ks = stencil_vertices_list[count].ks;
        int bs = stencil_vertices_list[count].bs;
        stencil_vertices_offset[count] = overlap_vertices_list[other_index].get_index_nocheck(is, js, ks, bs)
                                       + offsets[other_index];
        super_stencil_vertices_list[count] = overlap_vertices_list[other_index];
        ++count;
    }
    for (uint kth = 1; kth < domain_counts[4] + 1; kth++) {
        int other_comm_k = (int)this->get_active_comm_k() - (int)(kth % (int)this->get_active_comm_nk());
        if (other_comm_k < 0) other_comm_k = other_comm_k + (int)this->get_active_comm_nk();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(), this->get_active_comm_j(),
                                                      other_comm_k, this->get_active_comm_b());
        int is = stencil_vertices_list[count].is;
        int js = stencil_vertices_list[count].js;
        int ks = stencil_vertices_list[count].ks - stencil_vertices_block_shift[count] * (int)this->get_4D_shared_vertices().nk;
        int bs = stencil_vertices_list[count].bs;
        stencil_vertices_offset[count] = overlap_vertices_list[other_index].get_index_nocheck(is, js, ks, bs)
                                       + offsets[other_index];
        super_stencil_vertices_list[count] = overlap_vertices_list[other_index];
        ++count;
    }
    for (uint kth = 1; kth < domain_counts[5] + 1; kth++) {
        int other_comm_k = (int)this->get_active_comm_k() + (int)(kth % (int)this->get_active_comm_nk());
        if (other_comm_k > (int)this->get_active_comm_nk() - 1) other_comm_k = other_comm_k - (int)this->get_active_comm_nk();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(), this->get_active_comm_j(),
                                                     other_comm_k, this->get_active_comm_b());
        int is = stencil_vertices_list[count].is;
        int js = stencil_vertices_list[count].js;
        int ks = stencil_vertices_list[count].ks - stencil_vertices_block_shift[count] * (int)this->get_4D_shared_vertices().nk;
        int bs = stencil_vertices_list[count].bs;
        stencil_vertices_offset[count] = overlap_vertices_list[other_index].get_index_nocheck(is, js, ks, bs)
                                       + offsets[other_index];
        super_stencil_vertices_list[count] = overlap_vertices_list[other_index];
        ++count;
    }
    for (uint bth = 1; bth < domain_counts[6] + 1; bth++) {
        int other_comm_b = (int)this->get_active_comm_b() - (int)(bth % (int)this->get_active_comm_nb());
        if (other_comm_b < 0) other_comm_b = other_comm_b + (int)this->get_active_comm_nb();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(), this->get_active_comm_j(),
                                                      this->get_active_comm_k(), other_comm_b);
        int is = stencil_vertices_list[count].is;
        int js = stencil_vertices_list[count].js;
        int ks = stencil_vertices_list[count].ks;
        int bs = stencil_vertices_list[count].bs - stencil_vertices_block_shift[count] * (int)this->get_4D_shared_vertices().nb;
        stencil_vertices_offset[count] = overlap_vertices_list[other_index].get_index_nocheck(is, js, ks, bs)
                                       + offsets[other_index];
        super_stencil_vertices_list[count] = overlap_vertices_list[other_index];
        ++count;
    }
    for (uint bth = 1; bth < domain_counts[7] + 1; bth++) {
        int other_comm_b = (int)this->get_active_comm_b() + (int)(bth % (int)this->get_active_comm_nb());
        if (other_comm_b > (int)this->get_active_comm_nb() - 1) other_comm_b = other_comm_b - (int)this->get_active_comm_nb();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(), this->get_active_comm_j(),
                                                      this->get_active_comm_k(), other_comm_b);
        int is = stencil_vertices_list[count].is;
        int js = stencil_vertices_list[count].js;
        int ks = stencil_vertices_list[count].ks;
        int bs = stencil_vertices_list[count].bs - stencil_vertices_block_shift[count] * (int)this->get_4D_shared_vertices().nb;
        stencil_vertices_offset[count] = overlap_vertices_list[other_index].get_index_nocheck(is, js, ks, bs)
                                       + offsets[other_index];
        super_stencil_vertices_list[count] = overlap_vertices_list[other_index];
        ++count;
    }
    return;
}

void Domain_parallel_vertices_4D::ex_arr_send_data_transfer_prepare(const int* FDn,
                                                                    const uint* domain_counts,
                                                                    Vertices_4D* overlap_vertices_list,
                                                                    int* nnode_list) const {
    int comm_size = this->get_domain_4d_comm_size();
    int domain_counts_sum = domain_counts[0] + domain_counts[1]
                          + domain_counts[2] + domain_counts[3]
                          + domain_counts[4] + domain_counts[5]
                          + domain_counts[6] + domain_counts[7];
    std::vector<Vertices_4D> stencil_vertices_list(domain_counts_sum, Vertices_4D(0, 0, 0, 0));
    uint count = 0;
    for (uint ith = 1; ith < domain_counts[0] + 1; ith++) {
        this->generate_ith_need_vertices(stencil_vertices_list[count], -(int)ith, FDn[0]);
        int other_comm_i = (int)this->get_active_comm_i() - (int)(ith % (int)this->get_active_comm_ni());
        if (other_comm_i < 0) other_comm_i = other_comm_i + (int)this->get_active_comm_ni();
        uint other_index = this->get_active_comm_rank(other_comm_i, this->get_active_comm_j(),
                                                      this->get_active_comm_k(), this->get_active_comm_b());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count]);
        ++count;
    }
    for (uint ith = 1; ith < domain_counts[1] + 1; ith++) {
        this->generate_ith_need_vertices(stencil_vertices_list[count], (int)ith, FDn[0]);
        int other_comm_i = (int)this->get_active_comm_i() + (int)(ith % (int)this->get_active_comm_ni());
        if (other_comm_i > (int)this->get_active_comm_ni() - 1) other_comm_i = other_comm_i - (int)this->get_active_comm_ni();
        uint other_index = this->get_active_comm_rank(other_comm_i, this->get_active_comm_j(),
                                                      this->get_active_comm_k(), this->get_active_comm_b());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count]);
        ++count;
    }
    for (uint jth = 1; jth < domain_counts[2] + 1; jth++) {
        this->generate_jth_need_vertices(stencil_vertices_list[count], -(int)jth, FDn[1]);
        int other_comm_j = (int)this->get_active_comm_j() - (int)(jth % (int)this->get_active_comm_nj());
        if (other_comm_j < 0) other_comm_j = other_comm_j + (int)this->get_active_comm_nj();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(), other_comm_j,
                                                      this->get_active_comm_k(), this->get_active_comm_b());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count]);
        ++count;
    }
    for (uint jth = 1; jth < domain_counts[3] + 1; jth++) {
        this->generate_jth_need_vertices(stencil_vertices_list[count], (int)jth, FDn[1]);
        int other_comm_j = (int)this->get_active_comm_j() + (int)(jth % (int)this->get_active_comm_nj());
        if (other_comm_j > (int)this->get_active_comm_nj() - 1) other_comm_j = other_comm_j - (int)this->get_active_comm_nj();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(), other_comm_j,
                                                      this->get_active_comm_k(), this->get_active_comm_b());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count]);
        ++count;
    }
    for (uint kth = 1; kth < domain_counts[4] + 1; kth++) {
        this->generate_kth_need_vertices(stencil_vertices_list[count], -(int)kth, FDn[2]);
        int other_comm_k = (int)this->get_active_comm_k() - (int)(kth % (int)this->get_active_comm_nk());
        if (other_comm_k < 0) other_comm_k = other_comm_k + (int)this->get_active_comm_nk();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(), this->get_active_comm_j(),
                                                      other_comm_k, this->get_active_comm_b());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count]);
        ++count;
    }
    for (uint kth = 1; kth < domain_counts[5] + 1; kth++) {
        this->generate_kth_need_vertices(stencil_vertices_list[count], (int)kth, FDn[2]);
        int other_comm_k = (int)this->get_active_comm_k() + (int)(kth % (int)this->get_active_comm_nk());
        if (other_comm_k > (int)this->get_active_comm_nk() - 1) other_comm_k = other_comm_k - (int)this->get_active_comm_nk();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(), this->get_active_comm_j(),
                                                      other_comm_k, this->get_active_comm_b());
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count]);
        ++count;
    }
    for (uint bth = 1; bth < domain_counts[6] + 1; bth++) {
        this->generate_bth_need_vertices(stencil_vertices_list[count], -(int)bth, FDn[3]);
        int other_comm_b = (int)this->get_active_comm_b() - (int)(bth % (int)this->get_active_comm_nb());
        if (other_comm_b < 0) other_comm_b = other_comm_b + (int)this->get_active_comm_nb();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(), this->get_active_comm_j(),
                                                      this->get_active_comm_k(), other_comm_b);
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count]);
        ++count;
    }
    for (uint bth = 1; bth < domain_counts[7] + 1; bth++) {
        this->generate_bth_need_vertices(stencil_vertices_list[count], (int)bth, FDn[3]);
        int other_comm_b = (int)this->get_active_comm_b() + (int)(bth % (int)this->get_active_comm_nb());
        if (other_comm_b > (int)this->get_active_comm_nb() - 1) other_comm_b = other_comm_b - (int)this->get_active_comm_nb();
        uint other_index = this->get_active_comm_rank(this->get_active_comm_i(), this->get_active_comm_j(),
                                                      this->get_active_comm_k(), other_comm_b);
        overlap_vertices_list[other_index] =
                overlap_vertices_list[other_index].get_super_vertices(stencil_vertices_list[count]);
        ++count;
    }
    for (uint i = 0; i < (uint)comm_size; i++) {
        nnode_list[i] = overlap_vertices_list[i].get_size();
    }
    return;
}

void Domain_parallel_vertices_4D::init(const Vertices_4D& shared_vertices, const MPI_Comm& comm, const uint& max_size) {
    this->set_shared_vertices(shared_vertices);
    this->deepcopy_mpi_comm(comm);
    if (shared_vertices.get_size() != 0) {
        this->generate_chunk_sizes(max_size);
        if (this->get_active_comm_nb() == 1) {
            this->band_comm = MPI_COMM_SELF;
        } else {
            int band_comm_index = this->Domain_parallel_vertices_3D::get_active_comm_rank(this->get_active_comm_i(),
                                                                                       this->get_active_comm_j(),
                                                                                       this->get_active_comm_k());
            MPI_Comm_split(this->comm, band_comm_index, band_comm_index, &(this->band_comm));
        }

        if (this->get_active_comm_ni() == 1 && this->get_active_comm_nj() == 1 && this->get_active_comm_nk() == 1) {
            this->domain_3d_comm = MPI_COMM_SELF;
        } else {
            int domain_3d_comm_index = this->is_active ? this->get_active_comm_b() : INT_MAX;
            MPI_Comm_split(this->comm, domain_3d_comm_index, domain_3d_comm_index, &(this->domain_3d_comm));
        }
        // int band_comm_index = this->Domain_parallel_vertices_3D::get_active_comm_rank(this->get_active_comm_i(),
        //                                                                               this->get_active_comm_j(),
        //                                                                               this->get_active_comm_k());
        // int domain_3d_comm_index = this->is_active ? this->get_active_comm_b() : INT_MAX;
        // MPI_Comm_split(this->comm, domain_3d_comm_index, domain_3d_comm_index, &(this->domain_3d_comm));
        // MPI_Comm_split(this->comm, band_comm_index, band_comm_index, &(this->band_comm));
    } else {
        this->local_vertices.set_vertices(shared_vertices);
    }
    return;
}

void Domain_parallel_vertices_4D::init(const Domain_parallel_vertices_4D& domain_parallel_vertices_4D) {
    this->deepcopy(domain_parallel_vertices_4D);
    return;
}

void Domain_parallel_vertices_4D::destructor() {
    if (this->comm != MPI_COMM_SELF && this->comm != MPI_COMM_WORLD && this->comm != MPI_COMM_NULL) {
        MPI_Comm_free(&(this->comm));
    }
    this->comm = MPI_COMM_NULL;
    if (this->domain_3d_comm != MPI_COMM_SELF && this->domain_3d_comm != MPI_COMM_WORLD && this->domain_3d_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&(this->domain_3d_comm));
    }
    this->domain_3d_comm = MPI_COMM_NULL;
    if (this->band_comm != MPI_COMM_SELF && this->band_comm != MPI_COMM_WORLD && this->band_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&(this->band_comm));
    }
    this->band_comm = MPI_COMM_NULL;
    return;
}

void Domain_parallel_vertices_4D::show() const {
    std::cout << "mpi rank = " << this->get_comm_rank() 
              << ", in size = " << this->get_comm_size() << std::endl;
    std::cout << "is_active = " << this->is_active << std::endl;
    std::cout << "this->get_active_comm_size() = " << this->get_active_comm_size() << std::endl;
    std::cout << "size - this->get_active_comm_size() = "
              << this->get_comm_size() - this->get_active_comm_size() << std::endl;
    std::cout << "[get_active_comm_ni, nj, nk, nb] = [" << this->get_active_comm_ni() << ", " 
                                                        << this->get_active_comm_nj() << ", " 
                                                        << this->get_active_comm_nk() << ", " 
                                                        << this->get_active_comm_nb() << "] " << std::endl;
    std::cout << "[cart_coords i, j, k, b] = [" << this->get_active_comm_i() << ", " 
                                                << this->get_active_comm_j() << ", " 
                                                << this->get_active_comm_k() << ", " 
                                                << this->get_active_comm_b() << "] " << std::endl;
    std::cout << "[chunk_size_i, j, k, b] = [" << this->chunk_size_i << ", "
                                               << this->chunk_size_j << ", "
                                               << this->chunk_size_k << ", "
                                               << this->chunk_size_b << "] " << std::endl;
    std::cout << "shared_vertices: " << std::endl;
    this->get_4D_shared_vertices().show();
    std::cout << "this->shared_vertices.get_size() = " << this->get_4D_shared_vertices().get_size() << std::endl;
    std::cout << "get_4D_local_vertices(): " << std::endl;
    this->get_4D_local_vertices().show();
    std::cout << "this->local_vertices.get_size() = " << this->get_4D_local_vertices().get_size() << std::endl;
    std::cout << "domain_3d_comm_index = " << this->Domain_parallel_vertices_3D::get_active_comm_rank() << std::endl;
    std::cout << "band_comm_index = " << this->get_active_comm_b() << std::endl;
    return;
}

Exarr_3D_mpi_package::Exarr_3D_mpi_package(const Domain_parallel_vertices_3D& domain_vertices)
                                         : domain_vertices(domain_vertices) {}

Exarr_3D_mpi_package::~Exarr_3D_mpi_package() {}

template<typename T>
void Exarr_3D_mpi_package::fill_domain_par_ex_arr(const Array_3D<T>& send_arr, Array_3D<T>& recv_arr) const {
    int comm_size;
    MPI_Comm_size(this->comm, &comm_size);
    if (this->need_comm) {
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();

        Array_0D<T> send_array(this->send_offsets[comm_size]);
        for (uint i = 0; i < (uint)comm_size; i++) {
            send_arr.fill_vector(send_array, this->send_overlap_vertices_list[i], this->send_offsets[i]);
        }
        Array_0D<T> receive_array(this->receive_offsets[comm_size]);
        MPI_Request request;
        MPI_Ialltoallv(send_array.data, this->send_nnode_list.data(), this->send_offsets.data(), mpi_datatype,
                    receive_array.data, this->receive_nnode_list.data(), this->receive_offsets.data(),
                    mpi_datatype, this->comm, &request);
        send_arr.fill_overlap(recv_arr);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        for (uint i = 0; i < this->domain_counts_sum; i++) {
            recv_arr.be_filled_vector(receive_array, this->receive_super_stencil_vertices_list[i],
                                    this->receive_stencil_vertices_list[i], this->receive_stencil_vertices_offset[i]);
        }
    } else {
        send_arr.fill_overlap(recv_arr);
        for (uint i = 0; i < this->domain_counts_sum; i++) {
            recv_arr.be_filled_vector(send_arr, send_arr.get_vertices(),
                                    this->receive_stencil_vertices_list[i], this->receive_stencil_vertices_offset[i]);
        }
    }
    return;
}
template void Exarr_3D_mpi_package::fill_domain_par_ex_arr<int>(const Array_3D<int>& send_arr, Array_3D<int>& recv_arr) const;
template void Exarr_3D_mpi_package::fill_domain_par_ex_arr<float>(const Array_3D<float>& send_arr, Array_3D<float>& recv_arr) const;
template void Exarr_3D_mpi_package::fill_domain_par_ex_arr<double>(const Array_3D<double>& send_arr, Array_3D<double>& recv_arr) const;

template<typename T>
void Exarr_3D_mpi_package::fill_domain_par_ex_arr(T const* const __restrict__ send_arr, const Vertices_3D& send_arr_vertices,
                                                  T* const __restrict__ recv_arr, const Vertices_3D& recv_arr_vertices) const {
    #ifdef USE_OPENMP
        int comm_size;
        MPI_Comm_size(this->comm, &comm_size);
        if (this->need_comm) {
            MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
            static T* send_array_static = nullptr;
            #pragma omp single
            send_array_static = new T [this->send_offsets[comm_size]];
            T* const send_array = send_array_static;
            for (uint i = 0; i < (uint)comm_size; i++) {
                Vertices_method::fill_vector(send_arr, send_arr_vertices, send_array,
                                            this->send_overlap_vertices_list[i], this->send_offsets[i]);
            }
            static T* receive_array_static = nullptr;
            #pragma omp single
            receive_array_static = new T [this->receive_offsets[comm_size]];
            T* const receive_array = receive_array_static;
            MPI_Request request;
            #pragma omp master
            {
            MPI_Ialltoallv(send_array, this->send_nnode_list.data(), this->send_offsets.data(), mpi_datatype,
                        receive_array, this->receive_nnode_list.data(), this->receive_offsets.data(),
                        mpi_datatype, this->comm, &request);
            }
            Vertices_method::fill_region(send_arr, send_arr_vertices, recv_arr, recv_arr_vertices,
                                        send_arr_vertices.get_overlap_vertices(recv_arr_vertices));
            #pragma omp master
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            #pragma omp barrier
            for (uint i = 0; i < this->domain_counts_sum; i++) {
                Vertices_method::be_filled_vector(recv_arr, recv_arr_vertices, receive_array, this->receive_super_stencil_vertices_list[i],
                                                this->receive_stencil_vertices_list[i], this->receive_stencil_vertices_offset[i]);
            }
            #pragma omp barrier
            #pragma omp single nowait
            {
                delete [] send_array_static;
                send_array_static = nullptr;
            }
            #pragma omp single nowait
            {
                delete [] receive_array_static;
                receive_array_static = nullptr;
            }
        } else {
            // send_arr.fill_overlap(recv_arr);
            Vertices_method::fill_region(send_arr, send_arr_vertices, recv_arr, recv_arr_vertices,
                                        send_arr_vertices.get_overlap_vertices(recv_arr_vertices));
            for (uint i = 0; i < this->domain_counts_sum; i++) {
                // recv_arr.be_filled_vector(send_arr, send_arr.get_vertices(),
                //                         this->receive_stencil_vertices_list[i], this->receive_stencil_vertices_offset[i]);
                Vertices_method::be_filled_vector(recv_arr, recv_arr_vertices, send_arr, send_arr_vertices,
                                                this->receive_stencil_vertices_list[i], this->receive_stencil_vertices_offset[i]);
            }
        }
    #else //USE_OPENMP
        this->fill_domain_par_ex_arr_sequential<T>(send_arr, send_arr_vertices, recv_arr, recv_arr_vertices);
    #endif //USE_OPENMP
    return;
}
template void Exarr_3D_mpi_package::fill_domain_par_ex_arr<int>(int const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                                int* const recv_arr, const Vertices_3D& recv_arr_vertices) const;
template void Exarr_3D_mpi_package::fill_domain_par_ex_arr<float>(float const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                                  float* const recv_arr, const Vertices_3D& recv_arr_vertices) const;
template void Exarr_3D_mpi_package::fill_domain_par_ex_arr<double>(double const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                                   double* const recv_arr, const Vertices_3D& recv_arr_vertices) const;

template<typename T>
void Exarr_3D_mpi_package::fill_domain_par_ex_arr_sequential(T const* const __restrict__ send_arr, const Vertices_3D& send_arr_vertices,
                                                  T* const __restrict__ recv_arr, const Vertices_3D& recv_arr_vertices) const {
    int comm_size;
    MPI_Comm_size(this->comm, &comm_size);
    if (this->need_comm) {
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        // Array_0D<T> send_array(this->send_offsets[comm_size]);
        T* send_array = new T [this->send_offsets[comm_size]];
        for (uint i = 0; i < (uint)comm_size; i++) {
            // send_arr.fill_vector(send_array, this->send_overlap_vertices_list[i], this->send_offsets[i]);
            Vertices_method::fill_vector(send_arr, send_arr_vertices, send_array,
                                         this->send_overlap_vertices_list[i], this->send_offsets[i]);
        }
        // Array_0D<T> receive_array(this->receive_offsets[comm_size]);
        T* receive_array = new T [this->receive_offsets[comm_size]];
        MPI_Request request;
        MPI_Ialltoallv(send_array, this->send_nnode_list.data(), this->send_offsets.data(), mpi_datatype,
                       receive_array, this->receive_nnode_list.data(), this->receive_offsets.data(),
                       mpi_datatype, this->comm, &request);
        // send_arr.fill_overlap(recv_arr);
        Vertices_method::fill_region(send_arr, send_arr_vertices, recv_arr, recv_arr_vertices,
                                     send_arr_vertices.get_overlap_vertices(recv_arr_vertices));
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        for (uint i = 0; i < this->domain_counts_sum; i++) {
            // recv_arr.be_filled_vector(receive_array, this->receive_super_stencil_vertices_list[i],
            //                         this->receive_stencil_vertices_list[i], this->receive_stencil_vertices_offset[i]);
            Vertices_method::be_filled_vector(recv_arr, recv_arr_vertices, receive_array, this->receive_super_stencil_vertices_list[i],
                                              this->receive_stencil_vertices_list[i], this->receive_stencil_vertices_offset[i]);
        }
        
        delete [] send_array;
        delete [] receive_array;
    } else {
        // send_arr.fill_overlap(recv_arr);
        Vertices_method::fill_region(send_arr, send_arr_vertices, recv_arr, recv_arr_vertices,
                                     send_arr_vertices.get_overlap_vertices(recv_arr_vertices));
        for (uint i = 0; i < this->domain_counts_sum; i++) {
            // recv_arr.be_filled_vector(send_arr, send_arr.get_vertices(),
            //                         this->receive_stencil_vertices_list[i], this->receive_stencil_vertices_offset[i]);
            Vertices_method::be_filled_vector(recv_arr, recv_arr_vertices, send_arr, send_arr_vertices,
                                              this->receive_stencil_vertices_list[i], this->receive_stencil_vertices_offset[i]);
        }
    }
    return;
}
template void Exarr_3D_mpi_package::fill_domain_par_ex_arr_sequential<int>(int const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                                int* const recv_arr, const Vertices_3D& recv_arr_vertices) const;
template void Exarr_3D_mpi_package::fill_domain_par_ex_arr_sequential<float>(float const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                                  float* const recv_arr, const Vertices_3D& recv_arr_vertices) const;
template void Exarr_3D_mpi_package::fill_domain_par_ex_arr_sequential<double>(double const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                                   double* const recv_arr, const Vertices_3D& recv_arr_vertices) const;

void Exarr_3D_mpi_package::init(const bool* is_periodic, const int* FDn) {
    this->comm = this->domain_vertices.get_domain_3d_comm();

    std::array<uint,6> domain_counts;
    int comm_size = this->domain_vertices.get_domain_3d_comm_size();

    // find how many domains is needed
    this->domain_counts_sum = this->domain_vertices.generate_domain_counts(domain_counts.data(), is_periodic, FDn);

    //receive mpi data
    this->receive_stencil_vertices_list.resize(this->domain_counts_sum, Vertices_3D(0, 0, 0));
    this->receive_super_stencil_vertices_list.resize(this->domain_counts_sum, Vertices_3D(0, 0, 0));
    this->receive_stencil_vertices_offset.resize(this->domain_counts_sum, 0);
    this->receive_nnode_list.resize(comm_size, 0);
    this->domain_vertices.ex_arr_receive_data_transfer_prepare(FDn, domain_counts.data(),
                                                               this->receive_stencil_vertices_list.data(),
                                                               this->receive_super_stencil_vertices_list.data(),
                                                               this->receive_stencil_vertices_offset.data(),
                                                               this->receive_nnode_list.data());
    
    this->need_comm = Parallel_vertices::need_comm(this->receive_nnode_list.data(), this->comm);
    if (!this->need_comm) {
        for (std::vector<Vertices_3D>::iterator it = this->receive_super_stencil_vertices_list.begin();
             it != this->receive_super_stencil_vertices_list.end(); ++it) {
            assert(*it == this->domain_vertices.get_3D_local_vertices());
        }
        std::vector<Vertices_3D>().swap(this->receive_super_stencil_vertices_list);
        std::vector<int>().swap(this->receive_nnode_list);
        return;
    }
    this->receive_offsets.resize(comm_size + 1, 0);
    this->receive_offsets[0] = 0;
    for (uint i = 0; i < (uint)comm_size; i++) {
        this->receive_offsets[i + 1] = this->receive_nnode_list[i] + this->receive_offsets[i];
    }

    //send mpi data
    this->send_overlap_vertices_list.resize(comm_size, Vertices_3D(0, 0, 0));
    this->send_nnode_list.resize(comm_size, 0);
    this->send_offsets.resize(comm_size + 1, 0);
    this->domain_vertices.ex_arr_send_data_transfer_prepare(FDn, domain_counts.data(),
                                                            this->send_overlap_vertices_list.data(),
                                                            this->send_nnode_list.data());
    this->send_offsets[0] = 0;
    for (uint i = 0; i < (uint)comm_size; i++) {
        this->send_offsets[i + 1] = this->send_nnode_list[i] + this->send_offsets[i];
    }
    return;
}

void Exarr_3D_mpi_package::init_full(const bool* is_periodic, const int* FDn) {
    this->comm = this->domain_vertices.get_domain_3d_comm();

    std::array<uint,6> domain_counts;
    int comm_size = this->domain_vertices.get_domain_3d_comm_size();

    // find how many domains is needed
    this->domain_counts_sum = this->domain_vertices.generate_domain_counts_full(domain_counts.data(), is_periodic, FDn);

    //receive mpi data
    this->receive_stencil_vertices_list.resize(this->domain_counts_sum, Vertices_3D(0, 0, 0));
    this->receive_super_stencil_vertices_list.resize(this->domain_counts_sum, Vertices_3D(0, 0, 0));
    this->receive_stencil_vertices_offset.resize(this->domain_counts_sum, 0);
    this->receive_nnode_list.resize(comm_size, 0);
    this->domain_vertices.ex_arr_receive_data_transfer_prepare_full(FDn, domain_counts.data(),
                                                                    this->receive_stencil_vertices_list.data(),
                                                                    this->receive_super_stencil_vertices_list.data(),
                                                                    this->receive_stencil_vertices_offset.data(),
                                                                    this->receive_nnode_list.data());
    
    this->need_comm = Parallel_vertices::need_comm(this->receive_nnode_list.data(), this->comm);
    if (!this->need_comm) {
        for (std::vector<Vertices_3D>::iterator it = this->receive_super_stencil_vertices_list.begin();
             it != this->receive_super_stencil_vertices_list.end(); ++it) {
            assert(*it == this->domain_vertices.get_3D_local_vertices());
        }
        std::vector<Vertices_3D>().swap(this->receive_super_stencil_vertices_list);
        std::vector<int>().swap(this->receive_nnode_list);
        return;
    }
    this->receive_offsets.resize(comm_size + 1, 0);
    this->receive_offsets[0] = 0;
    for (uint i = 0; i < (uint)comm_size; i++) {
        this->receive_offsets[i + 1] = this->receive_nnode_list[i] + this->receive_offsets[i];
    }

    //send mpi data
    this->send_overlap_vertices_list.resize(comm_size, Vertices_3D(0, 0, 0));
    this->send_nnode_list.resize(comm_size, 0);
    this->send_offsets.resize(comm_size + 1, 0);
    this->domain_vertices.ex_arr_send_data_transfer_prepare_full(FDn, domain_counts.data(),
                                                                 this->send_overlap_vertices_list.data(),
                                                                 this->send_nnode_list.data());
    this->send_offsets[0] = 0;
    for (uint i = 0; i < (uint)comm_size; i++) {
        this->send_offsets[i + 1] = this->send_nnode_list[i] + this->send_offsets[i];
    }
    return;
}

void Exarr_3D_mpi_package::init(const Exarr_3D_mpi_package& exarr_3D_mpi_package) {
    this->comm = this->domain_vertices.get_domain_3d_comm();
    assert(Parallel_vertices::is_comm_equal(this->comm, exarr_3D_mpi_package.comm));
    this->domain_counts_sum = exarr_3D_mpi_package.domain_counts_sum;
    this->receive_stencil_vertices_list = exarr_3D_mpi_package.receive_stencil_vertices_list;
    this->receive_stencil_vertices_offset = exarr_3D_mpi_package.receive_stencil_vertices_offset;
    this->need_comm = exarr_3D_mpi_package.need_comm;
    if (this->need_comm) {
        this->receive_super_stencil_vertices_list = exarr_3D_mpi_package.receive_super_stencil_vertices_list;
        this->receive_nnode_list = exarr_3D_mpi_package.receive_nnode_list;
        this->receive_offsets = exarr_3D_mpi_package.receive_offsets;
        this->send_overlap_vertices_list = exarr_3D_mpi_package.send_overlap_vertices_list;
        this->send_nnode_list = exarr_3D_mpi_package.send_nnode_list;
        this->send_offsets = exarr_3D_mpi_package.send_offsets;
    }
    return;
}

void Exarr_3D_mpi_package::destructor() {
    std::vector<Vertices_3D>().swap(this->send_overlap_vertices_list);
    std::vector<int>().swap(this->send_nnode_list);
    std::vector<int>().swap(this->send_offsets);
    std::vector<Vertices_3D>().swap(this->receive_stencil_vertices_list);
    std::vector<Vertices_3D>().swap(this->receive_super_stencil_vertices_list);
    std::vector<int>().swap(this->receive_stencil_vertices_offset);
    std::vector<int>().swap(this->receive_nnode_list);
    std::vector<int>().swap(this->receive_offsets);
    return;
}

void Exarr_3D_mpi_package::show() const {
    std::cout << "domain_counts_sum = " << domain_counts_sum << std::endl;
    std::cout << "need_comm = " << this->need_comm << std::endl;
    std::cout << "send_overlap_vertices_list.size() = " << this->send_overlap_vertices_list.size() <<std::endl;
    for (std::vector<Vertices_3D>::const_iterator it = this->send_overlap_vertices_list.begin();
            it != this->send_overlap_vertices_list.end(); it++) {
        it->show();
    }
    std::cout << "send_nnode_list.size() = " << this->send_nnode_list.size() << std::endl;
    for (std::vector<int>::const_iterator it = this->send_nnode_list.begin();
            it != this->send_nnode_list.end(); it++) {
        std::cout << "send node = " << *it << std::endl;
    }
    std::cout << "send_offsets.size() = " << this->send_offsets.size() << std::endl;
    for (std::vector<int>::const_iterator it = this->send_offsets.begin();
            it != this->send_offsets.end(); it++) {
        std::cout << "send_offsets = " << *it << std::endl;
    }
    std::cout << std::endl;
    std::cout << "receive_stencil_vertices_list.size() = " << this->receive_stencil_vertices_list.size() << std::endl;
    for (std::vector<Vertices_3D>::const_iterator it = this->receive_stencil_vertices_list.begin();
            it != this->receive_stencil_vertices_list.end(); it++) {
        it->show();
    }
    std::cout << "receive_super_stencil_vertices_list.size() = " << this->receive_super_stencil_vertices_list.size() << std::endl;
    for (std::vector<Vertices_3D>::const_iterator it = this->receive_super_stencil_vertices_list.begin();
            it != this->receive_super_stencil_vertices_list.end(); it++) {
        it->show();
    }
    std::cout << "receive_stencil_vertices_offset.size() = " << this->receive_stencil_vertices_offset.size() << std::endl;
    for (std::vector<int>::const_iterator it = this->receive_stencil_vertices_offset.begin();
            it != this->receive_stencil_vertices_offset.end(); it++) {
        std::cout << "receive_stencil_vertices_offset = " << *it << std::endl;
    }
    std::cout << "receive_nnode_list.size() = " << this->receive_nnode_list.size() << std::endl;
    for (std::vector<int>::const_iterator it = this->receive_nnode_list.begin();
            it != this->receive_nnode_list.end(); it++) {
        std::cout << "receive node = " << *it << std::endl;
    }
    std::cout << "receive_offsets.size() = " << this->receive_offsets.size() <<  std::endl;
    for (std::vector<int>::const_iterator it = this->receive_offsets.begin();
            it != this->receive_offsets.end(); it++) {
        std::cout << "receive_offsets = " << *it << std::endl;
    }
    return;
}

Exarr_4D_mpi_package::Exarr_4D_mpi_package(const Domain_parallel_vertices_4D& domain_vertices)
                                         : domain_vertices(domain_vertices) {}

Exarr_4D_mpi_package::~Exarr_4D_mpi_package() {}

template<typename T>
void Exarr_4D_mpi_package::fill_domain_par_ex_arr(const Array_4D<T>& send_arr, Array_4D<T>& recv_arr) const {
    int comm_size;
    MPI_Comm_size(this->comm, &comm_size);
    if (this->need_comm) {
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();

        Array_0D<T> send_array(this->send_offsets[comm_size]);
        for (uint i = 0; i < (uint)comm_size; i++) {
            send_arr.fill_vector(send_array, this->send_overlap_vertices_list[i], this->send_offsets[i]);
        }
        Array_0D<T> receive_array(this->receive_offsets[comm_size]);
        MPI_Request request;
        MPI_Ialltoallv(send_array.data, this->send_nnode_list.data(), this->send_offsets.data(), mpi_datatype,
                    receive_array.data, this->receive_nnode_list.data(), this->receive_offsets.data(),
                    mpi_datatype, this->comm, &request);
        send_arr.fill_overlap(recv_arr);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        for (uint i = 0; i < this->domain_counts_sum; i++) {
            recv_arr.be_filled_vector(receive_array, this->receive_super_stencil_vertices_list[i],
                                    this->receive_stencil_vertices_list[i], this->receive_stencil_vertices_offset[i]);
        }
    } else {
        send_arr.fill_overlap(recv_arr);
        for (uint i = 0; i < this->domain_counts_sum; i++) {
            recv_arr.be_filled_vector(send_arr, send_arr.get_vertices(),
                                    this->receive_stencil_vertices_list[i], this->receive_stencil_vertices_offset[i]);
        }
    }
    return;
}
template void Exarr_4D_mpi_package::fill_domain_par_ex_arr<int>(const Array_4D<int>& send_arr, Array_4D<int>& recv_arr) const;
template void Exarr_4D_mpi_package::fill_domain_par_ex_arr<float>(const Array_4D<float>& send_arr, Array_4D<float>& recv_arr) const;
template void Exarr_4D_mpi_package::fill_domain_par_ex_arr<double>(const Array_4D<double>& send_arr, Array_4D<double>& recv_arr) const;

template<typename T>
void Exarr_4D_mpi_package::fill_domain_par_ex_arr(T const* const& __restrict__ send_arr, Vertices_4D send_arr_vertices,
                                                  T* const& __restrict__ recv_arr, Vertices_4D recv_arr_vertices) const {
    int comm_size;
    MPI_Comm_size(this->comm, &comm_size);
    if (this->need_comm) {
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        #ifdef USE_OPENMP //without anycheck be carefull
        static T* send_array_static = nullptr;
        #pragma omp single
        send_array_static = new T [this->send_offsets[comm_size]];
        T* const send_array = send_array_static;
        for (uint i = 0; i < (uint)comm_size; i++) {
            Vertices_method::fill_vector(send_arr, send_arr_vertices, send_array,
                                         this->send_overlap_vertices_list[i], this->send_offsets[i]);
        }
        static T* receive_array_static = nullptr;
        #pragma omp single
        receive_array_static = new T [this->receive_offsets[comm_size]];
        T* const receive_array = receive_array_static;
        MPI_Request request;
        #pragma omp master
        {
        MPI_Ialltoallv(send_array, this->send_nnode_list.data(), this->send_offsets.data(), mpi_datatype,
                       receive_array, this->receive_nnode_list.data(), this->receive_offsets.data(),
                       mpi_datatype, this->comm, &request);
        }
        Vertices_method::fill_region(send_arr, send_arr_vertices, recv_arr, recv_arr_vertices,
                                     send_arr_vertices.get_overlap_vertices(recv_arr_vertices));
        #pragma omp master
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        #pragma omp barrier
        for (uint i = 0; i < this->domain_counts_sum; i++) {
            Vertices_method::be_filled_vector(recv_arr, recv_arr_vertices, receive_array, this->receive_super_stencil_vertices_list[i],
                                              this->receive_stencil_vertices_list[i], this->receive_stencil_vertices_offset[i]);
        }

        #pragma omp barrier
        #pragma omp single nowait
        {
            delete [] send_array_static;
            send_array_static = nullptr;
        }
        #pragma omp single nowait
        {
            delete [] receive_array_static;
            receive_array_static = nullptr;
        }

        #else //USE_OPENMP

        // Array_0D<T> send_array(this->send_offsets[comm_size]);
        T* send_array = new T [this->send_offsets[comm_size]];
        for (uint i = 0; i < (uint)comm_size; i++) {
            // send_arr.fill_vector(send_array, this->send_overlap_vertices_list[i], this->send_offsets[i]);
            Vertices_method::fill_vector(send_arr, send_arr_vertices, send_array,
                                         this->send_overlap_vertices_list[i], this->send_offsets[i]);
        }
        // Array_0D<T> receive_array(this->receive_offsets[comm_size]);
        T* receive_array = new T [this->receive_offsets[comm_size]];
        MPI_Request request;
        MPI_Ialltoallv(send_array, this->send_nnode_list.data(), this->send_offsets.data(), mpi_datatype,
                       receive_array, this->receive_nnode_list.data(), this->receive_offsets.data(),
                       mpi_datatype, this->comm, &request);
        // send_arr.fill_overlap(recv_arr);
        Vertices_method::fill_region(send_arr, send_arr_vertices, recv_arr, recv_arr_vertices,
                                     send_arr_vertices.get_overlap_vertices(recv_arr_vertices));
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        for (uint i = 0; i < this->domain_counts_sum; i++) {
            // recv_arr.be_filled_vector(receive_array, this->receive_super_stencil_vertices_list[i],
            //                         this->receive_stencil_vertices_list[i], this->receive_stencil_vertices_offset[i]);
            Vertices_method::be_filled_vector(recv_arr, recv_arr_vertices, receive_array, this->receive_super_stencil_vertices_list[i],
                                              this->receive_stencil_vertices_list[i], this->receive_stencil_vertices_offset[i]);
        }
        
        delete [] send_array;
        delete [] receive_array;

        #endif //USE_OPENMP
    } else {
        // send_arr.fill_overlap(recv_arr);
        Vertices_method::fill_region(send_arr, send_arr_vertices, recv_arr, recv_arr_vertices,
                                     send_arr_vertices.get_overlap_vertices(recv_arr_vertices));
        for (uint i = 0; i < this->domain_counts_sum; i++) {
            // recv_arr.be_filled_vector(send_arr, send_arr.get_vertices(),
            //                         this->receive_stencil_vertices_list[i], this->receive_stencil_vertices_offset[i]);
            Vertices_method::be_filled_vector(recv_arr, recv_arr_vertices, send_arr, send_arr_vertices,
                                              this->receive_stencil_vertices_list[i], this->receive_stencil_vertices_offset[i]);
        }
    }
    return;
}
template void Exarr_4D_mpi_package::fill_domain_par_ex_arr<int>(int const* const& send_arr, Vertices_4D send_arr_vertices,
                                                                int* const& recv_arr, Vertices_4D recv_arr_vertices) const;
template void Exarr_4D_mpi_package::fill_domain_par_ex_arr<float>(float const* const& send_arr, Vertices_4D send_arr_vertices,
                                                                  float* const& recv_arr, Vertices_4D recv_arr_vertices) const;
template void Exarr_4D_mpi_package::fill_domain_par_ex_arr<double>(double const* const& send_arr, Vertices_4D send_arr_vertices,
                                                                   double* const& recv_arr, Vertices_4D recv_arr_vertices) const;

void Exarr_4D_mpi_package::init(const bool* is_periodic, const int* FDn) {
    this->comm = this->domain_vertices.get_domain_4d_comm();

    std::array<uint, 8> domain_counts;
    int comm_size = this->domain_vertices.get_domain_4d_comm_size();

    // find how many domains is needed
    this->domain_counts_sum = this->domain_vertices.generate_domain_counts(domain_counts.data(), is_periodic, FDn);

    //receive mpi data
    this->receive_stencil_vertices_list.resize(this->domain_counts_sum, Vertices_4D(0, 0, 0, 0));
    this->receive_super_stencil_vertices_list.resize(this->domain_counts_sum, Vertices_4D(0, 0, 0, 0));
    this->receive_stencil_vertices_offset.resize(this->domain_counts_sum, 0);
    this->receive_nnode_list.resize(comm_size, 0);
    this->domain_vertices.ex_arr_receive_data_transfer_prepare(FDn, domain_counts.data(),
                                                         this->receive_stencil_vertices_list.data(),
                                                         this->receive_super_stencil_vertices_list.data(),
                                                         this->receive_stencil_vertices_offset.data(),
                                                         this->receive_nnode_list.data());
    
    this->need_comm = Parallel_vertices::need_comm(this->receive_nnode_list.data(), this->comm);
    if (!this->need_comm) {
        for (std::vector<Vertices_4D>::iterator it = this->receive_super_stencil_vertices_list.begin();
             it != this->receive_super_stencil_vertices_list.end(); ++it) {
            assert(*it == this->domain_vertices.get_4D_local_vertices());
        }
        std::vector<Vertices_4D>().swap(this->receive_super_stencil_vertices_list);
        std::vector<int>().swap(this->receive_nnode_list);
        return;
    }
    this->receive_offsets.resize(comm_size + 1, 0);
    this->receive_offsets[0] = 0;
    for (uint i = 0; i < (uint)comm_size; i++) {
        this->receive_offsets[i + 1] = this->receive_nnode_list[i] + this->receive_offsets[i];
    }

    //send mpi data
    this->send_overlap_vertices_list.resize(comm_size, Vertices_4D(0, 0, 0, 0));
    this->send_nnode_list.resize(comm_size, 0);
    this->send_offsets.resize(comm_size + 1, 0);
    this->domain_vertices.ex_arr_send_data_transfer_prepare(FDn, domain_counts.data(),
                                                      this->send_overlap_vertices_list.data(),
                                                      this->send_nnode_list.data());
    this->send_offsets[0] = 0;
    for (uint i = 0; i < (uint)comm_size; i++) {
        this->send_offsets[i + 1] = this->send_nnode_list[i] + this->send_offsets[i];
    }
    return;
}

void Exarr_4D_mpi_package::init(const Exarr_4D_mpi_package& exarr_4D_mpi_package) {
    this->comm = this->domain_vertices.get_domain_4d_comm();
    assert(Parallel_vertices::is_comm_equal(this->comm, exarr_4D_mpi_package.comm));
    this->domain_counts_sum = exarr_4D_mpi_package.domain_counts_sum;
    this->receive_stencil_vertices_list = exarr_4D_mpi_package.receive_stencil_vertices_list;
    this->receive_stencil_vertices_offset = exarr_4D_mpi_package.receive_stencil_vertices_offset;
    this->need_comm = exarr_4D_mpi_package.need_comm;
    if (this->need_comm) {
        this->receive_super_stencil_vertices_list = exarr_4D_mpi_package.receive_super_stencil_vertices_list;
        this->receive_nnode_list = exarr_4D_mpi_package.receive_nnode_list;
        this->receive_offsets = exarr_4D_mpi_package.receive_offsets;
        this->send_overlap_vertices_list = exarr_4D_mpi_package.send_overlap_vertices_list;
        this->send_nnode_list = exarr_4D_mpi_package.send_nnode_list;
        this->send_offsets = exarr_4D_mpi_package.send_offsets;
    }
    return;
}

void Exarr_4D_mpi_package::destructor() {
    std::vector<Vertices_4D>().swap(this->send_overlap_vertices_list);
    std::vector<int>().swap(this->send_nnode_list);
    std::vector<int>().swap(this->send_offsets);
    std::vector<Vertices_4D>().swap(this->receive_stencil_vertices_list);
    std::vector<Vertices_4D>().swap(this->receive_super_stencil_vertices_list);
    std::vector<int>().swap(this->receive_stencil_vertices_offset);
    std::vector<int>().swap(this->receive_nnode_list);
    std::vector<int>().swap(this->receive_offsets);
    return;
}

void Exarr_4D_mpi_package::show() const {
    std::cout << "domain_counts_sum = " << domain_counts_sum << std::endl;
    std::cout << "need_comm = " << this->need_comm << std::endl;
    std::cout << "send_overlap_vertices_list.size() = " << this->send_overlap_vertices_list.size() <<std::endl;
    for (std::vector<Vertices_4D>::const_iterator it = this->send_overlap_vertices_list.begin();
            it != this->send_overlap_vertices_list.end(); it++) {
        it->show();
    }
    std::cout << "send_nnode_list.size() = " << this->send_nnode_list.size() << std::endl;
    for (std::vector<int>::const_iterator it = this->send_nnode_list.begin();
            it != this->send_nnode_list.end(); it++) {
        std::cout << "send node = " << *it << std::endl;
    }
    std::cout << "send_offsets.size() = " << this->send_offsets.size() << std::endl;
    for (std::vector<int>::const_iterator it = this->send_offsets.begin();
            it != this->send_offsets.end(); it++) {
        std::cout << "send_offsets = " << *it << std::endl;
    }
    std::cout << std::endl;
    std::cout << "receive_stencil_vertices_list.size() = " << this->receive_stencil_vertices_list.size() << std::endl;
    for (std::vector<Vertices_4D>::const_iterator it = this->receive_stencil_vertices_list.begin();
            it != this->receive_stencil_vertices_list.end(); it++) {
        it->show();
    }
    std::cout << "receive_super_stencil_vertices_list.size() = " << this->receive_super_stencil_vertices_list.size() << std::endl;
    for (std::vector<Vertices_4D>::const_iterator it = this->receive_super_stencil_vertices_list.begin();
            it != this->receive_super_stencil_vertices_list.end(); it++) {
        it->show();
    }
    std::cout << "receive_stencil_vertices_offset.size() = " << this->receive_stencil_vertices_offset.size() << std::endl;
    for (std::vector<int>::const_iterator it = this->receive_stencil_vertices_offset.begin();
            it != this->receive_stencil_vertices_offset.end(); it++) {
        std::cout << "receive_stencil_vertices_offset = " << *it << std::endl;
    }
    std::cout << "receive_nnode_list.size() = " << this->receive_nnode_list.size() << std::endl;
    for (std::vector<int>::const_iterator it = this->receive_nnode_list.begin();
            it != this->receive_nnode_list.end(); it++) {
        std::cout << "receive node = " << *it << std::endl;
    }
    std::cout << "receive_offsets.size() = " << this->receive_offsets.size() <<  std::endl;
    for (std::vector<int>::const_iterator it = this->receive_offsets.begin();
            it != this->receive_offsets.end(); it++) {
        std::cout << "receive_offsets = " << *it << std::endl;
    }
    return;
}

Domain_3D_to_3D_mpi_package::Domain_3D_to_3D_mpi_package(const Domain_parallel_vertices_3D& my_domain_vertices,
                                                         const Domain_parallel_vertices_3D& other_domain_vertices)
                                                       : my_domain_vertices(my_domain_vertices),
                                                         other_domain_vertices(other_domain_vertices) {}

Domain_3D_to_3D_mpi_package::~Domain_3D_to_3D_mpi_package() {}

template<typename T>
void Domain_3D_to_3D_mpi_package::send_data(const Array_3D<T>& send_arr, Array_3D<T>& recv_arr) const {
    this->send_data(send_arr.data, send_arr.get_vertices(), recv_arr.data, recv_arr.get_vertices());
    // if (this->need_comm) {
    //     int comm_size;
    //     MPI_Comm_size(this->comm, &comm_size);
    //     MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
    //     Array_0D<T> send_array(this->send_offsets[comm_size]);
    //     for (uint i = 0; i < (uint)comm_size; i++) {
    //         send_arr.fill_vector(send_array, this->send_overlap_vertices_list[i], this->send_offsets[i]);
    //     }
    //     Array_0D<T> receive_array(this->receive_offsets[comm_size]);
    //     MPI_Alltoallv(send_array.data, this->send_nnode_list.data(), this->send_offsets.data(), mpi_datatype,
    //                 receive_array.data, this->receive_nnode_list.data(), this->receive_offsets.data(), mpi_datatype,
    //                 this->comm);
    //     for (uint i = 0; i < (uint)comm_size; i++) {
    //         recv_arr.be_filled_vector(receive_array, this->receive_overlap_vertices_list[i], this->receive_offsets[i]);
    //     }
    // } else {
    //     send_arr.fill_overlap(recv_arr);
    // }
    return;
}
template void Domain_3D_to_3D_mpi_package::send_data<int>(const Array_3D<int>& send_arr, Array_3D<int>& recv_arr) const;
template void Domain_3D_to_3D_mpi_package::send_data<float>(const Array_3D<float>& send_arr, Array_3D<float>& recv_arr) const;
template void Domain_3D_to_3D_mpi_package::send_data<double>(const Array_3D<double>& send_arr, Array_3D<double>& recv_arr) const;

template<typename T>
void Domain_3D_to_3D_mpi_package::send_data(T const* const send_arr, const Vertices_3D& send_arr_vertices,
                                        T* const recv_arr, const Vertices_3D& recv_arr_vertices) const {
    if (this->need_comm) {
        int comm_size;
        MPI_Comm_size(this->comm, &comm_size);
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        // Array_0D<T> send_array(this->send_offsets[comm_size]);
        T* send_array = new (std::align_val_t(64)) T [this->send_offsets[comm_size]];
        for (uint i = 0; i < (uint)comm_size; i++) {
            // send_arr.fill_vector(send_array, this->send_overlap_vertices_list[i], this->send_offsets[i]);
            Vertices_method::fill_vector(send_arr, send_arr_vertices, send_array,
                this->send_overlap_vertices_list[i], this->send_offsets[i]);
        }
        // Array_0D<T> receive_array(this->receive_offsets[comm_size]);
        T* receive_array = new (std::align_val_t(64)) T [this->receive_offsets[comm_size]];
        MPI_Alltoallv(send_array, this->send_nnode_list.data(), this->send_offsets.data(), mpi_datatype,
                    receive_array, this->receive_nnode_list.data(), this->receive_offsets.data(), mpi_datatype,
                    this->comm);
        for (uint i = 0; i < (uint)comm_size; i++) {
            // recv_arr.be_filled_vector(receive_array, this->receive_overlap_vertices_list[i], this->receive_offsets[i]);
            Vertices_method::be_filled_vector(recv_arr, recv_arr_vertices, receive_array,
                this->receive_overlap_vertices_list[i], this->receive_overlap_vertices_list[i],
                this->receive_offsets[i]);
        }
        ::operator delete[](send_array, std::align_val_t(64));
        ::operator delete[](receive_array, std::align_val_t(64));
    } else {
        // send_arr.fill_overlap(recv_arr);
        const Vertices_3D temp = send_arr_vertices.get_overlap_vertices(recv_arr_vertices);
        Vertices_method::fill_region(send_arr, send_arr_vertices,
                                 recv_arr, recv_arr_vertices, temp);
    }
    return;
}
template void Domain_3D_to_3D_mpi_package::send_data<int>(int const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                            int* const recv_arr, const Vertices_3D& recv_arr_vertices) const;
template void Domain_3D_to_3D_mpi_package::send_data<float>(float const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                            float* const recv_arr, const Vertices_3D& recv_arr_vertices) const;
template void Domain_3D_to_3D_mpi_package::send_data<double>(double const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                            double* const recv_arr, const Vertices_3D& recv_arr_vertices) const;


template<typename T>
void Domain_3D_to_3D_mpi_package::send_data_mp(T const* const send_arr, const Vertices_3D& send_arr_vertices,
                                        T* const recv_arr, const Vertices_3D& recv_arr_vertices,
                                        Memory_pool<T, Fast_memory>& pool_fast,
                                        Memory_pool<T, Capacity_memory>& pool_cap) const {
    if (this->need_comm) {
        Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
        Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
        int comm_size;
        MPI_Comm_size(this->comm, &comm_size);
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        // Array_0D<T> send_array(this->send_offsets[comm_size]);
        // T* send_array = new (std::align_val_t(64)) T [this->send_offsets[comm_size]];
        T* send_array = pool_cap.allocate(this->send_offsets[comm_size]);
        for (uint i = 0; i < (uint)comm_size; i++) {
            // send_arr.fill_vector(send_array, this->send_overlap_vertices_list[i], this->send_offsets[i]);
            Vertices_method::fill_vector(send_arr, send_arr_vertices, send_array,
                this->send_overlap_vertices_list[i], this->send_offsets[i]);
        }
        // Array_0D<T> receive_array(this->receive_offsets[comm_size]);
        // T* receive_array = new (std::align_val_t(64)) T [this->receive_offsets[comm_size]];
        T* receive_array = pool_cap.allocate(this->receive_offsets[comm_size]);
        MPI_Alltoallv(send_array, this->send_nnode_list.data(), this->send_offsets.data(), mpi_datatype,
                    receive_array, this->receive_nnode_list.data(), this->receive_offsets.data(), mpi_datatype,
                    this->comm);
        for (uint i = 0; i < (uint)comm_size; i++) {
            // recv_arr.be_filled_vector(receive_array, this->receive_overlap_vertices_list[i], this->receive_offsets[i]);
            Vertices_method::be_filled_vector(recv_arr, recv_arr_vertices, receive_array,
                this->receive_overlap_vertices_list[i], this->receive_overlap_vertices_list[i],
                this->receive_offsets[i]);
        }
        // ::operator delete[](send_array, std::align_val_t(64));
        // ::operator delete[](receive_array, std::align_val_t(64));
    } else {
        // send_arr.fill_overlap(recv_arr);
        const Vertices_3D temp = send_arr_vertices.get_overlap_vertices(recv_arr_vertices);
        Vertices_method::fill_region(send_arr, send_arr_vertices,
                                 recv_arr, recv_arr_vertices, temp);
    }
    return;
}
template void Domain_3D_to_3D_mpi_package::send_data_mp<int>(int const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                            int* const recv_arr, const Vertices_3D& recv_arr_vertices,
                                                            Memory_pool<int, Fast_memory>& pool_fast,
                                                            Memory_pool<int, Capacity_memory>& pool_cap) const;
template void Domain_3D_to_3D_mpi_package::send_data_mp<float>(float const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                            float* const recv_arr, const Vertices_3D& recv_arr_vertices,
                                                            Memory_pool<float, Fast_memory>& pool_fast,
                                                            Memory_pool<float, Capacity_memory>& pool_cap) const;
template void Domain_3D_to_3D_mpi_package::send_data_mp<double>(double const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                            double* const recv_arr, const Vertices_3D& recv_arr_vertices,
                                                            Memory_pool<double, Fast_memory>& pool_fast,
                                                            Memory_pool<double, Capacity_memory>& pool_cap) const;

template<typename T>
void Domain_3D_to_3D_mpi_package::recv_data(Array_3D<T>& recv_arr, const Array_3D<T>& send_arr) const {
    if (this->need_comm) {
        int comm_size;
        MPI_Comm_size(this->comm, &comm_size);
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        Array_0D<T> send_array(this->receive_offsets[comm_size]);
        for (uint i = 0; i < (uint)comm_size; i++) {
            send_arr.fill_vector(send_array, this->receive_overlap_vertices_list[i], this->receive_offsets[i]);
        }
        Array_0D<T> receive_array(this->send_offsets[comm_size]);
        MPI_Alltoallv(send_array.data, this->receive_nnode_list.data(), this->receive_offsets.data(), mpi_datatype,
                    receive_array.data, this->send_nnode_list.data(), this->send_offsets.data(), mpi_datatype,
                    this->comm);
        for (uint i = 0; i < (uint)comm_size; i++) {
            recv_arr.be_filled_vector(receive_array, this->send_overlap_vertices_list[i], this->send_offsets[i]);
        }
    } else {
        send_arr.fill_overlap(recv_arr);
    }
    return;
}
template void Domain_3D_to_3D_mpi_package::recv_data<int>(Array_3D<int>& recv_arr, const Array_3D<int>& send_arr) const;
template void Domain_3D_to_3D_mpi_package::recv_data<float>(Array_3D<float>& recv_arr, const Array_3D<float>& send_arr) const;
template void Domain_3D_to_3D_mpi_package::recv_data<double>(Array_3D<double>& recv_arr, const Array_3D<double>& send_arr) const;

void Domain_3D_to_3D_mpi_package::init() {
    // assert(my_domain_vertices.get_3D_shared_vertices() == other_domain_vertices.get_3D_shared_vertices()
    //        && my_domain_vertices.comm == other_domain_vertices.comm);
    assert(this->my_domain_vertices.get_3D_shared_vertices() == this->other_domain_vertices.get_3D_shared_vertices()
        && Parallel_vertices::is_comm_equal(this->my_domain_vertices.comm, this->other_domain_vertices.comm));
    this->comm = this->my_domain_vertices.comm;
    int comm_size = this->my_domain_vertices.get_comm_size();

    //receive mpi data
    this->receive_overlap_vertices_list.resize(comm_size, Vertices_3D(0, 0, 0));
    this->receive_nnode_list.resize(comm_size, 0);
    if (this->other_domain_vertices.get_active_comm_b() == 0 && this->other_domain_vertices.is_active) {
        this->other_domain_vertices.data_transfer_prepare(this->my_domain_vertices,
                                                    this->receive_overlap_vertices_list.data(),
                                                    this->receive_nnode_list.data());
    }
    this->need_comm = Parallel_vertices::need_comm(this->receive_nnode_list.data(), this->comm);
    if (!this->need_comm) {
        std::vector<Vertices_3D>().swap(this->receive_overlap_vertices_list);
        std::vector<int>().swap(this->receive_nnode_list);
        return;
    }
    this->receive_offsets.resize(comm_size + 1, 0);
    this->receive_offsets[0] = 0;
    for (uint i = 0; i < (uint)comm_size; i++) {
        this->receive_offsets[i + 1] =  this->receive_nnode_list[i] + this->receive_offsets[i];
    }

    //send mpi data
    this->send_overlap_vertices_list.resize(comm_size, Vertices_3D(0, 0, 0));
    this->send_nnode_list.resize(comm_size, 0);
    this->send_offsets.resize(comm_size + 1, 0);
    if (this->my_domain_vertices.get_active_comm_b() == 0 && this->my_domain_vertices.is_active) {
        this->my_domain_vertices.data_transfer_prepare(this->other_domain_vertices,
                                                       this->send_overlap_vertices_list.data(),
                                                       this->send_nnode_list.data());
    }
    this->send_offsets[0] = 0;
    for (uint i = 0; i < (uint)comm_size; i++) {
        this->send_offsets[i + 1] =  this->send_nnode_list[i] + this->send_offsets[i];
    }
    return;
}

void Domain_3D_to_3D_mpi_package::init(const Domain_3D_to_3D_mpi_package& domain_3D_to_3D_mpi_package) {
    this->comm = this->my_domain_vertices.comm;
    this->need_comm = domain_3D_to_3D_mpi_package.need_comm;
    if (this->need_comm) {
        this->receive_overlap_vertices_list = domain_3D_to_3D_mpi_package.receive_overlap_vertices_list;
        this->receive_nnode_list = domain_3D_to_3D_mpi_package.receive_nnode_list;
        this->receive_offsets = domain_3D_to_3D_mpi_package.receive_offsets;
        this->send_overlap_vertices_list = domain_3D_to_3D_mpi_package.send_overlap_vertices_list;
        this->send_nnode_list = domain_3D_to_3D_mpi_package.send_nnode_list;
        this->send_offsets = domain_3D_to_3D_mpi_package.send_offsets;
    }
    return;
}

void Domain_3D_to_3D_mpi_package::destructor() {
    std::vector<Vertices_3D>().swap(this->send_overlap_vertices_list);
    std::vector<int>().swap(this->send_nnode_list);
    std::vector<int>().swap(this->send_offsets);
    std::vector<Vertices_3D>().swap(this->receive_overlap_vertices_list);
    std::vector<int>().swap(this->receive_nnode_list);
    std::vector<int>().swap(this->receive_offsets);
    return;
}

void Domain_3D_to_3D_mpi_package::show() const {
    std::cout << "need_comm = " << this->need_comm << std::endl;
    std::cout << "send_overlap_vertices_list" << std::endl;
    for (std::vector<Vertices_3D>::const_iterator it = this->send_overlap_vertices_list.begin();
            it != this->send_overlap_vertices_list.end(); it++) {
        it->show();
    }
    std::cout << "send_nnode_list" << std::endl;
    for (std::vector<int>::const_iterator it = this->send_nnode_list.begin();
            it != this->send_nnode_list.end(); it++) {
        std::cout << "send node = " << *it << std::endl;
    }
    std::cout << "send_offsets" << std::endl;
    for (std::vector<int>::const_iterator it = this->send_offsets.begin();
            it != this->send_offsets.end(); it++) {
        std::cout << "send_offsets = " << *it << std::endl;
    }
    std::cout << std::endl;
    std::cout << "receive_overlap_vertices_list" << std::endl;
    for (std::vector<Vertices_3D>::const_iterator it = this->receive_overlap_vertices_list.begin();
            it != this->receive_overlap_vertices_list.end(); it++) {
        it->show();
    }
    std::cout << "receive_nnode_list" << std::endl;
    for (std::vector<int>::const_iterator it = this->receive_nnode_list.begin();
            it != this->receive_nnode_list.end(); it++) {
        std::cout << "receive node = " << *it << std::endl;
    }
    std::cout << "receive_offsets" << std::endl;
    for (std::vector<int>::const_iterator it = this->receive_offsets.begin();
            it != this->receive_offsets.end(); it++) {
        std::cout << "receive_offsets = " << *it << std::endl;
    }
    return;
}

Domain_3D_to_4D_mpi_package::Domain_3D_to_4D_mpi_package(const Domain_parallel_vertices_3D& my_domain_vertices,
                                                         const Domain_parallel_vertices_4D& other_domain_vertices)
                                                       : Domain_3D_to_3D_mpi_package(my_domain_vertices, other_domain_vertices) {}

Domain_3D_to_4D_mpi_package::~Domain_3D_to_4D_mpi_package() {}

template<typename T>
void Domain_3D_to_4D_mpi_package::send_data(const Array_3D<T>& send_arr, Array_3D<T>& recv_arr) const {
    this->send_data(send_arr.data, send_arr.get_vertices(), recv_arr.data, recv_arr.get_vertices());
    // this->Domain_3D_to_3D_mpi_package::send_data(send_arr, recv_arr);
    // int band_comm_size;
    // MPI_Comm_size(this->band_comm, &band_comm_size);
    // if (this->need_band_comm) {
    //     MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
    //     MPI_Bcast(recv_arr.data, (int)recv_arr.length, mpi_datatype, 0, this->band_comm);
    // }
    return;
}
template void Domain_3D_to_4D_mpi_package::send_data<int>(const Array_3D<int>& send_arr, Array_3D<int>& recv_arr) const;
template void Domain_3D_to_4D_mpi_package::send_data<float>(const Array_3D<float>& send_arr, Array_3D<float>& recv_arr) const;
template void Domain_3D_to_4D_mpi_package::send_data<double>(const Array_3D<double>& send_arr, Array_3D<double>& recv_arr) const;

template<typename T>
void Domain_3D_to_4D_mpi_package::send_data(T const* const send_arr, const Vertices_3D& send_arr_vertices,
                                            T* const recv_arr, const Vertices_3D& recv_arr_vertices) const {
    // this->Domain_3D_to_3D_mpi_package::send_data(send_arr, recv_arr);
    this->Domain_3D_to_3D_mpi_package::send_data(send_arr, send_arr_vertices, recv_arr, recv_arr_vertices);
    if (this->need_band_comm) {
        MPI_Bcast(recv_arr, int(recv_arr_vertices.get_size()), Linalg::get_mpi_datatype<T>(), 0, this->band_comm);
    }
    return;
}
template void Domain_3D_to_4D_mpi_package::send_data<int>(int const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                          int* const recv_arr, const Vertices_3D& recv_arr_vertices) const;
template void Domain_3D_to_4D_mpi_package::send_data<float>(float const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                            float* const recv_arr, const Vertices_3D& recv_arr_vertices) const;
template void Domain_3D_to_4D_mpi_package::send_data<double>(double const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                            double* const recv_arr, const Vertices_3D& recv_arr_vertices) const;

template<typename T>
void Domain_3D_to_4D_mpi_package::send_data_mp(T const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                T* const recv_arr, const Vertices_3D& recv_arr_vertices,
                                                Memory_pool<T, Fast_memory>& pool_fast,
                                                Memory_pool<T, Capacity_memory>& pool_cap) const {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    // this->Domain_3D_to_3D_mpi_package::send_data(send_arr, recv_arr);
    this->Domain_3D_to_3D_mpi_package::send_data_mp(send_arr, send_arr_vertices, recv_arr, recv_arr_vertices,
                                                    pool_fast, pool_cap);
    if (this->need_band_comm) {
        MPI_Bcast(recv_arr, int(recv_arr_vertices.get_size()), Linalg::get_mpi_datatype<T>(), 0, this->band_comm);
    }
    return;
}
template void Domain_3D_to_4D_mpi_package::send_data_mp<int>(int const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                            int* const recv_arr, const Vertices_3D& recv_arr_vertices,
                                                            Memory_pool<int, Fast_memory>& pool_fast,
                                                            Memory_pool<int, Capacity_memory>& pool_cap) const;
template void Domain_3D_to_4D_mpi_package::send_data_mp<float>(float const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                            float* const recv_arr, const Vertices_3D& recv_arr_vertices,
                                                            Memory_pool<float, Fast_memory>& pool_fast,
                                                            Memory_pool<float, Capacity_memory>& pool_cap) const;
template void Domain_3D_to_4D_mpi_package::send_data_mp<double>(double const* const send_arr, const Vertices_3D& send_arr_vertices,
                                                                double* const recv_arr, const Vertices_3D& recv_arr_vertices,
                                                                Memory_pool<double, Fast_memory>& pool_fast,
                                                                Memory_pool<double, Capacity_memory>& pool_cap) const;

template<typename T>
void Domain_3D_to_4D_mpi_package::recv_data(Array_3D<T>& recv_arr, const Array_3D<T>& send_arr) const {
    this->Domain_3D_to_3D_mpi_package::recv_data(recv_arr, send_arr);
    return;
}
template void Domain_3D_to_4D_mpi_package::recv_data<int>(Array_3D<int>& recv_arr, const Array_3D<int>& send_arr) const;
template void Domain_3D_to_4D_mpi_package::recv_data<float>(Array_3D<float>& recv_arr, const Array_3D<float>& send_arr) const;
template void Domain_3D_to_4D_mpi_package::recv_data<double>(Array_3D<double>& recv_arr, const Array_3D<double>& send_arr) const;

void Domain_3D_to_4D_mpi_package::init() {
    this->band_comm = this->other_domain_vertices.get_band_comm();
    int band_comm_size;
    MPI_Comm_size(this->band_comm, &band_comm_size);
    if (band_comm_size > 1) this->need_band_comm = true;
    this->Domain_3D_to_3D_mpi_package::init();
    return;
}

void Domain_3D_to_4D_mpi_package::init(const Domain_3D_to_4D_mpi_package& domain_3D_to_4D_mpi_package) {
    this->band_comm = this->other_domain_vertices.get_band_comm();
    this->need_band_comm = domain_3D_to_4D_mpi_package.need_band_comm;
    this->Domain_3D_to_3D_mpi_package::init(domain_3D_to_4D_mpi_package);
    return;
}

void Domain_3D_to_4D_mpi_package::destructor() {
    this->Domain_3D_to_3D_mpi_package::destructor();
    return;
}

void Domain_3D_to_4D_mpi_package::show() const {
    std::cout << "need_band_comm = " << this->need_band_comm << std::endl;
    this->Domain_3D_to_3D_mpi_package::show();
    return;
}

Domain_4D_to_4D_mpi_package::Domain_4D_to_4D_mpi_package(const Domain_parallel_vertices_4D& my_domain_vertices,
                                                         const Domain_parallel_vertices_4D& other_domain_vertices)
                                                       : my_domain_vertices(my_domain_vertices),
                                                         other_domain_vertices(other_domain_vertices) {}

Domain_4D_to_4D_mpi_package::~Domain_4D_to_4D_mpi_package() {}

template<typename T>
void Domain_4D_to_4D_mpi_package::send_data(const Array_4D<T>& send_arr, Array_4D<T>& recv_arr) const {
    if (this->need_comm) {
        int comm_size;
        MPI_Comm_size(this->comm, &comm_size);
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        Array_0D<T> send_array(this->send_offsets[comm_size]);
        for (uint i = 0; i < (uint)comm_size; i++) {
            send_arr.fill_vector(send_array, this->send_overlap_vertices_list[i], this->send_offsets[i]);
        }
        Array_0D<T> receive_array(this->receive_offsets[comm_size]);
        MPI_Alltoallv(send_array.data, this->send_nnode_list.data(), this->send_offsets.data(), mpi_datatype,
                    receive_array.data, this->receive_nnode_list.data(), this->receive_offsets.data(), mpi_datatype,
                    this->comm);
        for (uint i = 0; i < (uint)comm_size; i++) {
            recv_arr.be_filled_vector(receive_array, this->receive_overlap_vertices_list[i], this->receive_offsets[i]);
        }
    } else {
        send_arr.fill_overlap(recv_arr);
    }
    return;
}
template void Domain_4D_to_4D_mpi_package::send_data<int>(const Array_4D<int>& send_arr, Array_4D<int>& recv_arr) const;
template void Domain_4D_to_4D_mpi_package::send_data<float>(const Array_4D<float>& send_arr, Array_4D<float>& recv_arr) const;
template void Domain_4D_to_4D_mpi_package::send_data<double>(const Array_4D<double>& send_arr, Array_4D<double>& recv_arr) const;

template<typename T> void Domain_4D_to_4D_mpi_package::send_data(T const* const& send_arr, const Vertices_4D& send_arr_vertices,
                                                                 T* const& recv_arr, const Vertices_4D& recv_arr_vertices) const {
    if (this->need_comm) {
        int comm_size;
        MPI_Comm_size(this->comm, &comm_size);
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        #ifdef USE_OPENMP

        static T* send_array_static = nullptr;
        #pragma omp single
        send_array_static = new T [this->send_offsets[comm_size]];
        T* const send_array = send_array_static;

        for (uint i = 0; i < (uint)comm_size; i++) {
            // send_arr.fill_vector(send_array, this->send_overlap_vertices_list[i], this->send_offsets[i]);
            Vertices_method::fill_vector(send_arr, send_arr_vertices, send_array,
                                         this->send_overlap_vertices_list[i], this->send_offsets[i]);
        }
        static T* receive_array_static = nullptr;
        #pragma omp single
        receive_array_static = new T [this->receive_offsets[comm_size]];
        T* const receive_array = receive_array_static;
        #pragma omp master
        MPI_Alltoallv(send_array, this->send_nnode_list.data(), this->send_offsets.data(), mpi_datatype,
                      receive_array, this->receive_nnode_list.data(), this->receive_offsets.data(), mpi_datatype,
                      this->comm);
        #pragma omp barrier
        for (uint i = 0; i < (uint)comm_size; i++) {
            // recv_arr.be_filled_vector(receive_array, this->receive_overlap_vertices_list[i], this->receive_offsets[i]);
            Vertices_method::be_filled_vector(recv_arr, recv_arr_vertices, receive_array,
                                              this->receive_overlap_vertices_list[i],
                                              this->receive_overlap_vertices_list[i],
                                              this->receive_offsets[i]);
        }
        #pragma omp barrier
        #pragma omp single nowait
        {
            delete [] send_array_static;
            send_array_static = nullptr;
        }
        #pragma omp single nowait
        {
            delete [] receive_array_static;
            receive_array_static = nullptr;
        }

        #else //USE_OPENMP

        // Array_0D<T> send_array(this->send_offsets[comm_size]);
        T* send_array = new T [this->send_offsets[comm_size]];
        for (uint i = 0; i < (uint)comm_size; i++) {
            // send_arr.fill_vector(send_array, this->send_overlap_vertices_list[i], this->send_offsets[i]);
            Vertices_method::fill_vector(send_arr, send_arr_vertices, send_array,
                                         this->send_overlap_vertices_list[i], this->send_offsets[i]);
        }
        // Array_0D<T> receive_array(this->receive_offsets[comm_size]);
        T* receive_array = new T [this->receive_offsets[comm_size]];
        MPI_Alltoallv(send_array, this->send_nnode_list.data(), this->send_offsets.data(), mpi_datatype,
                      receive_array, this->receive_nnode_list.data(), this->receive_offsets.data(), mpi_datatype,
                      this->comm);
        for (uint i = 0; i < (uint)comm_size; i++) {
            // recv_arr.be_filled_vector(receive_array, this->receive_overlap_vertices_list[i], this->receive_offsets[i]);
            Vertices_method::be_filled_vector(recv_arr, recv_arr_vertices, receive_array,
                                              this->receive_overlap_vertices_list[i],
                                              this->receive_overlap_vertices_list[i],
                                              this->receive_offsets[i]);
        }
        delete [] receive_array;
        delete [] send_array;

        #endif //USE_OPENMP
    } else {
        // send_arr.fill_overlap(recv_arr);
        Vertices_method::fill_region(send_arr, send_arr_vertices, recv_arr, recv_arr_vertices,
                                     send_arr_vertices.get_overlap_vertices(recv_arr_vertices));
    }
    return;
}
template void Domain_4D_to_4D_mpi_package::send_data<int>(int const* const& send_arr, const Vertices_4D& send_arr_vertices,
                                                          int* const& recv_arr, const Vertices_4D& recv_arr_vertices) const;
template void Domain_4D_to_4D_mpi_package::send_data<float>(float const* const& send_arr, const Vertices_4D& send_arr_vertices,
                                                            float* const& recv_arr, const Vertices_4D& recv_arr_vertices) const;
template void Domain_4D_to_4D_mpi_package::send_data<double>(double const* const& send_arr, const Vertices_4D& send_arr_vertices,
                                                             double* const& recv_arr, const Vertices_4D& recv_arr_vertices) const;

template<typename T>
void Domain_4D_to_4D_mpi_package::recv_data(Array_4D<T>& recv_arr, const Array_4D<T>& send_arr) const {
    if (this->need_comm) {
        int comm_size;
        MPI_Comm_size(this->comm, &comm_size);
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        Array_0D<T> send_array(this->receive_offsets[comm_size]);
        for (uint i = 0; i < (uint)comm_size; i++) {
            send_arr.fill_vector(send_array, this->receive_overlap_vertices_list[i], this->receive_offsets[i]);
        }
        Array_0D<T> receive_array(this->send_offsets[comm_size]);
        MPI_Alltoallv(send_array.data, this->receive_nnode_list.data(), this->receive_offsets.data(), mpi_datatype,
                    receive_array.data, this->send_nnode_list.data(), this->send_offsets.data(), mpi_datatype,
                    this->comm);
        for (uint i = 0; i < (uint)comm_size; i++) {
            recv_arr.be_filled_vector(receive_array, this->send_overlap_vertices_list[i], this->send_offsets[i]);
        }
    } else {
        send_arr.fill_overlap(recv_arr);
    }
    return;
}
template void Domain_4D_to_4D_mpi_package::recv_data<int>(Array_4D<int>& recv_arr, const Array_4D<int>& send_arr) const;
template void Domain_4D_to_4D_mpi_package::recv_data<float>(Array_4D<float>& recv_arr, const Array_4D<float>& send_arr) const;
template void Domain_4D_to_4D_mpi_package::recv_data<double>(Array_4D<double>& recv_arr, const Array_4D<double>& send_arr) const;

template<typename T>
void Domain_4D_to_4D_mpi_package::recv_data(T* const& recv_arr, const Vertices_4D& recv_arr_vertices,
                                            T const* const& send_arr, const Vertices_4D& send_arr_vertices) const {
    if (this->need_comm) {
        int comm_size;
        MPI_Comm_size(this->comm, &comm_size);
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        // Array_0D<T> send_array(this->receive_offsets[comm_size]);
        #ifdef USE_OPENMP

        static T* send_array_static = nullptr;
        #pragma omp single
        send_array_static = new T [this->receive_offsets[comm_size]];
        T* const send_array = send_array_static;

        for (uint i = 0; i < (uint)comm_size; i++) {
            // send_arr.fill_vector(send_array, this->receive_overlap_vertices_list[i], this->receive_offsets[i]);
            Vertices_method::fill_vector(send_arr, send_arr_vertices, send_array,
                                         this->receive_overlap_vertices_list[i], this->receive_offsets[i]);
        }

        static T* receive_array_static = nullptr;
        #pragma omp single
        receive_array_static = new T [this->send_offsets[comm_size]];
        T* const receive_array = receive_array_static;

        #pragma omp master
        MPI_Alltoallv(send_array, this->receive_nnode_list.data(), this->receive_offsets.data(), mpi_datatype,
                      receive_array, this->send_nnode_list.data(), this->send_offsets.data(), mpi_datatype,
                      this->comm);
        #pragma omp barrier
        for (uint i = 0; i < (uint)comm_size; i++) {
            // recv_arr.be_filled_vector(receive_array, this->send_overlap_vertices_list[i], this->send_offsets[i]);
            Vertices_method::be_filled_vector(recv_arr, recv_arr_vertices, receive_array,
                                              this->send_overlap_vertices_list[i], this->send_overlap_vertices_list[i],
                                              this->send_offsets[i]);
        }
        #pragma omp barrier
        #pragma omp single nowait
        {
            delete [] send_array_static;
            send_array_static = nullptr;
        }
        #pragma omp single nowait
        {
            delete [] receive_array_static;
            receive_array_static = nullptr;
        }

        #else //USE_OPENMP

        T* send_array = new T [this->receive_offsets[comm_size]];
        for (uint i = 0; i < (uint)comm_size; i++) {
            // send_arr.fill_vector(send_array, this->receive_overlap_vertices_list[i], this->receive_offsets[i]);
            Vertices_method::fill_vector(send_arr, send_arr_vertices, send_array,
                                         this->receive_overlap_vertices_list[i], this->receive_offsets[i]);
        }
        // Array_0D<T> receive_array(this->send_offsets[comm_size]);
        T* receive_array = new T [this->send_offsets[comm_size]];
        MPI_Alltoallv(send_array, this->receive_nnode_list.data(), this->receive_offsets.data(), mpi_datatype,
                      receive_array, this->send_nnode_list.data(), this->send_offsets.data(), mpi_datatype,
                      this->comm);
        for (uint i = 0; i < (uint)comm_size; i++) {
            // recv_arr.be_filled_vector(receive_array, this->send_overlap_vertices_list[i], this->send_offsets[i]);
            Vertices_method::be_filled_vector(recv_arr, recv_arr_vertices, receive_array,
                                              this->send_overlap_vertices_list[i], this->send_overlap_vertices_list[i],
                                              this->send_offsets[i]);
        }
        delete [] receive_array;
        delete [] send_array;

        #endif //USE_OPENMP

    } else {
        // send_arr.fill_overlap(recv_arr);
        Vertices_method::fill_region(send_arr, send_arr_vertices, recv_arr, recv_arr_vertices,
                                     send_arr_vertices.get_overlap_vertices(recv_arr_vertices));
    }
    return;
}
template void Domain_4D_to_4D_mpi_package::recv_data<int>(int* const& recv_arr, const Vertices_4D& recv_arr_vertices,
                                                          int const* const& send_arr, const Vertices_4D& send_arr_vertices) const;
template void Domain_4D_to_4D_mpi_package::recv_data<float>(float* const& recv_arr, const Vertices_4D& recv_arr_vertices,
                                                            float const* const& send_arr, const Vertices_4D& send_arr_vertices) const;
template void Domain_4D_to_4D_mpi_package::recv_data<double>(double* const& recv_arr, const Vertices_4D& recv_arr_vertices,
                                                             double const* const& send_arr, const Vertices_4D& send_arr_vertices) const;

void Domain_4D_to_4D_mpi_package::init() {
    // assert(my_domain_vertices.get_4D_shared_vertices() == other_domain_vertices.get_4D_shared_vertices()
    //        && my_domain_vertices.comm == other_domain_vertices.comm);
    assert(this->my_domain_vertices.get_4D_shared_vertices() == this->other_domain_vertices.get_4D_shared_vertices()
        && Parallel_vertices::is_comm_equal(this->my_domain_vertices.comm, this->other_domain_vertices.comm));
    this->comm = this->my_domain_vertices.comm;
    int comm_size = this->my_domain_vertices.get_comm_size();

    //receive mpi data
    this->receive_overlap_vertices_list.resize(comm_size, Vertices_4D(0, 0, 0, 0));
    this->receive_nnode_list.resize(comm_size, 0);
    this->other_domain_vertices.data_transfer_prepare(this->my_domain_vertices,
                                                this->receive_overlap_vertices_list.data(),
                                                this->receive_nnode_list.data());
    this->need_comm = Parallel_vertices::need_comm(this->receive_nnode_list.data(), this->comm);
    if (!this->need_comm) {
        std::vector<Vertices_4D>().swap(this->receive_overlap_vertices_list);
        std::vector<int>().swap(this->receive_nnode_list);
        return;
    }
    this->receive_offsets.resize(comm_size + 1, 0);
    this->receive_offsets[0] = 0;
    for (uint i = 0; i < (uint)comm_size; i++) {
        this->receive_offsets[i + 1] =  this->receive_nnode_list[i] + this->receive_offsets[i];
    }

    //send mpi data
    this->send_overlap_vertices_list.resize(comm_size, Vertices_4D(0, 0, 0, 0));
    this->send_nnode_list.resize(comm_size, 0);
    this->send_offsets.resize(comm_size + 1, 0);
    this->my_domain_vertices.data_transfer_prepare(this->other_domain_vertices,
                                             this->send_overlap_vertices_list.data(),
                                             this->send_nnode_list.data());
    this->send_offsets[0] = 0;
    for (uint i = 0; i < (uint)comm_size; i++) {
        this->send_offsets[i + 1] =  this->send_nnode_list[i] + this->send_offsets[i];
    }
    return;
}

void Domain_4D_to_4D_mpi_package::init(const Domain_4D_to_4D_mpi_package& domain_4D_to_4D_mpi_package) {
    this->comm = this->my_domain_vertices.comm;
    this->need_comm = domain_4D_to_4D_mpi_package.need_comm;
    if (this->need_comm) {
        this->receive_overlap_vertices_list = domain_4D_to_4D_mpi_package.receive_overlap_vertices_list;
        this->receive_nnode_list = domain_4D_to_4D_mpi_package.receive_nnode_list;
        this->receive_offsets = domain_4D_to_4D_mpi_package.receive_offsets;
        this->send_overlap_vertices_list = domain_4D_to_4D_mpi_package.send_overlap_vertices_list;
        this->send_nnode_list = domain_4D_to_4D_mpi_package.send_nnode_list;
        this->send_offsets = domain_4D_to_4D_mpi_package.send_offsets;
    }
    return;
}

void Domain_4D_to_4D_mpi_package::destructor() {
    std::vector<Vertices_4D>().swap(this->send_overlap_vertices_list);
    std::vector<int>().swap(this->send_nnode_list);
    std::vector<int>().swap(this->send_offsets);
    std::vector<Vertices_4D>().swap(this->receive_overlap_vertices_list);
    std::vector<int>().swap(this->receive_nnode_list);
    std::vector<int>().swap(this->receive_offsets);
    return;
}

void Domain_4D_to_4D_mpi_package::show() const {
    std::cout << "need_comm = " << this->need_comm << std::endl;
    std::cout << "send_overlap_vertices_list" << std::endl;
    for (std::vector<Vertices_4D>::const_iterator it = this->send_overlap_vertices_list.begin();
            it != this->send_overlap_vertices_list.end(); it++) {
        it->show();
    }
    std::cout << "send_nnode_list" << std::endl;
    for (std::vector<int>::const_iterator it = this->send_nnode_list.begin();
            it != this->send_nnode_list.end(); it++) {
        std::cout << "send node = " << *it << std::endl;
    }
    std::cout << "send_offsets" << std::endl;
    for (std::vector<int>::const_iterator it = this->send_offsets.begin();
            it != this->send_offsets.end(); it++) {
        std::cout << "send_offsets = " << *it << std::endl;
    }
    std::cout << std::endl;
    std::cout << "receive_overlap_vertices_list" << std::endl;
    for (std::vector<Vertices_4D>::const_iterator it = this->receive_overlap_vertices_list.begin();
            it != this->receive_overlap_vertices_list.end(); it++) {
        it->show();
    }
    std::cout << "receive_nnode_list" << std::endl;
    for (std::vector<int>::const_iterator it = this->receive_nnode_list.begin();
            it != this->receive_nnode_list.end(); it++) {
        std::cout << "receive node = " << *it << std::endl;
    }
    std::cout << "receive_offsets" << std::endl;
    for (std::vector<int>::const_iterator it = this->receive_offsets.begin();
            it != this->receive_offsets.end(); it++) {
        std::cout << "receive_offsets = " << *it << std::endl;
    }
    return;
}


uint64_t Parallel_vertices::cal_optimize_chunksize(const uint64_t ni, const uint64_t ideal_chuck, const uint64_t unit_pool_size) {
    if (ni <= ideal_chuck) return ni;
    uint64_t real_ideal_chuck = ideal_chuck > 0 ? ideal_chuck : 1;
    uint64_t nunit_test = (ni - 1)/ real_ideal_chuck + 1;
    uint64_t nunit_test_m1 = nunit_test - 1;
    uint64_t chunk_test = Parallel_vertices::cal_chunksize_1D(nunit_test, ni);
    uint64_t chunk_test_m1 = Parallel_vertices::cal_chunksize_1D(nunit_test_m1, ni);

    uint64_t temp_chuck = nunit_test * chunk_test <= nunit_test_m1 * chunk_test_m1 ? chunk_test : chunk_test_m1;
    uint64_t temp_nuint = nunit_test * chunk_test <= nunit_test_m1 * chunk_test_m1 ? nunit_test : nunit_test_m1;

    if (temp_nuint + temp_nuint < unit_pool_size) {
        return temp_chuck;
    } else {
        return Parallel_vertices::cal_chunksize_1D(unit_pool_size, ni);
    }
}

uint64_t Parallel_vertices::cal_chunksize_1D(const uint64_t nunit, const uint64_t ni) {
    // assert(ni != 0);
    assert(nunit != 0);
    return (ni - 1 + nunit)/nunit;
}

std::array<uint64_t,3> Parallel_vertices::cal_chunk_sizes_3D(const uint64_t nunit, const uint64_t ni, const uint64_t nj, const uint64_t nk) {
    std::array<uint64_t, 3> result = {0, 0, 0};
    uint64_t size_0, size_1, size_2;
    uint64_t *chunk_size_0, *chunk_size_1, *chunk_size_2;
    if (ni <= nj && ni <= nk) {
        chunk_size_0 = &(result[0]);
        size_0 = ni;
        if (nj <= nk) {
            chunk_size_1 = &(result[1]);
            chunk_size_2 = &(result[2]);
            size_1 = nj;
            size_2 = nk;
        } else {
            chunk_size_1 = &(result[2]);
            chunk_size_2 = &(result[1]);
            size_1 = nk;
            size_2 = nj;
        }
    } else if (nj <= nk) {
        chunk_size_0 = &(result[1]);
        size_0 = nj;
        if (ni <= nk) {
            chunk_size_1 = &(result[0]);
            chunk_size_2 = &(result[2]);
            size_1 = ni;
            size_2 = nk;
        } else {
            chunk_size_1 = &(result[2]);
            chunk_size_2 = &(result[0]);
            size_1 = nk;
            size_2 = ni;
        }
    } else {
        chunk_size_0 = &(result[2]);
        size_0 = nk;
        if (ni <= nj) {
            chunk_size_1 = &(result[0]);
            chunk_size_2 = &(result[1]);
            size_1 = ni;
            size_2 = nj;
        } else {
            chunk_size_1 = &(result[1]);
            chunk_size_2 = &(result[0]);
            size_1 = nj;
            size_2 = ni;
        }
    }
    *chunk_size_2 = Parallel_vertices::cal_optimize_chunksize(size_2, (uint64_t) std::pow(size_0 * size_1 * size_2 / nunit, 1.0/3.0), nunit);
    uint64_t nunit_2 = (size_2 - 1)/ *chunk_size_2 + 1;
    uint64_t nunit_01 = nunit / nunit_2;
    *chunk_size_1 = Parallel_vertices::cal_optimize_chunksize(size_1, (uint64_t) std::pow(size_0 * size_1 / nunit_01, 0.5), nunit_01);
    uint64_t nunit_1 = (size_1 - 1)/ *chunk_size_1 + 1;
    uint64_t nunit_0 = nunit_01 / nunit_1;
    *chunk_size_0 = Parallel_vertices::cal_chunksize_1D(nunit_0, size_0);

    std::array<uint64_t, 3> result_temp = result;
    uint64_t nunit_0_temp = nunit_0;
    uint64_t nunit_1_temp = nunit_1;
    uint64_t nunit_2_temp = nunit_2;

    *chunk_size_0 = Parallel_vertices::cal_optimize_chunksize(size_0, (uint64_t) std::pow(size_0 * size_1 * size_2 / nunit, 1.0/3.0), nunit);
    nunit_0 = (size_0 - 1)/ *chunk_size_0 + 1;
    uint64_t nunit_12 = nunit / nunit_0;
    *chunk_size_1 = Parallel_vertices::cal_optimize_chunksize(size_1, (uint64_t) std::pow(size_1 * size_2 / nunit_12, 0.5), nunit_12);
    nunit_1 = (size_1 - 1)/ *chunk_size_1 + 1;
    nunit_2 = nunit_12 / nunit_1;
    *chunk_size_2 = Parallel_vertices::cal_chunksize_1D(nunit_2, size_2);
    
    if(nunit_0 * nunit_1 * nunit_2 > nunit_0_temp * nunit_1_temp * nunit_2_temp) {
        return result;
    } else {
        return result_temp;
    }
}

std::array<uint64_t,2> Parallel_vertices::cal_chunk_sizes_2D(const uint64_t nunit, const uint64_t ni, const uint64_t nj) {
    std::array<uint64_t, 2> result = {0, 0};
    
    uint64_t chunk_size_i;
    uint64_t chunk_size_j;

    chunk_size_i = Parallel_vertices::cal_optimize_chunksize(ni, (uint64_t) std::pow(ni * nj / nunit, 0.5), nunit);
    uint64_t nunit_i = (ni - 1 + chunk_size_i) / chunk_size_i;
    uint64_t nunit_j = nunit / nunit_i;
    chunk_size_j = Parallel_vertices::cal_chunksize_1D(nunit_j, nj);
    uint64_t nunit_i_first = nunit_i;
    uint64_t nunit_j_first = nunit_j;
    result[0] = chunk_size_i;
    result[1] = chunk_size_j;

    chunk_size_j = Parallel_vertices::cal_optimize_chunksize(nj, (uint64_t) std::pow(ni * nj / nunit, 0.5), nunit);
    nunit_j = (nj - 1 + chunk_size_j) / chunk_size_j;
    nunit_i = nunit / nunit_j;
    chunk_size_i = Parallel_vertices::cal_chunksize_1D(nunit_i, ni);
    if (nunit_i_first * nunit_j_first <= nunit_j * nunit_i) {
        result[0] = chunk_size_i;
        result[1] = chunk_size_j;
    }
    return result;
}

int64_t Parallel_vertices::generate_nth_vertices(int64_t* is, int64_t* ie, const int64_t ith, const uint64_t comm_i, const uint64_t comm_ni,
                                    const uint64_t chunk_size, const uint64_t last_block_size, const uint64_t nodes) {
    int64_t block_num = ith / (int64_t)comm_ni;
    int64_t ith_local = ith % (int64_t)comm_ni;
    if (ith_local < 0) {
        ith_local += (int64_t)comm_ni;
        block_num -= 1;
    }
    int64_t offset = block_num * (chunk_size * (comm_ni - 1) + last_block_size);
    int64_t is_local = (ith_local + comm_i <= comm_ni - 1) ? (ith_local + comm_i) * chunk_size : (ith_local + comm_i) * chunk_size - (chunk_size - last_block_size);
    int64_t ie_local = (ith_local + comm_i < comm_ni - 1) ? (ith_local + comm_i + 1) * chunk_size - 1 : (ith_local + comm_i + 1) * chunk_size - 1 - (chunk_size - last_block_size);
    int64_t this_is = chunk_size * comm_i;
    int64_t this_ie = comm_i == comm_ni - 1 ? chunk_size * (comm_ni - 1) + last_block_size - 1 : (comm_i + 1) * chunk_size - 1;
    if (ith >= 0) {
        *is = is_local + offset;
        *ie = (ie_local + (int64_t)offset - this_ie <= (int64_t) nodes) ? ie_local + offset : this_ie + (int64_t) nodes;
    } else {
        *is = (this_is - (is_local + offset) <= (int64_t) nodes) ? is_local + offset : this_is - (int64_t) nodes;
        *ie = ie_local + offset;
    }
    return (int64_t)(block_num + (ith_local + comm_i) / comm_ni);
}

uint64_t Parallel_vertices::generate_left_domain_n(const uint64_t comm_i, const uint64_t comm_ni, const uint64_t chunk_size,
                                               const uint64_t last_block_size, const uint64_t nodes, const bool is_periodic) {
    uint64_t nodes_local = nodes % ((comm_ni - 1) * chunk_size + last_block_size);
    uint64_t block_num = nodes / ((comm_ni - 1) * chunk_size + last_block_size);
    uint64_t n_local;
    if (nodes_local == 0) {
        n_local = 0;
    } else {
        n_local = ((comm_i == comm_ni - 1) || ((nodes_local + chunk_size - 1 ) / chunk_size) <= comm_i)
                ? (nodes_local + chunk_size - 1 ) / chunk_size
                : (nodes_local + chunk_size + chunk_size - last_block_size - 1 ) / chunk_size;
    }
    if (is_periodic) {
        return n_local + block_num * comm_ni;
    } else if (n_local + block_num * comm_ni <= comm_i) {
        return n_local + block_num * comm_ni;
    } else {
        return comm_i;
    }
}

uint64_t Parallel_vertices::generate_right_domain_n(const uint64_t comm_i, const uint64_t comm_ni, const uint64_t chunk_size,
                                               const uint64_t last_block_size, const uint64_t nodes, const bool is_periodic) {
    uint64_t nodes_local = nodes % ((comm_ni - 1) * chunk_size + last_block_size);
    uint64_t block_num = nodes / ((comm_ni - 1) * chunk_size + last_block_size);
    uint64_t n_local;
    if (nodes_local == 0) {
        n_local = 0;
    } else {
        n_local = ((comm_i == comm_ni - 1) || ((comm_i + 1) * chunk_size) + nodes_local <=  (comm_ni - 1) * chunk_size + last_block_size)
                ? (nodes_local + chunk_size - 1 ) / chunk_size
                : (nodes_local + chunk_size + chunk_size - last_block_size - 1 ) / chunk_size;
    }
    if (is_periodic) {
        return n_local + block_num * comm_ni;
    } else if (comm_i + n_local + block_num * comm_ni <= comm_ni - 1) {
        return n_local + block_num * comm_ni;
    } else {
        return (comm_ni - 1) - comm_i;
    }
}

template<typename T>
void Parallel_vertices::domain_vertices_rand(T* const arr, const Domain_parallel_vertices_3D& domain_vertices,
                                                   const T rand_min, const T rand_max) {
    uint64_t shared_origin = domain_vertices.get_3D_shared_vertices().get_index_nocheck(domain_vertices.get_3D_local_vertices().is,
                                                                           domain_vertices.get_3D_local_vertices().js,
                                                                           domain_vertices.get_3D_local_vertices().ks);
    for (uint64_t k = 0; k < domain_vertices.get_3D_local_vertices().nk; k++) {
        uint64_t shared_offset_k = k * domain_vertices.get_3D_shared_vertices().ni * domain_vertices.get_3D_shared_vertices().nj;
        uint64_t local_offset_k = k * domain_vertices.get_3D_local_vertices().ni * domain_vertices.get_3D_local_vertices().nj;
        for (uint64_t j = 0; j < domain_vertices.get_3D_local_vertices().nj; j++) {
            uint64_t shared_offset_j = j * domain_vertices.get_3D_shared_vertices().ni + shared_offset_k;
            uint64_t local_offset_j = j * domain_vertices.get_3D_local_vertices().ni + local_offset_k;
            uint64_t seed = shared_origin + shared_offset_j + 1;
            Linalg::seededrand_sequential(arr + local_offset_j,
                               domain_vertices.get_3D_local_vertices().ni,
                               rand_min, rand_max, seed);
        }
    }
    return;
}
template void Parallel_vertices::domain_vertices_rand<int>(int* const arr, const Domain_parallel_vertices_3D& domain_vertices,
                                                           const int rand_min, const int rand_max);
template void Parallel_vertices::domain_vertices_rand<float>(float* const arr, const Domain_parallel_vertices_3D& domain_vertices,
                                                             const float rand_min, const float rand_max);
template void Parallel_vertices::domain_vertices_rand<double>(double* const arr, const Domain_parallel_vertices_3D& domain_vertices,
                                                              const double rand_min, const double rand_max);

template<typename T>
void Parallel_vertices::domain_vertices_rand(Array_3D<T>& arr, const Domain_parallel_vertices_3D& domain_vertices,
                                             const T& rand_min, const T& rand_max) {
    uint64_t shared_origin = domain_vertices.get_3D_shared_vertices().get_index_nocheck(domain_vertices.get_3D_local_vertices().is,
                                                                           domain_vertices.get_3D_local_vertices().js,
                                                                           domain_vertices.get_3D_local_vertices().ks);
    for (uint64_t k = 0; k < domain_vertices.get_3D_local_vertices().nk; k++) {
        uint64_t shared_offset_k = k * domain_vertices.get_3D_shared_vertices().ni * domain_vertices.get_3D_shared_vertices().nj;
        uint64_t local_offset_k = k * domain_vertices.get_3D_local_vertices().ni * domain_vertices.get_3D_local_vertices().nj;
        for (uint64_t j = 0; j < domain_vertices.get_3D_local_vertices().nj; j++) {
            uint64_t shared_offset_j = j * domain_vertices.get_3D_shared_vertices().ni + shared_offset_k;
            uint64_t local_offset_j = j * domain_vertices.get_3D_local_vertices().ni + local_offset_k;
            uint64_t seed = shared_origin + shared_offset_j + 1;
            Linalg::seededrand_sequential(arr.data + local_offset_j,
                               domain_vertices.get_3D_local_vertices().ni,
                               rand_min, rand_max, seed);
        }
    }
    return;
}
template void Parallel_vertices::domain_vertices_rand<int>(Array_3D<int>& arr, const Domain_parallel_vertices_3D& domain_vertices,
                                                           const int& rand_min, const int& rand_max);
template void Parallel_vertices::domain_vertices_rand<float>(Array_3D<float>& arr, const Domain_parallel_vertices_3D& domain_vertices,
                                                             const float& rand_min, const float& rand_max);
template void Parallel_vertices::domain_vertices_rand<double>(Array_3D<double>& arr, const Domain_parallel_vertices_3D& domain_vertices,
                                                              const double& rand_min, const double& rand_max);

template<typename T>
void Parallel_vertices::domain_vertices_rand(Array_4D<T>& arr, const Domain_parallel_vertices_4D& domain_vertices,
                                             const T& rand_min, const T& rand_max) {
    uint64_t shared_origin = domain_vertices.get_4D_shared_vertices().get_index_nocheck(domain_vertices.get_4D_local_vertices().is,
                                                                                    domain_vertices.get_4D_local_vertices().js,
                                                                                    domain_vertices.get_4D_local_vertices().ks,
                                                                                    domain_vertices.get_4D_local_vertices().bs);
    for (uint64_t b = 0; b < domain_vertices.get_4D_local_vertices().nb; b++) {
        uint64_t shared_offset_b = b * domain_vertices.get_4D_shared_vertices().ni
                                 * domain_vertices.get_4D_shared_vertices().nj
                                 * domain_vertices.get_4D_shared_vertices().nk;
        uint64_t local_offset_b = b * domain_vertices.get_4D_local_vertices().ni
                                * domain_vertices.get_4D_local_vertices().nj
                                * domain_vertices.get_4D_local_vertices().nk;
        for (uint64_t k = 0; k < domain_vertices.get_4D_local_vertices().nk; k++) {
            uint64_t shared_offset_k = k * domain_vertices.get_4D_shared_vertices().ni
                                     * domain_vertices.get_4D_shared_vertices().nj
                                     + shared_offset_b;
            uint64_t local_offset_k = k * domain_vertices.get_4D_local_vertices().ni
                                    * domain_vertices.get_4D_local_vertices().nj
                                    + local_offset_b;
            for (uint64_t j = 0; j < domain_vertices.get_4D_local_vertices().nj; j++) {
                uint64_t shared_offset_j = j * domain_vertices.get_4D_shared_vertices().ni + shared_offset_k;
                uint64_t local_offset_j = j * domain_vertices.get_4D_local_vertices().ni + local_offset_k;
                uint64_t seed = shared_origin + shared_offset_j + 1;
                Linalg::seededrand_sequential(arr.data + local_offset_j,
                                domain_vertices.get_4D_local_vertices().ni,
                                rand_min, rand_max, seed);
            }
        }
    }
    return;
}
template void Parallel_vertices::domain_vertices_rand<int>(Array_4D<int>& arr, const Domain_parallel_vertices_4D& domain_vertices,
                                                           const int& rand_min, const int& rand_max);
template void Parallel_vertices::domain_vertices_rand<float>(Array_4D<float>& arr, const Domain_parallel_vertices_4D& domain_vertices,
                                                             const float& rand_min, const float& rand_max);
template void Parallel_vertices::domain_vertices_rand<double>(Array_4D<double>& arr, const Domain_parallel_vertices_4D& domain_vertices,
                                                              const double& rand_min, const double& rand_max);

bool Parallel_vertices::need_comm(int const* const& receive_nnode_list, const MPI_Comm& comm) {
    int rank;
    int size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    bool need_comm = false;
    for (int i = 0; i < size; i++) {
        if (i == rank) continue;
        if(receive_nnode_list[i] > 0) {
            need_comm = true;
            break;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &need_comm, 1, MPI_C_BOOL, MPI_LOR, comm);
    return need_comm;
}

#if (defined(USE_MKL) || defined(USE_SCALAPACK))
void Parallel_vertices::Cblacs_gridmap_subcomm(int* icontxt, const int ldup, const int nprow,
                                               const int npcol, const bool& is_active,
                                               const MPI_Comm subcomm) {
    int size;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!is_active) {
        int usermap[1] = {rank};
        int ldup = 1;
        int nprow = 1;
        int npcol = 1;
        Cblacs_gridmap(icontxt, usermap, ldup, nprow, npcol);
    } else if (subcomm == MPI_COMM_NULL) {
        assert(ldup == 1 && nprow == 1 && npcol == 1);
        int usermap[1] = {rank};
        Cblacs_gridmap(icontxt, usermap, ldup, nprow, npcol);
    } else {
        int sub_size;
        MPI_Comm_size(subcomm, &sub_size);
        bool* is_in_subcomm = new bool[size]();
        is_in_subcomm[rank] = true;
        MPI_Allreduce(MPI_IN_PLACE, is_in_subcomm, size, MPI_C_BOOL, MPI_LOR, subcomm);
        assert(ldup > 0 && npcol > 0 && nprow > 0);
        int contxt_size = ldup * (npcol-1) + nprow;
        assert(sub_size >= contxt_size);
        int* usermap = new int [sub_size];
        int count = 0;
        for (int i = 0; i < size; i++) {
            if (is_in_subcomm[i]) {
                usermap[count] = i;
                count++;
            }
        }
        Cblacs_gridmap(icontxt, usermap, ldup, nprow, npcol);
        delete [] usermap;
        delete [] is_in_subcomm;
    }
    return;
}
#endif

/**
 * @brief  For the given parallelization params, find how many count of 
 *         work load each process is assigned, we take the ceiling since
 *         at least one process would have that many columns and will
 *         dominate the cost.
 *         
 **/
int workload_assigned(const int Nk, const int Ns, 
    const int np1, const int np2)
{
    return Linalg::get_chunksize(np1,Nk) * Linalg::get_chunksize(np2,Ns);
}

double skbd_band_efficiency(const int x) {
    double eff = 1.00;
    double x_table[5] = {1,   8,   27,  125, 512};
    double y_table[5] = {1.00,0.95,0.75,0.55,0.45};
    int len = 5;
    for (int i = 0; i < len - 1; i++) {
        if (x >= x_table[i] && x < x_table[i+1]) {
            double slope = (y_table[i+1]-y_table[i])/(x_table[i+1]-x_table[i]);
            eff = y_table[i] + (x - x_table[i]) * slope;
        }
    }

    if (x > x_table[len-1]) eff = 0.44;
    return eff;
}

double skbd_domain_efficiency(const int x) {
    double eff = 1.00;
    double x_table[7] = {1,   2,   8,   20,  27  ,120 ,200 };
    double y_table[7] = {1.00,0.80,0.75,0.65,0.60,0.40,0.30};
    int len = 7;
    for (int i = 0; i < len - 1; i++) {
        if (x >= x_table[i] && x < x_table[i+1]) {
            double slope = (y_table[i+1]-y_table[i])/(x_table[i+1]-x_table[i]);
            eff = y_table[i] + (x - x_table[i]) * slope;
        }
    }

    if (x > x_table[len-1]) eff = 0.29;
    return eff;
}

/**
 * @brief  The scaling factor introduced by DMndbyNband
 * 
 *         Adding a side effect for tall thin local psi where applying ACE operator is slow
 */
double scaling_DMndbyNband(const int Ns, const int *gridsizes, const int npb, const int npd)
{
    int Nd = gridsizes[0] * gridsizes[1] * gridsizes[2];
    double DMndbyNband = Linalg::get_chunksize(npd, Nd)/((double)Linalg::get_chunksize(npb, Ns));
    // empirically DMndbyNband around 40 to 80 is good
    double x = (DMndbyNband - 60)/20.0;
    double cheby4 = std::max(0.0,16*std::pow(x,4)-12*std::pow(x,2)+1);
    return pow(0.98,cheby4);
}

double skbd_weighted_efficiency(
    const int Nspin, const int Nk, const int Ns, 
    const int *gridsizes, const int np, const int nps, 
    const int npk, const int npb, const int npd, const int isfock)
{
    double eff;
    int ncols = workload_assigned(Nspin*Nk, Ns, nps*npk, npb);
    eff = Nspin * Nk * Ns * npd / (double)(ncols * np);
    // apply weight for band
    eff *= skbd_band_efficiency(npb);
    // apply weight for domain
    eff *= skbd_domain_efficiency(npd);
    
    if (isfock) {
        // scale by DMndbyNband coefficients 
        eff *= scaling_DMndbyNband(Ns, gridsizes, npb, npd);
    }
    return eff;
}

/**
 * @brief  Caluclate the efficiency for a given set of parameters.
 *         This is derived based on the assumption that there are
 *         infinite grid points (Nd) so that np can not be larger
 *         than Nk*Ns*Nd in any case. In practice, Nd ~ O(1e6), so
 *         this assumption is quite reasonable.
 *         1. Ideal load: Nk*Ns*Nd/np
 *         2. Actual load: ceil(Nk/np1)*ceil(Ns/np2)*ceil(Nd/np3) 
 *                      ~= ceil(Nk/np1)*ceil(Ns/np2)* (Nd/np3) #since Nd is large
 *         3. efficiency = Ideal load / Actual load
 *                      ~= Nk*Ns*np3/ (ceil(Nk/np1)*ceil(Ns/np2)*np)
 * @param Nk   Number of kpoints (after symmetry reduction).
 * @param Ns   Number of states/bands.
 * @param np   Number of processors available.
 * @param np1  Number of kpoint groups for kpoint parallelization.
 * @param np2  Number of band groups for band parallelization.
 * @param np3  Number of domain groups for domain parallelization.
 *
 **/
double work_load_efficiency(
    const int Nk, const int Ns, const int np, 
    const int np1, const int np2, const int np3
)
{
    int ncols = workload_assigned(Nk, Ns, np1, np2);
    return Nk * Ns * np3 / (double)(ncols * np);
}

typedef struct {
    int *list;
    short count; 
} Factors;

/**
 * @brief   The following code for factorizing an integer is copied from Rosetta Code
 *          (https://rosettacode.org/wiki/Factors_of_an_integer#C).
 */
// The defined type is moved to the header file: tools.h
// typedef struct {
// 	int *list;
// 	short count; 
// } Factors;
void xferFactors( Factors *fctrs, int *flist, int flix ) 
{
    int ix, ij;
    int newSize = fctrs->count + flix;
    if (newSize > flix)  {
        // fctrs->list = realloc( fctrs->list, newSize * sizeof(int));
        delete [] fctrs->list;
        fctrs->list = new int [newSize];
    }
    else {
        // fctrs->list = malloc(  newSize * sizeof(int));
        fctrs->list = new int [newSize];
    }
    for (ij=0,ix=fctrs->count; ix<newSize; ij++,ix++) {
        fctrs->list[ix] = flist[ij];
    }
    fctrs->count = newSize;
}

Factors *factor( int num, Factors *fctrs)
{
    int flist[301], flix;
    int dvsr;
    flix = 0;
    fctrs->count = 0;
    // free(fctrs->list);
    delete [] fctrs->list;
    fctrs->list = NULL;
    for (dvsr=1; dvsr*dvsr < num; dvsr++) {
        if (num % dvsr != 0) continue;
        if ( flix == 300) {
            xferFactors( fctrs, flist, flix );
            flix = 0;
        }
        flist[flix++] = dvsr;
        flist[flix++] = num/dvsr;
    }
    if (dvsr*dvsr == num) 
        flist[flix++] = dvsr;
    if (flix > 0)
        xferFactors( fctrs, flist, flix );
 
    return fctrs;
}

void sorted_factor( int num, Factors *fctrs) {
    // call factors to do the calculation
    factor(num, fctrs);

    short len = fctrs->count;
    // int *new_list = (int *)malloc(len * sizeof(int));
    int *new_list = new int [len];

    // the returned list comes in pairs, f1xf2, f3xf4, ...
    // sort the list in ascending order
    // copy the 1st half
    for (int i = 0; i < (len+1)/2; i++) {
        new_list[i] = fctrs->list[2*i];
    }

    short ind = len-1;
    // copy the 2nd half
    for (int i = 1; i < len; i+=2) {
        new_list[ind--] = fctrs->list[i];
    }   

    // free(fctrs->list);
    delete [] fctrs->list;
    fctrs->list = new_list;

}

/**
 * @brief  Caluclate a division of processors in a 2D Cartesian grid.
 *
 *         For a given number of processors, choose np1 and np2 so that the maximum 
 *         work assigned to each process is the smallest, i.e., choose x * y <= np, 
 *         s.t., ceil(N1/x) * ceil(N2/y) is minimum.
 *
 *         Here we assume the parallelization of the two properties are the same,
 *         therefore there's no preference to use more or less processes in any
 *         dimension. 
 *         
 *         The objective function and the constraint are both quadratic, which 
 *         makes this a very hard problem in general.
 *         
 *         In this func, we try to find a reasonably good solution using the following 
 *         strategy: note that if x0 | N1 and y0 | N2 ( "|" means divides), then 
 *         (x0,y0) is a solution to the problem with a weaker constraint x*y <= x0*y0.
 *         We can then keep reducing the constraint from np down until we find a 
 *         solution for the subproblem, and choose the best of all the searched combo. 
 *
 * @param N1   Number of properties to parallelize on the grid in the 1st dimension.
 * @param N2   Number of properties to parallelize on the grid in the 2nd dimension.
 * @param np   Number of processors available.
 *
 **/
void dims_divide_2d(const int N1, const int N2, const int np, int *np1, int *np2)
{
    int search_count = 0;
    
    // initialize with the naive way, parallelize as much as possible for N1 first
    int cur1 = std::min(np, N1);
    int cur2 = std::min(np/cur1, N2);
    int best = workload_assigned(N1,N2,cur1,cur2); 

    int np_up = np > N1*N2 ? N1*N2 : np; // upper bound of np that can be used
    for (int p = np_up; p > 0; p--) {
        // find factors of p
        Factors facs = {NULL, 0};
        sorted_factor(p, &facs);
        int nfacs = facs.count; // total number of factors
        for (int i = 0; i < nfacs; i++) {
            int fac1 = facs.list[i]; // small to large
            int fac2 = p / fac1; // large to small
            // use larger factor for the first dim from the beginning
            int load = workload_assigned(N1,N2,fac2,fac1);
            if (load < best) {
                best = load;
                cur1 = fac2;
                cur2 = fac1;
            }
            if ((N1 % fac2 == 0 && N2 % fac1 == 0) || search_count > 1e5) {
                *np1 = cur1;
                *np2 = cur2;
                // free(facs.list);
                delete [] facs.list;
                return;
            }
        }
        // free(facs.list);
        delete [] facs.list;
    }
}


/**
 * @brief  Caluclate a division of processors in for kpoint, band, domain (KBD) 
 *         parallelization.
 *
 *         For a given number of processors, choose npkpt, npband, and npdomain so 
 *         that the maximum work assigned to each process is the smallest, i.e., 
 *         choose x * y * z <= np, s.t., 
 *         ceil(Nk/x) * ceil(Ns/y) * ceil(Nd/z) is minimum.
 *
 *         Here we assume the parallelization of the two properties are the same,
 *         therefore there's no preference to use more or less processes in any
 *         dimension. 
 *         
 *         The objective function and the constraint are both nonlinear, which 
 *         makes this a very hard problem in general.
 *         
 *         In this func, we try to find a reasonably good solution using the following 
 *         strategy: since the parallelization over kpoint and band are both very 
 *         efficient, we try to parallel over kpoint and band first, then we consider
 *         parallelization over domain. There are two cases: np <=  or > Nk * Ns.
 *         Case 1: np <= Nk * Ns. Then we try to fix npdomain = 1:10, and find the best
 *         parameters for the given npdomain, we always prefer to use less npdomain if
 *         possible.
 *         Case 2: np > Nk * Ns. Then we try to provide Nk*Ns ./ [1:10] for kpoint and
 *         band parallelization, and pick the best combination. Again we prefer to use
 *         as less npdomain (more for K & B) if work load is the similar.
 *
 * @param Nk        Number of kpoints (after symmetry reduction).
 * @param Ns        Number of states.
 * @param gridsizes Number of grid points in all three directions.
 * @param np        Number of processors available.
 * @param np1 (OUT) Number of kpoint groups.
 * @param np2 (OUT) Number of band groups.
 * @param np3 (OUT) Number of domain groups.
 **/
void dims_divide_kbd(
    const int Nk, const int Ns, //const int *gridsizes,
    const int np, int *np1, int *np2, int *np3)
{
#define LEN_NPDM 8

    // SPARC_Dims_create(nproc, ndims, gridsizes, minsize, int *dims, int *ierr); 
    int cur1 = std::min(np,Nk);
    int cur2 = std::min(np/cur1,Ns);
    int cur3 = np / (cur1 * cur2);
    double best_weight = work_load_efficiency(Nk,Ns,np,cur1,cur2,cur3);
    if (best_weight > 0.97) { // prefer kpoint to band
        *np1 = cur1; *np2 = cur2; *np3 = cur3;
        return;
    }

    int npkpt, npband, npdm;
    double weight;

    int npdm_list[LEN_NPDM];
    int npdm_count = LEN_NPDM;
    if (np > Nk * Ns) {
        int npdm_base = np / (Nk * Ns);
        int count = 0;
        for (int i = 1; i <= LEN_NPDM; i++) {
            if (npdm_base * i <= np) {
                npdm_list[count] = npdm_base * i;
                count++;
            }
        }
        npdm_count = count;
    } else {
        int count = 0;
        for (int i = 1; i <= LEN_NPDM; i++) {
            npdm_list[count] = i;
            count++;
        }
        npdm_count = count;
    }

    for (int i = 0; i < npdm_count; i++) {
        // # TODO: check domain parallelization first
        // # npNx,npNy,npNz = domain_paral(Nx,Ny,Nz,npdomain)
        // # if npNx*npNy*npNz != npdomain:
        // #     continue
        npdm = npdm_list[i];
        int np_2d = np / npdm;
        if (np_2d < 1) continue;
        
        dims_divide_2d(Nk, Ns, np_2d, &npkpt, &npband);
        npdm = np / (npkpt * npband);
        weight = work_load_efficiency(Nk,Ns,np,npkpt,npband,npdm);
        if (weight > best_weight) {
            cur1 = npkpt;
            cur2 = npband;
            cur3 = npdm; 
            best_weight = weight;
        }
        if (weight > 0.95) break;
    }

    *np1 = cur1;
    *np2 = cur2;
    *np3 = cur3;
}

double work_load_efficiency_fock(
    const int Nk, const int Ns, const int *gridsizes,
    const int np, const int np1, const int np2, const int np3
)
{
    int ncols = workload_assigned(Nk, Ns, np1, np2);
    double eff = Nk * Ns * np3 / (double)(ncols * np);
    eff *= scaling_DMndbyNband(Ns, gridsizes, np2, np3);
    return eff;
}

void dims_divide_kbd_fock(
    const int Nk, const int Ns, const int *gridsizes,
    const int np, int *np1, int *np2, int *np3, const int minsize)
{
#define LEN_NPB 20

    // SPARC_Dims_create(nproc, ndims, gridsizes, minsize, int *dims, int *ierr); 
    int cur1 = std::min(np,Nk);
    int npNdx_max = gridsizes[0]/minsize;
    int npNdy_max = gridsizes[1]/minsize;
    int npNdz_max = gridsizes[2]/minsize;
    int npd_max = npNdx_max * npNdy_max * npNdz_max;
    int cur3 = std::min(npd_max,np/cur1);
    int cur2 = std::min(np/(cur1*cur3),Ns);
    double best_weight = work_load_efficiency_fock(Nk,Ns,gridsizes,np,cur1,cur2,cur3);
    
    if (best_weight > 0.9) { // prefer kpt to domain to band
        *np1 = cur1; *np2 = cur2; *np3 = cur3;
        return;
    }

    int npkpt, npband, npdm;
    double weight;

    int npb_list[LEN_NPB];
    int npb_count = LEN_NPB;
    if (np > Nk * npd_max) {
        int npb_base = np / (Nk * npd_max);
        int count = 0;
        for (int i = 1; i <= LEN_NPB; i++) {
            if (npb_base * i <= np) {
                npb_list[count] = npb_base * i;
                count++;
            }
        }
        npb_count = count;
    } else {
        int count = 0;
        for (int i = 1; i <= LEN_NPB; i++) {
            npb_list[count] = i;
            count++;
        }
        npb_count = count;
    }

    for (int i = 0; i < npb_count; i++) {
        npband = npb_list[i];
        int np_2d = np / npband;
        if (np_2d < 1) continue;
        
        // Use as much npkpt as possible for fock
        npkpt = std::min(np_2d,Nk);
        npdm = np_2d / npkpt;
        npband = np / (npkpt * npdm);
        weight = work_load_efficiency_fock(Nk,Ns,gridsizes,np,npkpt,npband,npdm);
        if (weight > best_weight) {
            cur1 = npkpt;
            cur2 = npband;
            cur3 = npdm; 
            best_weight = weight;
        }
        if (weight > 0.93) break;
    }

    *np1 = cur1;
    *np2 = cur2;
    *np3 = cur3;
}

/**
 * @brief  Caluclate a division of processors in for spin, kpoint, band, 
 *         domain (SKBD). 
 *         parallelization.
 * 
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/parallelization.c#L1857
 * @param Nspin     Number of spin, 1 or 2.
 * @param Nk        Number of kpoints (after symmetry reduction).
 * @param Ns        Number of states.
 * @param gridsizes Number of grid points in all three directions.
 * @param np        Number of processors available.
 * @param nps (OUT) Number of spin groups.
 * @param npk (OUT) Number of kpoint groups.
 * @param npp (OUT) Number of band groups.
 * @param npd (OUT) Number of domain groups.
 * @param minsize   Minimum size in domain parallelization
 * @param isfock    Flag for if it's hybrid calculation
 **/
void Parallel_vertices::dims_divide_skbd(const int Nspin, const int Nk, const int Ns, const int *gridsizes, const int np, 
                                         int *nps, int *npk, int *npb, int *npd, int minsize, int isfock)
{
    if (Nspin != 1 && Nspin != 2) 
        exit(1);

    int nps_cur, npk_cur, npb_cur, npd_cur;
    double weight_cur;
    // first set naive way
    if (!isfock) {
        nps_cur = std::min(Nspin,np);
        npk_cur = std::min(Nk,np/nps_cur);
        npb_cur = std::min(Ns,np/(nps_cur*npk_cur));
        npd_cur = np/(nps_cur*npk_cur*npb_cur);
    } else {
        nps_cur = std::min(Nspin,np);
        npk_cur = std::min(Nk,np/nps_cur);
        int npNdx_max = gridsizes[0]/minsize;
        int npNdy_max = gridsizes[1]/minsize;
        int npNdz_max = gridsizes[2]/minsize;
        int npd_max = npNdx_max * npNdy_max * npNdz_max;
        npd_cur = std::min(npd_max,np/(nps_cur*npk_cur));
        npb_cur = np/(nps_cur*npk_cur*npd_cur);
    }
    weight_cur = skbd_weighted_efficiency(Nspin,Nk,Ns,gridsizes,np,nps_cur,npk_cur,npb_cur,npd_cur,isfock);


    // try with both Nspin = 1 
    int nps_1, npk_1, npb_1, npd_1;
    nps_1 = 1;
    if (!isfock) {
        // dims_divide_kbd(Nk,Ns,gridsizes,np/(nps_1),&npk_1,&npb_1,&npd_1);
        dims_divide_kbd(Nk,Ns,np/(nps_1),&npk_1,&npb_1,&npd_1);
    } else {
        dims_divide_kbd_fock(Nk,Ns,gridsizes,np/(nps_1),&npk_1,&npb_1,&npd_1,minsize);
    }
    // double weight_1 = work_load_efficiency(Nspin*Nk,Ns,np,nps_1*npk_1,npb_1,npd_1);
    double weight_1;
    weight_1 = skbd_weighted_efficiency(Nspin,Nk,Ns,gridsizes,np,nps_1,npk_1,npb_1,npd_1,isfock);
    if (weight_1 > weight_cur) {
        nps_cur = nps_1;
        npk_cur = npk_1;
        npb_cur = npb_1;
        npd_cur = npd_1;
        weight_cur = weight_1;
    }

    // if there's spin, also try npspin = 2, and see if gives a better result
    if (Nspin > 1 && np > 1) {
        int nps_2, npk_2, npb_2, npd_2;
        nps_2 = 2;
        if (!isfock) {
            // dims_divide_kbd(Nk,Ns,gridsizes,np/(nps_2),&npk_2,&npb_2,&npd_2);
            dims_divide_kbd(Nk,Ns,np/(nps_2),&npk_2,&npb_2,&npd_2);
        } else {
            dims_divide_kbd_fock(Nk,Ns,gridsizes,np/(nps_2),&npk_2,&npb_2,&npd_2,minsize);
        }
        double weight_2;
        weight_2 = skbd_weighted_efficiency(Nspin,Nk,Ns,gridsizes,np,nps_2,npk_2,npb_2,npd_2,isfock);
        if (weight_2 >= weight_cur) {
            nps_cur = nps_2;
            npk_cur = npk_2;
            npb_cur = npb_2;
            npd_cur = npd_2;
            weight_cur = weight_2;
        }
    }

    *nps = nps_cur;
    *npk = npk_cur;
    *npb = npb_cur;
    *npd = npd_cur;
}


/**
 * @brief   Creates a balanced division of processors/subset of processors in a
 *          Cartesian grid according to application size.
 *
 * @ref https://github.com/SPARC-X/SPARC/blob/1fa3d2d332dfe64600ed3734db1903dd1edd4166/src/parallelization.c#L1217
 */
void Parallel_vertices::SPARC_Dims_create(int nproc, int ndims, int *gridsizes, int minsize, int *dims, int *ierr) {
#define NPSTRIDE 5
#define NPSTRIDE3D 2
    *ierr = 0;
    minsize = std::max(minsize,0); // minsize should be greater than or equal to 0
    if (ndims == 1) {
        *dims = std::min(nproc, *gridsizes / minsize);
        *dims = std::max(*dims,1);
    } else if (ndims == 2) {
        double r, rnormi[(NPSTRIDE*2+1)*2],  min_rnorm, tmp1, tmp2;
        int i, j, count, best_ind, max_npi, tmpint1, tmpint2, dimx[(NPSTRIDE*2+1)*2], dimy[(NPSTRIDE*2+1)*2], npi[(NPSTRIDE*2+1)*2];
        r = sqrt((double)nproc) / sqrt( (double)gridsizes[0] * (double)gridsizes[1] );
        r = std::min(r, 1/(double)minsize);
        double r2 = r * r;

        // initial estimate
        dims[0] = round(gridsizes[0] * r);
        dims[1] = round(gridsizes[1] * r);
        count = 0;
        for (i = 0; i < (NPSTRIDE*2+1); i++) {
            tmpint1 = dims[0] + i - NPSTRIDE;
            dimx[count] = (tmpint1 > 0) ? (tmpint1 <= nproc ? tmpint1 : (tmpint1-(NPSTRIDE*2+1))) : (tmpint1+(NPSTRIDE*2+1));
            // dimx[count] > 0 ? (dimx[count] <= nproc ? : (dimx[count] = 1)) : (dimx[count] = 1);
            if (dimx[count] > 0 && dimx[count] <= nproc) {
                SPARC_Dims_create(std::max(1,nproc/dimx[count]), 1, &gridsizes[1], minsize, &dimy[count], ierr);
                npi[count] = dimx[count] * dimy[count];
                tmp1 = ((double) gridsizes[0]) / dimx[count];
                tmp2 = ((double) gridsizes[1]) / dimy[count];
                rnormi[count] = (tmp1 - tmp2) * (tmp1 - tmp2);
                rnormi[count] *= r2;
            } else {
                dimx[count] = 0; dimy[count] = 0; npi[count] = 0;
                rnormi[count] = 0;
            }
            count++;
        }
        for (j = 0; j < (NPSTRIDE*2+1); j++) {
            tmpint2 = dims[1] + j - NPSTRIDE;
            dimy[count] = (tmpint2 > 0) ? (tmpint2 <= nproc ? tmpint2 : (tmpint2-(NPSTRIDE*2+1))) : (tmpint2+(NPSTRIDE*2+1));
            //dimy[count] > 0 ? (dimy[count] <= nproc ? : (dimy[count] = 1)) : (dimy[count] = 1);
            if (dimy[count] > 0 && dimy[count] <= nproc) {
                SPARC_Dims_create(std::max(1,nproc/dimy[count]), 1, &gridsizes[0], minsize, &dimx[count], ierr);
                npi[count] = dimx[count] * dimy[count];
                tmp1 = ((double) gridsizes[0]) / dimx[count];
                tmp2 = ((double) gridsizes[1]) / dimy[count];
                rnormi[count] = (tmp1 - tmp2) * (tmp1 - tmp2);
                rnormi[count] *= r2;
            } else {
                dimx[count] = 0; dimy[count] = 0; npi[count] = 0;
                rnormi[count] = 0;
            }
            count++;
        }

        // check which one uses largest number of processes provided
        max_npi = 0;
        count = 0; best_ind = -1; min_rnorm = 1e4;
        for (i = 0; i < (NPSTRIDE*2+1)*2; i++) {
            if (npi[count] < max_npi || npi[count] > nproc || npi[count] <= 0) {
                count++;
                continue;
            }
            if (npi[count] > max_npi && gridsizes[0]/dimx[count] >= minsize && gridsizes[1]/dimy[count] >= minsize) {
                best_ind = count;
                max_npi = npi[count];
                min_rnorm = rnormi[count];
            } else if (npi[count] == max_npi && rnormi[count] < min_rnorm) {
                best_ind = count;
                min_rnorm = rnormi[count];
            }
            count++;
        }

        // TODO: after first scan, perhaps we can allow np to be up to 3% smaller, and choose the one with smaller rnormi,
        // the idea is that by reducing total number of process we lose speed by 3% or less, but we might gain speed up in
        // communication by more than 3%
        if (best_ind != -1) {
            dims[0] = dimx[best_ind]; dims[1] = dimy[best_ind];
        } else {
            dims[0] = nproc; dims[1] = 1;;
            *ierr = 1; // cannot find any admissable distribution
        }
    } else if (ndims == 3) {
        double r, rnormi[(NPSTRIDE3D*2+1)*3], min_rnorm, tmp1, tmp2, tmp3;
        int i, j, k, count, best_ind, max_npi, tmpint1, tmpint2, tmpint3, dims_temp[2], gridsizes_temp[2];
        int dimx[(NPSTRIDE3D*2+1)*3], dimy[(NPSTRIDE3D*2+1)*3], dimz[(NPSTRIDE3D*2+1)*3], npi[(NPSTRIDE3D*2+1)*3];
        r = cbrt((double)nproc) / cbrt( (double)gridsizes[0] * (double)gridsizes[1] * (double)gridsizes[2] );
        r = std::min(r, 1/(double)minsize);
        double r2 = r * r;

        // initial estimate
        dims[0] = round(gridsizes[0] * r);
        dims[1] = round(gridsizes[1] * r);
        dims[2] = round(gridsizes[2] * r);
        count = 0;
        for (i = 0; i < (NPSTRIDE3D*2+1); i++) {
            tmpint1 = dims[0] + i - NPSTRIDE3D;
            dimx[count] = (tmpint1 > 0) ? (tmpint1 <= nproc ? tmpint1 : (tmpint1-(NPSTRIDE3D*2+1))) : (tmpint1+(NPSTRIDE3D*2+1));
            // dimx[count] > 0 ? (dimx[count] <= nproc ? : (dimx[count] = 1)) : (dimx[count] = 1);
            if (dimx[count] > 0 && dimx[count] <= nproc) {
                SPARC_Dims_create(std::max(1,nproc/dimx[count]), 2, &gridsizes[1], minsize, dims_temp, ierr);
                dimy[count] = dims_temp[0];
                dimz[count] = dims_temp[1];
                npi[count] = dimx[count] * dimy[count] * dimz[count];
                tmp1 = ((double) gridsizes[0]) / dimx[count];
                tmp2 = ((double) gridsizes[1]) / dimy[count];
                tmp3 = ((double) gridsizes[2]) / dimz[count];
                rnormi[count] = (tmp1 - tmp2) * (tmp1 - tmp2) + (tmp1 - tmp3) * (tmp1 - tmp3) + (tmp2 - tmp3) * (tmp2 - tmp3);
                rnormi[count] *= r2;
            } else {
                dimx[count] = 0; dimy[count] = 0; dimz[count] = 0;
                npi[count] = 0;
                rnormi[count] = 0;
            }

            count++;
        }

        gridsizes_temp[0] = gridsizes[0];
        gridsizes_temp[1] = gridsizes[2];
        for (j = 0; j < (NPSTRIDE3D*2+1); j++) {
            tmpint2 = dims[1] + j - NPSTRIDE3D;
            dimy[count] = (tmpint2 > 0) ? (tmpint2 <= nproc ? tmpint2 : (tmpint2-(NPSTRIDE3D*2+1))) : (tmpint2+(NPSTRIDE3D*2+1));
           // dimy[count] > 0 ? (dimy[count] <= nproc ? : (dimy[count] = 1)) : (dimy[count] = 1);
            if (dimy[count] > 0 && dimy[count] <= nproc) {
                SPARC_Dims_create(std::max(1,nproc/dimy[count]), 2, gridsizes_temp, minsize, dims_temp, ierr);
                dimx[count] = dims_temp[0];
                dimz[count] = dims_temp[1];
                npi[count] = dimx[count] * dimy[count] * dimz[count];
                tmp1 = ((double) gridsizes[0]) / dimx[count];
                tmp2 = ((double) gridsizes[1]) / dimy[count];
                tmp3 = ((double) gridsizes[2]) / dimz[count];
                rnormi[count] = (tmp1 - tmp2) * (tmp1 - tmp2) + (tmp1 - tmp3) * (tmp1 - tmp3) + (tmp2 - tmp3) * (tmp2 - tmp3);
                rnormi[count] *= r2;
            } else {
                dimx[count] = 0; dimy[count] = 0; dimz[count] = 0;
                npi[count] = 0;
                rnormi[count] = 0;
            }
            count++;
        }

        for (k = 0; k < (NPSTRIDE3D*2+1); k++) {
            tmpint3 = dims[2] + k - NPSTRIDE3D;
            dimz[count] = (tmpint3 > 0) ? (tmpint3 <= nproc ? tmpint3 : (tmpint3-(NPSTRIDE3D*2+1))) : (tmpint3+(NPSTRIDE3D*2+1));
            // for these cases, one can actually skip the calculation
            // dimz[count] > 0 ? (dimz[count] <= nproc ? : (dimz[count] = 1)) : (dimz[count] = 1);
            if (dimz[count] > 0 && dimz[count] <= nproc) {
                SPARC_Dims_create(std::max(1,nproc/dimz[count]), 2, &gridsizes[0], minsize, dims_temp, ierr);
                dimx[count] = dims_temp[0];
                dimy[count] = dims_temp[1];
                npi[count] = dimx[count] * dimy[count] * dimz[count];
                tmp1 = ((double) gridsizes[0]) / dimx[count];
                tmp2 = ((double) gridsizes[1]) / dimy[count];
                tmp3 = ((double) gridsizes[2]) / dimz[count];
                rnormi[count] = (tmp1 - tmp2) * (tmp1 - tmp2) + (tmp1 - tmp3) * (tmp1 - tmp3) + (tmp2 - tmp3) * (tmp2 - tmp3);
                rnormi[count] *= r2;
            } else {
                dimx[count] = 0; dimy[count] = 0; dimz[count] = 0;
                npi[count] = 0;
                rnormi[count] = 0;
            }
            count++;
        }

        // check which one uses largest number of processes provided
        max_npi = 0;
        count = 0; best_ind = -1; min_rnorm = 1e4;
        for (i = 0; i < (NPSTRIDE3D*2+1)*3; i++) {
            if (npi[count] < max_npi || npi[count] > nproc || npi[count] <= 0) {
                count++;
                continue;
            }
            if (npi[count] > max_npi && gridsizes[0]/dimx[count] >= minsize && gridsizes[1]/dimy[count] >= minsize  && gridsizes[2]/dimz[count] >= minsize) {
                best_ind = count;
                max_npi = npi[count];
                min_rnorm = rnormi[count];
            } else if (npi[count] == max_npi && rnormi[count] < min_rnorm) {
                best_ind = count;
                min_rnorm = rnormi[count];
            }
            count++;
        }
        // TODO: after first scan, perhaps we can allow np to be up to 3% smaller, and choose the one with smaller rnormi,
        // the idea is that by reducing total number of process we lose speed by 3% or less, but we might gain speed up in
        // communication by more than 3%
        if (best_ind != -1) {
            dims[0] = dimx[best_ind]; dims[1] = dimy[best_ind]; dims[2] = dimz[best_ind];
        } else {
            dims[0] = nproc; dims[1] = 1; dims[2] = 1;
            *ierr = 1; // cannot find any admissable distribution
        }
    } else {
        *ierr = 1;
        exit(EXIT_FAILURE); // currently only works for 1d, 2d and 3d
    }
#undef NPSTRIDE
#undef NPSTRIDE3D
}

std::array<uint,4> Parallel_vertices::cal_comm_ns_4D_by_sparc(const Vertices_4D& shared_vertices,
                                                    const MPI_Comm& comm, const uint& max_size) {
    std::array<uint, 4> result = {0, 0, 0, 0};
    int size;
    MPI_Comm_size(comm, &size);
    int comm_size = (max_size == 0)
                  ? size
                  : (max_size < (uint)size)
                  ? (int)max_size
                  : size;
    int Nspin = 1;
    int Nk = 1;
    int Ns = shared_vertices.get_nb();
    int gridsizes[3] = {(int)shared_vertices.get_ni(),
                        (int)shared_vertices.get_nj(),
                        (int)shared_vertices.get_nk()};
    int np = comm_size;
    int nps;
    int npk;
    int npb;
    int npd;
    int minsize = 3;
    int isfock = 0;
    Parallel_vertices::dims_divide_skbd(Nspin, Nk, Ns, gridsizes, np, 
                                        &nps, &npk, &npb, &npd, minsize, isfock);
    int subsize = comm_size/npb;
    int nproc = subsize;
    int ndims = 3;
    int ierr;
    Parallel_vertices::SPARC_Dims_create(nproc, ndims, gridsizes, minsize, (int*)result.data(), &ierr);
    result[3] = npb;
    assert(result[0] * result[1] * result[2] * result[3] <= (uint)comm_size);
    return result;
}

std::array<uint,3> Parallel_vertices::cal_comm_ns_3D_by_sparc(const Vertices_3D& shared_vertices,
                                                    const MPI_Comm& comm, const uint& max_size) {
    std::array<uint, 3> result = {0, 0, 0};
    int size;
    MPI_Comm_size(comm, &size);
    int comm_size = (max_size == 0)
                  ? size
                  : (max_size < (uint)size)
                  ? (int)max_size
                  : size;
    int gridsizes[3] = {(int)shared_vertices.get_ni(),
                        (int)shared_vertices.get_nj(),
                        (int)shared_vertices.get_nk()};
    int nproc = comm_size;
    int ndims = 3;
    int minsize = 3;
    int ierr;
    Parallel_vertices::SPARC_Dims_create(nproc, ndims, gridsizes, minsize, (int*)result.data(), &ierr);
    assert(result[0] * result[1] * result[2] <= (uint)comm_size);
    return result;
}

bool Parallel_vertices::is_comm_equal(const MPI_Comm& comm1, const MPI_Comm& comm2) {
    if (comm1 == MPI_COMM_NULL && comm2 == MPI_COMM_NULL) {
        return true;
    } else {
        int size1;
        int rank1;
        MPI_Comm_size(comm1, &size1);
        MPI_Comm_rank(comm1, &rank1);
        int size2;
        int rank2;
        MPI_Comm_size(comm2, &size2);
        MPI_Comm_rank(comm2, &rank2);
        bool flag = (size1 == size2) && (rank1 == rank2);
        MPI_Allreduce(MPI_IN_PLACE, &flag, 1, MPI_C_BOOL, MPI_LAND, comm1);
        return flag;
    }
}

template<typename T>
void Parallel_vertices::cal_gradient_d3(T const* const& arr, const Vertices_3D& vertices,
                                        const Stencil<T>& stencil, const Exarr_3D_mpi_package& exarr_mpi_package,
                                        T* const& darr_x, T* const& darr_y, T* const& darr_z) {
    #ifdef USE_OPENMP

    Vertices_3D ex_vertice(vertices.generate_ex_vertices(stencil.FDn));
    static T* ex_arr_static = nullptr;
    #pragma omp single
    ex_arr_static = new T [ex_vertice.get_size()]();
    T* const ex_arr = ex_arr_static;
    exarr_mpi_package.fill_domain_par_ex_arr(arr, vertices, ex_arr, ex_vertice);
    #pragma omp barrier
    Stencil_method::calc_gradient(ex_arr, ex_vertice, 0, stencil,
                                    vertices, darr_x, vertices);
    Stencil_method::calc_gradient(ex_arr, ex_vertice, 1, stencil,
                                    vertices, darr_y, vertices);
    Stencil_method::calc_gradient(ex_arr, ex_vertice, 2, stencil,
                                    vertices, darr_z, vertices);
    #pragma omp barrier
    #pragma omp single
    {
        delete [] ex_arr_static;
        ex_arr_static = nullptr;
    }

    #else

    Vertices_3D ex_vertice(vertices.generate_ex_vertices(stencil.FDn));
    T* ex_arr = new T [ex_vertice.get_size()]();
    exarr_mpi_package.fill_domain_par_ex_arr(arr, vertices, ex_arr, ex_vertice);
    Stencil_method::calc_gradient(ex_arr, ex_vertice, 0, stencil,
                                    vertices, darr_x, vertices);
    Stencil_method::calc_gradient(ex_arr, ex_vertice, 1, stencil,
                                    vertices, darr_y, vertices);
    Stencil_method::calc_gradient(ex_arr, ex_vertice, 2, stencil,
                                    vertices, darr_z, vertices);
    delete [] ex_arr;

    #endif // USE_OPENMP
    return;
}
template void Parallel_vertices::cal_gradient_d3<float>(float const* const& arr, const Vertices_3D& vertices,
                                        const Stencil<float>& stencil, const Exarr_3D_mpi_package& exarr_mpi_package,
                                        float* const& darr_x, float* const& darr_y, float* const& darr_z);
template void Parallel_vertices::cal_gradient_d3<double>(double const* const& arr, const Vertices_3D& vertices,
                                        const Stencil<double>& stencil, const Exarr_3D_mpi_package& exarr_mpi_package,
                                        double* const& darr_x, double* const& darr_y, double* const& darr_z);
