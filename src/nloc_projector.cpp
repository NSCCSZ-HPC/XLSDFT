#include "nloc_projector.h"

#include "nloc_opt.h"

#ifdef USE_HBM
#include <hbwmalloc.h>

namespace {

void nloc_index_hbm_free(uint*& data, uint& length) {
    if (data != nullptr) {
        hbw_free(data);
        data = nullptr;
    }
    length = 0;
}

void nloc_index_hbm_alloc(uint*& data, uint& length, const uint nrow,
                          uint const* const src) {
    nloc_index_hbm_free(data, length);
    length = nrow;
    if (nrow == 0) {
        return;
    }
    data = static_cast<uint*>(hbw_malloc(static_cast<size_t>(nrow) * sizeof(uint)));
    if (data == nullptr) {
        length = 0;
        return;
    }
    if (src != nullptr) {
        for (uint i = 0; i < nrow; ++i) {
            data[i] = src[i];
        }
    }
}

}  // namespace
#endif

template<typename T>
uint const* Nloc_projector<T>::index_data() const {
#ifdef USE_HBM
    if (this->index_hbm_length > 0 && this->index_hbm != nullptr) {
        return this->index_hbm;
    }
#endif
    return this->index.data();
}

template<typename T>
Nloc_projector<T>::Nloc_projector(const Psp8_file& psp8_file)
                                : psp8_file(psp8_file) {}

template<typename T>
Nloc_projector<T>::Nloc_projector(const Nloc_projector<T>& nloc_projector)
                                : psp8_file(nloc_projector.psp8_file),
                                  ncol(nloc_projector.ncol),
                                  nrow(nloc_projector.nrow),
                                  atom_index(nloc_projector.atom_index),
                                  is_real(nloc_projector.is_real),
                                  is_in_domain(nloc_projector.is_in_domain),
                                  cell_shift_x(nloc_projector.cell_shift_x),
                                  cell_shift_y(nloc_projector.cell_shift_y),
                                  cell_shift_z(nloc_projector.cell_shift_z),
                                  chi(nloc_projector.chi),
                                  gamma(nloc_projector.gamma),
                                  index(nloc_projector.index)
#ifdef USE_HBM
                                  , index_hbm_length(0)
                                  , index_hbm(nullptr)
#endif
                                  {
#ifdef USE_HBM
    if (nloc_projector.index_hbm_length > 0) {
        nloc_index_hbm_alloc(this->index_hbm, this->index_hbm_length, this->nrow,
                             nloc_projector.index_hbm);
    }
#endif
}

template<typename T>
template<typename T2>
Nloc_projector<T>::Nloc_projector(const Nloc_projector<T2>& nloc_projector)
                                : psp8_file(nloc_projector.psp8_file) {
    *this = nloc_projector;
}
template Nloc_projector<float>::Nloc_projector(const Nloc_projector<double>& nloc_projector);
template Nloc_projector<double>::Nloc_projector(const Nloc_projector<float>& nloc_projector);

template<typename T>
Nloc_projector<T>::Nloc_projector(Nloc_projector<T>&& nloc_projector) 
                                : psp8_file(nloc_projector.psp8_file),
                                  ncol(nloc_projector.ncol),
                                  nrow(nloc_projector.nrow),
                                  atom_index(nloc_projector.atom_index),
                                  is_real(nloc_projector.is_real),
                                  is_in_domain(nloc_projector.is_in_domain),
                                  cell_shift_x(nloc_projector.cell_shift_x),
                                  cell_shift_y(nloc_projector.cell_shift_y),
                                  cell_shift_z(nloc_projector.cell_shift_z),
                                  chi(std::move(nloc_projector.chi)),
                                  gamma(std::move(nloc_projector.gamma)),
                                  index(std::move(nloc_projector.index))
#ifdef USE_HBM
                                  , index_hbm_length(nloc_projector.index_hbm_length)
                                  , index_hbm(nloc_projector.index_hbm)
#endif
                                  {
#ifdef USE_HBM
    nloc_projector.index_hbm = nullptr;
    nloc_projector.index_hbm_length = 0;
#endif
}

template<typename T>
Nloc_projector<T>::~Nloc_projector() {
    this->destructor();
}

template<typename T>
void Nloc_projector<T>::generate_ncol() {
    this->ncol = (uint)this->psp8_file.generate_ncol();
    return;
}

template<typename T>
void Nloc_projector<T>::set_nrow(const uint& nrow) {
    this->nrow = nrow;
    return;
}

template<typename T>
void Nloc_projector<T>::resize() {
#ifdef USE_HBM
    this->chi.reconstructor_hbm(this->nrow, this->ncol);
    nloc_index_hbm_alloc(this->index_hbm, this->index_hbm_length, this->nrow,
                         this->index.data());
    std::vector<uint>().swap(this->index);
#else
    this->chi.reconstructor(this->nrow, this->ncol);
    this->index.resize(this->nrow);
#endif
    return;
}

template<typename T>
Nloc_projector<T>& Nloc_projector<T>::operator=(const Nloc_projector<T>& other) {
    this->ncol = other.ncol;
    this->nrow = other.nrow;
    this->atom_index = other.atom_index;
    this->is_real = other.is_real;
    this->is_in_domain = other.is_in_domain;
    this->cell_shift_x = other.cell_shift_x;
    this->cell_shift_y = other.cell_shift_y;
    this->cell_shift_z = other.cell_shift_z;
    this->chi.deepcopy(other.chi);
    this->gamma.deepcopy(other.gamma);
#ifdef USE_HBM
    if (other.nrow > 0) {
        nloc_index_hbm_alloc(this->index_hbm, this->index_hbm_length, other.nrow,
                             other.index_data());
        std::vector<uint>().swap(this->index);
    } else {
        nloc_index_hbm_free(this->index_hbm, this->index_hbm_length);
        this->index = other.index;
    }
#else
    this->index = other.index;
#endif
    return *this;
}

template<typename T>
template<typename T2>
Nloc_projector<T>& Nloc_projector<T>::operator=(const Nloc_projector<T2>& other) {
    this->ncol = other.ncol;
    this->nrow = other.nrow;
    this->atom_index = other.atom_index;
    this->is_real = other.is_real;
    this->is_in_domain = other.is_in_domain;
    this->cell_shift_x = other.cell_shift_x;
    this->cell_shift_y = other.cell_shift_y;
    this->cell_shift_z = other.cell_shift_z;
    this->chi.deepcopy(std::move(other.chi.as_type(this->chi.data)));
    this->gamma.deepcopy(std::move(other.gamma.as_type(this->gamma.data)));
#ifdef USE_HBM
    if (other.nrow > 0) {
        nloc_index_hbm_alloc(this->index_hbm, this->index_hbm_length, other.nrow,
                             other.index_data());
        std::vector<uint>().swap(this->index);
    } else {
        nloc_index_hbm_free(this->index_hbm, this->index_hbm_length);
        this->index = other.index;
    }
#else
    this->index = other.index;
#endif
    return *this;
}
template Nloc_projector<float>& Nloc_projector<float>::operator=(const Nloc_projector<double>& other);
template Nloc_projector<double>& Nloc_projector<double>::operator=(const Nloc_projector<float>& other);

template<typename T>
void Nloc_projector<T>::copy_mp(const Nloc_projector<T>& other,
                                Memory_pool<T, Fast_memory>& pool_fast,
                                Memory_pool<T, Capacity_memory>& pool_cap) {
    (void) pool_cap; 
    this->ncol = other.ncol;
    this->nrow = other.nrow;
    this->atom_index = other.atom_index;
    this->is_real = other.is_real;
    this->is_in_domain = other.is_in_domain;
    this->cell_shift_x = other.cell_shift_x;
    this->cell_shift_y = other.cell_shift_y;
    this->cell_shift_z = other.cell_shift_z;
    const size_t chi_elems =
        static_cast<size_t>(this->ncol) * static_cast<size_t>(this->nrow);
    if (chi_elems > 0) {
        this->chi.data = pool_fast.allocate(chi_elems);
        Linalg::set_value_general(this->chi.data, other.chi.data, chi_elems);
    } else {
        this->chi.data = nullptr;
    }
    if (this->ncol > 0) {
        this->gamma.data = pool_fast.allocate(this->ncol);
        Linalg::set_value_general(this->gamma.data, other.gamma.data, this->ncol);
    } else {
        this->gamma.data = nullptr;
    }
    if (other.nrow > 0) {
#ifdef USE_HBM
        this->index_hbm = pool_alloc<uint32_t>(pool_fast, other.nrow);
        this->index_hbm_length = other.nrow;
        std::memcpy(this->index_hbm, other.index_data(),
                    static_cast<size_t>(other.nrow) * sizeof(uint32_t));
        std::vector<uint>().swap(this->index);
#else
        this->index.assign(other.index_data(), other.index_data() + other.nrow);
#endif
    } else {
#ifdef USE_HBM
        nloc_index_hbm_free(this->index_hbm, this->index_hbm_length);
#endif
        this->index = other.index;
    }
    return;
}

template<typename T>
void Nloc_projector<T>::init() {
    this->generate_basic();
    return;
}

template<typename T>
template<typename T2>
void Nloc_projector<T>::init(const Nloc_projector<T2>& other) {
    this->operator=(other);
    return;
}
template void Nloc_projector<float>::init(const Nloc_projector<double>& other);
template void Nloc_projector<double>::init(const Nloc_projector<float>& other);
template void Nloc_projector<float>::init(const Nloc_projector<float>& other);
template void Nloc_projector<double>::init(const Nloc_projector<double>& other);

template<typename T>
void Nloc_projector<T>::generate_basic() {
    this->generate_ncol();
#ifdef USE_HBM
    this->gamma.reconstructor_hbm(this->ncol);
#else
    this->gamma.reconstructor(this->ncol);
#endif
    uint count = 0;
    for (uint l = 0; l < 5; l++) {
        for (uint i = 0; i < this->psp8_file.nproj[l]; i++) {
            for (int m = -(int)l; m <= (int) l; m++) {
                this->gamma[count] = T(this->psp8_file.ekb[l][i]);
                count++;
            }
        }
    }
    return;
}

template<typename T>
void Nloc_projector<T>::destructor() {
#ifdef USE_HBM
    this->chi.destructor_hbm();
    this->gamma.destructor_hbm();
    nloc_index_hbm_free(this->index_hbm, this->index_hbm_length);
#else
    this->chi.destructor();
    this->gamma.destructor();
#endif
    std::vector<uint>().swap(this->index);
    return;
}

template<typename T>
void Nloc_projector<T>::destructor_mp() {
    this->chi.data = nullptr;
    this->gamma.data = nullptr;
#ifdef USE_HBM
    this->index_hbm = nullptr;
    this->index_hbm_length = 0;
#endif
    std::vector<uint>().swap(this->index);
    return;
}

template<typename T>
void Nloc_projector<T>::show(std::ostream& output) const {
    output << "ncol = " << this->ncol << std::endl;
    output << "nrow = " << this->nrow << std::endl;
    output << "atom_index = " << this->atom_index << std::endl;
    output << "is_real = " << this->is_real << std::endl;
    output << "is_in_domain = " << this->is_in_domain << std::endl;
    output << "cell_shift_x,y,z = " << cell_shift_x << cell_shift_y << cell_shift_z << this->nrow << std::endl;
    for (uint icol = 0; icol < this->ncol; icol++) {
        output << "for " << icol << " icol:" << std::endl;
        output << "gamma = " << this->gamma[icol] << std::endl;
        output << "sum(chi) = " << std::fixed << std::setprecision(15) << Linalg::vector_sum(this->chi.data + icol * this->nrow, this->nrow) << std::endl;
    }
    return;
}

template class Nloc_projector<float>;
template class Nloc_projector<double>;

template<typename T> Nloc_projector<T> Nloc_projector_method::merge(const std::vector<Nloc_projector<T>>& nloc_projectors,
                                                                    const Psp8_file& psp8_file) {
    Nloc_projector<T> result(psp8_file);
    result.init();
    result.nrow = 0;
    int atom_index = nloc_projectors.begin()->atom_index;
    result.atom_index = atom_index;
    for (typename std::vector<Nloc_projector<T>>::const_iterator it = nloc_projectors.begin(); it != nloc_projectors.end(); ++it) {
        result.nrow += it->nrow;
        assert(atom_index == it->atom_index);
    }
    result.chi.reconstructor(result.nrow, result.ncol);
    result.index.reserve(result.nrow);
    int irow = 0; 
    for (typename std::vector<Nloc_projector<T>>::const_iterator it = nloc_projectors.begin(); it != nloc_projectors.end(); ++it) {
        result.index.insert(result.index.end(), it->index_data(), it->index_data() + it->nrow);
        uint nrow = it->nrow;
        Vertices_2D be_filled_region(irow, irow + nrow - 1, 0, result.ncol - 1);
        Vertices_method::be_filled_vector(result.chi.data, result.chi.get_vertices(),
                                          it->chi.data, it->chi.get_vertices(), be_filled_region, 0);
        irow += nrow;
    }
    return result;
}
template Nloc_projector<float> Nloc_projector_method::merge<float>(const std::vector<Nloc_projector<float>>& nloc_projectors,
                                                                   const Psp8_file& psp8_file);
template Nloc_projector<double> Nloc_projector_method::merge<double>(const std::vector<Nloc_projector<double>>& nloc_projectors,
                                                                     const Psp8_file& psp8_file);

template<typename T>
void Nloc_projector_method::nloc_projector_product_vectors(T* const& __restrict__ result, T const* const& __restrict__ vectors,
                                                           const Nloc_projector<T>& nloc_projector, const T& dv) {
    const uint& nrow = nloc_projector.nrow;
    if (nrow * nloc_projector.ncol == 0) return;
    #ifdef USE_OPENMP

    static T* filtered_vectors_static = nullptr;
    #pragma omp single
    filtered_vectors_static = new T [nrow];
    T* const __restrict__ filtered_vectors = filtered_vectors_static;
    uint const* const& __restrict__ Nloc_projector_index_data= nloc_projector.index_data();
    #ifdef USE_OPENMP_SIMD
    #pragma omp for simd schedule(static, (nrow - 1)/omp_get_num_threads() + 1)
    #else
    #pragma omp for schedule(static, (nrow - 1)/omp_get_num_threads() + 1)
    #endif
    for (uint i = 0; i < nrow; i++) {
        filtered_vectors[i] = vectors[Nloc_projector_index_data[i]];
    }
    Linalg::matrix_vector_product(nloc_projector.chi.data, 0, filtered_vectors, result, nloc_projector.ncol, nrow, dv, (T)1.0);
    #pragma omp barrier
    #pragma omp single
    {
        delete [] filtered_vectors_static;
        filtered_vectors_static = nullptr;
    }

    #else //USE_OPENMP

    T* const __restrict__ filtered_vectors = new T [nrow];
    uint const* __restrict__ Nloc_projector_index_data = nloc_projector.index_data();
    T* __restrict__ filtered_vectors_0 = filtered_vectors;
    #ifdef USE_OPENMP_SIMD
    #pragma omp simd
    #endif
    for (uint i = 0; i < nrow; i++) {
        *filtered_vectors_0++ = vectors[*Nloc_projector_index_data++];
    }
    T const* const nonrestricted_filtered_vectors = filtered_vectors;
    Linalg::matrix_vector_product(nloc_projector.chi.data, 0, nonrestricted_filtered_vectors, result, nloc_projector.ncol, nrow, dv, (T)1.0);
    delete [] filtered_vectors;

    #endif //USE_OPENMP
    return;
}
template void Nloc_projector_method::nloc_projector_product_vectors<float>(float* const& result, float const* const& vectors,
                                                                           const Nloc_projector<float>& nloc_projector, const float& dv);
template void Nloc_projector_method::nloc_projector_product_vectors<double>(double* const& result, double const* const& vectors,
                                                                            const Nloc_projector<double>& nloc_projector, const double& dv);

template<typename T>
void Nloc_projector_method::nloc_projector_product_vectors(T* const& __restrict__ result, T const* const& __restrict__ vectors,
                                                           const Vertices_4D& vertices, const Nloc_projector<T>& nloc_projector,
                                                           const T& dv) {
    #ifdef USE_OPENMP

    const uint& nrow = nloc_projector.nrow;
    if (nrow * nloc_projector.ncol == 0) return;

    const uint& ncol = vertices.nb;
    uint Nd = nrow * ncol;
    static T* filtered_vectors_static = nullptr;
    #pragma omp single
    filtered_vectors_static = new T [Nd];
    T* const __restrict__ filtered_vectors = filtered_vectors_static;
    uint const* const& __restrict__ Nloc_projector_index_data= nloc_projector.index_data();
    const uint vertices_3d_size = vertices.Vertices_3D::get_size();
    #pragma omp for schedule(static, (ncol - 1)/omp_get_num_threads() + 1)
    for (uint icol = 0; icol < ncol; icol++) {
        T const* const vectors_data_icol = vectors + icol * vertices_3d_size;
        T* __restrict__ filtered_vectors_data_icol = filtered_vectors + icol * nrow;
        uint const* __restrict__ Nloc_projector_index_data_icol = Nloc_projector_index_data;
        #ifdef USE_OPENMP_SIMD
        #pragma omp simd
        #endif
        for (uint irow = 0; irow < nrow; irow++) {
            *filtered_vectors_data_icol++ = vectors_data_icol[*Nloc_projector_index_data_icol++];
        }
    }
    Linalg::matrix_product(nloc_projector.chi.data, 0, filtered_vectors, 1, result, 1, nloc_projector.ncol, ncol, nrow, dv, (T)1.0);
    // Linalg::__cblas_gemm(CblasColMajor, CblasTrans, CblasNoTrans, nloc_projector.ncol, ncol, nrow,
                                //  dv, nloc_projector.chi.data, nrow,
                                //  filtered_vectors, nrow, (T)0.0, result, nloc_projector.ncol);
    #pragma omp barrier
    #pragma omp single
    {
        delete [] filtered_vectors_static;
        filtered_vectors_static = nullptr;
    }

    #else //USE_OPENMP

    Nloc_projector_method::nloc_projector_product_vectors_sequential(result, vectors, vertices, nloc_projector, dv);

    #endif //USE_OPENMP
    return;
}
template void Nloc_projector_method::nloc_projector_product_vectors(float* const& result, float const* const& vectors, const Vertices_4D& vertices,
                                                                    const Nloc_projector<float>& nloc_projector, const float& dv);
template void Nloc_projector_method::nloc_projector_product_vectors(double* const& result, double const* const& vectors, const Vertices_4D& vertices,
                                                                    const Nloc_projector<double>& nloc_projector, const double& dv);

template<typename T>
void Nloc_projector_method::nloc_projector_product_vectors_sequential(T* const __restrict__ result, T const* const __restrict__ vectors,
                                                           const Vertices_4D& vertices, const Nloc_projector<T>& nloc_projector,
                                                           const T dv) {
    T* const __restrict__ filtered_vectors = new T [nloc_projector.nrow * vertices.nb];
    Nloc_projector_method::nloc_projector_product_vectors_sequential_with_buffer(result, vectors, vertices, nloc_projector, dv, filtered_vectors);
    delete [] filtered_vectors;

    return;
}
template void Nloc_projector_method::nloc_projector_product_vectors_sequential(float* const result, float const* const vectors, const Vertices_4D& vertices,
                                                                                const Nloc_projector<float>& nloc_projector, const float dv);
template void Nloc_projector_method::nloc_projector_product_vectors_sequential(double* const result, double const* const vectors, const Vertices_4D& vertices,
                                                                                const Nloc_projector<double>& nloc_projector, const double dv);

template<typename T>
void Nloc_projector_method::nloc_projector_product_vectors_sequential_with_buffer(T* const __restrict__ result, T const* const __restrict__ vectors,
                                                           const Vertices_4D& vertices, const Nloc_projector<T>& nloc_projector,
                                                           const T dv, T* const buffer) {
    const uint& nrow = nloc_projector.nrow;
    if (nrow * nloc_projector.ncol == 0) return;

    const uint& ncol = vertices.nb;
    T* const __restrict__ filtered_vectors = buffer;
    uint const* const& __restrict__ Nloc_projector_index_data= nloc_projector.index_data();
    const uint vertices_3d_size = vertices.Vertices_3D::get_size();
    for (uint icol = 0; icol < ncol; icol++) {
        T const* const vectors_data_icol = vectors + icol * vertices_3d_size;
        T* __restrict__ filtered_vectors_data_icol = filtered_vectors + icol * nrow;
        uint const* __restrict__ Nloc_projector_index_data_icol = Nloc_projector_index_data;
        #pragma omp simd
        for (uint irow = 0; irow < nrow; irow++) {
            *filtered_vectors_data_icol++ = vectors_data_icol[*Nloc_projector_index_data_icol++];
        }
    }
    // T const* const nonrestricted_filtered_vectors = filtered_vectors;
    // Linalg::matrix_product(nloc_projector.chi.data, 0, nonrestricted_filtered_vectors, 1, result, 1, nloc_projector.ncol, ncol, nrow, dv, (T)1.0);
    #pragma omp parallel if(0)
    Linalg::matrix_product(nloc_projector.chi.data, 0, filtered_vectors, 1, result, 1,
                           nloc_projector.ncol, ncol, nrow, dv, T(1.0));

    return;
}
template void Nloc_projector_method::nloc_projector_product_vectors_sequential_with_buffer(float* const result, float const* const vectors, const Vertices_4D& vertices,
                                                                                const Nloc_projector<float>& nloc_projector, const float dv, float* const buffer);
template void Nloc_projector_method::nloc_projector_product_vectors_sequential_with_buffer(double* const result, double const* const vectors, const Vertices_4D& vertices,
                                                                                const Nloc_projector<double>& nloc_projector, const double dv, double* const buffer);
