#include "atom.h"

Atom::Atom() {}

Atom::Atom(const Atom& atom) {
    *this = atom;
}

Atom::Atom(Atom&& atom) {
    *this = std::move(atom);
}

Atom::Atom(const Atom_real x, const Atom_real y, const Atom_real z, const uint type) {
    this->x = x;
    this->y = y;
    this->z = z;
    this->type = type;
}

Atom::Atom(const Atom_real x, const Atom_real y, const Atom_real z, const uint type, const uint index) {
    this->x = x;
    this->y = y;
    this->z = z;
    this->type = type;
    this->index = index;
}

Atom::~Atom() {}

Atom& Atom::operator=(const Atom& other) {
    this->x = other.x;
    this->y = other.y;
    this->z = other.z;
    this->type = other.type;
    this->index = other.index;
    // this->is_spin = other.is_spin;
    // this->atom_spin[0] = other.atom_spin[0];
    // this->atom_spin[1] = other.atom_spin[1];
    // this->atom_spin[2] = other.atom_spin[2];
    return *this;
}

Atom& Atom::operator=(Atom&& other) {
    this->x = other.x;
    this->y = other.y;
    this->z = other.z;
    this->type = other.type;
    this->index = other.index;
    // this->is_spin = other.is_spin;
    // this->atom_spin[0] = other.atom_spin[0];
    // this->atom_spin[1] = other.atom_spin[1];
    // this->atom_spin[2] = other.atom_spin[2];
    return *this;
}

Vertices_3D Atom::generate_rc_vertices(const uint cell_type, const double dx, const double dy, const double dz,
                                       const double r_x, const double r_y, const double r_z) const {
    if (cell_type <= 2) {
        double left_bound_x = this->x - r_x;
        double left_bound_y = this->y - r_y;
        double left_bound_z = this->z - r_z;
        double right_bound_x = this->x + r_x;
        double right_bound_y = this->y + r_y;
        double right_bound_z = this->z + r_z;
        int is = std::ceil(left_bound_x/dx);
        int js = std::ceil(left_bound_y/dy);
        int ks = std::ceil(left_bound_z/dz);
        int ie = std::floor(right_bound_x/dx);
        int je = std::floor(right_bound_y/dy);
        int ke = std::floor(right_bound_z/dz);
        return(Vertices_3D(is, ie, js, je, ks, ke));
    } else {
        assert(!"ERROR:: only Orthorhombi is supported~");
        return(Vertices_3D(0, 0, 0));
    }
}

template<typename T>
Array_3D<T> Atom::generate_R_array(const uint cell_type, const Vertices_3D& vertices,
                                   const double dx, const double dy, const double dz) const {
    if (cell_type <= 2) {
        Array_3D<T> result(vertices);
        uint vertices_ninj = vertices.ni * vertices.nj;
        T x_origin = vertices.is * (T)dx;
        T y_origin = vertices.js * (T)dy;
        T z_origin = vertices.ks * (T)dz;
        T* const& __restrict__ result_data = result.data;
        // #ifdef USE_OPENMP_SIMD
        // #pragma omp for schedule(static, Linalg::get_chunksize(omp_get_num_threads(), vertices.nk))
        // #endif //USE_OPENMP_SIMD
        // for (uint k = 0; k < vertices.nk; ++k) {
        //     uint vertices_offset_k = k * vertices_ninj;
        //     double z = z_origin + (int)k * (T)dz - (T)this->z;
        //     double y = y_origin - (T)this->y;
        //     for (uint j = 0; j < vertices.nj; ++j) {
        //         uint vertices_offset_j = j * vertices.ni + vertices_offset_k;
        //         double x = x_origin - (T)this->x;
        //         for (uint i = 0; i < vertices.ni; ++i) {
        //             result_data[i + vertices_offset_j] = std::sqrt(x*x + y*y + z*z);
        //             x += dx;
        //         }
        //         y += dy;
        //     }
        // }
        for (uint k = 0; k < vertices.nk; ++k) {
            uint vertices_offset_k = k * vertices_ninj;
            double z = (int)k * (T)dz - ((T)this->z - z_origin);
            for (uint j = 0; j < vertices.nj; ++j) {
                uint vertices_offset_j = j * vertices.ni + vertices_offset_k;
                double y = (int)j * (T)dy - ((T)this->y - y_origin);
                for (uint i = 0; i < vertices.ni; ++i) {
                    double x = (int)i * (T)dx - ((T)this->x - x_origin);
                    result_data[i + vertices_offset_j] = std::sqrt(x*x + y*y + z*z);
                }
            }
        }
        return result;
    } else {
        assert(!"ERROR:: only Orthorhombi is supported~");
        return Array_3D<T>(Vertices_3D(0, 0, 0));
    }
}
template Array_3D<float> Atom::generate_R_array<float>(const uint cell_type, const Vertices_3D& vertices,
                                                       const double dx, const double dy, const double dz) const;
template Array_3D<double> Atom::generate_R_array<double>(const uint cell_type, const Vertices_3D& vertices,
                                                         const double dx, const double dy, const double dz) const;

template<typename T>
Array_3D<T> Atom::generate_relative_x_array(const uint cell_type, const Vertices_3D& vertices,
                                            const double dx) const {
    if (cell_type <= 2) {
        Array_3D<T> result(vertices);
        uint vertices_ninj = vertices.ni * vertices.nj;
        T x_origin = vertices.is * (T)dx;
        T* const& __restrict__ result_data = result.data;
        #ifdef USE_OPENMP_SIMD
        #pragma omp for schedule(static, Linalg::get_chunksize(omp_get_num_threads(), vertices.nk))
        #endif //USE_OPENMP_SIMD
        for (uint k = 0; k < vertices.nk; ++k) {
            uint vertices_offset_k = k * vertices_ninj;
            for (uint j = 0; j < vertices.nj; ++j) {
                uint vertices_offset_j = j * vertices.ni + vertices_offset_k;
                T x = x_origin - (T)this->x;
                for (uint i = 0; i < vertices.ni; ++i) {
                    result_data[i + vertices_offset_j] = x;
                    x += dx;
                }
            }
        }
        return result;
    } else {
        assert(!"ERROR:: only Orthorhombi is supported~");
        return Array_3D<T>(Vertices_3D(0, 0, 0));
    }
}
template Array_3D<float> Atom::generate_relative_x_array<float>(const uint cell_type, const Vertices_3D& vertices,
                                                                const double dx) const;
template Array_3D<double> Atom::generate_relative_x_array<double>(const uint cell_type, const Vertices_3D& vertices,
                                                                  const double dx) const;

template<typename T>
Array_3D<T> Atom::generate_relative_y_array(const uint cell_type, const Vertices_3D& vertices,
                                            const double dy) const {
    if (cell_type <= 2) {
        Array_3D<T> result(vertices);
        uint vertices_ninj = vertices.ni * vertices.nj;
        T y_origin = vertices.js * (T)dy;
        T* const& __restrict__ result_data = result.data;
        #ifdef USE_OPENMP_SIMD
        #pragma omp for schedule(static, Linalg::get_chunksize(omp_get_num_threads(), vertices.nk))
        #endif //USE_OPENMP_SIMD
        for (uint k = 0; k < vertices.nk; ++k) {
            uint vertices_offset_k = k * vertices_ninj;
            T y = y_origin - (T)this->y;
            for (uint j = 0; j < vertices.nj; ++j) {
                uint vertices_offset_j = j * vertices.ni + vertices_offset_k;
                for (uint i = 0; i < vertices.ni; ++i) {
                    result_data[i + vertices_offset_j] = y;
                }
                y += dy;
            }
        }
        return result;
    } else {
        assert(!"ERROR:: only Orthorhombi is supported~");
        return Array_3D<T>(Vertices_3D(0, 0, 0));
    }
}
template Array_3D<float> Atom::generate_relative_y_array<float>(const uint cell_type, const Vertices_3D& vertices,
                                                                const double dy) const;
template Array_3D<double> Atom::generate_relative_y_array<double>(const uint cell_type, const Vertices_3D& vertices,
                                                                  const double dy) const;

template<typename T>
Array_3D<T> Atom::generate_relative_z_array(const uint cell_type, const Vertices_3D& vertices,
                                            const double dz) const {
    if (cell_type <= 2) {
        Array_3D<T> result(vertices);
        uint vertices_ninj = vertices.ni * vertices.nj;
        T z_origin = vertices.ks * (T)dz;
        T* const& __restrict__ result_data = result.data;
        #ifdef USE_OPENMP_SIMD
        #pragma omp for schedule(static, Linalg::get_chunksize(omp_get_num_threads(), vertices.nk))
        #endif //USE_OPENMP_SIMD
        for (uint k = 0; k < vertices.nk; ++k) {
            uint vertices_offset_k = k * vertices_ninj;
            T z = z_origin + (int)k * dz - (T)this->z;
            for (uint j = 0; j < vertices.nj; ++j) {
                uint vertices_offset_j = j * vertices.ni + vertices_offset_k;
                for (uint i = 0; i < vertices.ni; ++i) {
                    result_data[i + vertices_offset_j] = z;
                }
            }
        }
        return result;
    } else {
        assert(!"ERROR:: only Orthorhombi is supported~");
        return Array_3D<T>(Vertices_3D(0, 0, 0));
    }
}
template Array_3D<float> Atom::generate_relative_z_array<float>(const uint cell_type, const Vertices_3D& vertices,
                                                                const double dz) const;
template Array_3D<double> Atom::generate_relative_z_array<double>(const uint cell_type, const Vertices_3D& vertices,
                                                                  const double dz) const;

template<typename T>
void Atom::generate_xyz_r_array(Array_3D<T>& ref_x, Array_3D<T>& ref_y, Array_3D<T>& ref_z, Array_3D<T>& R,
                                const uint cell_type, const Vertices_3D& vertices,
                                const double dx, const double dy, const double dz) const {
    this->generate_xyz_r_array(ref_x.data, ref_y.data, ref_z.data, R.data, cell_type, vertices, dx, dy, dz);
}
template void Atom::generate_xyz_r_array<float>(Array_3D<float>& ref_x, Array_3D<float>& ref_y, Array_3D<float>& ref_z,
                                                Array_3D<float>& R, const uint cell_type, const Vertices_3D& vertices,
                                                const double dx, const double dy, const double dz) const;
template void Atom::generate_xyz_r_array<double>(Array_3D<double>& ref_x, Array_3D<double>& ref_y, Array_3D<double>& ref_z,
                                                 Array_3D<double>& R, const uint cell_type, const Vertices_3D& vertices,
                                                 const double dx, const double dy, const double dz) const;

template<typename T>
void Atom::generate_xyz_r_array(T* const __restrict__ ref_x, T* const __restrict__ ref_y, T* const __restrict__ ref_z, T* const __restrict__ R,
                               const uint cell_type, const Vertices_3D& vertices,
                               const double dx, const double dy, const double dz) const {
    if (cell_type <= 2) {
        uint vertices_ninj = vertices.ni * vertices.nj;
        T x_origin = vertices.is * (T)dx;
        T y_origin = vertices.js * (T)dy;
        T z_origin = vertices.ks * (T)dz;
        #ifdef USE_OPENMP_SIMD
        #pragma omp for schedule(static, Linalg::get_chunksize(omp_get_num_threads(), vertices.nk))
        #endif //USE_OPENMP_SIMD
        for (uint k = 0; k < vertices.nk; ++k) {
            uint vertices_offset_k = k * vertices_ninj;
            double z = z_origin + (int)k * (T)dz - (T)this->z;
            double y = y_origin - (T)this->y;
            for (uint j = 0; j < vertices.nj; ++j) {
                uint vertices_offset_j = j * vertices.ni + vertices_offset_k;
                double x = x_origin - (T)this->x;
                for (uint i = 0; i < vertices.ni; ++i) {
                    R[i + vertices_offset_j] = std::sqrt(x*x + y*y + z*z);
                    ref_x[i + vertices_offset_j] = x;
                    ref_y[i + vertices_offset_j] = y;
                    ref_z[i + vertices_offset_j] = z;
                    x += dx;
                }
                y += dy;
            }
        }
        return;
    } else {
        assert(!"ERROR:: only Orthorhombi is supported~");
        return;
    }
}
template void Atom::generate_xyz_r_array<float>(float* const ref_x, float* const ref_y, float* const ref_z, float* const R,
                                                const uint cell_type, const Vertices_3D& vertices,
                                                const double dx, const double dy, const double dz) const;
template void Atom::generate_xyz_r_array<double>(double* const ref_x, double* const ref_y, double* const ref_z, double* const R,
                                                const uint cell_type, const Vertices_3D& vertices,
                                                const double dx, const double dy, const double dz) const;

template<typename T>
Nloc_projector<T> Atom::generate_nloc_projector_chi(const uint cell_type, const Psp8_file& psp8_file,
                                                    const Vertices_3D& vertices, const Vertices_3D& local_vertices,
                                                    const double dx, const double dy, const double dz,
                                                    const int atom_index, const int cell_shift_x,
                                                    const int cell_shift_y, const int cell_shift_z,
                                                    const bool is_in_domain) const {
    Nloc_projector<T> result(psp8_file);
    result.init();
    result.atom_index = atom_index;
    result.is_real = (cell_shift_x == 0 && cell_shift_y == 0 && cell_shift_z == 0) ? true : false;
    result.is_in_domain = is_in_domain;
    result.cell_shift_x = cell_shift_x;
    result.cell_shift_y = cell_shift_y;
    result.cell_shift_z = cell_shift_z;
    if (cell_type <= 2) {
        double rc = psp8_file.r_core[0];
        for (int i = 1; i <= psp8_file.lmax; i++) {
            if (rc < psp8_file.r_core[i]) rc = psp8_file.r_core[i];
        }
        uint Nd = vertices.get_size();
        // Array_3D<T> R = this->generate_R_array<T>(2, vertices, dx, dy, dz);
        // Array_3D<T> ref_x = this->generate_relative_x_array<T>(2, vertices, dx);
        // Array_3D<T> ref_y = this->generate_relative_y_array<T>(2, vertices, dy);
        // Array_3D<T> ref_z = this->generate_relative_z_array<T>(2, vertices, dz);
        Array_3D<T> R(vertices);
        Array_3D<T> ref_x(vertices);
        Array_3D<T> ref_y(vertices);
        Array_3D<T> ref_z(vertices);
        #ifdef USE_OPENMP
        omp_set_max_active_levels(2);
        #pragma omp parallel
        this->generate_xyz_r_array(ref_x, ref_y, ref_z, R, 2, vertices, dx, dy, dz);
        omp_set_max_active_levels(1);
        #else
        this->generate_xyz_r_array(ref_x, ref_y, ref_z, R, 2, vertices, dx, dy, dz);
        #endif
        std::vector<T> R_in_sphere;
        std::vector<T> ref_x_in_sphere;
        std::vector<T> ref_y_in_sphere;
        std::vector<T> ref_z_in_sphere;
        T const* const& __restrict__ R_data = R.data;
        T const* const& __restrict__ ref_x_data = ref_x.data;
        T const* const& __restrict__ ref_y_data = ref_y.data;
        T const* const& __restrict__ ref_z_data = ref_z.data;
        for (uint index = 0; index < Nd; index++) {
            if (R_data[index] < rc) {
                int vertice_i = vertices.get_i_nocheck(index);
                int vertice_j = vertices.get_j_nocheck(index);
                int vertice_k = vertices.get_k_nocheck(index);
                uint index_ex = local_vertices.get_index_nocheck(vertice_i, vertice_j, vertice_k);
                result.index.emplace_back(index_ex);
                R_in_sphere.emplace_back(R_data[index]);
                ref_x_in_sphere.emplace_back(ref_x_data[index]);
                ref_y_in_sphere.emplace_back(ref_y_data[index]);
                ref_z_in_sphere.emplace_back(ref_z_data[index]);
            }
        }

        uint nrow = result.index.size();
        result.set_nrow(nrow);
        result.resize();
        Array_0D<T> temp(nrow);
        Array_0D<T> Ylm(nrow);
        Array_0D<T> ref_x_r_in_sphere(nrow);
        Array_0D<T> ref_y_r_in_sphere(nrow);
        Array_0D<T> ref_z_r_in_sphere(nrow);
        #ifdef USE_OPENMP
        omp_set_max_active_levels(2);
        #pragma omp parallel
        {
        Linalg::hadamard_divide_general(ref_x_r_in_sphere.data, ref_x_in_sphere.data(), R_in_sphere.data(), nrow);
        Linalg::hadamard_divide_general(ref_y_r_in_sphere.data, ref_y_in_sphere.data(), R_in_sphere.data(), nrow);
        Linalg::hadamard_divide_general(ref_z_r_in_sphere.data, ref_z_in_sphere.data(), R_in_sphere.data(), nrow);
        }
        omp_set_max_active_levels(1);
        #else
        Linalg::hadamard_divide_general(ref_x_r_in_sphere.data, ref_x_in_sphere.data(), R_in_sphere.data(), nrow);
        Linalg::hadamard_divide_general(ref_y_r_in_sphere.data, ref_y_in_sphere.data(), R_in_sphere.data(), nrow);
        Linalg::hadamard_divide_general(ref_z_r_in_sphere.data, ref_z_in_sphere.data(), R_in_sphere.data(), nrow);
        #endif
        uint icol = 0;
        for (uint l = 0; l < 5; l++) {
            for (uint i = 0; i < psp8_file.nproj[l]; i++) {
                if (psp8_file.is_rgrid_uniform[l]) {
                    Tools::SplineInterpUniform(psp8_file.rgrid[l].data(), psp8_file.kbk_projector_R[l][i].data(), psp8_file.mmax,
                                               R_in_sphere.data(), temp.data, nrow, psp8_file.kbk_projector_RD[l][i].data());
                } else {
                    Tools::SplineInterpNonuniform(psp8_file.rgrid[l].data(), psp8_file.kbk_projector_R[l][i].data(), psp8_file.mmax,
                                                  R_in_sphere.data(), temp.data, nrow, psp8_file.kbk_projector_RD[l][i].data());
                }
                for (int m = - (int) l; m <= (int) l; m++) {
                    // Tools::RealSphericalHarmonic(Ylm.data, (int)l, m, (int)nrow, ref_x_in_sphere.data(), ref_y_in_sphere.data(),
                    //                              ref_z_in_sphere.data(), R_in_sphere.data());
                    Tools::RealSphericalHarmonic2(Ylm.data, (int)l, m, (int)nrow, ref_x_r_in_sphere.data, ref_y_r_in_sphere.data,
                                                 ref_z_r_in_sphere.data, R_in_sphere.data());
                                                 #ifdef USE_OPENMP
                    omp_set_max_active_levels(2);
                    #pragma omp parallel
                    Linalg::hadamard_product_general(result.chi.data + icol * nrow, temp.data, Ylm.data, nrow);
                    omp_set_max_active_levels(1);
                    #else
                    Linalg::hadamard_product_general(result.chi.data + icol * nrow, temp.data, Ylm.data, nrow);
                    #endif
                    icol++;
                }
            }
        }
    } else {
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return result;
}
template Nloc_projector<float> Atom::generate_nloc_projector_chi<float>(const uint cell_type, const Psp8_file& psp8_file,
                                                                        const Vertices_3D& vertices, const Vertices_3D& local_vertices,
                                                                        const double dx, const double dy, const double dz,
                                                                        const int atom_index, const int cell_shift_x,
                                                                        const int cell_shift_y, const int cell_shift_z,
                                                                        const bool is_in_domain) const;
template Nloc_projector<double> Atom::generate_nloc_projector_chi<double>(const uint cell_type, const Psp8_file& psp8_file,
                                                                          const Vertices_3D& vertices, const Vertices_3D& local_vertices,
                                                                          const double dx, const double dy, const double dz,
                                                                          const int atom_index, const int cell_shift_x,
                                                                          const int cell_shift_y, const int cell_shift_z,
                                                                          const bool is_in_domain) const;

template<typename T>
Array_3D<T> Atom::generate_pseudo_charge_density(const uint cell_type, const Vertices_3D& vertices,
                                                 const Psp8_file& psp8_file, const Stencil<T>& stencil,
                                                 const double dx, const double dy, const double dz) const {
    if (cell_type <= 2) {
        Vertices_3D ex_vertices = vertices.generate_ex_vertices(stencil.FDn);
        uint nd_ex = ex_vertices.get_size();
        Array_3D<T> ex_VlocR(ex_vertices);
        Array_3D<T> ex_Vloc(ex_vertices);
        Array_3D<T> ex_R = this->generate_R_array<T>(2, ex_vertices, dx, dy, dz);
        double rchrg = psp8_file.rchrg;
        if (psp8_file.is_rgrid_local_potential_uniform) {
            Tools::SplineInterpUniform(psp8_file.rgrid_local_potential.data(), psp8_file.local_potential_R.data(), psp8_file.mmax,
                                       ex_R.data, ex_VlocR.data, nd_ex, psp8_file.local_potential_RD.data());
        } else {
            Tools::SplineInterpNonuniform(psp8_file.rgrid_local_potential.data(), psp8_file.local_potential_R.data(), psp8_file.mmax,
                                          ex_R.data, ex_VlocR.data, nd_ex, psp8_file.local_potential_RD.data());
        }
        for (uint i = 0; i < nd_ex; i++) {
            // rearrange VJ back to original order
            T* const& __restrict__ ex_Vloc_data = ex_Vloc.data;
            T const* const& __restrict__ ex_VlocR_data = ex_VlocR.data;
            T const* const& __restrict__ ex_R_data = ex_R.data;
            const double mzion = -psp8_file.zion;
            if (fabs(ex_R_data[i]) < 1e-12) {
                ex_Vloc_data[i] = psp8_file.local_potential[0];
            } else if (ex_R_data[i] > rchrg) {
                ex_Vloc_data[i] = mzion / ex_R_data[i];
            } else {
                ex_Vloc_data[i] = ex_VlocR_data[i] / ex_R_data[i];
            }
        }
        Stencil<T> stencil_tem = stencil.coeffs_scale(-0.25 * M_1_PI, 2);
        // return ex_Vloc.calc_laplacian(vertices, stencil_tem);
        Array_3D<T> result(vertices);
        Stencil_method::calc_laplacian(ex_Vloc.data, ex_vertices, stencil_tem, vertices, result.data, vertices);
        return result;
    } else {
        assert(!"ERROR:: only Orthorhombi is supported~");
        return Array_3D<T>(Vertices_3D(0, 0, 0));
    }
}
template Array_3D<float> Atom::generate_pseudo_charge_density<float>(const uint cell_type, const Vertices_3D& vertices,
                                                                     const Psp8_file& psp8_file, const Stencil<float>& stencil,
                                                                     const double dx, const double dy, const double dz) const;
template Array_3D<double> Atom::generate_pseudo_charge_density<double>(const uint cell_type, const Vertices_3D& vertices,
                                                                       const Psp8_file& psp8_file, const Stencil<double>& stencil,
                                                                       const double dx, const double dy, const double dz) const;

template<typename T>
Array_3D<T> Atom::generate_pseudo_charge_density_ref(const uint cell_type, const Vertices_3D& vertices,
                                                     const Psp8_file& psp8_file, const Stencil<T>& stencil,
                                                     const double dx, const double dy, const double dz) const {
    if (cell_type <= 2) {
        Vertices_3D ex_vertices = vertices.generate_ex_vertices(stencil.FDn);
        Array_3D<T> ex_V_ref(ex_vertices);
        Array_3D<T> ex_R = this->generate_R_array<T>(2, ex_vertices, dx, dy, dz);
        double ref_cut = 0.5;
        Atom_method::Calculate_Pseudopot_Ref(ex_R.data, ex_R.length, ref_cut, -psp8_file.zion, ex_V_ref.data);
        Stencil<T> stencil_tem = stencil.coeffs_scale(-0.25 * M_1_PI, 2);
        Array_3D<T> result(vertices);
        Stencil_method::calc_laplacian(ex_V_ref.data, ex_vertices, stencil_tem, vertices, result.data, vertices);
        return result;
    } else {
        assert(!"ERROR:: only Orthorhombi is supported~");
        return Array_3D<T>(Vertices_3D(0, 0, 0));
    }
}
template Array_3D<float> Atom::generate_pseudo_charge_density_ref<float>(const uint cell_type, const Vertices_3D& vertices,
                                                                         const Psp8_file& psp8_file, const Stencil<float>& stencil,
                                                                         const double dx, const double dy, const double dz) const;
template Array_3D<double> Atom::generate_pseudo_charge_density_ref<double>(const uint cell_type, const Vertices_3D& vertices,
                                                                           const Psp8_file& psp8_file, const Stencil<double>& stencil,
                                                                           const double dx, const double dy, const double dz) const;

template<typename T>
T Atom::generate_pseudo_charge_density_related(Array_3D<T>& pseudo_charge_density,
                                               Array_3D<T>& pseudo_charge_density_ref,
                                               Array_3D<T>& pseudo_charge_density_potiential_correction,
                                               const uint cell_type, const Vertices_3D& vertices,
                                               const Psp8_file& psp8_file, const Stencil<T>& stencil,
                                               const double dx, const double dy, const double dz) const {
    if (cell_type <= 2) {
        Vertices_3D ex_vertices = vertices.generate_ex_vertices(stencil.FDn);
        uint nd_ex = ex_vertices.get_size();
        Array_3D<T> ex_VlocR(ex_vertices, 0);
        Array_3D<T> ex_Vloc(ex_vertices);
        Array_3D<T> ex_V_ref(ex_vertices);
        Array_3D<T> ex_R = this->generate_R_array<T>(2, ex_vertices, dx, dy, dz);

        double rchrg = psp8_file.rchrg;
        if (psp8_file.is_rgrid_local_potential_uniform) {
            Tools::SplineInterpUniform(psp8_file.rgrid_local_potential.data(), psp8_file.local_potential_R.data(), psp8_file.mmax,
                                       ex_R.data, ex_VlocR.data, nd_ex, psp8_file.local_potential_RD.data());
        } else {
            Tools::SplineInterpNonuniform(psp8_file.rgrid_local_potential.data(), psp8_file.local_potential_R.data(), psp8_file.mmax,
                                          ex_R.data, ex_VlocR.data, nd_ex, psp8_file.local_potential_RD.data());
        }

        // rearrange VJ back to original order
        T* const& __restrict__ ex_Vloc_data = ex_Vloc.data;
        T const* const& __restrict__ ex_VlocR_data = ex_VlocR.data;
        T const* const& __restrict__ ex_R_data = ex_R.data;
        const double mzion = -psp8_file.zion;
        for (uint i = 0; i < nd_ex; i++) {
            if (fabs(ex_R_data[i]) < 1e-12) {
                ex_Vloc_data[i] = psp8_file.local_potential[0];
            } else if (ex_R_data[i] > rchrg) {
                ex_Vloc_data[i] = mzion / ex_R_data[i];
            } else {
                ex_Vloc_data[i] = ex_VlocR_data[i] / ex_R_data[i];
            }
        }

        double ref_cut = 0.5;
        Atom_method::Calculate_Pseudopot_Ref(ex_R.data, ex_R.length, ref_cut, -psp8_file.zion, ex_V_ref.data);

        Array_3D<T> Vloc(vertices);
        Array_3D<T> V_ref(vertices);
        Vertices_method::fill_region(ex_Vloc.data, ex_vertices, Vloc.data, vertices, vertices);
        Vertices_method::fill_region(ex_V_ref.data, ex_vertices, V_ref.data, vertices, vertices);

        Stencil<T> stencil_tem = stencil.coeffs_scale(-0.25 * M_1_PI, 2);
        Stencil_method::calc_laplacian(ex_Vloc.data, ex_vertices, stencil_tem, vertices, pseudo_charge_density.data, vertices);
        Stencil_method::calc_laplacian(ex_V_ref.data, ex_vertices, stencil_tem, vertices, pseudo_charge_density_ref.data, vertices);
        Linalg::hadamard_minus_general(pseudo_charge_density_potiential_correction.data, V_ref.data, Vloc.data, vertices.get_size());
        Linalg::hadamard_product_general(V_ref.data, pseudo_charge_density_ref.data, vertices.get_size());
        return -Linalg::vector_sum(V_ref.data, vertices.get_size());
    } else {
        assert(!"ERROR:: only Orthorhombi is supported~");
        return (T)0.0;
    }
}
template float Atom::generate_pseudo_charge_density_related<float>(Array_3D<float>& psoducharge_density,
                                                                   Array_3D<float>& psoducharge_density_ref,
                                                                   Array_3D<float>& pseudo_charge_density_potiential_correction,
                                                                   const uint cell_type, const Vertices_3D& vertices,
                                                                   const Psp8_file& psp8_file, const Stencil<float>& stencil,
                                                                   const double dx, const double dy, const double dz) const;
template double Atom::generate_pseudo_charge_density_related<double>(Array_3D<double>& psoducharge_density,
                                                                     Array_3D<double>& psoducharge_density_ref,
                                                                     Array_3D<double>& pseudo_charge_density_potiential_correction,
                                                                     const uint cell_type, const Vertices_3D& vertices,
                                                                     const Psp8_file& psp8_file, const Stencil<double>& stencil,
                                                                     const double dx, const double dy, const double dz) const;

template<typename T>
void Atom::generate_pseudo_charge_density_local_force_related(Array_3D<T>& psoducharge_density,
                                                              Array_3D<T>& psoducharge_density_ref,
                                                              Array_3D<T>& Dpotiential_correction_x,
                                                              Array_3D<T>& Dpotiential_correction_y,
                                                              Array_3D<T>& Dpotiential_correction_z,
                                                              const uint cell_type, const Vertices_3D& vertices,
                                                              const Psp8_file& psp8_file, const Stencil<T>& stencil,
                                                              const double dx, const double dy, const double dz) const {
    if (cell_type <= 2) {
        Vertices_3D ex_vertices = vertices.generate_ex_vertices(stencil.FDn);
        uint nd_ex = ex_vertices.get_size();
        Array_3D<T> ex_VlocR(ex_vertices);
        Array_3D<T> ex_Vloc(ex_vertices);
        Array_3D<T> ex_V_ref(ex_vertices);
        Array_3D<T> ex_R = this->generate_R_array<T>(2, ex_vertices, dx, dy, dz);

        double rchrg = psp8_file.rchrg;
        if (psp8_file.is_rgrid_local_potential_uniform) {
            Tools::SplineInterpUniform(psp8_file.rgrid_local_potential.data(), psp8_file.local_potential_R.data(), psp8_file.mmax,
                                       ex_R.data, ex_VlocR.data, nd_ex, psp8_file.local_potential_RD.data());
        } else {
            Tools::SplineInterpNonuniform(psp8_file.rgrid_local_potential.data(), psp8_file.local_potential_R.data(), psp8_file.mmax,
                                          ex_R.data, ex_VlocR.data, nd_ex, psp8_file.local_potential_RD.data());
        }
        for (uint i = 0; i < nd_ex; i++) {
            // rearrange VJ back to original order
            T* const& __restrict__ ex_Vloc_data = ex_Vloc.data;
            T const* const& __restrict__ ex_VlocR_data = ex_VlocR.data;
            T const* const& __restrict__ ex_R_data = ex_R.data;
            const double mzion = -psp8_file.zion;
            if (fabs(ex_R_data[i]) < 1e-12) {
                ex_Vloc_data[i] = psp8_file.local_potential[0];
            } else if (ex_R_data[i] > rchrg) {
                ex_Vloc_data[i] = mzion / ex_R_data[i];
            } else {
                ex_Vloc_data[i] = ex_VlocR_data[i] / ex_R_data[i];
            }
        }

        double ref_cut = 0.5;
        Atom_method::Calculate_Pseudopot_Ref(ex_R.data, ex_R.length, ref_cut, -psp8_file.zion, ex_V_ref.data);

        // Array_3D<T> Vloc(vertices);
        // Array_3D<T> V_ref(vertices);
        // Vertices_method::fill_region(ex_Vloc.data, ex_vertices, Vloc.data, vertices, vertices);
        // Vertices_method::fill_region(ex_V_ref.data, ex_vertices, V_ref.data, vertices, vertices);
        Array_3D<T> ex_potiential_correction(ex_vertices);
        Linalg::hadamard_minus_general(ex_potiential_correction.data, ex_V_ref.data, ex_Vloc.data, ex_vertices.get_size());

        Stencil<T> stencil_tem = stencil.coeffs_scale(-0.25 * M_1_PI, 2);
        Stencil_method::calc_laplacian(ex_Vloc.data, ex_vertices, stencil_tem, vertices, psoducharge_density.data, vertices);
        Stencil_method::calc_laplacian(ex_V_ref.data, ex_vertices, stencil_tem, vertices, psoducharge_density_ref.data, vertices);
        Stencil_method::calc_gradient(ex_potiential_correction.data, ex_vertices, 0, stencil,
                                      vertices, Dpotiential_correction_x.data, vertices);
        Stencil_method::calc_gradient(ex_potiential_correction.data, ex_vertices, 1, stencil,
                                      vertices, Dpotiential_correction_y.data, vertices);
        Stencil_method::calc_gradient(ex_potiential_correction.data, ex_vertices, 2, stencil,
                                      vertices, Dpotiential_correction_z.data, vertices);
        return;
    } else {
        assert(!"ERROR:: only Orthorhombi is supported~");
        return;
    }
}
template void Atom::generate_pseudo_charge_density_local_force_related<float>(Array_3D<float>& psoducharge_density,
                                                                              Array_3D<float>& psoducharge_density_ref,
                                                                              Array_3D<float>& Dpotiential_correction_x,
                                                                              Array_3D<float>& Dpotiential_correction_y,
                                                                              Array_3D<float>& Dpotiential_correction_z,
                                                                              const uint cell_type, const Vertices_3D& vertices,
                                                                              const Psp8_file& psp8_file, const Stencil<float>& stencil,
                                                                              const double dx, const double dy, const double dz) const;
template void Atom::generate_pseudo_charge_density_local_force_related<double>(Array_3D<double>& psoducharge_density,
                                                                               Array_3D<double>& psoducharge_density_ref,
                                                                               Array_3D<double>& Dpotiential_correction_x,
                                                                               Array_3D<double>& Dpotiential_correction_y,
                                                                               Array_3D<double>& Dpotiential_correction_z,
                                                                               const uint cell_type, const Vertices_3D& vertices,
                                                                               const Psp8_file& psp8_file, const Stencil<double>& stencil,
                                                                               const double dx, const double dy, const double dz) const;

template<typename T>
Array_3D<T> Atom::generate_initial_electron_density_array(const uint cell_type, const Vertices_3D& vertices,
                                                          const Psp8_file& psp8_file,
                                                          const double dx, const double dy, const double dz) const {
    if (cell_type <= 2) {
        uint nd = vertices.get_size();
        Array_3D<T> R = this->generate_R_array<T>(2, vertices, dx, dy, dz);
        Array_3D<T> initial_electron_density(vertices, 0);
        if (psp8_file.is_rgrid_density_uniform) {
            Tools::SplineInterpUniform(psp8_file.rgrid_density.data(), psp8_file.r_density_4pi.data(), psp8_file.mmax,
                                       R.data, initial_electron_density.data, nd, psp8_file.r_density_4piD.data());
        } else {
            Tools::SplineInterpNonuniform(psp8_file.rgrid_density.data(), psp8_file.r_density_4pi.data(), psp8_file.mmax,
                                          R.data, initial_electron_density.data, nd, psp8_file.r_density_4piD.data());
        }
        double rchrg = psp8_file.rchrg;
        T const* const& __restrict__ R_data = R.data;
        T* const& __restrict__ initial_electron_density_data = initial_electron_density.data;
        for (uint i = 0; i < nd; i++) {
            if (R_data[i] > rchrg) {
                initial_electron_density_data[i] = 0.0;
            }
        }
        return initial_electron_density;
    } else {
        assert(!"ERROR:: only Orthorhombi is supported~");
        return Array_3D<T>(Vertices_3D(0, 0, 0));
    }
}
template Array_3D<float> Atom::generate_initial_electron_density_array<float>(const uint cell_type, const Vertices_3D& vertices,
                                                                              const Psp8_file& psp8_file, const double dx,
                                                                              const double dy, const double dz) const;
template Array_3D<double> Atom::generate_initial_electron_density_array<double>(const uint cell_type, const Vertices_3D& vertices,
                                                                                const Psp8_file& psp8_file, const double dx,
                                                                                const double dy, const double dz) const;

template<typename T>
Array_3D<T> Atom::generate_initial_electron_density_core_array(const uint cell_type, const Vertices_3D& vertices,
                                                          const Psp8_file& psp8_file,
                                                          const double dx, const double dy, const double dz) const {
    if (cell_type <= 2) {
        uint nd = vertices.get_size();
        Array_3D<T> R = this->generate_R_array<T>(2, vertices, dx, dy, dz);
        Array_3D<T> initial_electron_density(vertices);
        if (psp8_file.is_rgrid_charge_uniform) {
            Tools::SplineInterpUniform(psp8_file.rgrid_charge.data(), psp8_file.rho_c_table.data(), psp8_file.mmax,
                                       R.data, initial_electron_density.data, nd, psp8_file.rho_c_tableD.data());
        } else {
            Tools::SplineInterpNonuniform(psp8_file.rgrid_charge.data(), psp8_file.rho_c_table.data(), psp8_file.mmax,
                                          R.data, initial_electron_density.data, nd, psp8_file.rho_c_tableD.data());
        }
        double rchrg = psp8_file.rchrg;
        T const* const& __restrict__ R_data = R.data;
        T* const& __restrict__ initial_electron_density_data = initial_electron_density.data;
        for (uint i = 0; i < nd; i++) {
            if (R_data[i] > rchrg) {
                initial_electron_density_data[i] = 0.0;
            }
        }
        return initial_electron_density;
    } else {
        assert(!"ERROR:: only Orthorhombi is supported~");
        return Array_3D<T>(Vertices_3D(0, 0, 0));
    }
}
template Array_3D<float> Atom::generate_initial_electron_density_core_array<float>(const uint cell_type, const Vertices_3D& vertices,
                                                                              const Psp8_file& psp8_file, const double dx,
                                                                              const double dy, const double dz) const;
template Array_3D<double> Atom::generate_initial_electron_density_core_array<double>(const uint cell_type, const Vertices_3D& vertices,
                                                                                const Psp8_file& psp8_file, const double dx,
                                                                                const double dy, const double dz) const;

template<typename T>
Array_4D<T> Atom::generate_atomic_orbitals_nlm_array(const uint cell_type, const uint n_atomic_orbital_Ynlm,
                                                    const Vertices_3D& vertices, const Upf_file& upf_file, const T cutoff,
                                                    const double dx, const double dy, const double dz) const {
    if (cell_type <= 2) {
        const uint nd = vertices.get_size();
        Array_0D<T> ref_x_r(nd);
        Array_0D<T> ref_y_r(nd);
        Array_0D<T> ref_z_r(nd);
        Array_0D<T> R(nd);
        this->generate_xyz_r_array(ref_x_r.data, ref_y_r.data, ref_z_r.data, R.data, 2, vertices, dx, dy, dz);
        Linalg::hadamard_divide_general(ref_x_r.data, R.data, nd);
        Linalg::hadamard_divide_general(ref_y_r.data, R.data, nd);
        Linalg::hadamard_divide_general(ref_z_r.data, R.data, nd);
        Array_4D<T> atomic_orbitals_nlm(Vertices_4D(vertices, n_atomic_orbital_Ynlm));
        Array_0D<T> chi_radius(nd, 0);
        Array_0D<T> Ylm(nd);
        uint count = 0;
        for (int i_chi = 0; i_chi < upf_file.n_chi; i_chi++) {
            const Atomic_orbital& atomic_orbital = upf_file.atomic_orbitals[i_chi];
            const int& l = atomic_orbital.l;
            if (true) {
                Tools::SplineInterpUniform(upf_file.r_grid.data(), atomic_orbital.chi_r.data(), upf_file.n_grid,
                                        R.data, chi_radius.data, nd, atomic_orbital.chi_rD.data());
            } else {
                Tools::SplineInterpNonuniform(upf_file.r_grid.data(), atomic_orbital.chi_r.data(), upf_file.n_grid,
                                        R.data, chi_radius.data, nd, atomic_orbital.chi_rD.data());
            }
            // Linalg::hadamard_product_general(chi_radius.data, R.data, nd);
            for (int m = -l; m <= l; m++) {
                Tools::RealSphericalHarmonic2(Ylm.data, l, m, nd, ref_x_r.data, ref_y_r.data,
                                                 ref_z_r.data, R.data);
                Linalg::hadamard_product_general(atomic_orbitals_nlm.data + count * nd, Ylm.data, chi_radius.data, nd);
                for (uint i = 0; i < nd; i++) {
                    if (R[i] >= cutoff) {
                        atomic_orbitals_nlm.data[count * nd + i] = T(0);
                    }
                }
                count++;
            }
        }
        assert(count == n_atomic_orbital_Ynlm);
        return atomic_orbitals_nlm;
    } else {
        assert(!"ERROR:: only Orthorhombi is supported~");
        return Array_4D<T>(Vertices_4D(0, 0, 0, 0));
    }
}
template Array_4D<float> Atom::generate_atomic_orbitals_nlm_array<float>(const uint cell_type, const uint n_atomic_orbital_Ynlm,
                                                    const Vertices_3D& vertices, const Upf_file& upf_file, const float cutoff,
                                                    const double dx, const double dy, const double dz) const;
template Array_4D<double> Atom::generate_atomic_orbitals_nlm_array<double>(const uint cell_type, const uint n_atomic_orbital_Ynlm,
                                                    const Vertices_3D& vertices, const Upf_file& upf_file, const double cutoff,
                                                    const double dx, const double dy, const double dz) const;

uint Atom::generate_valid_images(std::vector<Atom>& atoms, const uint cell_type, 
                                 const Domain_parallel_vertices_3D& domain_vertices, const bool* is_periodic,
                                 const double dx, const double dy, const double dz,
                                 const double lx, const double ly, const double lz,
                                 const double r_x, const double r_y, const double r_z) const {
    return this->generate_valid_images(atoms, cell_type, domain_vertices.get_3D_shared_vertices(),
                                       domain_vertices.get_3D_local_vertices(), is_periodic,
                                       dx, dy, dz, lx, ly, lz, r_x, r_y, r_z);
}

uint Atom::generate_valid_images(std::vector<Atom>& atoms, const uint cell_type, 
                                 const Vertices_3D& shared_vertices, const Vertices_3D& local_vertices,
                                 const bool* is_periodic,
                                 const double dx, const double dy, const double dz,
                                 const double lx, const double ly, const double lz,
                                 const double r_x, const double r_y, const double r_z) const {
    if (shared_vertices.get_size() == 0 || local_vertices.get_size() == 0) return 0;
    if (cell_type <= 2) {
        // double lx = shared_vertices.ni * dx;
        // double ly = shared_vertices.nj * dy;
        // double lz = shared_vertices.nk * dz;
        double shared_x_left_bound = (shared_vertices.is < local_vertices.is ? shared_vertices.is : local_vertices.is) * dx - r_x;
        double shared_y_left_bound = (shared_vertices.js < local_vertices.js ? shared_vertices.js : local_vertices.js) * dy - r_y;
        double shared_z_left_bound = (shared_vertices.ks < local_vertices.ks ? shared_vertices.ks : local_vertices.ks) * dz - r_z;
        int shared_x_right_point = std::max(shared_vertices.get_ie(), local_vertices.get_ie());
        int shared_y_right_point = std::max(shared_vertices.get_je(), local_vertices.get_je());
        int shared_z_right_point = std::max(shared_vertices.get_ke(), local_vertices.get_ke());
        if (shared_x_right_point == shared_vertices.get_ie() && is_periodic[0]) shared_x_right_point += 1;
        if (shared_y_right_point == shared_vertices.get_je() && is_periodic[1]) shared_y_right_point += 1;
        if (shared_z_right_point == shared_vertices.get_ke() && is_periodic[2]) shared_z_right_point += 1;
        double shared_x_right_bound = shared_x_right_point * dx + r_x;
        double shared_y_right_bound = shared_y_right_point * dy + r_y;
        double shared_z_right_bound = shared_z_right_point * dz + r_z;
        int imax = is_periodic[0] ? floor((shared_x_right_bound - this->x) / lx + 1e-12) : 0;
        int jmax = is_periodic[1] ? floor((shared_y_right_bound - this->y) / ly + 1e-12) : 0;
        int kmax = is_periodic[2] ? floor((shared_z_right_bound - this->z) / lz + 1e-12) : 0;
        int imin = is_periodic[0] ? -floor((this->x - shared_x_left_bound) / lx + 1e-12) : 0;
        int jmin = is_periodic[1] ? -floor((this->y - shared_y_left_bound) / ly + 1e-12) : 0;
        int kmin = is_periodic[2] ? -floor((this->z - shared_z_left_bound) / lz + 1e-12) : 0;

        double local_x_left_bound = local_vertices.is * dx - r_x;
        double local_y_left_bound = local_vertices.js * dy - r_y;
        double local_z_left_bound = local_vertices.ks * dz - r_z;
        // double local_x_right_bound = (local_vertices.get_ie() + 1) * dx + r_x;
        // double local_y_right_bound = (local_vertices.get_je() + 1) * dy + r_y;
        // double local_z_right_bound = (local_vertices.get_ke() + 1) * dz + r_z;
        double local_y_right_bound = (local_vertices.get_je()) * dy + r_y;
        double local_x_right_bound = (local_vertices.get_ie()) * dx + r_x;
        double local_z_right_bound = (local_vertices.get_ke()) * dz + r_z;

        uint count = 0;
        double x_image;
        double y_image;
        double z_image;
        for (int k = kmin; k <= kmax; k++) {
            z_image = this->z + k * lz;
            if (z_image < local_z_left_bound || z_image >= local_z_right_bound) continue;
            for (int j = jmin; j <= jmax; j++) {
                y_image = this->y + j * ly;
                if (y_image < local_y_left_bound || y_image >= local_y_right_bound) continue;
                for (int i = imin; i <= imax; i++) {
                    x_image = this->x + i * lx;
                    if (x_image < local_x_left_bound || x_image >= local_x_right_bound) continue;
                    atoms.emplace_back(x_image, y_image, z_image, this->type, this->index);
                    // atoms.back().is_spin = this->is_spin;
                    // atoms.back().atom_spin[0] = this->atom_spin[0];
                    // atoms.back().atom_spin[1] = this->atom_spin[1];
                    // atoms.back().atom_spin[2] = this->atom_spin[2];
                    ++count;
                }
            }
        }
        return count;
    } else {
        assert(!"ERROR:: only Orthorhombi is supported~");
        return 0;
    }
}

MPI_Datatype Atom::register_mpi_type() const {
    constexpr std::size_t num_members = 5;
    int lengths[num_members] = {1, 1, 1, 1, 1}; //,1, 3
    MPI_Aint offsets[num_members] = {offsetof(Atom, x), offsetof(Atom, y),
                                     offsetof(Atom, z), offsetof(Atom, type),
                                     offsetof(Atom, index)};
                                    //  , offsetof(Atom, is_spin),
                                    //  offsetof(Atom, atom_spin)};
    MPI_Datatype types[num_members] = {MPI_ATOM_REAL, MPI_ATOM_REAL,
                                       MPI_ATOM_REAL, MPI_UNSIGNED,
                                       MPI_UNSIGNED};
                                        // MPI_C_BOOL,
                                    //    MPI_ATOM_REAL};
    MPI_Datatype type;
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    return type;
}

void Atom::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
}

void Atom::bcast(const MPI_Comm comm, const int root) {
    MPI_Datatype Atom_type = this->register_mpi_type();
    MPI_Bcast(this, 1, Atom_type, root, comm);
    this->deregister_mpi_type(Atom_type);
    return;
}

void Atom::show() const {
    std::cout << "type = " << this->type
              << ", index = " << this->index
              << ", [x, y, z] = ["
              << std::setw(22) << std::setprecision(15) << std::fixed << std::right << this->x << " , "
              << std::setw(22) << std::setprecision(15) << std::fixed << std::right << this->y << " , "
              << std::setw(22) << std::setprecision(15) << std::fixed << std::right << this->z << " ]" << std::endl;
    // std::cout << "is_spin = " << this->is_spin
    //           << ", atom_spin = ["
    //           << this->atom_spin[0] << ", "
    //           << this->atom_spin[1] << ", "
    //           << this->atom_spin[2] << "]"
    //           << std::endl;
    std::cout << " std::is_standard_layout<Atom>::value = " << std::is_standard_layout<Atom>::value << std::endl;
    return;
}

template<typename T>
void Atom_method::print_pdos(std::ostream &file, const Atom& atom, const uint nspin,
                            const Array_2D<T>& pdos_atom, const Array_0D<T>& eigen_values,
                            const Psp8_file& psp8_file, const T chemical_potential,
                            const bool if_print_header) {
    assert(nspin == 1);
    if (if_print_header) {
        file << "Final PDOS (Ha) with index (ispin, n, l, m), with chemical potential "
                    << std::fixed << std::setprecision(12) << chemical_potential
                    << std::right << std::endl;
    }
    const uint& nstates = eigen_values.length;
    const uint width = 16;
    const uint precision = 8;
    if (nspin == 1) {
        const uint ispin = 0;
        file << std::setw(6) << std::left << "Atom #"
            << std::setw(8) << std::left << atom.index + 1
            << std::setw(6) << "type #"
            << std::setw(4) << std::left << atom.type + 1
            << std::setw(4) << "POS "
            << "("
            << std::setw(7) << std::fixed << std::setprecision(5) << atom.x
            << ", "
            << std::setw(7) << std::fixed << std::setprecision(5) << atom.y
            << ", "
            << std::setw(7) << std::fixed << std::setprecision(5) << atom.z
            << ")"
            << std::endl;
        if (!psp8_file.if_has_upf_file) {
            file << std::right << std::setw(8) << "n"
                    << std::setw(width) << "eigval" << std::endl;
            for (uint ib = 0; ib < nstates; ib++) {
                file << std::setw(8) << ib + 1
                        << std::setw(width) << std::setprecision(precision) << eigen_values[ib]
                        << std::endl;
            }
            file << std::endl;
        } else {
            const Upf_file& upf_file = psp8_file.upf_file;
            file << std::right <<std::setw(8) << "n"
                    << std::setw(width) << "eigval";
            uint n_atomic_orbital = 0;
            for (int i_chi = 0; i_chi < upf_file.n_chi; i_chi++) {
                const int& l = upf_file.atomic_orbitals[i_chi].l;
                const int& n = upf_file.atomic_orbitals[i_chi].n;
                n_atomic_orbital += 2 * l + 1;
                for (int m = -l; m <= l; m++) {
                    std::ostringstream oss;
                    oss << "(" << ispin << "," << n << "," << l << "," << m << ")";
                    std::string result = oss.str();
                    file << std::setw(width) << result;
                }
            }
            file << std::endl;
            for (uint ib = 0; ib < nstates; ib++) {
                file << std::setw(8) << ib + 1
                        << std::setw(width) << std::setprecision(precision) << eigen_values[ib];
                for (uint i_atomic_orbital = 0; i_atomic_orbital < n_atomic_orbital; i_atomic_orbital++) {
                    file << std::setw(width) << std::setprecision(precision) << pdos_atom[i_atomic_orbital + n_atomic_orbital * ib];
                }
                file << std::endl;
            }
            file << std::endl;
        }
    }
    return;
}
template void Atom_method::print_pdos<double>(std::ostream &file, const Atom& atom, const uint nspin,
                                            const Array_2D<double>& pdos_atom, const Array_0D<double>& eigen_values,
                                            const Psp8_file& psp8_file, const double chemical_potential,
                                            const bool if_print_header);
template void Atom_method::print_pdos<float>(std::ostream &file, const Atom& atom, const uint nspin,
                                            const Array_2D<float>& pdos_atom, const Array_0D<float>& eigen_values,
                                            const Psp8_file& psp8_file, const float chemical_potential,
                                            const bool if_print_header);

void Atom_method::generate_valid_images(const std::vector<Atom>& atoms, std::vector<Atom>& image_atoms, const uint cell_type,
                                        const Domain_parallel_vertices_3D& domain_vertices, const bool* is_periodic,
                                        const double dx, const double dy, const double dz,
                                        const double lx, const double ly, const double lz,
                                        const std::vector<Psp8_file>& psp8_files, const uint mode) {
    if (mode == 0) {
        for (std::vector<Atom>::const_iterator ptr = atoms.cbegin(); ptr != atoms.cend(); ++ptr) {
            ptr->generate_valid_images(image_atoms, cell_type, domain_vertices, is_periodic, dx, dy, dz, lx, ly, lz,
                                    psp8_files[ptr->type].rchrg, psp8_files[ptr->type].rchrg, psp8_files[ptr->type].rchrg);
        }
    } else if (mode == 1) {
        for (std::vector<Atom>::const_iterator ptr = atoms.cbegin(); ptr != atoms.cend(); ++ptr) {
            ptr->generate_valid_images(image_atoms, cell_type, domain_vertices, is_periodic, dx, dy, dz, lx, ly, lz,
                                    psp8_files[ptr->type].charge_cut_x, psp8_files[ptr->type].charge_cut_y,
                                    psp8_files[ptr->type].charge_cut_z);
        }
    }
    return;
}

/**
 * @brief   Calculate pseudopotential reference.
 *
 * @ref     Linear scaling solution of the all-electron Coulomb problem in solids.
 *          J.E.Pask, N. Sukumar and S.E. Mousavi
 */
template<typename T>
void Atom_method::Calculate_Pseudopot_Ref(T const* const __restrict__ R, const int len, const double ref_cut,
                                          const double zion, T* const __restrict__ Vref) {
    int i;
    T R2;
    T C1, C3, C6, C7, C8;
    C1 = (T)ref_cut;
    C3 = (T)ref_cut * (T)ref_cut * (T)ref_cut;
    C6 = C3 * C3;
    C7 = C6 * (T)ref_cut;
    C8 = C7 * (T)ref_cut;
    T Tzion = (T)zion;
    for (i = 0; i < len; i++) {
        if (R[i] <= C1) {
            R2 = R[i]*R[i];
            // use horner's rule 
            Vref[i] = R2 * ( R2*R[i] * (5.6/C6 + R[i] * (1.8*R[i]/(C8) - 6.0/(C7))) - 2.8/C3) + 2.4/C1; 
            //Vref[i] = pow(R[i],2) * ( pow(R[i],3) * (5.6 / pow(rcut,6) + R[i]*(1.8*R[i]/pow(rcut,8) - 6/pow(rcut,7) ) ) - 2.8/pow(rcut,3)) + 2.4/rcut;
            // // calculate directly
            // Vref[i] = (9.0*pow(R[i],7) - 30.0*pow(R[i],6)*rcut + 28.0*pow(R[i],5)*rcut*rcut - 14.0*pow(R[i],2)*pow(rcut,5) + 12.0*pow(rcut,7))/(5.0*pow(rcut,8));
            Vref[i] *= Tzion;
        } else {
            Vref[i] = Tzion / R[i];
        }
    }
}
