#include "stencil.h"
#include "stencil_kernel.h"

namespace Stencil_method {

template<>
void calc_laplacian<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                        const Vertices_3D& region, float* const ptr, const Vertices_3D& vertices, float const* const ptr_A) {
    const int64_t index_this_origin = ex_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t index_result_origin = vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t result_ni = vertices.ni;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int FDn = stencil.FDn;
    const float coef_0 = stencil.get_D2_coef0();
    switch(stencil.cell_type) {
    case 0:
    case 1:
    case 2:
        laplacian_3d_c2_o0<float>(ex_ptr + index_this_origin, this_ni, this_ninj,
                       ptr + index_result_origin, result_ni, result_ninj,
                       region.ni, region.nj, region.nk,
                       FDn,
                       stencil.get_D2_coeffs_x(),
                       stencil.get_D2_coeffs_y(),
                       stencil.get_D2_coeffs_z(),
                       coef_0,
                       ptr_A);
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}

template<>
void calc_laplacian<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                        const Vertices_3D& region, double* const ptr, const Vertices_3D& vertices, double const* const ptr_A) {
    const int64_t index_this_origin = ex_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t index_result_origin = vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t result_ni = vertices.ni;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int FDn = stencil.FDn;
    const double coef_0 = stencil.get_D2_coef0();
    switch(stencil.cell_type) {
    case 0:
    case 1:
    case 2:
        laplacian_3d_c2_o0<double>(ex_ptr + index_this_origin, this_ni, this_ninj,
                       ptr + index_result_origin, result_ni, result_ninj,
                       region.ni, region.nj, region.nk,
                       FDn,
                       stencil.get_D2_coeffs_x(),
                       stencil.get_D2_coeffs_y(),
                       stencil.get_D2_coeffs_z(),
                       coef_0,
                       ptr_A);
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}

namespace Boundary_safe {

template<>
void calc_laplacian_boundary_safe<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                        const Vertices_3D& region, float* const ptr, const Vertices_3D& vertices, float const* const ptr_A) {
    const int64_t index_this_origin = ex_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t index_result_origin = vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t result_ni = vertices.ni;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int FDn = stencil.FDn;
    const float coef_0 = stencil.get_D2_coef0();
    switch(stencil.cell_type) {
    case 0:
    case 1:
    case 2:
        laplacian_3d_c2_o0_boundary_safe<float>(ex_ptr + index_this_origin, this_ni, this_ninj,
                       ptr + index_result_origin, result_ni, result_ninj,
                       region.ni, region.nj, region.nk,
                       ex_vertices.is - region.is,
                       ex_vertices.js - region.js,
                       ex_vertices.ks - region.ks,
                       ex_vertices.get_ie() - region.is,
                       ex_vertices.get_je() - region.js,
                       ex_vertices.get_ke() - region.ks,
                       FDn,
                       stencil.get_D2_coeffs_x(),
                       stencil.get_D2_coeffs_y(),
                       stencil.get_D2_coeffs_z(),
                       coef_0,
                       ptr_A);
        // std::cout << "ex_vertices.is - region.is = " << ex_vertices.is - region.is << std::endl;
        // std::cout << "ex_vertices.get_ie() - region.is = " << ex_vertices.get_ie() - region.is << std::endl;
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}

template<>
void calc_laplacian_boundary_safe<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                        const Vertices_3D& region, double* const ptr, const Vertices_3D& vertices, double const* const ptr_A) {
    const int64_t index_this_origin = ex_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t index_result_origin = vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t result_ni = vertices.ni;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int FDn = stencil.FDn;
    const double coef_0 = stencil.get_D2_coef0();
    switch(stencil.cell_type) {
    case 0:
    case 1:
    case 2:
        laplacian_3d_c2_o0_boundary_safe<double>(ex_ptr + index_this_origin, this_ni, this_ninj,
                       ptr + index_result_origin, result_ni, result_ninj,
                       region.ni, region.nj, region.nk,
                       ex_vertices.is - region.is,
                       ex_vertices.js - region.js,
                       ex_vertices.ks - region.ks,
                       ex_vertices.get_ie() - region.is,
                       ex_vertices.get_je() - region.js,
                       ex_vertices.get_ke() - region.ks,
                       FDn,
                       stencil.get_D2_coeffs_x(),
                       stencil.get_D2_coeffs_y(),
                       stencil.get_D2_coeffs_z(),
                       coef_0,
                       ptr_A);
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}

template<>
void calc_laplacian_boundary_safe<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                        const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices, float const* const ptr_A) {
    const int64_t index_this_origin = ex_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t this_ninjnk = this_ninj * ex_vertices.nk;
    const int64_t index_result_origin = vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t result_ni = vertices.ni;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int64_t result_ninjnk = result_ninj * vertices.nk;
    const int FDn = stencil.FDn;
    const float coef_0 = stencil.get_D2_coef0();
    switch(stencil.cell_type) {
    case 0:
    case 1:
    case 2:
        laplacian_3d_c2_o0_boundary_safe<float>(ex_ptr + index_this_origin, this_ni, this_ninj, this_ninjnk,
                       ptr + index_result_origin, result_ni, result_ninj, result_ninjnk,
                       region.ni, region.nj, region.nk, region.nb,
                       ex_vertices.is - region.is,
                       ex_vertices.js - region.js,
                       ex_vertices.ks - region.ks,
                       ex_vertices.get_ie() - region.is,
                       ex_vertices.get_je() - region.js,
                       ex_vertices.get_ke() - region.ks,
                       FDn,
                       stencil.get_D2_coeffs_x(),
                       stencil.get_D2_coeffs_y(),
                       stencil.get_D2_coeffs_z(),
                       coef_0,
                       ptr_A);
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}

template<>
void calc_laplacian_boundary_safe<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                        const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices, double const* const ptr_A) {
    const int64_t index_this_origin = ex_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t this_ninjnk = this_ninj * ex_vertices.nk;
    const int64_t index_result_origin = vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t result_ni = vertices.ni;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int64_t result_ninjnk = result_ninj * vertices.nk;
    const int FDn = stencil.FDn;
    const double coef_0 = stencil.get_D2_coef0();
    switch(stencil.cell_type) {
    case 0:
    case 1:
    case 2:
        laplacian_3d_c2_o0_boundary_safe<double>(ex_ptr + index_this_origin, this_ni, this_ninj, this_ninjnk,
                       ptr + index_result_origin, result_ni, result_ninj, result_ninjnk,
                       region.ni, region.nj, region.nk, region.nb,
                       ex_vertices.is - region.is,
                       ex_vertices.js - region.js,
                       ex_vertices.ks - region.ks,
                       ex_vertices.get_ie() - region.is,
                       ex_vertices.get_je() - region.js,
                       ex_vertices.get_ke() - region.ks,
                       FDn,
                       stencil.get_D2_coeffs_x(),
                       stencil.get_D2_coeffs_y(),
                       stencil.get_D2_coeffs_z(),
                       coef_0,
                       ptr_A);
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}

} // Boundary_safe

}
