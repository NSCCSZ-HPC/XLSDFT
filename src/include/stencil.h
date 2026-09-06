#ifndef _STENCIL_H_
#define _STENCIL_H_

#include <iostream>
#include <vector>
#include "vertices.h"
#include "linalg.h"
#include "memory_pool.h"

template<typename T>
class Stencil{
public:
    uint cell_type = 0;
    uint optimization = 0;
    int order = 0;
    int FDn = 0;
    // std::vector<T> D1_coeffs_x;    // 1st derivative weights including mesh
    // std::vector<T> D1_coeffs_y;    // 1st derivative weights including mesh
    // std::vector<T> D1_coeffs_z;    // 1st derivative weights including mesh
    // std::vector<T> D2_coeffs_x;    // 2nd derivative weights including mesh
    // std::vector<T> D2_coeffs_y;    // 2nd derivative weights including mesh
    // std::vector<T> D2_coeffs_z;    // 2nd derivative weights including mesh
    // std::vector<T> D2_coeffs_xyz;    // transpose of (D2_coeffs_x,y,z)
    T* D1_coeffs_x = nullptr;    // 1st derivative weights including mesh
    T* D1_coeffs_y = nullptr;    // 1st derivative weights including mesh
    T* D1_coeffs_z = nullptr;    // 1st derivative weights including mesh
    T* D2_coeffs_x = nullptr;    // 2nd derivative weights including mesh
    T* D2_coeffs_y = nullptr;    // 2nd derivative weights including mesh
    T* D2_coeffs_z = nullptr;    // 2nd derivative weights including mesh
    T* D2_coeffs_xyz = nullptr;    // transpose of (D2_coeffs_x,y,z)
    Stencil();
    Stencil(const int& order, const uint& cell_type = 0);
    Stencil(const Stencil& other);
    Stencil(Stencil&& other);
    ~Stencil();
    Stencil& operator=(const Stencil& other);
    Stencil& deepcopy(const Stencil& other);
    Stencil& deepcopy(Stencil&& other);
    Stencil& deepcopy_mp(const Stencil& other, Memory_pool<T, Fast_memory>& pool_fast);
    void set_cell_type(const uint& cell_type);
    void set_optimization(const uint& optimization);
    void set_order(const int& order);
    void set_FDn(const int& FDn);
    void set_D1_coeffs(double* deltas);
    void set_D1_coeffs(double dx, double dy, double dz);
    void set_D2_coeffs(double* deltas);
    void set_D2_coeffs(double dx, double dy, double dz);
    T const* get_D1_coeffs_x() const;
    T const* get_D1_coeffs_y() const;
    T const* get_D1_coeffs_z() const;
    T const* get_D2_coeffs_x() const;
    T const* get_D2_coeffs_y() const;
    T const* get_D2_coeffs_z() const;
    T const* get_D2_coeffs_xyz() const;
    uint get_D2_coeffs_xyz_length() const;
    T get_D2_coef0() const;
    Stencil& operator*=(const double& ratio);
    Stencil operator*(const double& ratio);
    Stencil coeffs_scale(const double& ratio) const;
    Stencil coeffs_scale(const double& ratio, const int& dim) const;
    Stencil& coeffs_scale_self(const double& ratio);
    Stencil& shift_D2_coeffs(const T& c);
    void init(const uint& cell_type, const int& order, const double& dx, const double& dy, const double& dz);
    template<typename T2> void init(const Stencil<T2>& stencil);
    void destructor();
    void destructor_mp();
    void show() const;
};

namespace Stencil_method {
    double fract(const int& n, const int& k);
    void set_FDweights_D1(const int& FDn, double* FDweights_D1);
    template<typename T> void set_D1_coeffs(const int& FDn, const double& delta, const double* FDweights_D1, T* D1_coeffs);
    void set_FDweights_D2(const int& FDn, double* FDweights_D2);
    template<typename T> void set_D2_coeffs(const int& FDn, const double& delta, const double* FDweights_D2, T* D2_coeffs);
    template<typename T> void jacobi_preconditioner(const Stencil<T>& stencil, const int& N, const T c, const T *r, T *f);

    /**
     * @brief DST-I preconditioner data (for Dirichlet boundary conditions)
     * @author Qimen Xu
     */
    // 
    template<typename T>
    struct DST_Preconditioner_Data {
        uint N1 = 0;
        uint N2 = 0;
        uint N3 = 0;
        uint N = 0;
        std::vector<T> d_hat;           // Eigenvalues: λ_k = w0 + 2Σ wp·cos(πkp/(N+1))
        std::vector<T> work;            // Real work array for DST
        #if defined(USE_FFTW)
        // fftw_plan dst_plan_x = nullptr;
        // fftw_plan dst_plan_y = nullptr;
        // fftw_plan dst_plan_z = nullptr;
        std::vector<fftw_plan> dst_plans_x;
        std::vector<fftw_plan> dst_plans_y;
        std::vector<fftw_plan> dst_plans_z;
        #elif defined(USE_MKL)
        // MKL FFTW interface plans (same as FFTW)
        // fftw_plan dst_plan_x = nullptr;
        // fftw_plan dst_plan_y = nullptr;
        // fftw_plan dst_plan_z = nullptr;
        std::vector<fftw_plan> dst_plans_x;
        std::vector<fftw_plan> dst_plans_y;
        std::vector<fftw_plan> dst_plans_z;
        #elif defined(USE_KML)
        // KML placeholder (copy of FFTW implementation for now)
        // fftw_plan dst_plan_x = nullptr;
        // fftw_plan dst_plan_y = nullptr;
        // fftw_plan dst_plan_z = nullptr;
        std::vector<fftw_plan> dst_plans_x;
        std::vector<fftw_plan> dst_plans_y;
        std::vector<fftw_plan> dst_plans_z;
        #endif

        void init(const Vertices_3D& local_vertices,
                  const Stencil<T>& stencil);
        template<typename T2> void init(const DST_Preconditioner_Data<T2>& DST_preconditioner_data);
        void destructor();
        void show() const;
    };

    /**
     * @brief DST-I based preconditioner: applies inv(L_local) * r using DST-I (Dirichlet BCs)
     */
    template<typename T>
    void dst_poisson_preconditioner(DST_Preconditioner_Data<T>& data,
                                     const int& N, const T* r, T* f, const Stencil<T>& stencil);

    int64_t generate_gradient_stride(const Vertices_3D& ex_vertices, const uint& dir);
    template<typename T> const T* generate_gradient_stencil_coeffs(const Stencil<T>& stencil, const uint dir);
    template<typename T1, typename T2>
    void calc_gradient(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                       const uint dir, const Stencil<T2>& stencil, const Vertices_3D& region,
                       T1* const ptr, const Vertices_3D& vertices);
    template<typename T1, typename T2>
    void calc_gradient_d3_c2(T1 const* const ex_ptr, const Vertices_3D& ex_vertices, const int64_t stride,
                             T2 const* const stencil_coeffs, const int FDn,
                             const Vertices_3D& region, T1* const ptr, const Vertices_3D& vertices);
    template<typename T1, typename T2>
    void calc_gradient_d4(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                        const uint dir, const Stencil<T2>& stencil, const Vertices_4D& region,
                        T1* const ptr, const Vertices_4D& vertices);
    template<typename T1, typename T2>
    void calc_gradient_d4_c2(T1 const* const ex_ptr, const Vertices_4D& ex_vertices, const int64_t stride,
                            T2 const* const stencil_coeffs, const int FDn,
                            const Vertices_4D& region, T1* const ptr, const Vertices_4D& vertices);
    template<typename T1, typename T2>
    void calc_laplacian(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                        const Stencil<T2>& stencil, const Vertices_3D& region,
                        T1* const ptr, const Vertices_3D& vertices);
    template<typename T1, typename T2, typename T3>
    void calc_laplacian(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                        const Stencil<T2>& stencil, const Vertices_3D& region,
                        T1* const ptr, const Vertices_3D& vertices,
                        T3 const* const ptr_A);
    template<typename T1, typename T2>
    void calc_laplacian(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                        const Stencil<T2>& stencil, const Vertices_3D& region,
                        T1* const ptr, const Vertices_3D& vertices,
                        T1 const* const ptr_A, const T2 alpha);
    template<typename T1, typename T2>
    void calc_laplacian(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                        const Stencil<T2>& stencil, const Vertices_3D& region,
                        T1* const ptr, const Vertices_3D& vertices,
                        T1 const* const ptr_A, const T2 alpha,
                        T1 const* const ptr_B, const T2 beta);
    template<typename T1, typename T2>
    void calc_laplacian(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                        const Stencil<T2>& stencil, const Vertices_3D& region,
                        T1* const ptr, const Vertices_3D& vertices,
                        T1 const* const ptr_A, const T2 alpha,
                        T1 const* const ptr_B, const T2 beta,
                        const T2 gamma);
    template<typename T1, typename T2>
    void calc_laplacian_d3_c2_o0(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_3D& region,
                                T1* const ptr, const Vertices_3D& vertices);
    template<typename T1, typename T2, typename T3>
    void calc_laplacian_d3_c2_o0(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_3D& region,
                                T1* const ptr, const Vertices_3D& vertices,
                                T3 const* const ptr_A);
    template<typename T1, typename T2>
    void calc_laplacian_d3_c2_o0(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_3D& region,
                                T1* const ptr, const Vertices_3D& vertices,
                                T1 const* const ptr_A, const T2 alpha);
    template<typename T1, typename T2>
    void calc_laplacian_d3_c2_o0(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_3D& region,
                                T1* const ptr, const Vertices_3D& vertices,
                                T1 const* const ptr_A, const T2 alpha,
                                T1 const* const ptr_B, const T2 beta);
    template<typename T1, typename T2>
    void calc_laplacian_d3_c2_o0(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_3D& region,
                                T1* const ptr, const Vertices_3D& vertices,
                                T1 const* const ptr_A, const T2 alpha,
                                T1 const* const ptr_B, const T2 beta,
                                const T2 gamma);
    template<typename T1, typename T2>
    void calc_laplacian_d3_c2_o4(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_3D& region,
                                T1* const ptr, const Vertices_3D& vertices);
    template<typename T1, typename T2, typename T3>
    void calc_laplacian_d3_c2_o4(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_3D& region,
                                T1* const ptr, const Vertices_3D& vertices,
                                T3 const* const ptr_A);
    template<typename T1, typename T2>
    void calc_laplacian_d3_c2_o4(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_3D& region,
                                T1* const ptr, const Vertices_3D& vertices,
                                T1 const* const ptr_A, const T2 alpha);
    template<typename T1, typename T2>
    void calc_laplacian_d3_c2_o4(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_3D& region,
                                T1* const ptr, const Vertices_3D& vertices,
                                T1 const* const ptr_A, const T2 alpha,
                                T1 const* const ptr_B, const T2 beta);
    template<typename T1, typename T2>
    void calc_laplacian_d3_c2_o4(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_3D& region,
                                T1* const ptr, const Vertices_3D& vertices,
                                T1 const* const ptr_A, const T2 alpha,
                                T1 const* const ptr_B, const T2 beta,
                                const T2 gamma);
    template<typename T1, typename T2>
    void calc_laplacian_d4(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                            const Stencil<T2>& stencil, const Vertices_4D& region,
                            T1* const ptr, const Vertices_4D& vertices);
    template<typename T1, typename T2>
    void calc_laplacian_d4(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                            const Stencil<T2>& stencil, const Vertices_4D& region,
                            T1* const ptr, const Vertices_4D& vertices,
                            T1 const* const ptr_A);
    template<typename T1, typename T2>
    void calc_laplacian_d4(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                            const Stencil<T2>& stencil, const Vertices_4D& region,
                            T1* const ptr, const Vertices_4D& vertices,
                            T1 const* const ptr_A, const T2 alpha);
    template<typename T1, typename T2>
    void calc_laplacian_d4(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                        const Stencil<T2>& stencil, const Vertices_4D& region,
                        T1* const ptr, const Vertices_4D& vertices,
                        T1 const* const ptr_A, const T2 alpha,
                        T1 const* const ptr_B, const T2 beta);
    template<typename T1, typename T2>
    void calc_laplacian_d4(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                            const Stencil<T2>& stencil, const Vertices_4D& region,
                            T1* const ptr, const Vertices_4D& vertices,
                            T1 const* const ptr_A, const T2 alpha,
                            T1 const* const&ptr_B, const T2 beta,
                            const T2 gamma);
    template<typename T1, typename T2>
    void calc_laplacian_d4_c2_o0(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_4D& region,
                                T1* const ptr, const Vertices_4D& vertices);
    template<typename T1, typename T2>
    void calc_laplacian_d4_c2_o0(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_4D& region,
                                T1* const ptr, const Vertices_4D& vertices,
                                T1 const* const ptr_A);
    template<typename T1, typename T2>
    void calc_laplacian_d4_c2_o0(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_4D& region,
                                T1* const ptr, const Vertices_4D& vertices,
                                T1 const* const ptr_A, const T2 alpha);
    template<typename T1, typename T2>
    void calc_laplacian_d4_c2_o0(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_4D& region,
                                T1* const ptr, const Vertices_4D& vertices,
                                T1 const* const ptr_A, const T2 alpha,
                                T1 const* const ptr_B, const T2 beta);
    template<typename T1, typename T2>
    void calc_laplacian_d4_c2_o0(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_4D& region,
                                T1* const ptr, const Vertices_4D& vertices,
                                T1 const* const ptr_A, const T2 alpha,
                                T1 const* const ptr_B, const T2 beta,
                                const T2 gamma);
    template<typename T1, typename T2>
    void calc_laplacian_d4_c2_o4(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_4D& region,
                                T1* const ptr, const Vertices_4D& vertices);
    template<typename T1, typename T2>
    void calc_laplacian_d4_c2_o4(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_4D& region,
                                T1* const ptr, const Vertices_4D& vertices,
                                T1 const* const ptr_A);
    template<typename T1, typename T2>
    void calc_laplacian_d4_c2_o4(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_4D& region,
                                T1* const ptr, const Vertices_4D& vertices,
                                T1 const* const ptr_A, const T2 alpha);
    template<typename T1, typename T2>
    void calc_laplacian_d4_c2_o4(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_4D& region,
                                T1* const ptr, const Vertices_4D& vertices,
                                T1 const* const ptr_A, const T2 alpha,
                                T1 const* const ptr_B, const T2 beta);
    template<typename T1, typename T2>
    void calc_laplacian_d4_c2_o4(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_4D& region,
                                T1* const ptr, const Vertices_4D& vertices,
                                T1 const* const ptr_A, const T2 alpha,
                                T1 const* const ptr_B, const T2 beta,
                                const T2 gamma);
    namespace Special {
        template<typename T1, typename T2>
        void calc_laplacian_d4(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                                const Stencil<T2>& stencil, const Vertices_4D& region,
                                T1* const ptr, const Vertices_4D& vertices,
                                T1 const* const ptr_A);
        template<typename T1, typename T2>
        void calc_laplacian_d4_c2_o0(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                                    const Stencil<T2>& stencil, const Vertices_4D& region,
                                    T1* const ptr, const Vertices_4D& vertices,
                                    T1 const* const ptr_A);
        template<typename T1, typename T2>
        void calc_laplacian_d4_c2_o4(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                                    const Stencil<T2>& stencil, const Vertices_4D& region,
                                    T1* const ptr, const Vertices_4D& vertices,
                                    T1 const* const ptr_A);
        template<typename T1, typename T2>
        void calc_laplacian_d4_c2_o4_rowmaj(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                                            const Stencil<T2>& stencil, const Vertices_4D& region,
                                            T1* const ptr, const Vertices_4D& vertices,
                                            T1 const* const ptr_A);
    }

    namespace Boundary_safe {
        template<typename T1, typename T2>
        void calc_laplacian_boundary_safe(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                                            const Stencil<T2>& stencil, const Vertices_3D& region,
                                            T1* const ptr, const Vertices_3D& vertices,
                                            T1 const* const ptr_A);
        template<typename T1, typename T2>
        void calc_laplacian_boundary_safe_d3_c2_o0(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                                                    const Stencil<T2>& stencil, const Vertices_3D& region,
                                                    T1* const ptr, const Vertices_3D& vertices,
                                                    T1 const* const ptr_A);
        template<typename T1, typename T2>
        void calc_laplacian_boundary_safe_d3_c2_o4(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                                                    const Stencil<T2>& stencil, const Vertices_3D& region,
                                                    T1* const ptr, const Vertices_3D& vertices,
                                                    T1 const* const ptr_A);
        template<typename T>
        void calc_laplacian_boundary_safe(T const* const ex_ptr, const Vertices_4D& ex_vertices,
                                        const Stencil<T>& stencil, const Vertices_4D& region,
                                        T* const ptr, const Vertices_4D& vertices,
                                        T const* const ptr_A);
    } // Boundary_safe
}

#endif //_STENCIL_H_
