#include "hamiltonian.h"

namespace Hamiltonian {

template<typename T>
void hamiltonian_product_vectors(Array_3D<T>& result, const Array_3D<T>& eigen_vector, const Array_3D<T>& Vloc,
                                 const Effective_potential_nloc<T>& Vnloc, const Stencil<T>& stencil, const T& dv,
                                 const Exarr_3D_mpi_package& exarr_mpi_package) {
    Stencil<T> stencil_temp(stencil.coeffs_scale(-0.5, 2));
    Vertices_3D ex_vertice(eigen_vector.get_vertices().generate_ex_vertices(stencil_temp.FDn));
    Array_3D<T> ex_eigen_vectors(ex_vertice, 0);
    exarr_mpi_package.fill_domain_par_ex_arr(eigen_vector, ex_eigen_vectors);
    // eigen_vectors_temp = (-0.5 \nabla + Vloc ) eigen_vectors
    Stencil_method::calc_laplacian(ex_eigen_vectors.data, ex_vertice, stencil_temp, eigen_vector.get_vertices(),
                                               result.data, eigen_vector.get_vertices(), Vloc.data);
    // eigen_vectors_temp += Vnloc*eigen_vectors;
    Hamiltonian::nloc_project_vectors<T>(result, eigen_vector, Vnloc, dv, exarr_mpi_package.comm);
    return;
}
template void hamiltonian_product_vectors<float>(Array_3D<float>& result, const Array_3D<float>& eigen_vector, const Array_3D<float>& Vloc,
                                                 const Effective_potential_nloc<float>& Vnloc, const Stencil<float>& stencil, const float& dv,
                                                 const Exarr_3D_mpi_package& exarr_mpi_package);
template void hamiltonian_product_vectors<double>(Array_3D<double>& result, const Array_3D<double>& eigen_vector, const Array_3D<double>& Vloc,
                                                  const Effective_potential_nloc<double>& Vnloc, const Stencil<double>& stencil, const double& dv,
                                                  const Exarr_3D_mpi_package& exarr_mpi_package);

template<typename T>
void hamiltonian_product_vectors(T* const& result, const Vertices_3D& vertices,
                                 T const* const& eigen_vector, T const* const& Vloc,
                                 const Effective_potential_nloc<T>& Vnloc, const Stencil<T>& stencil, 
                                 const T& dv, const Exarr_3D_mpi_package& exarr_mpi_package) {
    #ifdef USE_OPENMP

    static Stencil<T> stencil_temp;
    #pragma omp single
    stencil_temp.deepcopy(stencil);
    stencil_temp.coeffs_scale_self(-0.5);
    #pragma omp barrier
    Vertices_3D ex_vertice(vertices.generate_ex_vertices(stencil_temp.FDn));
    // Array_3D<T> ex_eigen_vector(ex_vertice, 0);
    static T* ex_eigen_vector_static = nullptr;
    #pragma omp single
    ex_eigen_vector_static = new T [ex_vertice.get_size()]();
    T* const ex_eigen_vector = ex_eigen_vector_static;

    exarr_mpi_package.fill_domain_par_ex_arr(eigen_vector, vertices,
                                             ex_eigen_vector, ex_vertice);
    #pragma omp barrier
    Stencil_method::calc_laplacian(ex_eigen_vector, ex_vertice, stencil_temp, vertices,
                                   result, vertices, Vloc);
    
    #pragma omp barrier
    #pragma omp single nowait
    {
        stencil_temp.destructor();
    }
    #pragma omp single nowait
    {
        delete [] ex_eigen_vector_static;
        ex_eigen_vector_static = nullptr;
    }
    
    Hamiltonian::nloc_project_vectors<T>(result, eigen_vector, Vnloc, dv, exarr_mpi_package.comm);

    #else //USE_OPENMP

    Stencil<T> stencil_temp(stencil.coeffs_scale(-0.5, 2));
    Vertices_3D ex_vertice(vertices.generate_ex_vertices(stencil_temp.FDn));
    // Array_3D<T> ex_eigen_vector(ex_vertice, 0);
    T* ex_eigen_vector = new T[ex_vertice.get_size()]();
    exarr_mpi_package.fill_domain_par_ex_arr(eigen_vector, vertices,
                                             ex_eigen_vector, ex_vertice);
    // eigen_vectors_temp = (-0.5 \nabla + Vloc ) eigen_vectors
    Stencil_method::calc_laplacian(ex_eigen_vector, ex_vertice, stencil_temp, vertices,
                                   result, vertices, Vloc);
    // eigen_vectors_temp += Vnloc*eigen_vectors;
    Hamiltonian::nloc_project_vectors<T>(result, eigen_vector, Vnloc, dv, exarr_mpi_package.comm);
    delete [] ex_eigen_vector;

    #endif //USE_OPENMP
    return;
}
template void hamiltonian_product_vectors<float>(float* const& result, const Vertices_3D& vertices,
                                                 float const* const& eigen_vectors, float const* const& Vloc,
                                                 const Effective_potential_nloc<float>& Vnloc, const Stencil<float>& stencil, 
                                                 const float& dv, const Exarr_3D_mpi_package& exarr_mpi_package);
template void hamiltonian_product_vectors<double>(double* const& result, const Vertices_3D& vertices,
                                                  double const* const& eigen_vectors, double const* const& Vloc,
                                                  const Effective_potential_nloc<double>& Vnloc, const Stencil<double>& stencil, 
                                                  const double& dv, const Exarr_3D_mpi_package& exarr_mpi_package);

template<typename T>
void hamiltonian_product_vectors(Array_4D<T>& result, const Array_4D<T>& eigen_vectors, const Array_3D<T>& Vloc,
                                 const Effective_potential_nloc<T>& Vnloc, const Stencil<T>& stencil,
                                 const T& dv, const Exarr_4D_mpi_package& exarr_mpi_package) {
    Stencil<T> stencil_temp(stencil.coeffs_scale(-0.5, 2));
    Vertices_4D ex_vertices(eigen_vectors.Vertices_3D::get_vertices().generate_ex_vertices(stencil_temp.FDn),
                           eigen_vectors.bs, eigen_vectors.get_be());
    Array_4D<T> ex_eigen_vectors(ex_vertices, 0);
    exarr_mpi_package.fill_domain_par_ex_arr(eigen_vectors, ex_eigen_vectors);
    // eigen_vectors_temp = (-0.5 \nabla + Vloc ) eigen_vectors
    Stencil_method::Special::calc_laplacian_d4(ex_eigen_vectors.data, ex_vertices, stencil_temp, eigen_vectors.get_vertices(),
                                               result.data, eigen_vectors.get_vertices(), Vloc.data);
    // eigen_vectors_temp += Vnloc*eigen_vectors;
    Hamiltonian::nloc_project_vectors<T>(result, eigen_vectors, Vnloc, dv, exarr_mpi_package.comm);
    return;
}
template void hamiltonian_product_vectors<float>(Array_4D<float>& result, const Array_4D<float>& eigen_vectors, const Array_3D<float>& Vloc,
                                                 const Effective_potential_nloc<float>& Vnloc, const Stencil<float>& stencil,
                                                 const float& dv, const Exarr_4D_mpi_package& exarr_mpi_package);
template void hamiltonian_product_vectors<double>(Array_4D<double>& result, const Array_4D<double>& eigen_vectors, const Array_3D<double>& Vloc,
                                                  const Effective_potential_nloc<double>& Vnloc, const Stencil<double>& stencil,
                                                  const double& dv, const Exarr_4D_mpi_package& exarr_mpi_package);

template<typename T>
void hamiltonian_product_vectors(T* const& result,  const Vertices_4D& vertices, T const* const& eigen_vectors,
                                 T const* const& Vloc, const Effective_potential_nloc<T>& Vnloc, const Stencil<T>& stencil, 
                                 const T& dv, const Exarr_4D_mpi_package& exarr_mpi_package) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    Stencil<T> stencil_temp(stencil.coeffs_scale(-0.5, 2));
    Vertices_4D ex_vertice(vertices.Vertices_3D::get_vertices().generate_ex_vertices(stencil_temp.FDn),
                           vertices.bs, vertices.get_be());
    // Array_4D<T> ex_eigen_vectors(ex_vertice, 0);
    T* ex_eigen_vectors = new T [ex_vertice.get_size()]();
    exarr_mpi_package.fill_domain_par_ex_arr(eigen_vectors, vertices,
                                             ex_eigen_vectors, ex_vertice);
    // eigen_vectors_temp = (-0.5 \nabla + Vloc ) eigen_vectors
    Stencil_method::Special::calc_laplacian_d4(ex_eigen_vectors, ex_vertice, stencil_temp, vertices,
                                               result, vertices, Vloc);
    // eigen_vectors_temp += Vnloc*eigen_vectors;
    Hamiltonian::nloc_project_vectors<T>(result, vertices, eigen_vectors, Vnloc, dv, exarr_mpi_package.comm);
    delete [] ex_eigen_vectors;
    int rank;
    MPI_Comm_rank(exarr_mpi_package.comm, &rank);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (rank == 0) {
        std::cout << "The hamiltonian_product_vectors took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}
template void hamiltonian_product_vectors<float>(float* const& result,  const Vertices_4D& vertices, float const* const& eigen_vectors,
                                                 float const* const& Vloc, const Effective_potential_nloc<float>& Vnloc, const Stencil<float>& stencil, 
                                                 const float& dv, const Exarr_4D_mpi_package& exarr_mpi_package);
template void hamiltonian_product_vectors<double>(double* const& result,  const Vertices_4D& vertices, double const* const& eigen_vectors,
                                                  double const* const& Vloc, const Effective_potential_nloc<double>& Vnloc, const Stencil<double>& stencil, 
                                                  const double& dv, const Exarr_4D_mpi_package& exarr_mpi_package);

template<typename T>
void hamiltonian_product_vectors_column_wise(T* const& result,  const Vertices_4D& vertices, T const* const& eigen_vectors,
                                             T const* const& Vloc, const Effective_potential_nloc<T>& Vnloc, const Stencil<T>& stencil, 
                                             const T& dv, const Exarr_3D_mpi_package& exarr_mpi_package) {
    Vertices_3D vertices_3d = vertices.Vertices_3D::get_vertices();
    const uint Nd = vertices_3d.get_size();
    if (Nd == 0 || vertices.nb == 0) return;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (uint ib = 0; ib < vertices.nb; ib++) {
        hamiltonian_product_vectors(result + ib * Nd, vertices_3d, eigen_vectors + ib * Nd,
                                    Vloc, Vnloc, stencil, dv, exarr_mpi_package);
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (rank == 0) {
        std::cout << "The hamiltonian_product_vectors_column_wise took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}
template void hamiltonian_product_vectors_column_wise<float>(float* const& result,  const Vertices_4D& vertices, float const* const& eigen_vectors,
                                                             float const* const& Vloc, const Effective_potential_nloc<float>& Vnloc, const Stencil<float>& stencil, 
                                                             const float& dv, const Exarr_3D_mpi_package& exarr_mpi_package);
template void hamiltonian_product_vectors_column_wise<double>(double* const& result,  const Vertices_4D& vertices, double const* const& eigen_vectors,
                                                              double const* const& Vloc, const Effective_potential_nloc<double>& Vnloc, const Stencil<double>& stencil, 
                                                              const double& dv, const Exarr_3D_mpi_package& exarr_mpi_package);

template<typename T>
void hamiltonian_product_vectors_column_wise2(T* const& result,  const Vertices_4D& vertices, T const* const& eigen_vectors,
                                              T const* const& Vloc, const Effective_potential_nloc<T>& Vnloc, const Stencil<T>& stencil, 
                                              const T& dv, const Exarr_3D_mpi_package& exarr_mpi_package, const bool& print_flag) {

    #ifdef USE_OPENMP

    // hamiltonian_product_vectors_column_wise2_omp<T>(result, vertices, eigen_vectors, Vloc, Vnloc, stencil, dv, exarr_mpi_package, print_flag);
    hamiltonian_product_vectors_column_wise2_omp_comm_self<T>(result, vertices, eigen_vectors, Vloc, Vnloc, stencil, dv, exarr_mpi_package, print_flag);

    #else //USE_OPENMP

    Vertices_3D vertices_3d = vertices.Vertices_3D::get_vertices();
    const uint Nd = vertices_3d.get_size();
    if (Nd == 0 || vertices.nb == 0) return;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    Stencil<T> stencil_temp(stencil.coeffs_scale(-0.5, 2));
    Vertices_3D ex_vertice(vertices_3d.generate_ex_vertices(stencil_temp.FDn));
    T* ex_eigen_vector = new T[ex_vertice.get_size()]();
    // result = (-0.5 \nabla + Vloc ) eigen_vectors
    for (uint ib = 0; ib < vertices.nb; ib++) {
        exarr_mpi_package.fill_domain_par_ex_arr(eigen_vectors + ib * Nd, vertices_3d,
                                                 ex_eigen_vector, ex_vertice);
        Stencil_method::calc_laplacian(ex_eigen_vector, ex_vertice, stencil_temp, vertices_3d,
                                       result + ib * Nd, vertices_3d, Vloc);
    }
    delete [] ex_eigen_vector;
     // result += Vnloc*eigen_vectors;
    Hamiltonian::nloc_project_vectors<T>(result, vertices, eigen_vectors, Vnloc, dv, exarr_mpi_package.comm);
    int rank;
    MPI_Comm_rank(exarr_mpi_package.comm, &rank);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (rank == 0 && print_flag) {
        std::cout << "The hamiltonian_product_vectors_column_wise2 took " << Tools::time_cost(begin, end) << "." << std::endl;
    }

    #endif //USE_OPENMP

    return;
}
template void hamiltonian_product_vectors_column_wise2<float>(float* const& result,  const Vertices_4D& vertices, float const* const& eigen_vectors,
                                                              float const* const& Vloc, const Effective_potential_nloc<float>& Vnloc, const Stencil<float>& stencil,
                                                              const float& dv, const Exarr_3D_mpi_package& exarr_mpi_package, const bool& print_flag);
template void hamiltonian_product_vectors_column_wise2<double>(double* const& result,  const Vertices_4D& vertices, double const* const& eigen_vectors,
                                                               double const* const& Vloc, const Effective_potential_nloc<double>& Vnloc, const Stencil<double>& stencil,
                                                               const double& dv, const Exarr_3D_mpi_package& exarr_mpi_package, const bool& print_flag);
template<typename T>
void hamiltonian_product_vectors_column_wise2_specialization(T* const& result,  const Vertices_4D& vertices, T const* const& eigen_vectors,
                                              T const* const& Vloc, const Effective_potential_nloc<T>& Vnloc, const Stencil<T>& stencil, 
                                              const T& dv, const Exarr_3D_mpi_package& exarr_mpi_package, const bool& print_flag) {
    #ifdef USE_OPENMP
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    Vertices_3D vertices_3d = vertices.Vertices_3D::get_vertices();
    const uint Nd = vertices_3d.get_size();
    if (Nd == 0 || vertices.nb == 0) return;
    
    static Stencil<T> stencil_temp;
    stencil_temp.deepcopy(stencil);
    stencil_temp.coeffs_scale_self(-0.5);

    // result = (-0.5 \nabla + Vloc ) eigen_vectors
    #pragma omp parallel
    Stencil_method::Boundary_safe::calc_laplacian_boundary_safe(eigen_vectors, vertices, stencil_temp, vertices,
                                            result, vertices, Vloc);
    #pragma omp parallel
    Hamiltonian::nloc_project_vectors<T>(result, vertices, eigen_vectors, Vnloc, dv, exarr_mpi_package.comm);
    stencil_temp.destructor();
    int rank;
    MPI_Comm_rank(exarr_mpi_package.comm, &rank);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (rank == 0 && print_flag) {
        std::cout << "The hamiltonian_product_vectors_column_wise2_specialization took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    #else //USE_OPENMP
    (void) result;
    (void) vertices;
    (void) eigen_vectors;
    (void) Vloc;
    (void) Vnloc;
    (void) stencil;
    (void) dv;
    (void) exarr_mpi_package;
    (void) print_flag;
    assert(false);
    #endif //USE_OPENMP

    return;
}
template void hamiltonian_product_vectors_column_wise2_specialization<float>(float* const& result,  const Vertices_4D& vertices, float const* const& eigen_vectors,
                                                              float const* const& Vloc, const Effective_potential_nloc<float>& Vnloc, const Stencil<float>& stencil,
                                                              const float& dv, const Exarr_3D_mpi_package& exarr_mpi_package, const bool& print_flag);
template void hamiltonian_product_vectors_column_wise2_specialization<double>(double* const& result,  const Vertices_4D& vertices, double const* const& eigen_vectors,
                                                               double const* const& Vloc, const Effective_potential_nloc<double>& Vnloc, const Stencil<double>& stencil,
                                                               const double& dv, const Exarr_3D_mpi_package& exarr_mpi_package, const bool& print_flag);

#ifdef USE_OPENMP
template<typename T>
inline void hamiltonian_product_vectors_column_wise2_omp(T* const& result,  const Vertices_4D& vertices, T const* const& eigen_vectors,
                                              T const* const& Vloc, const Effective_potential_nloc<T>& Vnloc, const Stencil<T>& stencil,
                                              const T& dv, const Exarr_3D_mpi_package& exarr_mpi_package, const bool& print_flag) {
    Vertices_3D vertices_3d = vertices.Vertices_3D::get_vertices();
    const uint Nd = vertices_3d.get_size();
    if (Nd == 0 || vertices.nb == 0) return;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    static Stencil<T> stencil_temp;
    #pragma omp single
    stencil_temp.deepcopy(stencil);
    stencil_temp.coeffs_scale_self(-0.5);
    #pragma omp barrier
    Vertices_3D ex_vertice(vertices_3d.generate_ex_vertices(stencil_temp.FDn));
    static T* ex_eigen_vector_static = nullptr;
    #pragma omp single
    ex_eigen_vector_static = new T [ex_vertice.get_size()]();
    T* const ex_eigen_vector = ex_eigen_vector_static;

    // result = (-0.5 \nabla + Vloc ) eigen_vectors
    for (uint ib = 0; ib < vertices.nb; ib++) {
        exarr_mpi_package.fill_domain_par_ex_arr(eigen_vectors + ib * Nd, vertices_3d,
                                                 ex_eigen_vector, ex_vertice);
        #pragma omp barrier
        Stencil_method::calc_laplacian(ex_eigen_vector, ex_vertice, stencil_temp, vertices_3d,
                                       result + ib * Nd, vertices_3d, Vloc);
        #pragma omp barrier
    }
    #pragma omp single nowait
    {
        stencil_temp.destructor();
    }
    #pragma omp single nowait
    {
        delete [] ex_eigen_vector_static;
        ex_eigen_vector_static = nullptr;
    }
     // result += Vnloc*eigen_vectors;
    Hamiltonian::nloc_project_vectors<T>(result, vertices, eigen_vectors, Vnloc, dv, exarr_mpi_package.comm);
    #pragma omp barrier
    #pragma omp single nowait
    {
    int rank;
    MPI_Comm_rank(exarr_mpi_package.comm, &rank);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (rank == 0 && print_flag) {
        std::cout << "The hamiltonian_product_vectors_column_wise2 took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    }
    return;
}
template void hamiltonian_product_vectors_column_wise2_omp<float>(float* const& result,  const Vertices_4D& vertices, float const* const& eigen_vectors,
                                                              float const* const& Vloc, const Effective_potential_nloc<float>& Vnloc, const Stencil<float>& stencil,
                                                              const float& dv, const Exarr_3D_mpi_package& exarr_mpi_package, const bool& print_flag);
template void hamiltonian_product_vectors_column_wise2_omp<double>(double* const& result,  const Vertices_4D& vertices, double const* const& eigen_vectors,
                                                               double const* const& Vloc, const Effective_potential_nloc<double>& Vnloc, const Stencil<double>& stencil,
                                                               const double& dv, const Exarr_3D_mpi_package& exarr_mpi_package, const bool& print_flag);
template<typename T>
inline void hamiltonian_product_vectors_column_wise2_omp_comm_self(T* const& result,  const Vertices_4D& vertices, T const* const& eigen_vectors,
                                              T const* const& Vloc, const Effective_potential_nloc<T>& Vnloc, const Stencil<T>& stencil,
                                              const T& dv, const Exarr_3D_mpi_package& exarr_mpi_package, const bool& print_flag) {
    if (unlikely(exarr_mpi_package.need_comm)) {
        if (unlikely(print_flag)) {
            std::cout << RED << "WARNING: hamiltonian_product_vectors_column_wise2_omp_comm_self only support single MPI process in the communicator." << RESET << std::endl;
            std::cout << RED << "WARNING: calling hamiltonian_product_vectors_column_wise2_omp instead." << RESET << std::endl;
        }
        hamiltonian_product_vectors_column_wise2_omp<T>(result, vertices, eigen_vectors, Vloc, Vnloc, stencil, dv, exarr_mpi_package, print_flag);
        return;
    }
    Vertices_3D vertices_3d = vertices.Vertices_3D::get_vertices();
    const uint Nd = vertices_3d.get_size();
    if (Nd == 0 || vertices.nb == 0) return;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    static Stencil<T> stencil_temp;
    #pragma omp single
    stencil_temp.deepcopy(stencil);
    stencil_temp.coeffs_scale_self(-0.5);
    #pragma omp barrier

    // result = (-0.5 \nabla + Vloc ) eigen_vectors
    Stencil_method::Boundary_safe::calc_laplacian_boundary_safe(eigen_vectors, vertices, stencil_temp, vertices,
                                            result, vertices, Vloc);
    #pragma omp barrier
    // result += Vnloc*eigen_vectors;
    Hamiltonian::nloc_project_vectors<T>(result, vertices, eigen_vectors, Vnloc, dv, exarr_mpi_package.comm);
    #pragma omp barrier
    #pragma omp single nowait
    {
        stencil_temp.destructor();
    }
    #pragma omp single nowait
    {
    int rank;
    MPI_Comm_rank(exarr_mpi_package.comm, &rank);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (rank == 0 && print_flag) {
        std::cout << "The hamiltonian_product_vectors_column_wise2 took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    }
    return;
}
template void hamiltonian_product_vectors_column_wise2_omp_comm_self<float>(float* const& result,  const Vertices_4D& vertices, float const* const& eigen_vectors,
                                                              float const* const& Vloc, const Effective_potential_nloc<float>& Vnloc, const Stencil<float>& stencil,
                                                              const float& dv, const Exarr_3D_mpi_package& exarr_mpi_package, const bool& print_flag);
template void hamiltonian_product_vectors_column_wise2_omp_comm_self<double>(double* const& result,  const Vertices_4D& vertices, double const* const& eigen_vectors,
                                                               double const* const& Vloc, const Effective_potential_nloc<double>& Vnloc, const Stencil<double>& stencil,
                                                               const double& dv, const Exarr_3D_mpi_package& exarr_mpi_package, const bool& print_flag);
#endif //USE_OPENMP

template<typename T>
void nloc_project_vectors(Array_3D<T>& result,
                          const Array_3D<T>& eigen_vector,
                          const Effective_potential_nloc<T>& effective_potential_nloc,
                          const T& dv,
                          const MPI_Comm& comm) {
    nloc_project_vectors<T>(result.data, eigen_vector.data, effective_potential_nloc, dv, comm);
    return;
}
template void nloc_project_vectors<float>(Array_3D<float>& result, const Array_3D<float>& eigen_vector,
                                          const Effective_potential_nloc<float>& effective_potential_nloc,
                                          const float& dv, const MPI_Comm& comm);
template void nloc_project_vectors<double>(Array_3D<double>& result, const Array_3D<double>& eigen_vector,
                                           const Effective_potential_nloc<double>& effective_potential_nloc,
                                           const double& dv, const MPI_Comm& comm);

template<typename T>
void nloc_project_vectors(T* const& __restrict__ result, T const* const& __restrict__ eigen_vector,
                          const Effective_potential_nloc<T>& effective_potential_nloc,
                          const T& dv, const MPI_Comm& comm) {
    #ifdef USE_OPENMP

    uint length = *(effective_potential_nloc.offsets.end() - 1);
    static T* chi_vector_static = nullptr;
    #pragma omp single
    chi_vector_static = new T [length]();
    T* const chi_vector = chi_vector_static;

    for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++)
    {
        Nloc_projector_method::nloc_projector_product_vectors<T>(chi_vector + effective_potential_nloc.offsets[iprojector], eigen_vector,
                                                                 effective_potential_nloc.nloc_projectors[iprojector], dv);
    }

    #pragma omp barrier
    #pragma omp master
    {
    int commsize;
    MPI_Comm_size(comm, &commsize);
    if (commsize > 1) {
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        MPI_Allreduce(MPI_IN_PLACE, chi_vector, length, mpi_datatype, MPI_SUM, comm);
    }
    }
    #pragma omp barrier

    int atom_index = effective_potential_nloc.nloc_projectors.cbegin()->atom_index-1;
    for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
        if (atom_index != effective_potential_nloc.nloc_projectors[iprojector].atom_index) {
            atom_index = effective_potential_nloc.nloc_projectors[iprojector].atom_index;
            Linalg::hadamard_product_general(chi_vector + effective_potential_nloc.offsets[iprojector],
                effective_potential_nloc.nloc_projectors[iprojector].gamma.data, effective_potential_nloc.nloc_projectors[iprojector].ncol);
        }
    }
    #pragma omp barrier

    for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
        const uint& ncol = effective_potential_nloc.nloc_projectors[iprojector].ncol;
        const uint& nrow = effective_potential_nloc.nloc_projectors[iprojector].nrow;
        if (ncol * nrow > 0) {
            #pragma omp barrier
            static T* temp_static = nullptr;
            #pragma omp single
            temp_static = new T [nrow];
            T* const __restrict__ temp = temp_static;

            Linalg::matrix_vector_product(effective_potential_nloc.nloc_projectors[iprojector].chi.data, 1,
                                          chi_vector + effective_potential_nloc.offsets[iprojector], temp, nrow, ncol);
            #pragma omp barrier
            uint const* __restrict__ index_data = effective_potential_nloc.nloc_projectors[iprojector].index_data();
            #ifdef USE_OPENMP_SIMD
            #pragma omp for simd schedule(static, (nrow - 1)/omp_get_num_threads() + 1) nowait
            #else
            #pragma omp for schedule(static, (nrow - 1)/omp_get_num_threads() + 1) nowait
            #endif
            for (uint i = 0; i < nrow; i++) {
                result[index_data[i]] += temp[i];
            }
            #pragma omp barrier
            #pragma omp single
            {
                delete [] temp_static;
                temp_static = nullptr;
            }
        }
    }

    #pragma omp barrier
    #pragma omp single nowait
    {
        delete [] chi_vector_static;
        chi_vector_static = nullptr;
    }

    #else //USE_OPENMP

    uint length = *(effective_potential_nloc.offsets.end() - 1);
    // Array_0D<T> chi_vector(length, 0);
    T* const chi_vector = new T [length]();
    std::vector<uint>::const_iterator it_offset = effective_potential_nloc.offsets.cbegin();
    for (typename std::vector<Nloc_projector<T>>::const_iterator it_nloc_projector = effective_potential_nloc.nloc_projectors.cbegin();
        it_nloc_projector != effective_potential_nloc.nloc_projectors.cend(); ++it_nloc_projector) {
        Nloc_projector_method::nloc_projector_product_vectors<T>(chi_vector + *it_offset, eigen_vector,
                                                                 *it_nloc_projector, dv);
        ++it_offset;
    }
    int commsize;
    MPI_Comm_size(comm, &commsize);
    if (commsize > 1) {
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        MPI_Allreduce(MPI_IN_PLACE, chi_vector, length, mpi_datatype, MPI_SUM, comm);
    }

    it_offset = effective_potential_nloc.offsets.cbegin();
    int atom_index = effective_potential_nloc.nloc_projectors.cbegin()->atom_index-1;
    for (typename std::vector<Nloc_projector<T>>::const_iterator it_nloc_projector = effective_potential_nloc.nloc_projectors.cbegin();
        it_nloc_projector != effective_potential_nloc.nloc_projectors.cend(); ++it_nloc_projector) {
        if (atom_index != it_nloc_projector->atom_index) {
            atom_index = it_nloc_projector->atom_index;
            Linalg::hadamard_product_general(chi_vector + *it_offset, it_nloc_projector->gamma.data, it_nloc_projector->ncol);
        }
        ++it_offset;
    }


    it_offset = effective_potential_nloc.offsets.cbegin();
    for (typename std::vector<Nloc_projector<T>>::const_iterator it_nloc_projector = effective_potential_nloc.nloc_projectors.cbegin();
        it_nloc_projector != effective_potential_nloc.nloc_projectors.cend(); ++it_nloc_projector) {
        const uint& ncol = it_nloc_projector->ncol;
        const uint& nrow = it_nloc_projector->nrow;
        if (ncol * nrow > 0) {
            // Linalg::hadamard_product_general(chi_vector + *it_offset,
            //                                  it_nloc_projector->gamma.data,
            //                                  it_nloc_projector->ncol);
            // Array_1D<T> temp(nrow);
            T* const __restrict__ temp = new T [nrow];
            Linalg::matrix_vector_product(it_nloc_projector->chi.data, 1,
                                          chi_vector + *it_offset, temp, nrow, ncol);
            uint const* __restrict__ index_data = it_nloc_projector->index.data();
            T const* __restrict__ temp_local = temp;
            for (uint i = 0; i < nrow; i++) {
                result[*index_data++] += *temp_local++;
            }
            delete [] temp;
        }
        ++it_offset;
    }
    delete [] chi_vector;

    #endif //USE_OPENMP
    return;
}
template void nloc_project_vectors<float>(float* const& result, float const* const& eigen_vector,
                                          const Effective_potential_nloc<float>& effective_potential_nloc,
                                          const float& dv, const MPI_Comm& comm);
template void nloc_project_vectors<double>(double* const& result, double const* const& eigen_vector,
                                           const Effective_potential_nloc<double>& effective_potential_nloc,
                                           const double& dv, const MPI_Comm& comm);

template<typename T>
void nloc_project_vectors(Array_4D<T>& result,
                          const Array_4D<T>& eigen_vectors,
                          const Effective_potential_nloc<T>& effective_potential_nloc,
                          const T& dv, const MPI_Comm& comm) {
    nloc_project_vectors(result.data, result.get_vertices(), eigen_vectors.data, effective_potential_nloc, dv, comm);
    return;
}
template void nloc_project_vectors<float>(Array_4D<float>& result, const Array_4D<float>& eigen_vectors,
                                          const Effective_potential_nloc<float>& effective_potential_nloc,
                                          const float& dv, const MPI_Comm& comm);
template void nloc_project_vectors<double>(Array_4D<double>& result, const Array_4D<double>& eigen_vectors,
                                           const Effective_potential_nloc<double>& effective_potential_nloc,
                                           const double& dv, const MPI_Comm& comm);

template<typename T>
void nloc_project_vectors(T* const& __restrict__ result, const Vertices_4D& vertices,
                          T const* const& __restrict__ eigen_vectors,
                          const Effective_potential_nloc<T>& effective_potential_nloc,
                          const T& dv, const MPI_Comm& comm) {
    #ifdef USE_OPENMP

    // nloc_project_vectors_omp_domain(result, vertices, eigen_vectors, effective_potential_nloc, dv, comm);
    // nloc_project_vectors_omp_for(result, vertices, eigen_vectors, effective_potential_nloc, dv, comm);
    nloc_project_vectors_omp_for_comm_self_with_chunk(result, vertices, eigen_vectors, effective_potential_nloc, dv, comm);
    // nloc_project_vectors_omp_task(result, vertices, eigen_vectors, effective_potential_nloc, dv, comm);

    #else //USE_OPENMP

    const uint& nb = vertices.nb;
    uint length = *(effective_potential_nloc.offsets.end() - 1) * nb;
    // Array_0D<T> chi_vector(length, 0);
    T* const chi_vector = new T [length]();
    std::vector<uint>::const_iterator it_offset = effective_potential_nloc.offsets.cbegin();
    for (typename std::vector<Nloc_projector<T>>::const_iterator it_nloc_projector = effective_potential_nloc.nloc_projectors.cbegin();
        it_nloc_projector != effective_potential_nloc.nloc_projectors.cend(); ++it_nloc_projector) {
        Nloc_projector_method::nloc_projector_product_vectors(chi_vector + (*it_offset) * nb, eigen_vectors,
                                                              vertices, *it_nloc_projector, dv);
        ++it_offset;
    }
    int commsize;
    MPI_Comm_size(comm, &commsize);
    if (commsize > 1) {
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        MPI_Allreduce(MPI_IN_PLACE, chi_vector, length, mpi_datatype, MPI_SUM, comm);
    }

    int atom_index = effective_potential_nloc.nloc_projectors.cbegin()->atom_index-1;

    it_offset = effective_potential_nloc.offsets.cbegin();
    for (typename std::vector<Nloc_projector<T>>::const_iterator it_nloc_projector = effective_potential_nloc.nloc_projectors.cbegin();
        it_nloc_projector != effective_potential_nloc.nloc_projectors.cend(); ++it_nloc_projector) {
        const uint& ncol = it_nloc_projector->ncol;
        const uint& nrow = it_nloc_projector->nrow;
        if (ncol * nrow > 0) {
            T* const __restrict__ chi_vector_local = chi_vector + (*it_offset) * nb;
            T const* const& __restrict__ gamma_data = it_nloc_projector->gamma.data;
            if (atom_index != it_nloc_projector->atom_index) {
                atom_index = it_nloc_projector->atom_index;
                uint count = 0;
                for (uint ib = 0; ib < nb; ib++) {
                    for (size_t icol = 0; icol < ncol; icol++) {
                        chi_vector_local[count] *= gamma_data[icol];
                        count++;
                    }
                }
            }

            // Array_2D<T> temp(Vertices_2D(nrow, nb));
            T* const __restrict__ temp = new T [nrow * nb];
            Linalg::matrix_product(it_nloc_projector->chi.data, 1, chi_vector_local, 1, temp, 1, nrow, nb, ncol);
            // uint const* const& __restrict__ index_data = it_nloc_projector->index.data();
            const uint vertices_3d_size = vertices.Vertices_3D::get_size();
            for (uint ib = 0; ib < nb; ib++) {
                uint const* __restrict__ index_data = it_nloc_projector->index.data();
                T* __restrict__ result_local = result + ib * vertices_3d_size;
                T const* __restrict__ temp_local = temp + ib * nrow;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (uint i = 0; i < nrow; i++) {
                    result_local[*index_data++] += *temp_local++;
                }
            }
            delete [] temp;
        }
        ++it_offset;
    }
    delete [] chi_vector;

    #endif //USE_OPENMP
    return;
}
template void nloc_project_vectors<float>(float* const& result, const Vertices_4D& vertices, float const* const& eigen_vectors,
                                          const Effective_potential_nloc<float>& effective_potential_nloc,
                                          const float& dv, const MPI_Comm& comm);
template void nloc_project_vectors<double>(double* const& result, const Vertices_4D& vertices, double const* const& eigen_vectors,
                                           const Effective_potential_nloc<double>& effective_potential_nloc,
                                           const double& dv, const MPI_Comm& comm);

#ifdef USE_OPENMP
template<typename T>
inline void nloc_project_vectors_omp_domain(T* const& __restrict__ result, const Vertices_4D& vertices,
                                        T const* const& __restrict__ eigen_vectors,
                                        const Effective_potential_nloc<T>& effective_potential_nloc,
                                        const T& dv, const MPI_Comm& comm) {

    const uint& nb = vertices.nb;
    uint length = *(effective_potential_nloc.offsets.end() - 1) * nb;
    // Array_0D<T> chi_vector(length, 0);
    static T* chi_vector_static = nullptr;
    #pragma omp single
    chi_vector_static = new (std::align_val_t(64)) T [length]();
    T* const chi_vector = chi_vector_static;

    for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
        Nloc_projector_method::nloc_projector_product_vectors(chi_vector + effective_potential_nloc.offsets[iprojector] * nb,
                                                        eigen_vectors, vertices, effective_potential_nloc.nloc_projectors[iprojector], dv);
    }

    #pragma omp barrier
    #pragma omp master
    {
    int commsize;
    MPI_Comm_size(comm, &commsize);
    if (commsize > 1) {
        MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
        MPI_Allreduce(MPI_IN_PLACE, chi_vector, length, mpi_datatype, MPI_SUM, comm);
    }
    }
    #pragma omp barrier

    int atom_index = effective_potential_nloc.nloc_projectors.cbegin()->atom_index-1;
    for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
        const uint& ncol = effective_potential_nloc.nloc_projectors[iprojector].ncol;
        const uint& nrow = effective_potential_nloc.nloc_projectors[iprojector].nrow;
        if (ncol * nrow > 0) {
            T* const __restrict__ chi_vector_local = chi_vector + effective_potential_nloc.offsets[iprojector] * nb;
            T const* const& __restrict__ gamma_data = effective_potential_nloc.nloc_projectors[iprojector].gamma.data;
            if (atom_index != effective_potential_nloc.nloc_projectors[iprojector].atom_index) {
                atom_index = effective_potential_nloc.nloc_projectors[iprojector].atom_index;
                #ifdef USE_OPENMP_SIMD
                #pragma omp for simd schedule(static, (nb * ncol - 1)/omp_get_num_threads() + 1) collapse(2) nowait
                #else
                #pragma omp for schedule(static, (nb * ncol - 1)/omp_get_num_threads() + 1) collapse(2) nowait
                #endif
                for (uint ib = 0; ib < nb; ib++) {
                    for (size_t icol = 0; icol < ncol; icol++) {
                        chi_vector_local[ib * ncol + icol] *= gamma_data[icol];
                    }
                }
            }

            #pragma omp barrier
            static T* temp_static = nullptr;
            #pragma omp single
            temp_static = new T [nrow * nb]();
            T* const __restrict__ temp = temp_static;

            Linalg::matrix_product(effective_potential_nloc.nloc_projectors[iprojector].chi.data, 1, chi_vector_local, 1, temp, 1, nrow, nb, ncol);
            #pragma omp barrier
            const uint vertices_3d_size = vertices.Vertices_3D::get_size();
            uint const* __restrict__ index_data = effective_potential_nloc.nloc_projectors[iprojector].index_data();
            #pragma omp barrier
            #ifdef USE_OPENMP_SIMD
            #pragma omp for simd schedule(static, (nb * nrow - 1)/omp_get_num_threads() + 1) collapse(2) nowait
            #else
            #pragma omp for schedule(static, (nb * nrow - 1)/omp_get_num_threads() + 1) collapse(2) nowait
            #endif
            for (uint ib = 0; ib < nb; ib++) {
                for (uint i = 0; i < nrow; i++) {
                    result[ib * vertices_3d_size + index_data[i]] += temp[ib * nrow + i];
                }
            }

            #pragma omp barrier
            #pragma omp single
            {
                delete [] temp_static;
                temp_static = nullptr;
            }
        }
    }

    #pragma omp barrier
    #pragma omp single nowait
    {
        delete [] chi_vector_static;
        chi_vector_static = nullptr;
    }
    
    return;
}
template void nloc_project_vectors_omp_domain<float>(float* const& result, const Vertices_4D& vertices, float const* const& eigen_vectors,
                                                    const Effective_potential_nloc<float>& effective_potential_nloc,
                                                    const float& dv, const MPI_Comm& comm);
template void nloc_project_vectors_omp_domain<double>(double* const& result, const Vertices_4D& vertices, double const* const& eigen_vectors,
                                                    const Effective_potential_nloc<double>& effective_potential_nloc,
                                                    const double& dv, const MPI_Comm& comm);
template<typename T>
inline void nloc_project_vectors_omp_for(T* const& __restrict__ result, const Vertices_4D& vertices,
                                            T const* const& __restrict__ eigen_vectors,
                                            const Effective_potential_nloc<T>& effective_potential_nloc,
                                            const T& dv, const MPI_Comm& comm) {
    const uint& nb = vertices.nb;
    uint length = *(effective_potential_nloc.offsets.end() - 1) * nb;
    static T* chi_vector_static = nullptr;
    #pragma omp single
    chi_vector_static = new (std::align_val_t(64)) T [length]();
    T* const chi_vector = chi_vector_static;
    const uint vertices_3d_size = vertices.Vertices_3D::get_size();
    #pragma omp for
    for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
        #pragma omp parallel if(0)
        {
            const uint chi_vector_task_length = nb * effective_potential_nloc.nloc_projectors[iprojector].ncol;
            T *const chi_vector_task = new (std::align_val_t(64)) T [chi_vector_task_length]();
            Nloc_projector_method::nloc_projector_product_vectors_sequential(chi_vector_task,
                        eigen_vectors, vertices, effective_potential_nloc.nloc_projectors[iprojector], dv);
            #pragma omp critical
            {
                Linalg::hadamard_plus_general(chi_vector + effective_potential_nloc.offsets[iprojector] * nb,
                                                chi_vector_task, chi_vector_task_length);
            }
            delete [] chi_vector_task;
        }
    }

    #pragma omp barrier
    #pragma omp master
    {
        int commsize;
        MPI_Comm_size(comm, &commsize);
        if (commsize > 1) {
            MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
            MPI_Allreduce(MPI_IN_PLACE, chi_vector, length, mpi_datatype, MPI_SUM, comm);
        }
    }
    #pragma omp barrier

    int atom_index = effective_potential_nloc.nloc_projectors.cbegin()->atom_index-1;
    for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
        const uint& ncol = effective_potential_nloc.nloc_projectors[iprojector].ncol;
        const uint& nrow = effective_potential_nloc.nloc_projectors[iprojector].nrow;
        if (likely(ncol * nrow > 0)) {
            T* const __restrict__ chi_vector_local = chi_vector + effective_potential_nloc.offsets[iprojector] * nb;
            if (atom_index != effective_potential_nloc.nloc_projectors[iprojector].atom_index) {
                atom_index = effective_potential_nloc.nloc_projectors[iprojector].atom_index;
                T const* const __restrict__ gamma_data = effective_potential_nloc.nloc_projectors[iprojector].gamma.data;
                #pragma omp for
                for (uint ib = 0; ib < nb; ib++) {
                    #pragma omp simd
                    for (uint icol = 0; icol < ncol; icol++) {
                        chi_vector_local[ib * ncol + icol] *= gamma_data[icol];
                    }
                }
            }
        }
    }

    #pragma omp for nowait
    for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
        const uint& ncol = effective_potential_nloc.nloc_projectors[iprojector].ncol;
        const uint& nrow = effective_potential_nloc.nloc_projectors[iprojector].nrow;
        if (ncol * nrow > 0) {
            T* const __restrict__ chi_vector_local = chi_vector + effective_potential_nloc.offsets[iprojector] * nb;
            #pragma omp parallel if(0)
            {
                T* const __restrict__ temp = new T [nrow * nb]();
                Linalg::matrix_product(effective_potential_nloc.nloc_projectors[iprojector].chi.data, 1, chi_vector_local, 1, temp, 1, nrow, nb, ncol);
                uint const* __restrict__ index_data = effective_potential_nloc.nloc_projectors[iprojector].index_data();
                #pragma omp critical
                for (uint ib = 0; ib < nb; ib++) {
                    #pragma omp simd
                    for (uint i = 0; i < nrow; i++) {
                        result[ib * vertices_3d_size + index_data[i]] += temp[ib * nrow + i];
                    }
                }
                delete [] temp;
            }
        }
    }

    #pragma omp barrier
    #pragma omp single nowait
    {
        delete [] chi_vector_static;
        chi_vector_static = nullptr;
    }

    return;
}
template void nloc_project_vectors_omp_for<float>(float* const& result, const Vertices_4D& vertices, float const* const& eigen_vectors,
                                                        const Effective_potential_nloc<float>& effective_potential_nloc,
                                                        const float& dv, const MPI_Comm& comm);
template void nloc_project_vectors_omp_for<double>(double* const& result, const Vertices_4D& vertices, double const* const& eigen_vectors,
                                                        const Effective_potential_nloc<double>& effective_potential_nloc,
                                                        const double& dv, const MPI_Comm& comm);

template<typename T>
inline void nloc_project_vectors_omp_for_comm_self_with_chunk(T* const& __restrict__ result, const Vertices_4D& vertices,
                                            T const* const& __restrict__ eigen_vectors,
                                            const Effective_potential_nloc<T>& effective_potential_nloc,
                                            const T& dv, const MPI_Comm& comm) {
    (void) comm;
    constexpr uint chunk_ib = 8;
    const uint nb = vertices.nb;
    const Vertices_3D vertices_3d = vertices;
    const uint vertices_3d_size = vertices_3d.get_size();

    #pragma omp for
    for (uint ib = 0; ib < nb; ib += chunk_ib) {
        const uint nb_local = std::min(ib + chunk_ib, nb) - ib;
        const uint length_local = *(effective_potential_nloc.offsets.end() - 1) * nb_local;
        const Vertices_4D vertices_local(vertices, nb_local);
        T* const chi_vector = new (std::align_val_t(64)) T [length_local]();
        for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
            const uint chi_vector_task_length = nb_local * effective_potential_nloc.nloc_projectors[iprojector].ncol;
            T *const chi_vector_task = new (std::align_val_t(64)) T [chi_vector_task_length]();
            T *const buffer = new (std::align_val_t(64)) T [nb_local * effective_potential_nloc.nloc_projectors[iprojector].nrow];
            Nloc_projector_method::nloc_projector_product_vectors_sequential_with_buffer(
                                        chi_vector_task, eigen_vectors + ib * vertices_3d_size,
                                        vertices_local, effective_potential_nloc.nloc_projectors[iprojector], dv, buffer);
            #pragma omp parallel if(0)
            Linalg::hadamard_plus_general(chi_vector + effective_potential_nloc.offsets[iprojector] * nb_local,
                                                chi_vector_task, chi_vector_task_length);
            ::operator delete[](buffer, std::align_val_t(64));
            ::operator delete[](chi_vector_task, std::align_val_t(64));
        }

        int atom_index = effective_potential_nloc.nloc_projectors.cbegin()->atom_index-1;
        for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
            const uint ncol = effective_potential_nloc.nloc_projectors[iprojector].ncol;
            const uint nrow = effective_potential_nloc.nloc_projectors[iprojector].nrow;
            if (likely(ncol * nrow > 0)) {
                T* const __restrict__ chi_vector_local = chi_vector + effective_potential_nloc.offsets[iprojector] * nb_local;
                if (atom_index != effective_potential_nloc.nloc_projectors[iprojector].atom_index) {
                    atom_index = effective_potential_nloc.nloc_projectors[iprojector].atom_index;
                    T const* const __restrict__ gamma_data = effective_potential_nloc.nloc_projectors[iprojector].gamma.data;
                    for (uint ib_local = 0; ib_local < nb_local; ib_local++) {
                        #pragma omp simd
                        for (uint icol = 0; icol < ncol; icol++) {
                            chi_vector_local[ib_local * ncol + icol] *= gamma_data[icol];
                        }
                    }
                }
            }
        }

        for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
            const uint ncol = effective_potential_nloc.nloc_projectors[iprojector].ncol;
            const uint nrow = effective_potential_nloc.nloc_projectors[iprojector].nrow;
            if (likely(ncol * nrow > 0)) {
                T* const __restrict__ chi_vector_local = chi_vector + effective_potential_nloc.offsets[iprojector] * nb_local;
                T* const __restrict__ temp = new (std::align_val_t(64)) T [nrow * nb_local]();
                #pragma omp parallel if(0)
                Linalg::matrix_product(
                    effective_potential_nloc.nloc_projectors[iprojector].chi.data,
                    1, chi_vector_local, 1, temp, 1, nrow, nb_local, ncol);
                uint const* __restrict__ index_data = effective_potential_nloc.nloc_projectors[iprojector].index_data();
                for (uint ib_local = 0; ib_local < nb_local; ib_local++) {
                    #pragma omp simd
                    for (uint i = 0; i < nrow; i++) {
                        result[(ib+ib_local) * vertices_3d_size + index_data[i]] += temp[ib_local * nrow + i];
                    }
                }
                ::operator delete[](temp, std::align_val_t(64));
            }
        }
        ::operator delete[](chi_vector, std::align_val_t(64));
    }
    return;
}
template void nloc_project_vectors_omp_for_comm_self_with_chunk<float>(float* const& result, const Vertices_4D& vertices, float const* const& eigen_vectors,
                                                                    const Effective_potential_nloc<float>& effective_potential_nloc,
                                                                    const float& dv, const MPI_Comm& comm);
template void nloc_project_vectors_omp_for_comm_self_with_chunk<double>(double* const& result, const Vertices_4D& vertices, double const* const& eigen_vectors,
                                                                    const Effective_potential_nloc<double>& effective_potential_nloc,
                                                                    const double& dv, const MPI_Comm& comm);

template<typename T>
inline void nloc_project_vectors_omp_task(T* const& __restrict__ result, const Vertices_4D& vertices,
                                            T const* const& __restrict__ eigen_vectors,
                                            const Effective_potential_nloc<T>& effective_potential_nloc,
                                            const T& dv, const MPI_Comm& comm) {
    const uint& nb = vertices.nb;
    uint length = *(effective_potential_nloc.offsets.end() - 1) * nb;
    static T* chi_vector_static = nullptr;
    #pragma omp single
    chi_vector_static = new (std::align_val_t(64)) T [length]();
    T* const chi_vector = chi_vector_static;
    #pragma omp master
    {
        const uint vertices_3d_size = vertices.Vertices_3D::get_size();
        for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
            #pragma omp task shared(effective_potential_nloc)
            {
                #pragma omp parallel if(0)
                {
                    const uint chi_vector_task_length = nb * effective_potential_nloc.nloc_projectors[iprojector].ncol;
                    T *const chi_vector_task = new (std::align_val_t(64)) T [chi_vector_task_length]();
                    Nloc_projector_method::nloc_projector_product_vectors_sequential(chi_vector_task,
                                eigen_vectors, vertices, effective_potential_nloc.nloc_projectors[iprojector], dv);
                    #pragma omp critical
                    {
                        Linalg::hadamard_plus_general(chi_vector + effective_potential_nloc.offsets[iprojector] * nb,
                                                        chi_vector_task, chi_vector_task_length);
                    }
                    delete [] chi_vector_task;
                }
            }
        }
        #pragma omp taskwait

        int commsize;
        MPI_Comm_size(comm, &commsize);
        if (commsize > 1) {
            MPI_Datatype mpi_datatype = Linalg::get_mpi_datatype<T>();
            MPI_Allreduce(MPI_IN_PLACE, chi_vector, length, mpi_datatype, MPI_SUM, comm);
        }

        int atom_index = effective_potential_nloc.nloc_projectors.cbegin()->atom_index-1;
        for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
            const uint& ncol = effective_potential_nloc.nloc_projectors[iprojector].ncol;
            const uint& nrow = effective_potential_nloc.nloc_projectors[iprojector].nrow;
            if (ncol * nrow > 0) {
                T* const __restrict__ chi_vector_local = chi_vector + effective_potential_nloc.offsets[iprojector] * nb;
                if (atom_index != effective_potential_nloc.nloc_projectors[iprojector].atom_index) {
                    atom_index = effective_potential_nloc.nloc_projectors[iprojector].atom_index;
                    T const* const __restrict__ gamma_data = effective_potential_nloc.nloc_projectors[iprojector].gamma.data;
                    for (uint ib = 0; ib < nb; ib++) {
                        #pragma omp simd
                        for (uint icol = 0; icol < ncol; icol++) {
                            chi_vector_local[ib * ncol + icol] *= gamma_data[icol];
                        }
                    }
                }
                #pragma omp task shared(effective_potential_nloc)
                #pragma omp parallel if(0)
                {
                    T* const __restrict__ temp = new T [nrow * nb]();
                    Linalg::matrix_product(effective_potential_nloc.nloc_projectors[iprojector].chi.data, 1, chi_vector_local, 1, temp, 1, nrow, nb, ncol);
                    uint const* __restrict__ index_data = effective_potential_nloc.nloc_projectors[iprojector].index_data();
                    #pragma omp critical
                    for (uint ib = 0; ib < nb; ib++) {
                        #pragma omp simd
                        for (uint i = 0; i < nrow; i++) {
                            result[ib * vertices_3d_size + index_data[i]] += temp[ib * nrow + i];
                        }
                    }
                    delete [] temp;
                }
            }
        }
    }

    #pragma omp barrier
    #pragma omp single nowait
    {
        delete [] chi_vector_static;
        chi_vector_static = nullptr;
    }

    return;
}
template void nloc_project_vectors_omp_task<float>(float* const& result, const Vertices_4D& vertices, float const* const& eigen_vectors,
                                                        const Effective_potential_nloc<float>& effective_potential_nloc,
                                                        const float& dv, const MPI_Comm& comm);
template void nloc_project_vectors_omp_task<double>(double* const& result, const Vertices_4D& vertices, double const* const& eigen_vectors,
                                                        const Effective_potential_nloc<double>& effective_potential_nloc,
                                                        const double& dv, const MPI_Comm& comm);
#endif //USE_OPENMP


template<typename T>
void nloc_project_vectors_omp_for_comm_self_with_chunk_mp(T* const __restrict__ result, const Vertices_4D& vertices,
                                            T const* const __restrict__ eigen_vectors,
                                            const Effective_potential_nloc<T>& effective_potential_nloc,
                                            const T dv, const MPI_Comm comm,
                                            Memory_pool<T, Fast_memory>& pool_fast,
                                            Memory_pool<T, Capacity_memory>& pool_cap) {
    (void) comm;
    constexpr uint chunk_ib = 8;

    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);

    const uint nb = vertices.nb;
    const Vertices_3D vertices_3d = vertices;
    const uint vertices_3d_size = vertices_3d.get_size();

    int nthread = 1;
    #ifdef USE_OPENMP
        #pragma omp parallel
        nthread = omp_get_num_threads();
    #endif

    uint ncol_max = 0;
    uint nrow_max = 0;
    for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
        ncol_max = std::max(ncol_max, effective_potential_nloc.nloc_projectors[iprojector].ncol);
        nrow_max = std::max(nrow_max, effective_potential_nloc.nloc_projectors[iprojector].nrow);
    } 

    const uint chi_vector_length_single_thread = *(effective_potential_nloc.offsets.end() - 1) * chunk_ib;
    // T* const chi_vector_nthread = new (std::align_val_t(64)) T [chi_vector_length_single_thread * nthread];
    T* const chi_vector_nthread = pool_fast.allocate(chi_vector_length_single_thread * nthread);

    const uint chi_vector_task_length_single_thread = ncol_max * chunk_ib;
    // T* const chi_vector_task_nthread = new (std::align_val_t(64)) T [chi_vector_task_length_single_thread * nthread];
    T* const chi_vector_task_nthread = pool_fast.allocate(chi_vector_task_length_single_thread * nthread);

    const uint buffer_length_single_thread = nrow_max * chunk_ib;
    // T* const buffer_nthread = new (std::align_val_t(64)) T [buffer_length_single_thread * nthread];
    T* const buffer_nthread = pool_fast.allocate(buffer_length_single_thread * nthread);

    #pragma omp parallel for
    for (uint ib = 0; ib < nb; ib += chunk_ib) {

        #ifdef USE_OPENMP
            const int tid = omp_get_thread_num();
        #else
            const int tid = 0;
        #endif

        const uint nb_local = std::min(ib + chunk_ib, nb) - ib;
        const uint length_local = *(effective_potential_nloc.offsets.end() - 1) * nb_local;
        const Vertices_4D vertices_local(vertices, nb_local);
        // T* const chi_vector = new (std::align_val_t(64)) T [length_local]();
        T* const chi_vector = chi_vector_nthread + chi_vector_length_single_thread * tid;
        #pragma omp parallel if(0)
        Linalg::set_value_general(chi_vector, T(0), length_local);
        for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
            const uint chi_vector_task_length = nb_local * effective_potential_nloc.nloc_projectors[iprojector].ncol;
            // T *const chi_vector_task = new (std::align_val_t(64)) T [chi_vector_task_length]();
            T *const chi_vector_task = chi_vector_task_nthread + chi_vector_task_length_single_thread * tid;
            #pragma omp parallel if(0)
            Linalg::set_value_general(chi_vector_task, T(0), chi_vector_task_length);
            // T *const buffer = new (std::align_val_t(64)) T [nb_local * effective_potential_nloc.nloc_projectors[iprojector].nrow];
            T *const buffer = buffer_nthread + buffer_length_single_thread * tid;
            Nloc_projector_method::nloc_projector_product_vectors_sequential_with_buffer(
                                        chi_vector_task, eigen_vectors + ib * vertices_3d_size,
                                        vertices_local, effective_potential_nloc.nloc_projectors[iprojector], dv, buffer);
            #pragma omp parallel if(0)
            Linalg::hadamard_plus_general(chi_vector + effective_potential_nloc.offsets[iprojector] * nb_local,
                                                chi_vector_task, chi_vector_task_length);
            // ::operator delete[](buffer, std::align_val_t(64));
            // ::operator delete[](chi_vector_task, std::align_val_t(64));
        }

        int atom_index = effective_potential_nloc.nloc_projectors.cbegin()->atom_index-1;
        for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
            const uint ncol = effective_potential_nloc.nloc_projectors[iprojector].ncol;
            const uint nrow = effective_potential_nloc.nloc_projectors[iprojector].nrow;
            if (likely(ncol * nrow > 0)) {
                T* const __restrict__ chi_vector_local = chi_vector + effective_potential_nloc.offsets[iprojector] * nb_local;
                if (atom_index != effective_potential_nloc.nloc_projectors[iprojector].atom_index) {
                    atom_index = effective_potential_nloc.nloc_projectors[iprojector].atom_index;
                    T const* const __restrict__ gamma_data = effective_potential_nloc.nloc_projectors[iprojector].gamma.data;
                    for (uint ib_local = 0; ib_local < nb_local; ib_local++) {
                        #pragma omp simd
                        for (uint icol = 0; icol < ncol; icol++) {
                            chi_vector_local[ib_local * ncol + icol] *= gamma_data[icol];
                        }
                    }
                }
            }
        }

        for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
            const uint ncol = effective_potential_nloc.nloc_projectors[iprojector].ncol;
            const uint nrow = effective_potential_nloc.nloc_projectors[iprojector].nrow;
            if (likely(ncol * nrow > 0)) {
                T* const __restrict__ chi_vector_local = chi_vector + effective_potential_nloc.offsets[iprojector] * nb_local;
                // T* const __restrict__ temp = new (std::align_val_t(64)) T [nrow * nb_local]();
                T* const __restrict__ temp = buffer_nthread + buffer_length_single_thread * tid;
                #pragma omp parallel if(0)
                Linalg::set_value_general(temp, T(0), nrow * nb_local);
                #pragma omp parallel if(0)
                Linalg::matrix_product(
                    effective_potential_nloc.nloc_projectors[iprojector].chi.data,
                    1, chi_vector_local, 1, temp, 1, nrow, nb_local, ncol);
                uint const* __restrict__ index_data = effective_potential_nloc.nloc_projectors[iprojector].index_data();
                for (uint ib_local = 0; ib_local < nb_local; ib_local++) {
                    #pragma omp simd
                    for (uint i = 0; i < nrow; i++) {
                        result[(ib+ib_local) * vertices_3d_size + index_data[i]] += temp[ib_local * nrow + i];
                    }
                }
                // ::operator delete[](temp, std::align_val_t(64));
            }
        }
        // ::operator delete[](chi_vector, std::align_val_t(64));
    }

    // ::operator delete[](chi_vector_nthread, std::align_val_t(64));
    // ::operator delete[](chi_vector_task_nthread, std::align_val_t(64));
    // ::operator delete[](buffer_nthread, std::align_val_t(64));
    return;
}
template void nloc_project_vectors_omp_for_comm_self_with_chunk_mp<float>(float* const result, const Vertices_4D& vertices, float const* const eigen_vectors,
                                                                    const Effective_potential_nloc<float>& effective_potential_nloc,
                                                                    const float dv, const MPI_Comm comm,
                                                                    Memory_pool<float, Fast_memory>& pool_fast,
                                                                    Memory_pool<float, Capacity_memory>& pool_cap);
template void nloc_project_vectors_omp_for_comm_self_with_chunk_mp<double>(double* const result, const Vertices_4D& vertices, double const* const eigen_vectors,
                                                                    const Effective_potential_nloc<double>& effective_potential_nloc,
                                                                    const double dv, const MPI_Comm comm,
                                                                    Memory_pool<double, Fast_memory>& pool_fast,
                                                                    Memory_pool<double, Capacity_memory>& pool_cap);

template<typename T>
void hamiltonian_product_vectors_column_wise2_specialization_mp(T* const result,  const Vertices_4D& vertices, T const* const eigen_vectors,
                                              T const* const Vloc, const Effective_potential_nloc<T>& Vnloc, const Stencil<T>& stencil, 
                                              const T dv, const Exarr_3D_mpi_package& exarr_mpi_package, const bool print_flag,
                                                Memory_pool<T, Fast_memory>& pool_fast,
                                                Memory_pool<T, Capacity_memory>& pool_cap) {
    (void) print_flag;
    Vertices_3D vertices_3d = vertices.Vertices_3D::get_vertices();
    const uint Nd = vertices_3d.get_size();
    if (Nd == 0 || vertices.nb == 0) return;
    
    Stencil<T> stencil_temp;
    stencil_temp.deepcopy_mp(stencil, pool_fast);
    stencil_temp.coeffs_scale_self(-0.5);

    // result = (-0.5 \nabla + Vloc ) eigen_vectors
    #pragma omp parallel
    Stencil_method::Boundary_safe::calc_laplacian_boundary_safe(eigen_vectors, vertices, stencil_temp, vertices,
                                            result, vertices, Vloc);
    // #pragma omp parallel
    Hamiltonian::nloc_project_vectors_omp_for_comm_self_with_chunk_mp<T>(result, vertices, eigen_vectors,
                                                                        Vnloc, dv, exarr_mpi_package.comm,
                                                                        pool_fast, pool_cap);
    stencil_temp.destructor_mp();
    return;
}
template void hamiltonian_product_vectors_column_wise2_specialization_mp<float>(float* const result,  const Vertices_4D& vertices, float const* const eigen_vectors,
                                                              float const* const Vloc, const Effective_potential_nloc<float>& Vnloc, const Stencil<float>& stencil,
                                                              const float dv, const Exarr_3D_mpi_package& exarr_mpi_package, const bool print_flag,
                                                                Memory_pool<float, Fast_memory>& pool_fast,
                                                                Memory_pool<float, Capacity_memory>& pool_cap);
template void hamiltonian_product_vectors_column_wise2_specialization_mp<double>(double* const result,  const Vertices_4D& vertices, double const* const eigen_vectors,
                                                               double const* const Vloc, const Effective_potential_nloc<double>& Vnloc, const Stencil<double>& stencil,
                                                               const double dv, const Exarr_3D_mpi_package& exarr_mpi_package, const bool print_flag,
                                                                Memory_pool<double, Fast_memory>& pool_fast,
                                                                Memory_pool<double, Capacity_memory>& pool_cap);


template<typename T>
void hamiltonian_product_vectors_column_wise2_compute_fusion_mp(
                T* const result,  const Vertices_4D& vertices, T const* const eigen_vectors,
                T const* const Vloc, const Effective_potential_nloc<T>& effective_potential_nloc,
                const Stencil<T>& stencil, const T shfit,
                const T dv, const Exarr_3D_mpi_package& exarr_mpi_package, const bool print_flag,
                Memory_pool<T, Fast_memory>& pool_fast,
                Memory_pool<T, Capacity_memory>& pool_cap) {
    (void) exarr_mpi_package;
    (void) print_flag;
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    constexpr uint chunk_ib = 8;
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    const Vertices_3D vertices_3d = vertices.Vertices_3D::get_vertices();
    const uint vertices_3d_size = vertices_3d.get_size();
    const uint nb = vertices.nb;
    if (vertices_3d_size == 0 || nb == 0) return;

    int nthread = 1;
    #ifdef USE_OPENMP
        #pragma omp parallel
        nthread = omp_get_num_threads();
    #endif

    Stencil<T> stencil_temp;
    stencil_temp.deepcopy(stencil);
    stencil_temp.coeffs_scale_self(-0.5);
    stencil_temp.shift_D2_coeffs(shfit);

    uint ncol_max = 0;
    uint nrow_max = 0;
    for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
        ncol_max = std::max(ncol_max, effective_potential_nloc.nloc_projectors[iprojector].ncol);
        nrow_max = std::max(nrow_max, effective_potential_nloc.nloc_projectors[iprojector].nrow);
    }

    const uint chi_vector_length_single_thread = *(effective_potential_nloc.offsets.end() - 1) * chunk_ib;
    // T* const chi_vector_nthread = new (std::align_val_t(64)) T [chi_vector_length_single_thread * nthread];
    T* const chi_vector_nthread = pool_fast.allocate(chi_vector_length_single_thread * nthread);

    const uint chi_vector_task_length_single_thread = ncol_max * chunk_ib;
    // T* const chi_vector_task_nthread = new (std::align_val_t(64)) T [chi_vector_task_length_single_thread * nthread];
    T* const chi_vector_task_nthread = pool_fast.allocate(chi_vector_task_length_single_thread * nthread);

    const uint buffer_length_single_thread = nrow_max * chunk_ib;
    // T* const buffer_nthread = new (std::align_val_t(64)) T [buffer_length_single_thread * nthread];
    T* const buffer_nthread = pool_fast.allocate(buffer_length_single_thread * nthread);

    #pragma omp parallel for
    for (uint ib = 0; ib < nb; ib += chunk_ib) {

        #ifdef USE_OPENMP
            const int tid = omp_get_thread_num();
        #else
            const int tid = 0;
        #endif

        const uint nb_local = std::min(ib + chunk_ib, nb) - ib;
        const uint length_local = *(effective_potential_nloc.offsets.end() - 1) * nb_local;
        const Vertices_4D vertices_local(vertices, nb_local);

        #pragma omp parallel if(0)
        Stencil_method::Boundary_safe::calc_laplacian_boundary_safe(eigen_vectors + ib * vertices_3d_size, vertices_local, stencil_temp, vertices_local,
                                                                    result + ib * vertices_3d_size, vertices_local, Vloc);
        T* const chi_vector = chi_vector_nthread + chi_vector_length_single_thread * tid;
        #pragma omp parallel if(0)
        Linalg::set_value_general(chi_vector, T(0), length_local);
        for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
            const uint chi_vector_task_length = nb_local * effective_potential_nloc.nloc_projectors[iprojector].ncol;
            // T *const chi_vector_task = new (std::align_val_t(64)) T [chi_vector_task_length]();
            T *const chi_vector_task = chi_vector_task_nthread + chi_vector_task_length_single_thread * tid;
            #pragma omp parallel if(0)
            Linalg::set_value_general(chi_vector_task, T(0), chi_vector_task_length);
            // T *const buffer = new (std::align_val_t(64)) T [nb_local * effective_potential_nloc.nloc_projectors[iprojector].nrow];
            T *const buffer = buffer_nthread + buffer_length_single_thread * tid;
            Nloc_projector_method::nloc_projector_product_vectors_sequential_with_buffer(
                                        chi_vector_task, eigen_vectors + ib * vertices_3d_size,
                                        vertices_local, effective_potential_nloc.nloc_projectors[iprojector], dv, buffer);
            #pragma omp parallel if(0)
            Linalg::hadamard_plus_general(chi_vector + effective_potential_nloc.offsets[iprojector] * nb_local,
                                                chi_vector_task, chi_vector_task_length);
            // ::operator delete[](buffer, std::align_val_t(64));
            // ::operator delete[](chi_vector_task, std::align_val_t(64));
        }

        int atom_index = effective_potential_nloc.nloc_projectors.cbegin()->atom_index-1;
        for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
            const uint ncol = effective_potential_nloc.nloc_projectors[iprojector].ncol;
            const uint nrow = effective_potential_nloc.nloc_projectors[iprojector].nrow;
            if (likely(ncol * nrow > 0)) {
                T* const __restrict__ chi_vector_local = chi_vector + effective_potential_nloc.offsets[iprojector] * nb_local;
                if (atom_index != effective_potential_nloc.nloc_projectors[iprojector].atom_index) {
                    atom_index = effective_potential_nloc.nloc_projectors[iprojector].atom_index;
                    T const* const __restrict__ gamma_data = effective_potential_nloc.nloc_projectors[iprojector].gamma.data;
                    for (uint ib_local = 0; ib_local < nb_local; ib_local++) {
                        #pragma omp simd
                        for (uint icol = 0; icol < ncol; icol++) {
                            chi_vector_local[ib_local * ncol + icol] *= gamma_data[icol];
                        }
                    }
                }
            }
        }

        for (uint iprojector = 0; iprojector < effective_potential_nloc.nloc_projectors.size(); iprojector++) {
            const uint ncol = effective_potential_nloc.nloc_projectors[iprojector].ncol;
            const uint nrow = effective_potential_nloc.nloc_projectors[iprojector].nrow;
            if (likely(ncol * nrow > 0)) {
                T* const __restrict__ chi_vector_local = chi_vector + effective_potential_nloc.offsets[iprojector] * nb_local;
                // T* const __restrict__ temp = new (std::align_val_t(64)) T [nrow * nb_local]();
                T* const __restrict__ temp = buffer_nthread + buffer_length_single_thread * tid;
                #pragma omp parallel if(0)
                Linalg::set_value_general(temp, T(0), nrow * nb_local);
                #pragma omp parallel if(0)
                Linalg::matrix_product(
                    effective_potential_nloc.nloc_projectors[iprojector].chi.data,
                    1, chi_vector_local, 1, temp, 1, nrow, nb_local, ncol);
                uint const* __restrict__ index_data = effective_potential_nloc.nloc_projectors[iprojector].index_data();
                for (uint ib_local = 0; ib_local < nb_local; ib_local++) {
                    #pragma omp simd
                    for (uint i = 0; i < nrow; i++) {
                        result[(ib+ib_local) * vertices_3d_size + index_data[i]] += temp[ib_local * nrow + i];
                    }
                }
                // ::operator delete[](temp, std::align_val_t(64));
            }
        }
        // ::operator delete[](chi_vector, std::align_val_t(64));
    }

    // // result = (-0.5 \nabla + Vloc ) eigen_vectors
    // #pragma omp parallel
    // Stencil_method::Boundary_safe::calc_laplacian_boundary_safe(eigen_vectors, vertices, stencil_temp, vertices,
    //                                         result, vertices, Vloc);
    // // #pragma omp parallel
    // Hamiltonian::nloc_project_vectors_omp_for_comm_self_with_chunk_mp<T>(result, vertices, eigen_vectors,
    //                                                                     Vnloc, dv, exarr_mpi_package.comm,
                                                                        // pool_fast, pool_cap);
    stencil_temp.destructor();
    // int rank;
    // MPI_Comm_rank(exarr_mpi_package.comm, &rank);
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // if (rank == 0 && print_flag) {
    //     std::cout << "The hamiltonian_product_vectors_column_wise2_specialization took " << Tools::time_cost(begin, end) << "." << std::endl;
    // }
    return;
}
template void hamiltonian_product_vectors_column_wise2_compute_fusion_mp<float>(float* const result,  const Vertices_4D& vertices, float const* const eigen_vectors,
                                                              float const* const Vloc, const Effective_potential_nloc<float>& Vnloc,
                                                              const Stencil<float>& stencil, const float shfit,
                                                              const float dv, const Exarr_3D_mpi_package& exarr_mpi_package, const bool print_flag,
                                                                Memory_pool<float, Fast_memory>& pool_fast,
                                                                Memory_pool<float, Capacity_memory>& pool_cap);
template void hamiltonian_product_vectors_column_wise2_compute_fusion_mp<double>(double* const result,  const Vertices_4D& vertices, double const* const eigen_vectors,
                                                               double const* const Vloc, const Effective_potential_nloc<double>& Vnloc, const Stencil<double>& stencil, const double shfit,
                                                               const double dv, const Exarr_3D_mpi_package& exarr_mpi_package, const bool print_flag,
                                                                Memory_pool<double, Fast_memory>& pool_fast,
                                                                Memory_pool<double, Capacity_memory>& pool_cap);

}
