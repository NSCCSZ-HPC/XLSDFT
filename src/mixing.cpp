#include "mixing.h"
#include "xlsdft_backend.h"

#include <limits>
#include <type_traits>

namespace {

template<typename T>
void solve_pulay_fallback(const uint history, T* matrix, T* rhs) {
    if (history == 0) return;
    T scale = T(0);
    for (uint i = 0; i < history; ++i) {
        scale = std::max(scale, std::abs(matrix[i + static_cast<std::size_t>(i) * history]));
    }
    const T regularization = std::max(scale, T(1))
                           * std::numeric_limits<T>::epsilon()
                           * static_cast<T>(64 * history);
    for (uint i = 0; i < history; ++i) {
        matrix[i + static_cast<std::size_t>(i) * history] += regularization;
    }

    for (uint pivot_column = 0; pivot_column < history; ++pivot_column) {
        uint pivot_row = pivot_column;
        T pivot_magnitude = std::abs(
            matrix[pivot_row + static_cast<std::size_t>(pivot_column) * history]);
        for (uint row = pivot_column + 1; row < history; ++row) {
            const T magnitude = std::abs(
                matrix[row + static_cast<std::size_t>(pivot_column) * history]);
            if (magnitude > pivot_magnitude) {
                pivot_magnitude = magnitude;
                pivot_row = row;
            }
        }
        if (pivot_row != pivot_column) {
            for (uint column = 0; column < history; ++column) {
                std::swap(matrix[pivot_column + static_cast<std::size_t>(column) * history],
                          matrix[pivot_row + static_cast<std::size_t>(column) * history]);
            }
            std::swap(rhs[pivot_column], rhs[pivot_row]);
        }
        const T pivot = matrix[pivot_column +
                               static_cast<std::size_t>(pivot_column) * history];
        if (std::abs(pivot) <= regularization) continue;
        for (uint row = pivot_column + 1; row < history; ++row) {
            const T factor = matrix[row +
                                    static_cast<std::size_t>(pivot_column) * history] / pivot;
            matrix[row + static_cast<std::size_t>(pivot_column) * history] = T(0);
            for (uint column = pivot_column + 1; column < history; ++column) {
                matrix[row + static_cast<std::size_t>(column) * history] -=
                    factor * matrix[pivot_column +
                                    static_cast<std::size_t>(column) * history];
            }
            rhs[row] -= factor * rhs[pivot_column];
        }
    }
    for (uint reverse = history; reverse-- > 0;) {
        T value = rhs[reverse];
        for (uint column = reverse + 1; column < history; ++column) {
            value -= matrix[reverse + static_cast<std::size_t>(column) * history]
                   * rhs[column];
        }
        const T pivot = matrix[reverse + static_cast<std::size_t>(reverse) * history];
        rhs[reverse] = std::abs(pivot) > regularization ? value / pivot : T(0);
    }
}

template<typename T>
int solve_pulay_via_f64(const uint history, T* matrix, T* rhs, T* singular_values) {
    const std::size_t matrix_count = static_cast<std::size_t>(history) * history;
    double* matrix_f64 = new (std::align_val_t(64)) double[matrix_count];
    double* rhs_f64 = new (std::align_val_t(64)) double[history];
    double* singular_values_f64 = new (std::align_val_t(64)) double[history];
    const std::size_t workspace_doubles =
        Xlsdft_backend::pulay_gelsd_workspace_doubles(static_cast<int>(history));
    double* workspace = new (std::align_val_t(64)) double[workspace_doubles];
    alignas(64) int rank = 0;
    for (std::size_t i = 0; i < matrix_count; ++i) {
        matrix_f64[i] = static_cast<double>(matrix[i]);
    }
    for (uint i = 0; i < history; ++i) rhs_f64[i] = static_cast<double>(rhs[i]);
    const int status = Xlsdft_backend::pulay_gelsd(
        static_cast<int>(history), matrix_f64, static_cast<int>(history), rhs_f64,
        singular_values_f64, -1.0, &rank, workspace, workspace_doubles);
    if (status == 0) {
        for (std::size_t i = 0; i < matrix_count; ++i) {
            matrix[i] = static_cast<T>(matrix_f64[i]);
        }
        for (uint i = 0; i < history; ++i) {
            rhs[i] = static_cast<T>(rhs_f64[i]);
            singular_values[i] = static_cast<T>(singular_values_f64[i]);
        }
    }
    ::operator delete[](workspace, std::align_val_t(64));
    ::operator delete[](singular_values_f64, std::align_val_t(64));
    ::operator delete[](rhs_f64, std::align_val_t(64));
    ::operator delete[](matrix_f64, std::align_val_t(64));
    return status;
}

}  // namespace

template<typename T>
Mixing<T>::Mixing(const Mixing_control& mixing_control,
                  const Spin& spin,
                  const Stencil<T>& stencil,
                  const Exarr_3D_mpi_package& exarr_mpi_package,
                  const uint& mixing_variable)
                : mixing_control(mixing_control),
                  spin(spin),
                  stencil(stencil),
                  exarr_mpi_package(exarr_mpi_package),
                  mixing_variable(mixing_variable) {}

template<typename T>
Mixing<T>::~Mixing(){}

template<typename T>
void Mixing<T>::set_x_km1(const Array_3D<T>& x_km1) {
    this->x_km1 = x_km1;
    return;
}

template<typename T>
void Mixing<T>::set_electron_densities(const std::vector<Array_3D<T>>& electron_densities) {
    const uint nspin = electron_densities.size();
    const uint Nd = electron_densities[0].length;
    if (nspin == 1) {
        Linalg::set_value_general(this->x_km1.data, electron_densities[0].data, Nd);
    } else if (nspin == 2) {
        Linalg::hadamard_plus_general(this->x_km1.data, electron_densities[0].data, electron_densities[1].data, Nd);
        Linalg::hadamard_minus_general(this->x_km1.data + Nd, electron_densities[0].data, electron_densities[1].data, Nd);
    } else {
        assert(nspin == 1 || nspin == 2);
    }
    return;
}

template<typename T>
void Mixing<T>::record_history(const Array_3D<T>& x_k_out, const Array_3D<T>& f_k, const uint& iter) {
    if (this->mixing_control.method != 0) {
        uint Nd = x_k_out.length;
        uint i_hist = iter % this->mixing_control.mixing_history;
        if (this->mixing_control.method == 1 && i_hist == 0 && this->mixing_control.pulay_control.pulay_restart) { //Pulay mixing Restart
            Linalg::set_value_general(this->R.data + Nd, (T)0.0, Nd * (this->mixing_control.mixing_history - 1));
            Linalg::set_value_general(this->F.data + Nd, (T)0.0, Nd * (this->mixing_control.mixing_history - 1));
        }
        Linalg::hadamard_minus_general(this->R.data + i_hist * Nd, x_k_out.data, this->x_km1.data, Nd);
        Linalg::hadamard_minus_general(this->F.data + i_hist * Nd, f_k.data, this->f_km1.data, Nd);
    }
    return;
}

template<typename T>
void Mixing<T>::record_F_history(const Array_3D<T>& f_k, const uint iter) {
    if (iter == 0) return;
    if (this->mixing_control.method != 0) {
        uint Nd = f_k.length;
        uint i_hist = (iter-1) % this->mixing_control.mixing_history;
        if (this->mixing_control.method == 1 && i_hist == 0 && this->mixing_control.pulay_control.pulay_restart) { //Pulay mixing Restart
            Linalg::set_value_general(this->F.data + Nd, (T)0.0, Nd * (this->mixing_control.mixing_history - 1));
            #pragma omp barrier
        }
        Linalg::hadamard_minus_general(this->F.data + i_hist * Nd, f_k.data, this->f_km1.data, Nd);
    }
    return;
}

template<typename T>
void Mixing<T>::record_F_history(T const* const f_k, const uint iter, uint const nd, const uint ncol) {
    if (iter == 0) return;
    if (this->mixing_control.method != 0) {
        const uint Nd = nd * ncol;
        uint i_hist = (iter-1) % this->mixing_control.mixing_history;
        if (this->mixing_control.method == 1 && i_hist == 0 && this->mixing_control.pulay_control.pulay_restart) { //Pulay mixing Restart
            #pragma omp parallel
            Linalg::set_value_general(this->F.data + Nd, (T)0.0, Nd * (this->mixing_control.mixing_history - 1));
        }
        #pragma omp parallel
        Linalg::hadamard_minus_general(this->F.data + i_hist * Nd, f_k, this->f_km1.data, Nd);
    }
    return;
}

template<typename T>
void Mixing<T>::record_R_history(const Array_3D<T>& x_k_out, const uint iter) {
    if (this->mixing_control.method != 0) {
        uint Nd = x_k_out.length;
        uint i_hist = iter % this->mixing_control.mixing_history;
        if (this->mixing_control.method == 1 && i_hist == 0 && this->mixing_control.pulay_control.pulay_restart) { //Pulay mixing Restart
            Linalg::set_value_general(this->R.data + Nd, (T)0.0, Nd * (this->mixing_control.mixing_history - 1));
            #pragma omp barrier
        }
        Linalg::hadamard_minus_general(this->R.data + i_hist * Nd, x_k_out.data, this->x_km1.data, Nd);
    }
    return;
}

template<typename T>
void Mixing<T>::record_R_history(T const* const x_k_out, const uint iter, uint const nd, const uint ncol) {
    if (this->mixing_control.method != 0) {
        uint Nd = nd * ncol;
        uint i_hist = iter % this->mixing_control.mixing_history;
        if (this->mixing_control.method == 1 && i_hist == 0 && this->mixing_control.pulay_control.pulay_restart) { //Pulay mixing Restart
            #pragma omp parallel
            Linalg::set_value_general(this->R.data + Nd, (T)0.0, Nd * (this->mixing_control.mixing_history - 1));
        }
        #pragma omp parallel
        Linalg::hadamard_minus_general(this->R.data + i_hist * Nd, x_k_out, this->x_km1.data, Nd);
    }
    return;
}

template<typename T>
void Mixing<T>::generate_weight_average(Array_3D<T>& x_wavg, Array_3D<T>& f_wavg,
                                        const Array_3D<T>& x_k, const Array_3D<T>& f_k,
                                        const uint iter, const MPI_Comm comm) {
    if (this->mixing_control.method == 0 || iter == 0) { //linear mixing
        // x_wavg = x_k;
        // x_wavg = this->x_km1;
        // f_wavg = f_k;
        #pragma omp parallel
        {
            Linalg::set_value_general(x_wavg.data, this->x_km1.data, x_wavg.length);
            Linalg::set_value_general(f_wavg.data, f_k.data, f_wavg.length);
        }
    } else if (this->mixing_control.method == 1) { //Pulay mixing
        if ((iter+1) % this->mixing_control.pulay_control.pulay_frequency == 0) {
            //Anderson extrapolation to generate x_wavg and f_wavg
            Mixing_method::Anderson_extrapolation_weighted_averaged_vectors((int)x_k.length, (int)this->mixing_control.mixing_history,
                                                                            x_wavg.data, f_wavg.data, x_k.data, f_k.data,
                                                                            this->R.data, this->F.data, comm);
        } else {    
            //x_wavg = x_k;
            // x_wavg = this->x_km1;
            // f_wavg = f_k;
            #pragma omp parallel
            {
                Linalg::set_value_general(x_wavg.data, this->x_km1.data, x_wavg.length);
                Linalg::set_value_general(f_wavg.data, f_k.data, f_wavg.length);
            }
        }
    } else {
        assert(this->mixing_control.method == 0 || this->mixing_control.method == 1);
    }
    return;
}

template<typename T>
void Mixing<T>::generate_weight_average_mp(T* const x_wavg, T* const f_wavg,
                                            T const* const x_k, T const* const f_k,
                                            const uint iter, const MPI_Comm comm,
                                            uint const nd, const uint ncol,
                                            Memory_pool<T, Fast_memory>& pool_fast,
                                            Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    const uint length = nd * ncol;
    if (this->mixing_control.method == 0 || iter == 0) { //linear mixing
        // x_wavg = x_k;
        // x_wavg = this->x_km1;
        // f_wavg = f_k;
        #pragma omp parallel
        {
            Linalg::set_value_general(x_wavg, this->x_km1.data, length);
            Linalg::set_value_general(f_wavg, f_k, length);
        }
    } else if (this->mixing_control.method == 1) { //Pulay mixing
        if ((iter+1) % this->mixing_control.pulay_control.pulay_frequency == 0) {
            //Anderson extrapolation to generate x_wavg and f_wavg
            Mixing_method::Anderson_extrapolation_weighted_averaged_vectors_mp(length, this->mixing_control.mixing_history,
                                                                            x_wavg, f_wavg, x_k, f_k,
                                                                            this->R.data, this->F.data, comm,
                                                                            pool_fast, pool_cap);
        } else {    
            //x_wavg = x_k;
            // x_wavg = this->x_km1;
            // f_wavg = f_k;
            #pragma omp parallel
            {
                Linalg::set_value_general(x_wavg, this->x_km1.data, length);
                Linalg::set_value_general(f_wavg, f_k, length);
            }
        }
    } else {
        assert(this->mixing_control.method == 0 || this->mixing_control.method == 1);
    }
    return;
}

template<typename T>
void Mixing<T>::apply_precondition(const Array_3D<T>& f_wavg, Array_3D<T>& pf_wavg, const uint iter) {
    //precondition
    if (this->mixing_control.precondition_method == 0) {  // no precondition
        // pf_wavg = f_wavg * (T)this->mixing_control.alpha;
        #pragma omp parallel
        Linalg::scalar_product_general(pf_wavg.data, f_wavg.data, (T)this->mixing_control.alpha, pf_wavg.length);
    } else if (this->mixing_control.precondition_method == 1) { // kerker precondition
        const T alpha = (iter + 1) % this->mixing_control.pulay_control.pulay_frequency == 0 && iter != 0 ?
                  (T) this->mixing_control.pulay_control.beta : (T) this->mixing_control.alpha;
        // Mixing_method::kerker_precondition(f_wavg, pf_wavg, alpha, this->stencil,
        //                                    this->mixing_control.kerker_control, this->exarr_mpi_package);
        // Mixing_method::kerker_precondition(f_wavg.data, pf_wavg.data, f_wavg.get_vertices(), alpha, this->stencil,
        //                                    this->mixing_control.kerker_control, this->exarr_mpi_package);
        if (this->spin.get_spin_type() == 0) {
            Mixing_method::kerker_precondition(f_wavg.data, pf_wavg.data, f_wavg.get_vertices(), alpha, this->stencil,
                                           this->mixing_control.kerker_control, this->exarr_mpi_package);
        } else if (this->spin.get_spin_type() == 1) {
            const uint ncol = this->spin.generate_nspin();
            Vertices_3D vertices_normal = f_wavg.get_vertices();
            // vertices_normal.set_nk(vertices_normal.get_nk() / nspin);
            vertices_normal.nk /= ncol;
            const uint Nd = vertices_normal.get_size();
            for (uint icol = 0; icol < ncol; icol++) {
                if (icol == 0) {
                    Mixing_method::kerker_precondition(f_wavg.data + icol * Nd, pf_wavg.data + icol * Nd,
                                                    vertices_normal, alpha, this->stencil,
                                                    this->mixing_control.kerker_control, this->exarr_mpi_package);
                } else if (icol == 1) {
                    #pragma omp parallel
                    Linalg::scalar_product_general(pf_wavg.data + icol * Nd, f_wavg.data + icol * Nd, (T)this->mixing_control.alpha_mag, Nd);
                } else {
                    assert(icol == 0 || icol == 1);
                }
            }
        } else {
            assert(this->spin.get_spin_type() == 0 || this->spin.get_spin_type() == 1);
        }
    } else {
        assert(this->mixing_control.precondition_method == 0 || this->mixing_control.precondition_method == 1);
    }
    return;
}

template<typename T>
void Mixing<T>::apply_precondition(T const* const f_wavg, T* const pf_wavg, const Vertices_3D& vertices, const uint iter, const uint ncol) {
    const uint nd = vertices.get_size();
    //precondition
    if (this->mixing_control.precondition_method == 0) {  // no precondition
        // pf_wavg = f_wavg * (T)this->mixing_control.alpha;
        #pragma omp parallel
        Linalg::scalar_product_general(pf_wavg, f_wavg, (T)this->mixing_control.alpha, nd * ncol);
    } else if (this->mixing_control.precondition_method == 1) { // kerker precondition
        const T alpha = (iter + 1) % this->mixing_control.pulay_control.pulay_frequency == 0 && iter != 0 ?
                  (T) this->mixing_control.pulay_control.beta : (T) this->mixing_control.alpha;
        // Mixing_method::kerker_precondition(f_wavg, pf_wavg, alpha, this->stencil,
        //                                    this->mixing_control.kerker_control, this->exarr_mpi_package);
        // Mixing_method::kerker_precondition(f_wavg.data, pf_wavg.data, f_wavg.get_vertices(), alpha, this->stencil,
        //                                    this->mixing_control.kerker_control, this->exarr_mpi_package);
        if (this->spin.get_spin_type() == 0) {
            Mixing_method::kerker_precondition(f_wavg, pf_wavg, vertices, alpha, this->stencil,
                                           this->mixing_control.kerker_control, this->exarr_mpi_package);
        } else if (this->spin.get_spin_type() == 1) {
            for (uint icol = 0; icol < ncol; icol++) {
                if (icol == 0) {
                    Mixing_method::kerker_precondition(f_wavg + icol * nd, pf_wavg + icol * nd,
                                                    vertices, alpha, this->stencil,
                                                    this->mixing_control.kerker_control, this->exarr_mpi_package);
                } else if (icol == 1) {
                    #pragma omp parallel
                    Linalg::scalar_product_general(pf_wavg + icol * nd, f_wavg + icol * nd, (T)this->mixing_control.alpha_mag, nd);
                } else {
                    assert(icol == 0 || icol == 1);
                }
            }
        } else {
            assert(this->spin.get_spin_type() == 0 || this->spin.get_spin_type() == 1);
        }
    } else {
        assert(this->mixing_control.precondition_method == 0 || this->mixing_control.precondition_method == 1);
    }
    return;
}

template<typename T>
void Mixing<T>::apply_precondition_mp(
    T const* const f_wavg,
    T* const pf_wavg,
    const Vertices_3D& vertices,
    const uint iter,
    const uint ncol,
    Memory_pool<T, Fast_memory>& pool_fast,
    Memory_pool<T, Capacity_memory>& pool_cap)
{
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);

    const uint nd = vertices.get_size();
    const uint spin_type = this->spin.get_spin_type();

    assert(spin_type == 0 || spin_type == 1);

    if (spin_type == 0) {
        assert(ncol == 1);
    } else {
        assert(ncol == 2);
    }

    /*
     * No precondition
     */
    if (this->mixing_control.precondition_method == 0) {
        #pragma omp parallel
        Linalg::scalar_product_general(
            pf_wavg,
            f_wavg,
            T(this->mixing_control.alpha),
            nd * ncol);

        return;
    }

    assert(this->mixing_control.precondition_method == 1);

    /*
     * Decide whether the charge channel uses
     * Kerker or simple linear mixing.
     */
    const uint pulay_frequency =
        this->mixing_control.pulay_control.pulay_frequency;

    assert(pulay_frequency > 0);

    const bool if_pulay =
        iter != 0 &&
        (iter + 1) % pulay_frequency == 0;

    const bool if_kerker =
        if_pulay ||
        this->mixing_control.simple_precondition_method == 1;

    /*
     * Charge channel
     */
    if (if_kerker) {
        const T alpha =
            if_pulay
            ? T(this->mixing_control.pulay_control.beta)
            : T(this->mixing_control.alpha);

        Mixing_method::kerker_precondition_mp(
            f_wavg,
            pf_wavg,
            vertices,
            alpha,
            this->stencil,
            this->mixing_control.kerker_control,
            this->exarr_mpi_package,
            pool_fast,
            pool_cap);
    } else {
        #pragma omp parallel
        Linalg::scalar_product_general(
            pf_wavg,
            f_wavg,
            T(this->mixing_control.alpha),
            nd);
    }

    /*
     * Magnetization channel
     */
    if (spin_type == 1) {
        #pragma omp parallel
        Linalg::scalar_product_general(
            pf_wavg + nd,
            f_wavg + nd,
            T(this->mixing_control.alpha_mag),
            nd);
    }
}

template<typename T>
void Mixing<T>::shift_pf(const Array_3D<T>& f_wavg, Array_3D<T>& pf_wavg, const MPI_Comm comm) {
    if (this->spin.get_spin_type() == 0) {
        return;
    } else if (this->spin.get_spin_type() == 1) {
        const uint ncol = this->spin.generate_nspin();
        Vertices_3D vertices_normal = f_wavg.get_vertices();
        vertices_normal.nk /= ncol;
        const uint64_t Nd = vertices_normal.get_size();
        uint64_t sum_Nd = 0;
        MPI_Allreduce(&Nd, &sum_Nd, 1, MPI_UINT64_T, MPI_SUM, comm);
        for (uint icol = 0; icol < ncol; icol++) {
            T sum_f;
            T sum_pf;
            #pragma omp parallel
            {
                sum_f = Linalg::vector_sum(f_wavg.data + icol * Nd, Nd, comm);
                #pragma omp barrier
                sum_pf = Linalg::vector_sum(pf_wavg.data + icol * Nd, Nd, comm);
                Linalg::scalar_plus_general(pf_wavg.data + icol * Nd, (sum_f - sum_pf)/T(sum_Nd), Nd);
            }
        }
    } else {
        assert(this->spin.get_spin_type() == 0 || this->spin.get_spin_type() == 1);
    }
    return;
}

template<typename T>
void Mixing<T>::shift_pf(T const* const f_wavg, T* const pf_wavg,
                        const Vertices_3D& vertices, const uint ncol, const MPI_Comm comm) {
    if (this->spin.get_spin_type() == 0) {
        return;
    } else if (this->spin.get_spin_type() == 1) {
        // const uint ncol = this->spin.generate_nspin();
        // Vertices_3D vertices_normal = f_wavg.get_vertices();
        // vertices_normal.nk /= ncol;
        const uint64_t Nd = vertices.get_size();
        uint64_t sum_Nd = 0;
        MPI_Allreduce(&Nd, &sum_Nd, 1, MPI_UINT64_T, MPI_SUM, comm);
        for (uint icol = 0; icol < ncol; icol++) {
            T sum_f;
            T sum_pf;
            #pragma omp parallel
            {
                sum_f = Linalg::vector_sum(f_wavg + icol * Nd, Nd, comm);
                #pragma omp barrier
                sum_pf = Linalg::vector_sum(pf_wavg + icol * Nd, Nd, comm);
                Linalg::scalar_plus_general(pf_wavg + icol * Nd, (sum_f - sum_pf)/T(sum_Nd), Nd);
            }
        }
    } else {
        assert(this->spin.get_spin_type() == 0 || this->spin.get_spin_type() == 1);
    }
    return;
}

template<typename T>
void Mixing<T>::scale_density(const Array_3D<T>& x_k, Array_3D<T>& x_k_out, const MPI_Comm comm) {
    if (this->mixing_variable == 0) {
        const uint ncol = this->spin.generate_nspin();
        Vertices_3D vertices_normal = x_k_out.get_vertices();
        vertices_normal.nk /= ncol;
        const uint Nd = vertices_normal.get_size();
        T x_k_sum = Linalg::vector_sum(x_k.data, Nd, comm);
        #pragma omp barrier
        T x_k_out_sum = Linalg::vector_sum(x_k_out.data, Nd, comm);
        Linalg::scalar_product_general(x_k_out.data, x_k_sum/x_k_out_sum, Nd);
        return;
    } else if (this->mixing_variable == 1) {
        return;
    } else {
        assert(0);
    }
    return;
}

template<typename T>
void Mixing<T>::scale_density(T const* const x_k, T* const x_k_out, const MPI_Comm comm, uint const nd) {
    if (this->mixing_variable == 0) {
        T x_k_sum = Linalg::vector_sum(x_k, nd, comm);
        T x_k_out_sum = Linalg::vector_sum(x_k_out, nd, comm);
        #pragma omp parallel
        Linalg::scalar_product_general(x_k_out, x_k_sum/x_k_out_sum, nd);
        return;
    } else if (this->mixing_variable == 1) {
        return;
    } else {
        assert(0);
    }
    return;
}

template<typename T>
void Mixing<T>::run(Array_3D<T>& x_k, const uint iter, const MPI_Comm comm) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    Vertices_3D local_vertices = x_k.get_vertices();
    
    // Array_3D<T> f_k = x_k - this->x_km1;
    Array_3D<T> f_k(local_vertices);
    #pragma omp parallel
    {
        Linalg::hadamard_minus_general(f_k.data, x_k.data, this->x_km1.data, x_k.length);
        //record the history when linear mixing dont record
        this->record_F_history(f_k, iter);
    }

    //generate history weight average x and f
    Array_3D<T> x_wavg(local_vertices);
    Array_3D<T> f_wavg(local_vertices);
    // this->generate_weight_average(x_wavg, f_wavg, x_k, f_k, iter, comm);
    this->generate_weight_average(x_wavg, f_wavg, this->x_km1, f_k, iter, comm);

    // apply preconditioner if required, Pf = amix * (P * f_tot)
    // Array_3D<T> pf_wavg(local_vertices);
    Array_3D<T>& pf_wavg = this->pf;
    this->apply_precondition(f_wavg, pf_wavg, iter);
    
    this->shift_pf(f_wavg, pf_wavg, comm);

    //generate x_k_out
    Array_3D<T> x_k_out(local_vertices);
    // x_k_out = x_wavg + pf_wavg;
    #pragma omp parallel
    {
    Linalg::hadamard_plus_general(x_k_out.data, x_wavg.data, pf_wavg.data, x_k_out.length);

    this->scale_density(x_k, x_k_out, comm);

    //record the history when linear mixing dont record
    // this->record_history(x_k_out, f_k, iter);
    this->record_R_history(x_k_out, iter);
    // std::swap(this->x_km1.data, x_k.data);
    // this->x_km1 = x_k_out;
    Linalg::set_value_general(this->x_km1.data, x_k_out.data, this->x_km1.length);
    }
    std::swap(this->f_km1.data, f_k.data);

    //modify the output x_k
    std::swap(x_k.data, x_k_out.data);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "The Mixing run took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

template<typename T>
void Mixing<T>::run_mp(T* const x_k, const uint iter, const MPI_Comm comm,
                        Memory_pool<T, Fast_memory>& pool_fast,
                        Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    const Vertices_3D local_vertices = this->exarr_mpi_package.domain_vertices.get_3D_local_vertices();
    const uint ncol = this->spin.generate_nspin();
    const uint nd = local_vertices.get_size();

    // Array_3D<T> f_k = x_k - this->x_km1;
    // Array_3D<T> f_k(local_vertices);
    T* f_k = pool_fast.allocate(ncol * nd);
    
    #pragma omp parallel
    Linalg::hadamard_minus_general(f_k, x_k, this->x_km1.data, ncol * nd);
    //record the history when linear mixing dont record
    this->record_F_history(f_k, iter, nd, ncol);

    //generate history weight average x and f
    // Array_3D<T> x_wavg(local_vertices);
    // Array_3D<T> f_wavg(local_vertices);
    T* x_wavg = pool_fast.allocate(ncol * nd);
    T* f_wavg = pool_fast.allocate(ncol * nd);
    // this->generate_weight_average(x_wavg, f_wavg, x_k, f_k, iter, comm);
    this->generate_weight_average_mp(x_wavg, f_wavg, this->x_km1.data, f_k, iter, comm,
                                    nd, ncol, pool_fast, pool_cap);

    // apply preconditioner if required, Pf = amix * (P * f_tot)
    // Array_3D<T> pf_wavg(local_vertices);
    T *const pf_wavg = this->pf.data;
    this->apply_precondition_mp(f_wavg, pf_wavg, local_vertices, iter, ncol, pool_fast, pool_cap);
    
    this->shift_pf(f_wavg, pf_wavg, local_vertices, ncol, comm);

    //generate x_k_out
    // Array_3D<T> x_k_out(local_vertices);
    T* x_k_out = pool_fast.allocate(ncol * nd);
    // x_k_out = x_wavg + pf_wavg;
    #pragma omp parallel
    Linalg::hadamard_plus_general(x_k_out, x_wavg, pf_wavg, ncol * nd);

    this->scale_density(x_k, x_k_out, comm, nd);

    //record the history when linear mixing dont record
    // this->record_history(x_k_out, f_k, iter);
    this->record_R_history(x_k_out, iter, nd, ncol);
    // std::swap(this->x_km1.data, x_k.data);
    // this->x_km1 = x_k_out;
    #pragma omp parallel
    Linalg::set_value_general(this->x_km1.data, x_k_out, nd * ncol);
    #pragma omp parallel
    // std::swap(this->f_km1.data, f_k.data);
    Linalg::set_value_general(this->f_km1.data, f_k, nd * ncol);

    // //modify the output x_k
    // std::swap(x_k.data, x_k_out.data);
    #pragma omp parallel
    Linalg::set_value_general(x_k, x_k_out, nd * ncol);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "The Mixing run took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

template<typename T>
void Mixing<T>::init(const Array_3D<T>& x_km1, char const* const& header) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (strcmp(header, "SCF") == 0) {
        Vertices_3D local_vertices = x_km1.get_vertices();
        this->x_km1.reconstructor(local_vertices);
        this->f_km1.reconstructor(local_vertices, 0);
        this->pf.reconstructor(local_vertices, 0);
        if (this->mixing_control.method == 0) {
            // do not reconstruct R and F
        } else if (this->mixing_control.method == 1) {
            Vertices_4D local_vertices_4d(local_vertices, this->mixing_control.mixing_history);
            this->R.reconstructor(local_vertices_4d, 0);
            this->F.reconstructor(local_vertices_4d, 0);
        } else {
            assert(this->mixing_control.method == 0 || this->mixing_control.method == 1);
        }
        this->set_x_km1(x_km1);
    } else {
        assert(!"header should be 'SCF'");
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "The mixing init took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

template<typename T>
template<typename T2>
void Mixing<T>::init(const Mixing<T2>& mixing) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    this->x_km1.deepcopy(std::move(mixing.x_km1.as_type(this->x_km1.data)));
    this->f_km1.deepcopy(std::move(mixing.f_km1.as_type(this->f_km1.data)));
    this->pf.deepcopy(std::move(mixing.pf.as_type(this->pf.data)));
    if (this->mixing_control.method == 0) {
        // do not reconstruct R and F
    } else if (this->mixing_control.method == 1) {
        this->R.deepcopy(std::move(mixing.R.as_type(this->R.data)));
        this->F.deepcopy(std::move(mixing.F.as_type(this->F.data)));
    } else {
        assert(this->mixing_control.method == 0 || this->mixing_control.method == 1);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "The mixing init from other took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}
template void Mixing<float>::init(const Mixing<float>& mixing);
template void Mixing<double>::init(const Mixing<double>& mixing);
template void Mixing<float>::init(const Mixing<double>& mixing);
template void Mixing<double>::init(const Mixing<float>& mixing);

template<typename T>
void Mixing<T>::destructor() {
    x_km1.destructor();
    f_km1.destructor();
    pf.destructor();
    R.destructor();
    F.destructor();
    return;
}


template<typename T>
void Mixing<T>::show() const {
    this->mixing_control.show();
}

template class Mixing<float>;
template class Mixing<double>;

/**
 * @brief Anderson extrapolation update.
 *
 *        x_{k+1} = (x_k - X * Gamma) + beta * P * (f_k - F * Gamma),
 *        where P is the preconditioner, and Gamma = inv(F^T * F) * F^T * f.
 *        Expanding above equation gives: 
 *        x_{k+1} = x_k + beta * P * f - (X + beta * P * F) * inv(F^T * F) * F^T * f
 * 
 * @ref https://github.com/SPARC-X/SPARC/blob/c1c792e66835b4f550f8c4eb2b1fe7e395a1e60e/src/mixing.c#L48
 * @ref https://www.sciencedirect.com/science/article/abs/pii/S001046551830256X Eq(3)
 */
template<typename T>
void Mixing_method::AndersonExtrapolation(const int N, const int m, T* const __restrict__ x_kp1,
                                          T const* const __restrict__ x_k, T const* const __restrict__ f_k,
                                          T const* const __restrict__ X, T const* const __restrict__ F,
                                          const T beta, const MPI_Comm comm) {
    #ifdef USE_OPENMP

    static T* history_v_static = nullptr;
    #pragma omp single
    history_v_static = new T [N];
    T* const history_v = history_v_static;

    // find the weighted average vectors
    Mixing_method::Anderson_history_vector(N, m, beta, f_k, X, F, history_v, comm);

    #pragma omp parallel
    {
    // x_kp1 = x_k + bete * f_k
    #pragma omp barrier
    Linalg::scalar_product_general(x_kp1, f_k, (T)beta, N, x_k);

    // x_kp1 = x_kp1 + history_v;
    Linalg::hadamard_plus_general(x_kp1, history_v, N);
    }
    #pragma omp barrier
    #pragma omp single nowait
    {
        delete [] history_v_static;
        history_v_static = nullptr;
    }

    #else //USE_OPENMP

    T* history_v = new T [N];
    // find the weighted average vectors
    Mixing_method::Anderson_history_vector(N, m, beta, f_k, X, F, history_v, comm);

    // x_kp1 = x_k + bete * f_k
    Linalg::scalar_product_general(x_kp1, f_k, (T)beta, N, x_k);


    // x_kp1 = x_kp1 + history_v;
    Linalg::hadamard_plus_general(x_kp1, history_v, N);
    delete [] history_v;

    #endif //USE_OPENMP
    return;
}
template void Mixing_method::AndersonExtrapolation<float>(const int N, const int m, float* const x_kp1,
                                                          float const* const x_k, float const* const f_k,
                                                          float const* const X, float const* const F,
                                                          const float beta, const MPI_Comm comm);
template void Mixing_method::AndersonExtrapolation<double>(const int N, const int m, double* const x_kp1,
                                                           double const* const x_k, double const* const f_k,
                                                           double const* const X, double const* const F,
                                                           const double beta, const MPI_Comm comm);

template<typename T>
void Mixing_method::AndersonExtrapolation_ompunnested(const int N, const int m, T* const __restrict__ x_kp1,
                                          T const* const __restrict__ x_k, T const* const __restrict__ f_k,
                                          T const* const __restrict__ X, T const* const __restrict__ F,
                                          const T beta, const MPI_Comm comm) {
    T* history_v = new T [N];
    // find the weighted average vectors
    Mixing_method::Anderson_history_vector_ompunnested(N, m, beta, f_k, X, F, history_v, comm);
    #pragma omp parallel
    {
        // x_kp1 = x_k + bete * f_k
        Linalg::scalar_product_general(x_kp1, f_k, (T)beta, N, x_k);
        // x_kp1 = x_kp1 + history_v;
        Linalg::hadamard_plus_general(x_kp1, history_v, N);
    }
    delete [] history_v;
    return;
}
template void Mixing_method::AndersonExtrapolation_ompunnested<float>(const int N, const int m, float* const x_kp1,
                                                          float const* const x_k, float const* const f_k,
                                                          float const* const X, float const* const F,
                                                          const float beta, const MPI_Comm comm);
template void Mixing_method::AndersonExtrapolation_ompunnested<double>(const int N, const int m, double* const x_kp1,
                                                           double const* const x_k, double const* const f_k,
                                                           double const* const X, double const* const F,
                                                           const double beta, const MPI_Comm comm);

template<typename T>
void Mixing_method::AndersonExtrapolation_ompunnested_mp(const int N, const int m, T* const __restrict__ x_kp1,
                                          T const* const __restrict__ x_k, T const* const __restrict__ f_k,
                                          T const* const __restrict__ X, T const* const __restrict__ F,
                                          const T beta, const MPI_Comm comm,
                                          Memory_pool<T, Fast_memory>& pool_fast,
                                          Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    // T* history_v = new T [N];
    T* history_v = pool_fast.allocate(N);
    // find the weighted average vectors
    Mixing_method::Anderson_history_vector_ompunnested_mp(N, m, beta, f_k, X, F, history_v, comm, pool_fast, pool_cap);
    #pragma omp parallel
    {
        // x_kp1 = x_k + bete * f_k
        Linalg::scalar_product_general(x_kp1, f_k, (T)beta, N, x_k);
        // x_kp1 = x_kp1 + history_v;
        Linalg::hadamard_plus_general(x_kp1, history_v, N);
    }
    // delete [] history_v;
    return;
}
template void Mixing_method::AndersonExtrapolation_ompunnested_mp<float>(const int N, const int m, float* const x_kp1,
                                                            float const* const x_k, float const* const f_k,
                                                            float const* const X, float const* const F,
                                                            const float beta, const MPI_Comm comm,
                                                            Memory_pool<float, Fast_memory>& pool_fast,
                                                            Memory_pool<float, Capacity_memory>& pool_cap);
template void Mixing_method::AndersonExtrapolation_ompunnested_mp<double>(const int N, const int m, double* const x_kp1,
                                                           double const* const x_k, double const* const f_k,
                                                           double const* const X, double const* const F,
                                                           const double beta, const MPI_Comm comm,
                                                            Memory_pool<double, Fast_memory>& pool_fast,
                                                            Memory_pool<double, Capacity_memory>& pool_cap);

template<typename T>
void Mixing_method::Anderson_extrapolation_weighted_averaged_vectors(const int N, const int m,
                                                                     T* const __restrict__ x_wavg, T* const __restrict__ f_wavg,
                                                                     T const* const __restrict__ x_k, T const* const __restrict__ f_k,
                                                                     T const* const __restrict__ R, T const* const __restrict__ F,
                                                                     const MPI_Comm comm) {
    #ifdef USE_OPENMP

    static T* Gamma_static = nullptr;
    #pragma omp single
    Gamma_static = new T [m]();
    T* const Gamma = Gamma_static;

    // find extrapolation weigths Gamma = inv(F^T * F) * F^T * f_k, dim(m,1)
    Mixing_method::cal_Gamma(N, m, F, f_k, Gamma, comm);

    #pragma omp parallel
    if (N > 0) {
        #pragma omp barrier
        Linalg::set_value_general(x_wavg, x_k, N);
        Linalg::set_value_general(f_wavg, f_k, N);
        #pragma omp barrier
        // find weighted average x_{k+1} = x_k - R*Gamma
        Linalg::matrix_vector_product(R, 1, Gamma, x_wavg, N, m, (T)-1.0, (T)1.0);
        // find weighted average f_{k+1} = f_k - F*Gamma
        Linalg::matrix_vector_product(F, 1, Gamma, f_wavg, N, m, (T)-1.0, (T)1.0);
    }
    #pragma omp barrier
    #pragma omp single nowait
    {
        delete [] Gamma_static;
        Gamma_static = nullptr;
    }

    #else //USE_OPENMP

    T* const Gamma = new T [m]();
    // find extrapolation weigths Gamma = inv(F^T * F) * F^T * f_k, dim(m,1)
    Mixing_method::cal_Gamma(N, m, F, f_k, Gamma, comm);

    if (N > 0) {
        // find weighted average x_{k+1} = x_k - R*Gamma
        Linalg::set_value_general(x_wavg, x_k, N);
        Linalg::matrix_vector_product(R, 1, Gamma, x_wavg, N, m, (T)-1.0, (T)1.0);
        // find weighted average f_{k+1} = f_k - F*Gamma
        Linalg::set_value_general(f_wavg, f_k, N);
        Linalg::matrix_vector_product(F, 1, Gamma, f_wavg, N, m, (T)-1.0, (T)1.0);
    }

    delete [] Gamma;

    #endif //USE_OPENMP
    return;
}
template void Mixing_method::Anderson_extrapolation_weighted_averaged_vectors<float>(const int N, const int m,
                                                                                     float* const x_wavg, float* const f_wavg,
                                                                                     float const* const x_k, float const* const f_k,
                                                                                     float const* const X, float const* const F,
                                                                                     const MPI_Comm comm);
template void Mixing_method::Anderson_extrapolation_weighted_averaged_vectors<double>(const int N, const int m,
                                                                                      double* const x_wavg, double* const f_wavg,
                                                                                      double const* const x_k, double const* const f_k,
                                                                                      double const* const X, double const* const F,
                                                                                      const MPI_Comm comm);

template<typename T>
void Mixing_method::Anderson_extrapolation_weighted_averaged_vectors_mp(const int N, const int m,
                                                                        T* const __restrict__ x_wavg, T* const __restrict__ f_wavg,
                                                                        T const* const __restrict__ x_k, T const* const __restrict__ f_k,
                                                                        T const* const __restrict__ R, T const* const __restrict__ F,
                                                                        const MPI_Comm comm,
                                                                        Memory_pool<T, Fast_memory>& pool_fast,
                                                                        Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);

    T* const Gamma = pool_fast.allocate(m);
    Linalg::set_value_general(Gamma, T(0), m);
    // find extrapolation weigths Gamma = inv(F^T * F) * F^T * f_k, dim(m,1)
    Mixing_method::cal_Gamma_ompunnested_mp(N, m, F, f_k, Gamma, comm, pool_fast, pool_cap);

    if (N > 0) {
        // find weighted average x_{k+1} = x_k - R*Gamma
        #pragma omp parallel
        Linalg::set_value_general(x_wavg, x_k, N);
        // find weighted average f_{k+1} = f_k - F*Gamma
        #pragma omp parallel
        Linalg::set_value_general(f_wavg, f_k, N);
        if constexpr (std::is_same_v<T, double>) {
            int status = Xlsdft_backend::pulay_update(N, m, R, N, Gamma, x_wavg);
            Xlsdft_backend::require_success(Xlsdft_backend::Operation::pulay_update,
                status, comm,
                {"pulay_update_x", N, 1, m, N, 0, N,
                 R, Gamma, x_wavg, nullptr, 0});
            status = Xlsdft_backend::pulay_update(N, m, F, N, Gamma, f_wavg);
            Xlsdft_backend::require_success(Xlsdft_backend::Operation::pulay_update,
                status, comm,
                {"pulay_update_f", N, 1, m, N, 0, N,
                 F, Gamma, f_wavg, nullptr, 0});
        } else {
            for (int column = 0; column < m; ++column) {
                const T weight = Gamma[column];
                for (int row = 0; row < N; ++row) {
                    x_wavg[row] -= R[row + static_cast<std::size_t>(column) * N] * weight;
                    f_wavg[row] -= F[row + static_cast<std::size_t>(column) * N] * weight;
                }
            }
        }
    }

    // delete [] Gamma;

    return;
}
template void Mixing_method::Anderson_extrapolation_weighted_averaged_vectors_mp<float>(const int N, const int m,
                                                                                        float* const x_wavg, float* const f_wavg,
                                                                                        float const* const x_k, float const* const f_k,
                                                                                        float const* const X, float const* const F,
                                                                                        const MPI_Comm comm,
                                                                                        Memory_pool<float, Fast_memory>& pool_fast,
                                                                                        Memory_pool<float, Capacity_memory>& pool_cap);
template void Mixing_method::Anderson_extrapolation_weighted_averaged_vectors_mp<double>(const int N, const int m,
                                                                                        double* const x_wavg, double* const f_wavg,
                                                                                        double const* const x_k, double const* const f_k,
                                                                                        double const* const X, double const* const F,
                                                                                        const MPI_Comm comm,
                                                                                        Memory_pool<double, Fast_memory>& pool_fast,
                                                                                        Memory_pool<double, Capacity_memory>& pool_cap);

template<typename T>
void Mixing_method::Anderson_history_vector(const uint N, const uint m, const T beta, T const* const __restrict__ f_k,
                                            T const* const __restrict__ X, T const* const __restrict__ F,
                                            T* const __restrict__ history_v, const MPI_Comm comm) {
    #ifdef USE_OPENMP

    static T* Gamma_static = nullptr;
    static T* XF_static = nullptr;
    #pragma omp single nowait
    Gamma_static = new T [m]();
    #pragma omp single nowait
    XF_static = new T [N * m];
    #pragma omp barrier
    T* const Gamma = Gamma_static;
    T* const XF = XF_static;

    // find extrapolation weigths Gamma = inv(F^T * F) * F^T * f_k, dim(m,1)
    Mixing_method::cal_Gamma(N, m, F, f_k, Gamma, comm);

    // find weighted average XF_{k} = -X_{k} - beta*F_{k}, dim(N,m)
    //XF = X * (T)-1.0 + (T)-beta * F
    #pragma omp parallel
    {
    #pragma omp barrier
    Linalg::scalar_product_general(XF, X, (T)-1.0, N*m, F, (T)-beta);

    // find weighted average history_v = XF_{k} * Gamma, dim(N,1)
    if (N > 0) {
        #pragma omp barrier
        Linalg::matrix_vector_product(XF, 1, Gamma, history_v, N, m);
    }
    }
    #pragma omp barrier
    #pragma omp single nowait
    {
        delete [] Gamma_static;
        Gamma_static = nullptr;
    }
    #pragma omp single nowait
    {
        delete [] XF_static;
        XF_static = nullptr;
    }

    #else //USE_OPENMP

    T* const Gamma = new T [m]();
    // find extrapolation weigths Gamma = inv(F^T * F) * F^T * f_k, dim(m,1)
    Mixing_method::cal_Gamma(N, m, F, f_k, Gamma, comm);
    
    // find weighted average XF_{k} = -X_{k} - beta*F_{k}, dim(N,m)
    //XF = X * (T)-1.0 + (T)-beta * F
    T* const XF = new T [N*m];
    Linalg::scalar_product_general(XF, X, (T)-1.0, N*m, F, (T)-beta);
    
    // find weighted average history_v = XF_{k} * Gamma, dim(N,1)
    if (N > 0) {
        Linalg::matrix_vector_product(XF, 1, Gamma, history_v, N, m);
    }
    delete [] XF;
    delete [] Gamma;

    #endif //USE_OPENMP
    return;
}
template void Mixing_method::Anderson_history_vector<float>(const uint N, const uint m, const float beta, float const* const f_k,
                                                            float const* const X, float const* const F, float* const history_v, const MPI_Comm comm);
template void Mixing_method::Anderson_history_vector<double>(const uint N, const uint m, const double beta, double const* const f_k,
                                                             double const* const X, double const* const F, double* const history_v, const MPI_Comm comm);

template<typename T>
void Mixing_method::Anderson_history_vector_ompunnested(const uint N, const uint m, const T beta, T const* const __restrict__ f_k,
                                            T const* const __restrict__ X, T const* const __restrict__ F,
                                            T* const __restrict__ history_v, const MPI_Comm comm) {

    T* const Gamma = new T [m]();
    // find extrapolation weigths Gamma = inv(F^T * F) * F^T * f_k, dim(m,1)
    Mixing_method::cal_Gamma_ompunnested(N, m, F, f_k, Gamma, comm);

    // find weighted average XF_{k} = -X_{k} - beta*F_{k}, dim(N,m)
    //XF = X * (T)-1.0 + (T)-beta * F
    T* const XF = new T [N*m];
    #pragma omp parallel
    Linalg::scalar_product_general(XF, X, T(-1.0), N*m, F, -beta);

    // find weighted average history_v = XF_{k} * Gamma, dim(N,1)
    if (N > 0) {
        #pragma omp parallel
        Linalg::matrix_vector_product(XF, 1, Gamma, history_v, N, m);
    }
    delete [] XF;
    delete [] Gamma;

    return;
}
template void Mixing_method::Anderson_history_vector_ompunnested<float>(const uint N, const uint m, const float beta, float const* const f_k,
                                                            float const* const X, float const* const F, float* const history_v, const MPI_Comm comm);
template void Mixing_method::Anderson_history_vector_ompunnested<double>(const uint N, const uint m, const double beta, double const* const f_k,
                                                             double const* const X, double const* const F, double* const history_v, const MPI_Comm comm);

template<typename T>
void Mixing_method::Anderson_history_vector_ompunnested_mp(const uint N, const uint m, const T beta, T const* const __restrict__ f_k,
                                            T const* const __restrict__ X, T const* const __restrict__ F,
                                            T* const __restrict__ history_v, const MPI_Comm comm,
                                        Memory_pool<T, Fast_memory>& pool_fast,
                                        Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);

    // T* const Gamma = new T [m]();
    T* const Gamma = pool_fast.allocate(m);
    Linalg::set_value_general(Gamma, T(0), m);

    // find extrapolation weigths Gamma = inv(F^T * F) * F^T * f_k, dim(m,1)
    Mixing_method::cal_Gamma_ompunnested_mp(N, m, F, f_k, Gamma, comm, pool_fast, pool_cap);

    // find weighted average XF_{k} = -X_{k} - beta*F_{k}, dim(N,m)
    //XF = X * (T)-1.0 + (T)-beta * F
    // T* const XF = new T [N*m];
    T* const XF = pool_fast.allocate(N*m);
    #pragma omp parallel
    Linalg::scalar_product_general(XF, X, T(-1.0), N*m, F, -beta);

    // find weighted average history_v = XF_{k} * Gamma, dim(N,1)
    if (N > 0) {
        #pragma omp parallel
        Linalg::matrix_vector_product(XF, 1, Gamma, history_v, N, m);
    }
    // delete [] XF;
    // delete [] Gamma;

    return;
}
template void Mixing_method::Anderson_history_vector_ompunnested_mp<float>(const uint N, const uint m, const float beta, float const* const f_k,
                                                                    float const* const X, float const* const F, float* const history_v, const MPI_Comm comm,
                                                                    Memory_pool<float, Fast_memory>& pool_fast,
                                                                    Memory_pool<float, Capacity_memory>& pool_cap);
template void Mixing_method::Anderson_history_vector_ompunnested_mp<double>(const uint N, const uint m, const double beta, double const* const f_k,
                                                                    double const* const X, double const* const F, double* const history_v, const MPI_Comm comm,
                                                                    Memory_pool<double, Fast_memory>& pool_fast,
                                                                    Memory_pool<double, Capacity_memory>& pool_cap);

/**
 * @brief Anderson extrapolation coefficiens.
 * 
 *        Gamma = inv(F^T * F) * F^T * f.
 * @param N, m F(N,m), f(N,1), Gamma(m, 1)
 * @param comm domain comm
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/mixing.c#L113
 */
template<typename T>
void Mixing_method::cal_Gamma(const uint N, const uint m,
                              T const* const __restrict__ F,
                              T const* const __restrict__ f,
                              T* const __restrict__ Gamma,
                              const MPI_Comm comm) {
    if (m == 0) return;
    T* FtF = new (std::align_val_t(64)) T[static_cast<std::size_t>(m) * m];
    if (N > 0) {
        #pragma omp parallel
        {
            Linalg::matrix_product(F, 0, F, 1, FtF, 1, m, m, N);
            Linalg::matrix_vector_product(F, 0, f, Gamma, m, N);
        }
    } else {
        std::fill_n(FtF, static_cast<std::size_t>(m) * m, T(0));
        std::fill_n(Gamma, m, T(0));
    }

    int comm_size = 1;
    MPI_Comm_size(comm, &comm_size);
    if (comm_size > 1) {
        const MPI_Datatype datatype = Linalg::get_mpi_datatype<T>();
        MPI_Allreduce(MPI_IN_PLACE, FtF, m * m, datatype, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, Gamma, m, datatype, MPI_SUM, comm);
    }
    solve_pulay_fallback(m, FtF, Gamma);
    ::operator delete[](FtF, std::align_val_t(64));
}
template void Mixing_method::cal_Gamma<float>(const uint N, const uint m, float const* const F,
                                              float const* const f, float* const Gamma, const MPI_Comm comm);
template void Mixing_method::cal_Gamma<double>(const uint N, const uint m, double const* const F,
                                               double const* const f, double* const Gamma, const MPI_Comm comm);

template<typename T>
void Mixing_method::cal_Gamma_ompunnested(
        const uint N, const uint m, T const* const __restrict__ F,
        T const* const __restrict__ f, T* const __restrict__ Gamma,
        const MPI_Comm comm) {
    Mixing_method::cal_Gamma(N, m, F, f, Gamma, comm);
}
template void Mixing_method::cal_Gamma_ompunnested<float>(const uint N, const uint m, float const* const F,
                                              float const* const f, float* const Gamma, const MPI_Comm comm);
template void Mixing_method::cal_Gamma_ompunnested<double>(const uint N, const uint m, double const* const F,
                                               double const* const f, double* const Gamma, const MPI_Comm comm);

template<typename T>
void Mixing_method::cal_Gamma_ompunnested_mp(
        const uint N, const uint m, T const* const __restrict__ F,
        T const* const __restrict__ f, T* const __restrict__ Gamma,
        const MPI_Comm comm, Memory_pool<T, Fast_memory>& pool_fast,
        Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    if (m == 0) return;
    #ifdef USE_OPENMP
    if (omp_in_parallel()) {
        Xlsdft_backend::require_success(
            Xlsdft_backend::Operation::pulay_gram,
            Xlsdft_backend::fixed_configuration_error, comm,
            {"pulay_requires_serial_caller", m, 1, N, N, 0, m,
             F, f, Gamma, nullptr, 0});
    }
    #endif

    T* FtF = pool_fast.allocate(static_cast<std::size_t>(m) * m);
    T* singular_values = pool_fast.allocate(m);
    if (N > 0) {
        if constexpr (std::is_same_v<T, double>) {
            int status = Xlsdft_backend::pulay_gram(
                static_cast<int>(N), static_cast<int>(m), F,
                static_cast<int>(N), FtF, static_cast<int>(m));
            Xlsdft_backend::require_success(Xlsdft_backend::Operation::pulay_gram,
                status, comm,
                {"pulay_gram", m, m, N, N, 0, m,
                 F, nullptr, FtF, nullptr, 0});
            status = Xlsdft_backend::pulay_tmv(
                static_cast<int>(N), static_cast<int>(m), F,
                static_cast<int>(N), f, Gamma);
            Xlsdft_backend::require_success(Xlsdft_backend::Operation::pulay_tmv,
                status, comm,
                {"pulay_tmv", m, 1, N, N, 0, m,
                 F, f, Gamma, nullptr, 0});
        } else {
            for (uint column = 0; column < m; ++column) {
                T projection = T(0);
                for (uint row = 0; row < N; ++row) {
                    projection += F[row + static_cast<std::size_t>(column) * N] * f[row];
                }
                Gamma[column] = projection;
                for (uint other = 0; other < m; ++other) {
                    T dot = T(0);
                    for (uint row = 0; row < N; ++row) {
                        dot += F[row + static_cast<std::size_t>(column) * N]
                             * F[row + static_cast<std::size_t>(other) * N];
                    }
                    FtF[column + static_cast<std::size_t>(other) * m] = dot;
                }
            }
        }
    } else {
        std::fill_n(FtF, static_cast<std::size_t>(m) * m, T(0));
        std::fill_n(Gamma, m, T(0));
    }

    int comm_size = 1;
    MPI_Comm_size(comm, &comm_size);
    if (comm_size > 1) {
        const MPI_Datatype datatype = Linalg::get_mpi_datatype<T>();
        MPI_Allreduce(MPI_IN_PLACE, FtF, m * m, datatype, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, Gamma, m, datatype, MPI_SUM, comm);
    }

    int status = 0;
    const void* failure_workspace = nullptr;
    std::size_t failure_workspace_doubles = 0U;
    if constexpr (std::is_same_v<T, double>) {
        const std::size_t workspace_doubles =
            Xlsdft_backend::pulay_gelsd_workspace_doubles(static_cast<int>(m));
        double* workspace = pool_fast.allocate(workspace_doubles);
        failure_workspace = workspace;
        failure_workspace_doubles = workspace_doubles;
        alignas(64) int rank = 0;
        status = Xlsdft_backend::pulay_gelsd(
            static_cast<int>(m), FtF, static_cast<int>(m), Gamma,
            singular_values, -1.0, &rank, workspace, workspace_doubles);
    } else {
        status = solve_pulay_via_f64(m, FtF, Gamma, singular_values);
    }
    Xlsdft_backend::require_success(Xlsdft_backend::Operation::pulay_gelsd,
        status, comm,
        {"pulay_gelsd", m, m, 1, m, 0, m,
         FtF, Gamma, singular_values,
         failure_workspace, failure_workspace_doubles});
}
template void Mixing_method::cal_Gamma_ompunnested_mp<float>(const uint N, const uint m, float const* const F,
                                                float const* const f, float* const Gamma, const MPI_Comm comm,
                                                Memory_pool<float, Fast_memory>& pool_fast,
                                                Memory_pool<float, Capacity_memory>& pool_cap);
template void Mixing_method::cal_Gamma_ompunnested_mp<double>(const uint N, const uint m, double const* const F,
                                               double const* const f, double* const Gamma, const MPI_Comm comm,
                                                Memory_pool<double, Fast_memory>& pool_fast,
                                                Memory_pool<double, Capacity_memory>& pool_cap);

/**
 * @brief  Perform Kerker preconditioner.
 *
 *         Apply Kerker preconditioner in real space. For given 
 *         function f, this function returns 
 *         Pf := a * (L - lambda_TF^2)^-1 * (L - idemac*lambda_TF^2)f, 
 *         where L is the discrete Laplacian operator, c is the 
 *         inverse of diemac (dielectric macroscopic constant).
 *         When c is 0, it's the original Kerker preconditioner.
 *         The result is written in Pf.
 * 
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/mixing.c#L460
 */
template<typename T>
void Mixing_method::kerker_precondition(const Array_3D<T>& f_wavg, Array_3D<T>& pf_wavg, const T& alpha,
                                        const Stencil<T> stencil, const Kerker_control& kerker_control,
                                        const Exarr_3D_mpi_package& exarr_mpi_package) {
    const MPI_Comm& comm = exarr_mpi_package.comm;
    const int aar_precondition_method = 1;  // TODO: should be adjustable outside
    //init aar
    Aar<T> aar(exarr_mpi_package.domain_vertices);
    aar.init();
    aar.aar_control.set_residual_method(0);
    aar.aar_control.set_precondition_method(aar_precondition_method);
    aar.init_res_method0(stencil, exarr_mpi_package, 1.0, - kerker_control.kerker_ktf * kerker_control.kerker_ktf);
    aar.init_pre_method0(stencil, 1.0, - kerker_control.kerker_ktf * kerker_control.kerker_ktf);
    aar.aar_control.set_tolerance(kerker_control.tolerance);
    aar.aar_control.set_max_iter(1000);
    if (aar_precondition_method == 0) {  // jacobi
    } else if (aar_precondition_method == 1) {
        aar.init_pre_dst();
    } else if (aar_precondition_method == 2) {
        // aar.init_pre_mg(mesh_control);
        assert(false);
    } else {
        assert(false);
    }

    const Vertices_3D vertices = f_wavg.get_vertices();
    const Vertices_3D ex_vertices = vertices.generate_ex_vertices(stencil.FDn);

    //Lf
    Stencil<T> stencil_temp = stencil.coeffs_scale(1.0, 2);
    stencil_temp.shift_D2_coeffs(-kerker_control.kerker_ktf * kerker_control.kerker_ktf * kerker_control.kerker_thresh);
    Array_3D<T> Lf(vertices);
    Array_3D<T> ex_f_wavg(ex_vertices, 0);
    exarr_mpi_package.fill_domain_par_ex_arr(f_wavg, ex_f_wavg);
    Stencil_method::calc_laplacian(ex_f_wavg.data, ex_vertices, stencil_temp, vertices, Lf.data, vertices);

    // T Lf_2norm = Linalg::vector_2norm(Lf.data, Lf.length, comm);

    aar.run(pf_wavg.data, Lf.data, vertices, comm);

    if (fabs(kerker_control.kerker_ktf) < 1e-14) {
        // in this case the result will be shifted by a constant
        uint64_t sum_Nd = exarr_mpi_package.domain_vertices.shared_vertices.Vertices_3D::get_size();
        T shift = Linalg::vector_sum(pf_wavg.data, pf_wavg.length, comm);
        // pf_wavg -= shift/(T)pf_wavg.length;
        Linalg::scalar_minus_general(pf_wavg.data, shift/T(sum_Nd), pf_wavg.length);
    }

    // pf_wavg *= -alpha;
    Linalg::scalar_product_general(pf_wavg.data, -alpha, pf_wavg.length);

    return;
}
template void Mixing_method::kerker_precondition<float>(const Array_3D<float>& f_wavg, Array_3D<float>& pf_wavg, const float& alpha,
                                                        const Stencil<float> stencil, const Kerker_control& kerker_control,
                                                        const Exarr_3D_mpi_package& exarr_mpi_package);
template void Mixing_method::kerker_precondition<double>(const Array_3D<double>& f_wavg, Array_3D<double>& pf_wavg, const double& alpha,
                                                         const Stencil<double> stencil, const Kerker_control& kerker_control,
                                                         const Exarr_3D_mpi_package& exarr_mpi_package);

/**
 * @brief  Perform Kerker preconditioner.
 *
 *         Apply Kerker preconditioner in real space. For given 
 *         function f, this function returns 
 *         Pf := a * (L - lambda_TF^2)^-1 * (L - idemac*lambda_TF^2)f, 
 *         where L is the discrete Laplacian operator, c is the 
 *         inverse of diemac (dielectric macroscopic constant).
 *         When c is 0, it's the original Kerker preconditioner.
 *         The result is written in Pf.
 * 
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/mixing.c#L460
 */
template<typename T>
void Mixing_method::kerker_precondition(T const* const __restrict__ f_wavg, T* const __restrict__ pf_wavg, const Vertices_3D& vertices,
                                        const T alpha, const Stencil<T> stencil, const Kerker_control& kerker_control,
                                        const Exarr_3D_mpi_package& exarr_mpi_package) {
    const MPI_Comm& comm = exarr_mpi_package.comm;
    const int aar_precondition_method = 1;  // TODO: should be adjustable outside
    //init aar
    Aar<T> aar(exarr_mpi_package.domain_vertices);
    aar.init();
    aar.aar_control.set_residual_method(0);
    aar.aar_control.set_precondition_method(aar_precondition_method);
    aar.init_res_method0(stencil, exarr_mpi_package, 1.0, - kerker_control.kerker_ktf * kerker_control.kerker_ktf);
    aar.init_pre_method0(stencil, 1.0, - kerker_control.kerker_ktf * kerker_control.kerker_ktf);
    aar.aar_control.set_tolerance(kerker_control.tolerance);
    aar.aar_control.set_max_iter(1000);
    if (aar_precondition_method == 0) {  // jacobi
    } else if (aar_precondition_method == 1) {
        aar.init_pre_dst();
    } else if (aar_precondition_method == 2) {
        // aar.init_pre_mg(mesh_control);
        assert(false);
    } else {
        assert(false);
    }

    const Vertices_3D ex_vertices = vertices.generate_ex_vertices(stencil.FDn);
    const uint ex_Nd = ex_vertices.get_size();
    const uint Nd = vertices.get_size();

    //Lf
    Stencil<T> stencil_temp = stencil.coeffs_scale(1.0, 2);
    stencil_temp.shift_D2_coeffs((T)(-kerker_control.kerker_ktf * kerker_control.kerker_ktf * kerker_control.kerker_thresh));
    T* const Lf = new T [Nd];
    T* const ex_f_wavg = new T [ex_Nd]();
    exarr_mpi_package.fill_domain_par_ex_arr(f_wavg, vertices, ex_f_wavg, ex_vertices);
    #pragma omp parallel
    Stencil_method::calc_laplacian(ex_f_wavg, ex_vertices, stencil_temp, vertices, Lf, vertices);
    delete [] ex_f_wavg;

    // T Lf_2norm = Linalg::vector_2norm(Lf.data, Lf.length, comm);

    aar.run(pf_wavg, Lf, vertices, comm);
    delete [] Lf;
    aar.destructor();

    #pragma omp parallel
    {
        if (fabs(kerker_control.kerker_ktf) < 1e-14) {
            // in this case the result will be shifted by a constant
            T shift = Linalg::vector_sum(pf_wavg, Nd, comm);
            uint64_t sum_Nd = exarr_mpi_package.domain_vertices.shared_vertices.Vertices_3D::get_size();
            Linalg::scalar_plus_general(pf_wavg, -shift/T(sum_Nd), Nd);
        }
        Linalg::scalar_product_general(pf_wavg, -alpha, Nd);
    }
    return;
}
template void Mixing_method::kerker_precondition<float>(float const* const f_wavg, float* const pf_wavg, const Vertices_3D& vertices,
                                                        const float alpha, const Stencil<float> stencil, const Kerker_control& kerker_control,
                                                        const Exarr_3D_mpi_package& exarr_mpi_package);
template void Mixing_method::kerker_precondition<double>(double const* const f_wavg, double* const pf_wavg, const Vertices_3D& vertices,
                                                         const double alpha, const Stencil<double> stencil, const Kerker_control& kerker_control,
                                                         const Exarr_3D_mpi_package& exarr_mpi_package);

/**
 * @brief  Perform Kerker preconditioner.
 *
 *         Apply Kerker preconditioner in real space. For given 
 *         function f, this function returns 
 *         Pf := a * (L - lambda_TF^2)^-1 * (L - idemac*lambda_TF^2)f, 
 *         where L is the discrete Laplacian operator, c is the 
 *         inverse of diemac (dielectric macroscopic constant).
 *         When c is 0, it's the original Kerker preconditioner.
 *         The result is written in Pf.
 * 
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/mixing.c#L460
 */
template<typename T>
void Mixing_method::kerker_precondition_mp(T const* const __restrict__ f_wavg, T* const __restrict__ pf_wavg, const Vertices_3D& vertices,
                                            const T alpha, const Stencil<T> stencil, const Kerker_control& kerker_control,
                                            const Exarr_3D_mpi_package& exarr_mpi_package,
                                            Memory_pool<T, Fast_memory>& pool_fast,
                                            Memory_pool<T, Capacity_memory>& pool_cap) {
    Memory_pool_scope<Memory_pool<T, Fast_memory>> scope_fast(pool_fast);
    Memory_pool_scope<Memory_pool<T, Capacity_memory>> scope_cap(pool_cap);
    const MPI_Comm& comm = exarr_mpi_package.comm;
    const int aar_precondition_method = 1;  // TODO: should be adjustable outside
    //init aar
    Aar<T> aar(exarr_mpi_package.domain_vertices);
    aar.init();
    aar.aar_control.set_residual_method(0);
    aar.aar_control.set_precondition_method(aar_precondition_method);
    aar.init_res_method0(stencil, exarr_mpi_package, 1.0, - kerker_control.kerker_ktf * kerker_control.kerker_ktf);
    aar.init_pre_method0(stencil, 1.0, - kerker_control.kerker_ktf * kerker_control.kerker_ktf);
    aar.aar_control.set_tolerance(kerker_control.tolerance);
    aar.aar_control.set_max_iter(1000);
    if (aar_precondition_method == 0) {  // jacobi
    } else if (aar_precondition_method == 1) {
        aar.init_pre_dst();
    } else if (aar_precondition_method == 2) {
        // aar.init_pre_mg(mesh_control);
        assert(false);
    } else {
        assert(false);
    }

    const Vertices_3D ex_vertices = vertices.generate_ex_vertices(stencil.FDn);
    const uint ex_Nd = ex_vertices.get_size();
    const uint Nd = vertices.get_size();

    //Lf
    Stencil<T> stencil_temp = stencil.coeffs_scale(1.0, 2);
    stencil_temp.shift_D2_coeffs((T)(-kerker_control.kerker_ktf * kerker_control.kerker_ktf * kerker_control.kerker_thresh));
    // T* const Lf = new T [Nd];
    // T* const ex_f_wavg = new T [ex_Nd]();
    T* const Lf = pool_fast.allocate(Nd);
    T* const ex_f_wavg = pool_fast.allocate(ex_Nd);
    Linalg::set_value_general(ex_f_wavg, T(0), ex_Nd);
    exarr_mpi_package.fill_domain_par_ex_arr(f_wavg, vertices, ex_f_wavg, ex_vertices);
    #pragma omp parallel
    Stencil_method::calc_laplacian(ex_f_wavg, ex_vertices, stencil_temp, vertices, Lf, vertices);
    // delete [] ex_f_wavg;

    // T Lf_2norm = Linalg::vector_2norm(Lf.data, Lf.length, comm);

    aar.run_ompunnested_mp(pf_wavg, Lf, vertices, comm, pool_fast, pool_cap);
    // delete [] Lf;
    aar.destructor();

    #pragma omp parallel
    {
        if (fabs(kerker_control.kerker_ktf) < 1e-14) {
            // in this case the result will be shifted by a constant
            T shift = Linalg::vector_sum(pf_wavg, Nd, comm);
            uint64_t sum_Nd = exarr_mpi_package.domain_vertices.shared_vertices.Vertices_3D::get_size();
            Linalg::scalar_plus_general(pf_wavg, -shift/T(sum_Nd), Nd);
        }
        Linalg::scalar_product_general(pf_wavg, -alpha, Nd);
    }
    return;
}
template void Mixing_method::kerker_precondition_mp<float>(float const* const f_wavg, float* const pf_wavg, const Vertices_3D& vertices,
                                                            const float alpha, const Stencil<float> stencil, const Kerker_control& kerker_control,
                                                            const Exarr_3D_mpi_package& exarr_mpi_package,
                                                            Memory_pool<float, Fast_memory>& pool_fast,
                                                            Memory_pool<float, Capacity_memory>& pool_cap);
template void Mixing_method::kerker_precondition_mp<double>(double const* const f_wavg, double* const pf_wavg, const Vertices_3D& vertices,
                                                            const double alpha, const Stencil<double> stencil, const Kerker_control& kerker_control,
                                                            const Exarr_3D_mpi_package& exarr_mpi_package,
                                                            Memory_pool<double, Fast_memory>& pool_fast,
                                                            Memory_pool<double, Capacity_memory>& pool_cap);
