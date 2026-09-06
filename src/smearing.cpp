#include "smearing.h"

Smearing::Smearing(const Smearing_control& smearing_control)
                    : smearing_control(smearing_control) {}

Smearing::~Smearing() {}

double Smearing::generate_kbt() const {
    return 1.0/this->smearing_control.beta;
}

void Smearing::init(const Smearing&) {
    return;
}

void Smearing::show() const {
    this->smearing_control.show();
    return;
}

template<typename T>
void Smearing_method::smear(T const* const& in_ptr, T* const& out_ptr, const T& mu_f, const Smearing& smearing, const uint& length) {
    switch (smearing.smearing_control.method)
    {
    case 0:
        Smearing_method::fermi_dirac_smearing(in_ptr, out_ptr, mu_f, (T)smearing.smearing_control.beta, length);
        break;
    case 1:
        Smearing_method::gaussian_smearing(in_ptr, out_ptr, mu_f, (T)smearing.smearing_control.beta, length);
        break;
    default:
        assert(smearing.smearing_control.method == 0 || smearing.smearing_control.method == 1);
        break;
    }
}
template void Smearing_method::smear<float>(float const* const& in_ptr, float* const& out_ptr,
                                            const float& mu_f, const Smearing& smearing, const uint& length);
template void Smearing_method::smear<double>(double const* const& in_ptr, double* const& out_ptr,
                                             const double& mu_f, const Smearing& smearing, const uint& length);

template<typename T>
void Smearing_method::generate_entropy_energy_arr(T const* const& __restrict__ occ, T* const& __restrict__ out_ptr,
                                                  const Smearing& smearing, const T& smearing_coef, const uint& length) {
    T coef = -smearing_coef * smearing.generate_kbt();
    for (uint i = 0; i < length; i++) {
        T f_i = occ[i];
        if (f_i > 1e-12 && (1.0-f_i) > 1e-12)
            out_ptr[i] = -(f_i * log(f_i) + (1.0 - f_i) * log(1.0 - f_i)) * coef;
        else 
            out_ptr[i] = (T) 0.0;
    }
    return;
}
template void Smearing_method::generate_entropy_energy_arr<float>(float const* const& occ, float* const& out_ptr,
                                                                  const Smearing& smearing, const float& smearing_coef,
                                                                  const uint& length);
template void Smearing_method::generate_entropy_energy_arr<double>(double const* const& occ, double* const& out_ptr,
                                                                   const Smearing& smearing, const double& smearing_coef,
                                                                   const uint& length);

template<typename T>
void Smearing_method::fermi_dirac_smearing(T const* const& __restrict__ in_ptr, T* const& __restrict__ out_ptr,
                                           const T& mu_f, const T& beta, const uint& length) {
    #ifdef USE_OPENMP_SIMD
    #pragma omp for schedule(static, Linalg::get_chunksize(omp_get_num_threads(), length))
    #endif
    for (uint i = 0; i < length; i++) {
        out_ptr[i] = (T)1.0 / ((T)1.0 + std::exp(beta * (in_ptr[i] - mu_f)));
    }
    return;
}
template void Smearing_method::fermi_dirac_smearing<float>(float const* const& __restrict__ in_ptr, float* const& __restrict__ out_ptr,
                                                           const float& mu_f, const float& beta, const uint& length);
template void Smearing_method::fermi_dirac_smearing<double>(double const* const& __restrict__ in_ptr, double* const& __restrict__ out_ptr,
                                                            const double& mu_f, const double& beta, const uint& length);

template<typename T>
void Smearing_method::gaussian_smearing(T const* const __restrict__ in_ptr, T* const __restrict__ out_ptr,
                                        const T mu_f, const T beta, const uint length) {
    #ifdef USE_OPENMP_SIMD
    #pragma omp for schedule(static, Linalg::get_chunksize(omp_get_num_threads(), length))
    #endif
    for (uint i = 0; i < length; i++) {
        out_ptr[i] = (T)0.5 * ((T)1.0 - erf(beta * (in_ptr[i] - mu_f)));
    }
    return;                               
}
template void Smearing_method::gaussian_smearing<float>(float const* const in_ptr, float* const out_ptr,
                                                        const float mu_f, const float beta, const uint length);
template void Smearing_method::gaussian_smearing<double>(double const* const in_ptr, double* const out_ptr,
                                                         const double mu_f, const double beta, const uint length);
template<typename T>
void Smearing_method::fermi_dirac_smearing_with_deriv(T const* const __restrict__ in_ptr, T* const __restrict__ out_occ, T* const __restrict__ out_docc,
                                                      const T mu_f, const T beta, const uint length) {
    #ifdef USE_OPENMP_SIMD
    #pragma omp for schedule(static, Linalg::get_chunksize(omp_get_num_threads(), length))
    #endif
    for (uint i = 0; i < length; i++) {
        T p = (T)1.0 / ((T)1.0 + std::exp(beta * (in_ptr[i] - mu_f)));
        out_occ[i] = p;
        out_docc[i] = beta * p * ((T)1.0 - p); // 解析导数 df/dmu
    }
}
template void Smearing_method::fermi_dirac_smearing_with_deriv<float>(float const* const in_ptr, float* const out_occ, float* const out_docc, const float mu_f, const float beta, const uint length);
template void Smearing_method::fermi_dirac_smearing_with_deriv<double>(double const* const in_ptr, double* const out_occ, double* const out_docc, const double mu_f, const double beta, const uint length);

template<typename T>
void Smearing_method::gaussian_smearing_with_deriv(T const* const __restrict__ in_ptr, T* const __restrict__ out_occ, T* const __restrict__ out_docc,
                                                   const T mu_f, const T beta, const uint length) {
    const T inv_sqrt_pi = T(1.12837916709551257390); // 2/sqrt(pi)
    #ifdef USE_OPENMP_SIMD
    #pragma omp for schedule(static, Linalg::get_chunksize(omp_get_num_threads(), length))
    #endif
    for (uint i = 0; i < length; i++) {
        T x = beta * (in_ptr[i] - mu_f);
        out_occ[i] = (T)0.5 * ((T)1.0 - erf(x));
        out_docc[i] = (T)0.5 * beta * inv_sqrt_pi * std::exp(-x * x); // 解析导数 df/dmu
    }
}
template void Smearing_method::gaussian_smearing_with_deriv<float>(float const* const __restrict__ in_ptr, float* const __restrict__ out_occ, float* const __restrict__ out_docc, const float mu_f, const float beta, const uint length);
template void Smearing_method::gaussian_smearing_with_deriv<double>(double const* const __restrict__ in_ptr, double* const __restrict__ out_occ, double* const __restrict__ out_docc, const double mu_f, const double beta, const uint length);

template<typename T>
void Smearing_method::smear_with_deriv(T const* const in_ptr, T* const out_occ, T* const out_docc, 
                                       const T mu_f, const Smearing& smearing, const uint length) {
    switch (smearing.smearing_control.method) {
    case 0:
        Smearing_method::fermi_dirac_smearing_with_deriv(in_ptr, out_occ, out_docc, mu_f, (T)smearing.smearing_control.beta, length);
        break;
    case 1:
        Smearing_method::gaussian_smearing_with_deriv(in_ptr, out_occ, out_docc, mu_f, (T)smearing.smearing_control.beta, length);
        break;
    default:
        assert(0 && "Unsupported smearing method for derivative calculation.");
        break;
    }
}
template void Smearing_method::smear_with_deriv<float>(float const* const in_ptr, float* const out_occ, float* const out_docc, const float mu_f, const Smearing& smearing, const uint length);
template void Smearing_method::smear_with_deriv<double>(double const* const in_ptr, double* const out_occ, double* const out_docc, const double mu_f, const Smearing& smearing, const uint length);