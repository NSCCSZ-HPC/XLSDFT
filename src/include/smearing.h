#ifndef _SMEARING_H_
#define _SMEARING_H_

#include <iostream>
#include "control.h"
#include "linalg.h"

class Smearing
{
public:
    const Smearing_control& smearing_control;
    Smearing(const Smearing_control& smearing_control);
    ~Smearing();
    double generate_kbt() const;
    void init(const Smearing& smearing);
    void show() const;
};

namespace Smearing_method {
    template<typename T> void smear(T const* const& in_ptr, T* const& out_ptr, const T& mu_f, const Smearing& smearing, const uint& length);
    template<typename T> void generate_entropy_energy_arr(T const* const& occ, T* const& out_ptr, const Smearing& smearing, const T& smearing_coef, const uint& length);
    template<typename T> void fermi_dirac_smearing(T const* const& in_ptr, T* const& out_ptr, const T& mu_f, const T& beta, const uint& length);
    template<typename T> void gaussian_smearing(T const* const in_ptr, T* const out_ptr, const T mu_f, const T beta, const uint length);
    template<typename T> void fermi_dirac_smearing_with_deriv(T const* const in_ptr, 
                                                      T* const out_occ, T* const out_docc,
                                                      const T mu_f, const T beta, const uint length);
    template<typename T> void gaussian_smearing_with_deriv(T const* const in_ptr, 
                                                            T* const out_occ, T* const out_docc,
                                                            const T mu_f, const T beta, const uint length);
    template<typename T> void smear_with_deriv(T const* const in_ptr, T* const out_occ, T* const out_docc, 
                                                const T mu_f, const Smearing& smearing, const uint length);
}

#endif //_SMEARING_H_