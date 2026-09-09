#ifndef _XLSDFT_NCHI_DIAG_HPP_
#define _XLSDFT_NCHI_DIAG_HPP_

#include <cstddef>
#include <cstdlib>

#include "effective_potential_nloc.h"

struct Xlsdft_element_workload_diag
{
    size_t nchi = 0;
    size_t n_projectors = 0;
    size_t projector_bytes = 0;
    double chi_build_ms = 0.0;
    double filter_nloc_ms = 0.0;
    double chefsi_ms = 0.0;
    double eigen_ms = 0.0;
    bool valid = false;
};

inline bool xlsdft_nchi_diag_enabled()
{
    static const bool enabled = std::getenv("XLSDFT_NCHI_DIAG") != nullptr;
    return enabled;
}

template<typename T>
inline size_t effective_potential_nloc_nchi(
    const Effective_potential_nloc<T>& vnloc)
{
    if (vnloc.offsets.empty()) {
        return 0;
    }
    return vnloc.offsets.back();
}

template<typename T>
inline size_t effective_potential_nloc_projector_bytes(
    const Effective_potential_nloc<T>& vnloc)
{
    size_t bytes = 0;
    for (const Nloc_projector<T>& projector : vnloc.nloc_projectors) {
        bytes += projector.chi.length * sizeof(T);
        bytes += projector.gamma.length * sizeof(T);
        bytes += projector.nrow * sizeof(uint);
    }
    return bytes;
}

#endif  // _XLSDFT_NCHI_DIAG_HPP_
