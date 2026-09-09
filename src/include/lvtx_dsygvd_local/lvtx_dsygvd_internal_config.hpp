#ifndef XLSDFT_LVTX_DSYGVD_INTERNAL_CONFIG_HPP_
#define XLSDFT_LVTX_DSYGVD_INTERNAL_CONFIG_HPP_

// XLSDFT-internal production choices.  This header is deliberately outside
// the public backend API so callers cannot retune the qualified solver.
namespace xlsdft_dsygvd_internal {

inline constexpr int production_threads = 36;
inline constexpr int production_tile = 64;

// Route public DSYGVD entry points through detail_opt/ (SME kernels there
// delegate to ref until replaced one phase at a time).
inline constexpr bool use_opt_path = false;

// Local invariant failure, deliberately outside the LVTX status range and
// distinct from Lvtx_backend::fixed_configuration_error (-1000).
inline constexpr int replica_mismatch_status = -1001;

}  // namespace xlsdft_dsygvd_internal

#endif  // XLSDFT_LVTX_DSYGVD_INTERNAL_CONFIG_HPP_
