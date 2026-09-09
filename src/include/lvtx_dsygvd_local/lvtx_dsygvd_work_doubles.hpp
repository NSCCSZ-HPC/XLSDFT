#ifndef LVTX_DSYGVD_WORK_DOUBLES_HPP_
#define LVTX_DSYGVD_WORK_DOUBLES_HPP_

#include "detail/lvtx_kml_dsygvd_detail.hpp"

#include <cstddef>

// Returns the number of double elements required by the DSYGVD entry points,
// or zero when n is outside [0,640].
extern "C" inline std::size_t lvtx_dsygvd_work_doubles(const int n) noexcept {
  using namespace lvtx_blas_detail::kml_dsygvd;
    if (n < 0 || n > kMaxOrder) {
        return 0;
    }
    const std::size_t order = static_cast<std::size_t>(n);
    // Must match kml_dsygvd_opt::kHouseholderPanelNb (detail_opt).
    constexpr std::size_t kPanelNb = 16U;
    const std::size_t panel_doubles = 2U * order * kPanelNb;
    return order * order + 6U * order + panel_doubles;
}

#endif  // LVTX_DSYGVD_WORK_DOUBLES_HPP_
