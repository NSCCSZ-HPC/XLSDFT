#ifndef XLSDFT_HAMILTONIAN_NLOC_DETAIL_H_
#define XLSDFT_HAMILTONIAN_NLOC_DETAIL_H_

#include "xlsdft_backend.h"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace Hamiltonian {
namespace Nloc_detail {

inline constexpr std::size_t chunk_bands = 8U;
inline constexpr std::size_t scratch_alignment_bytes = 64U;

enum class Scratch_layout_status {
    success = 0,
    invalid_configuration,
    overflow,
    insufficient_capacity,
};

struct Scratch_layout {
    std::size_t chunks = 0U;
    int requested_workers = 0;
    std::size_t accumulated_stride = 0U;
    std::size_t coefficient_stride = 0U;
    std::size_t gather_stride = 0U;
    std::size_t back_projection_stride = 0U;
    std::size_t coefficient_offset = 0U;
    std::size_t gather_offset = 0U;
    std::size_t back_projection_offset = 0U;
    std::size_t worker_stride = 0U;
    std::size_t total_elements = 0U;
    std::size_t allocation_requirement = 0U;
};

struct Backend_failure {
    int status = 0;
    Xlsdft_backend::Operation operation =
        Xlsdft_backend::Operation::nloc_forward;
    Xlsdft_backend::Failure_context context{};
};

// This is the single MPI-reporting boundary for failures captured by workers.
// Call it only after the worker team has joined on the MPI main thread.
void require_backend_success_after_parallel(
    const Backend_failure& failure, MPI_Comm comm);

inline bool checked_add(const std::size_t left, const std::size_t right,
                        std::size_t* const result) noexcept
{
    if (result == nullptr ||
        left > std::numeric_limits<std::size_t>::max() - right) {
        return false;
    }
    *result = left + right;
    return true;
}

inline bool checked_multiply(const std::size_t left,
                             const std::size_t right,
                             std::size_t* const result) noexcept
{
    if (result == nullptr ||
        (left != 0U &&
         right > std::numeric_limits<std::size_t>::max() / left)) {
        return false;
    }
    *result = left * right;
    return true;
}

inline bool padded_product(const std::size_t span,
                           const std::size_t count,
                           const std::size_t alignment_elements,
                           std::size_t* const result) noexcept
{
    std::size_t product = 0U;
    if (alignment_elements == 0U ||
        !checked_multiply(span, count, &product)) {
        return false;
    }
    const std::size_t remainder = product % alignment_elements;
    if (remainder == 0U) {
        *result = product;
        return true;
    }
    return checked_add(product, alignment_elements - remainder, result);
}

template<typename T>
Scratch_layout_status make_scratch_layout(
    const std::size_t bands_total,
    const std::size_t coefficient_span,
    const std::size_t max_columns,
    const std::size_t max_rows,
    const int max_threads,
    const std::size_t pool_available,
    Scratch_layout* const layout) noexcept
{
    if (layout == nullptr) {
        return Scratch_layout_status::invalid_configuration;
    }
    *layout = Scratch_layout{};
    if (
        scratch_alignment_bytes % sizeof(T) != 0U ||
        bands_total == 0U || coefficient_span == 0U ||
        max_columns == 0U || max_rows == 0U || max_threads <= 0) {
        return Scratch_layout_status::invalid_configuration;
    }

    Scratch_layout candidate{};
    candidate.chunks = bands_total / chunk_bands +
        static_cast<std::size_t>(bands_total % chunk_bands != 0U);
    candidate.requested_workers = static_cast<int>(
        std::min(candidate.chunks, static_cast<std::size_t>(max_threads)));
    if (candidate.requested_workers <= 0) {
        return Scratch_layout_status::invalid_configuration;
    }

    const std::size_t alignment_elements =
        scratch_alignment_bytes / sizeof(T);
    if (!padded_product(coefficient_span, chunk_bands,
                        alignment_elements,
                        &candidate.accumulated_stride) ||
        !padded_product(max_columns, chunk_bands, alignment_elements,
                        &candidate.coefficient_stride) ||
        !padded_product(max_rows, chunk_bands, alignment_elements,
                        &candidate.gather_stride) ||
        !padded_product(max_rows, chunk_bands, alignment_elements,
                        &candidate.back_projection_stride)) {
        *layout = candidate;
        return Scratch_layout_status::overflow;
    }

    candidate.coefficient_offset = candidate.accumulated_stride;
    if (!checked_add(candidate.coefficient_offset,
                     candidate.coefficient_stride,
                     &candidate.gather_offset) ||
        !checked_add(candidate.gather_offset, candidate.gather_stride,
                     &candidate.back_projection_offset) ||
        !checked_add(candidate.back_projection_offset,
                     candidate.back_projection_stride,
                     &candidate.worker_stride) ||
        !checked_multiply(candidate.worker_stride,
                          static_cast<std::size_t>(
                              candidate.requested_workers),
                          &candidate.total_elements) ||
        !checked_add(candidate.total_elements, alignment_elements - 1U,
                     &candidate.allocation_requirement)) {
        *layout = candidate;
        return Scratch_layout_status::overflow;
    }

    *layout = candidate;
    if (pool_available < candidate.allocation_requirement) {
        return Scratch_layout_status::insufficient_capacity;
    }
    return Scratch_layout_status::success;
}

inline const char* scratch_layout_status_name(
    const Scratch_layout_status status) noexcept
{
    switch (status) {
        case Scratch_layout_status::success:
            return "success";
        case Scratch_layout_status::invalid_configuration:
            return "invalid_configuration";
        case Scratch_layout_status::overflow:
            return "overflow";
        case Scratch_layout_status::insufficient_capacity:
            return "insufficient_capacity";
    }
    return "unknown";
}

}  // namespace Nloc_detail
}  // namespace Hamiltonian

#endif  // XLSDFT_HAMILTONIAN_NLOC_DETAIL_H_
