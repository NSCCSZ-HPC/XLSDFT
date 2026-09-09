#pragma once

#include <cstddef>

void transpose_16t_to_16(const size_t M, const size_t K,
                         double* __restrict__ out,
                         const double* __restrict__ in);

// In-place col-major -> row-major for square Q (nb must be a multiple of 16).
void transpose_square_colmajor_to_rowmajor(double* q, const size_t n);
