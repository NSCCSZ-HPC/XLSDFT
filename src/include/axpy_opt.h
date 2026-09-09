#pragma once

#include <cstddef>

void cheb_rec_scale(double* __restrict__ packed, const double a, const size_t M,
                    const size_t K);

template <int OutTile>
void cheb_rec_axpy(double* __restrict__ y_packed, const double a,
                   const double* __restrict__ z_packed, const double b,
                   const size_t M, const size_t K);
