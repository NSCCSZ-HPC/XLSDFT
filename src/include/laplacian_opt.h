#pragma once

#include <cstddef>

template <int OutTile>
void laplacian_4d(const size_t ni, const size_t nj, const size_t nk,
                  const size_t nb, double* __restrict__ h,
                  const double* __restrict__ psi,
                  const double* __restrict__ vloc,
                  const double* __restrict__ cx, const double* __restrict__ cy,
                  const double* __restrict__ cz, const double coef0,
                  double* __restrict__ psi_tile16t_out = nullptr);
