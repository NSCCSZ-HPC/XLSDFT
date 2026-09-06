#ifndef _INDEX_H_
#define _INDEX_H_

#include <iostream>
#include <climits>
#include <cassert>
#include "index.h"
#include "environment.h"

namespace Index {

    uint64_t get_index_Col_Maj(const uint64_t i, const uint64_t j, const uint64_t ni, const uint64_t nj);
    uint64_t get_index_Col_Maj(const uint64_t i, const uint64_t j, const uint64_t ni);
    uint64_t get_i_Col_Maj(const uint64_t index, const uint64_t ni, const uint64_t nj);
    uint64_t get_i_Col_Maj(const uint64_t index, const uint64_t ni);
    uint64_t get_j_Col_Maj(const uint64_t index, const uint64_t ni, const uint64_t nj);
    uint64_t get_j_Col_Maj(const uint64_t index, const uint64_t ni);

    uint64_t get_index_Row_Maj(const uint64_t i, const uint64_t j, const uint64_t ni, const uint64_t nj);
    uint64_t get_index_Row_Maj(const uint64_t i, const uint64_t j, const uint64_t nj);
    uint64_t get_i_Row_Maj(const uint64_t index, const uint64_t ni, const uint64_t nj);
    uint64_t get_i_Row_Maj(const uint64_t index, const uint64_t nj);
    uint64_t get_j_Row_Maj(const uint64_t index, const uint64_t ni, const uint64_t nj);
    uint64_t get_j_Row_Maj(const uint64_t index, const uint64_t nj);

    uint64_t get_index(const uint64_t i, const uint64_t j, const uint64_t ni, const uint64_t nj, const char Maj);
    uint64_t get_index(const uint64_t i, const uint64_t j, const uint64_t ld, const char Maj);
    uint64_t get_i(const uint64_t index, const uint64_t ni, const uint64_t nj, const char Maj);
    uint64_t get_i(const uint64_t index, const uint64_t ld, const char Maj);
    uint64_t get_j(const uint64_t index, const uint64_t ni, const uint64_t nj, const char Maj);
    uint64_t get_j(const uint64_t index, const uint64_t ld, const char Maj);

    #ifdef USE_CBLAS
    uint64_t get_index(const uint64_t i, const uint64_t j, const uint64_t ni, const uint64_t nj, const CBLAS_TRANSPOSE TRANSPOSE, const CBLAS_ORDER LAYOUT);
    uint64_t get_index(const uint64_t i, const uint64_t j, const uint64_t ni, const uint64_t nj, const CBLAS_ORDER LAYOUT, const CBLAS_TRANSPOSE TRANSPOSE);
    uint64_t get_index(const uint64_t i, const uint64_t j, const uint64_t ld, const CBLAS_TRANSPOSE TRANSPOSE, const CBLAS_ORDER LAYOUT);
    uint64_t get_index(const uint64_t i, const uint64_t j, const uint64_t ld, const CBLAS_ORDER LAYOUT, const CBLAS_TRANSPOSE TRANSPOSE);
    #endif // USE_CBLAS
}

#endif //_INDEX_H_