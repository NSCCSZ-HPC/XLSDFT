#include <stdio.h>
#include "index.h"
// uint64_t Index::get_index_Col_Maj(const uint64_t i, const uint64_t j, const uint64_t ni, const uint64_t64_t nj) {
//     return i + j * ni;
// }

uint64_t Index::get_index_Col_Maj(const uint64_t i, const uint64_t j, const uint64_t ni) {
    return i + j * ni;
}

// uint64_t Index::get_i_Col_Maj(const uint64_t index, const uint64_t ni, const uint64_t nj) {
//     return index % ni;
// }

uint64_t Index::get_i_Col_Maj(const uint64_t index, const uint64_t ni) {
    return index % ni;
}

// uint64_t Index::get_j_Col_Maj(const uint64_t index, const uint64_t ni, const uint64_t nj) {
//     return index / ni;
// }

uint64_t Index::get_j_Col_Maj(const uint64_t index, const uint64_t ni) {
    return index / ni;
}

// uint64_t Index::get_index_Row_Maj(const uint64_t i, const uint64_t j, const uint64_t ni, const uint64_t nj) {
//     return i * nj +j;
// }

uint64_t Index::get_index_Row_Maj(const uint64_t i, const uint64_t j, const uint64_t nj) {
    return i * nj +j;
}

// uint64_t Index::get_i_Row_Maj(const uint64_t index, const uint64_t ni, const uint64_t nj) {
//     return index / nj;
// }

uint64_t Index::get_i_Row_Maj(const uint64_t index, const uint64_t nj) {
    return index / nj;
}

// uint64_t Index::get_j_Row_Maj(const uint64_t index, const uint64_t ni, const uint64_t nj) {
//     return index % nj;
// }

uint64_t Index::get_j_Row_Maj(const uint64_t index, const uint64_t nj) {
    return index % nj;
}

uint64_t Index::get_index(const uint64_t i, const uint64_t j, const uint64_t ni, const uint64_t nj, const char Maj) {
    switch(Maj) {
        case 'C' :
            return Index::get_index_Col_Maj(i, j, ni);
            break;
        case 'R' :
            return Index::get_index_Row_Maj(i, j, nj);
            break;
        default :
            assert(Maj == 'C' || Maj == 'R');
    }
    return -1;
}

uint64_t Index::get_index(const uint64_t i, const uint64_t j, const uint64_t ld, const char Maj) {
    switch(Maj) {
        case 'C' :
            return Index::get_index_Col_Maj(i, j, ld);
            break;
        case 'R' :
            return Index::get_index_Row_Maj(i, j, ld);
            break;
        default :
            assert(Maj == 'C' || Maj == 'R');
    }
    return -1;
}

uint64_t Index::get_i(const uint64_t index, const uint64_t ni, const uint64_t nj, const char Maj) {
    switch(Maj) {
        case 'C' :
            return Index::get_i_Col_Maj(index, ni);
            break;
        case 'R' :
            return Index::get_i_Row_Maj(index, nj);
            break;
        default :
            assert(Maj == 'C' || Maj == 'R');
    }
    return -1;
}

uint64_t Index::get_i(const uint64_t index, const uint64_t ld, const char Maj) {
    switch(Maj) {
        case 'C' :
            return Index::get_i_Col_Maj(index, ld);
            break;
        case 'R' :
            return Index::get_i_Row_Maj(index, ld);
            break;
        default :
            assert(Maj == 'C' || Maj == 'R');
    }
    return -1;
}

uint64_t Index::get_j(const uint64_t index, const uint64_t ni, const uint64_t nj, const char Maj) {
    switch(Maj) {
        case 'C' :
            return Index::get_j_Col_Maj(index, ni);
            break;
        case 'R' :
            return Index::get_j_Row_Maj(index, nj);
            break;
        default :
            assert(Maj == 'C' || Maj == 'R');
    }
    return -1;
}

uint64_t Index::get_j(const uint64_t index, const uint64_t ld, const char Maj) {
    switch(Maj) {
        case 'C' :
            return Index::get_j_Col_Maj(index, ld);
            break;
        case 'R' :
            return Index::get_j_Row_Maj(index, ld);
            break;
        default :
            assert(Maj == 'C' || Maj == 'R');
    }
    return -1;
}

#ifdef USE_CBLAS
uint64_t Index::get_index(const uint64_t i, const uint64_t j, const uint64_t ni, const uint64_t nj, const CBLAS_TRANSPOSE TRANSPOSE, const CBLAS_ORDER LAYOUT) {
    return ((LAYOUT==CblasColMajor && TRANSPOSE==CblasNoTrans) || (LAYOUT==CblasRowMajor && TRANSPOSE==CblasTrans)) ?
            Index::get_index_Col_Maj(i, j, nj) : Index::get_index_Row_Maj(i, j, ni);
}
uint64_t Index::get_index(const uint64_t i, const uint64_t j, const uint64_t ni, const uint64_t nj, const CBLAS_ORDER LAYOUT, const CBLAS_TRANSPOSE TRANSPOSE) {
    return ((LAYOUT==CblasColMajor && TRANSPOSE==CblasNoTrans) || (LAYOUT==CblasRowMajor && TRANSPOSE==CblasTrans)) ?
            Index::get_index_Col_Maj(i, j, nj) : Index::get_index_Row_Maj(i, j, ni);
}
uint64_t Index::get_index(const uint64_t i, const uint64_t j, const uint64_t ld, const CBLAS_TRANSPOSE TRANSPOSE, const CBLAS_ORDER LAYOUT) {
    return ((LAYOUT==CblasColMajor && TRANSPOSE==CblasNoTrans) || (LAYOUT==CblasRowMajor && TRANSPOSE==CblasTrans)) ?
            Index::get_index_Col_Maj(i, j, ld) : Index::get_index_Row_Maj(i, j, ld);
}
uint64_t Index::get_index(const uint64_t i, const uint64_t j, const uint64_t ld, const CBLAS_ORDER LAYOUT, const CBLAS_TRANSPOSE TRANSPOSE) {
    return ((LAYOUT==CblasColMajor && TRANSPOSE==CblasNoTrans) || (LAYOUT==CblasRowMajor && TRANSPOSE==CblasTrans)) ?
            Index::get_index_Col_Maj(i, j, ld) : Index::get_index_Row_Maj(i, j, ld);
}
#endif // USE_CBLAS