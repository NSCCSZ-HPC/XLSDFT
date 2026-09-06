#include <cstring>
#include <stdio.h>
#include <cassert>
#include "vertices.h"

Vertices_1D::Vertices_1D() {
    this->is = 0;
    this->ni = 0;
}

Vertices_1D::Vertices_1D(const uint64_t ni) {
    this->set_i_vertices(0, (int64_t)ni - 1);
}

Vertices_1D::Vertices_1D(const int64_t is, const int64_t ie) {
    this->set_i_vertices(is, ie);
}

Vertices_1D::Vertices_1D(const Vertices_1D& other) {
    this->is = other.is;
    this->ni = other.ni;
}

Vertices_1D::~Vertices_1D() {}

void Vertices_1D::swap(Vertices_1D& other) {
    std::swap(this->is, other.is);
    std::swap(this->ni, other.ni);
    return;
}

void Vertices_1D::set_vertices(const int64_t is, const int64_t ie){
    this->set_i_vertices(is, ie);
    return;
}

void Vertices_1D::set_vertices(const Vertices_1D& other) {
    this->set_vertices(other.is, other.get_ie());
    return;
}

void Vertices_1D::set_i_vertices(const int64_t is, const int64_t ie) {
    assert(ie - is + 1 >= 0);
    this->is = is;
    this->ni = ie - is + 1;
    return;
}

void Vertices_1D::set_ni(const uint64_t ni) {
    this->set_i_vertices(0, ni == 0 ? -1 : (int64_t)ni - 1);
}

Vertices_1D& Vertices_1D::operator=(const Vertices_1D& other) {
    if (this != &other) {
        this->is = other.is;
        this->ni = other.ni;
    }
    return *this;
}

bool Vertices_1D::operator==(const Vertices_1D& other) const {
    return (this->is == other.is) && (this->ni == other.ni);
}

bool Vertices_1D::operator!=(const Vertices_1D& other) const {
    return !(*this == other);
}

int64_t Vertices_1D::get_is() const {
    return this->is;
}

int64_t Vertices_1D::get_ie() const {
    return this->is + (int64_t) this->ni - 1;
}

uint64_t Vertices_1D::get_ni() const {
    return this->ni;
}

Vertices_1D Vertices_1D::get_vertices() const {
    return *this;
}

Vertices_1D& Vertices_1D::get_vertices() {
    return *this;
}

bool Vertices_1D::contain_point(const int64_t i) const {
    return this->contain_i(i);
}

bool Vertices_1D::contain_i(const int64_t i) const {
    return (i >= this->is) && (i <= this->get_ie());
}

bool Vertices_1D::is_index_legal(const uint64_t index) const {
    return index < this->ni;
}

uint64_t Vertices_1D::get_size() const {
    return this->ni;
}

/**
 * @brief get the position in x direction start from "is", namely "i", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int  the position in x direction start from "is", namely "i".
 */
int64_t Vertices_1D::get_i(const uint64_t index) const {
    assert(this->is_index_legal(index));
    return this->get_i_nocheck(index);
}

/**
 * @brief get the position in x direction start from "is", namely "i", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int  the position in x direction start from "is", namely "i".
 */
int64_t Vertices_1D::get_i_nocheck(const uint64_t index) const {
    return index + this->is;
}

/**
 * @brief get the index, according to the position in x direction, namely "i".
 * 
 * @param i the position in x direction start from "is".
 * @return int the index_th position in the column major vertices, start from 0.
 */
uint64_t Vertices_1D::get_index(const int64_t i) const {
    assert(this->contain_point(i));
    return this->get_index_nocheck(i);
}

uint64_t Vertices_1D::get_index_DBC(const int64_t i) const {
    return this->contain_point(i) ? (int64_t) this->get_index_nocheck(i) + 1 : 0;
}

uint64_t Vertices_1D::get_index_PBC(const int64_t i) const {
    return this->get_index_nocheck(
           this->contain_i(i) ? i : this->map_i_into_vertices(i)
           );
}

/**
 * @brief get the index, according to the position in x direction, namely "i".
 * 
 * @param i the position in x direction start from "is".
 * @return int the index_th position in the column major vertices, start from 0.
 */
uint64_t Vertices_1D::get_index_nocheck(const int64_t i) const {
    return uint64_t(i - this->is);
}

int64_t Vertices_1D::map_i_into_vertices(const int64_t i) const {
    assert(this->ni != 0);
    const int64_t offset = (i - this->is) % (int64_t)this->ni;
    return this->is + (offset >= 0 ? offset : offset + (int64_t)this->ni);
}

bool Vertices_1D::is_ex_vertices(const Vertices_1D& other) const {
    return (other.get_size() == 0) || (this->is <= other.is && this->get_ie() >= other.get_ie());
}

bool Vertices_1D::is_sub_vertices(const Vertices_1D& other) const {
    return (this->get_size() == 0) || (this->is >= other.is && this->get_ie() <= other.get_ie());
}

bool Vertices_1D::is_overlaped(const Vertices_1D& other) const {
    return !((this->get_size() == 0) || (other.get_size() == 0) 
             ||(other.is>this->get_ie() || other.get_ie()< this->is));
}

std::vector<Vertices_1D> Vertices_1D::split(const uint64_t npart_i) const {
    assert(npart_i > 0 && npart_i <= this->ni);
    uint64_t chunksize_i = Linalg::get_chunksize(npart_i, this->ni);
    std::vector<Vertices_1D> result;
    result.reserve(npart_i);
    int64_t is = this->is;
    for (uint64_t i = 0; i < npart_i; i++) {
        int64_t ie = is + (int64_t)chunksize_i - 1 < this->get_ie() ? is + (int64_t)chunksize_i - 1 : this->get_ie();
        result.emplace_back(is, ie);
        is += chunksize_i;
    }
    return result;
}

Vertices_1D Vertices_1D::get_overlap_vertices(const Vertices_1D& other) const {
    if (!this->is_overlaped(other)) {
        return Vertices_1D(0);
    } else {
        return Vertices_1D(this->is >= other.is ? this->is : other.is,
                           this->get_ie() <= other.get_ie() ? this->get_ie() : other.get_ie());
    }
}

Vertices_1D Vertices_1D::get_overlap_vertices(const Vertices_1D& other, const int64_t i_shift) const {
    Vertices_1D temp(other.is + i_shift, other.get_ie() + i_shift);
    return this->get_overlap_vertices(temp);
}

Vertices_1D Vertices_1D::get_super_vertices(const Vertices_1D& other) const {
    if (this->get_size() == 0) {
        return Vertices_1D(other);
    } else if (other.get_size() == 0) {
        return Vertices_1D(*this);
    } else {
        return Vertices_1D(this->is <= other.is ? this->is : other.is,
                           this->get_ie() >= other.get_ie() ? this->get_ie() : other.get_ie());
    }
}

Vertices_1D Vertices_1D::get_super_vertices(const Vertices_1D& other, const int64_t i_shift) const {
    Vertices_1D temp(other.is + i_shift, other.get_ie() + i_shift);
    return this->get_super_vertices(temp);
}

template<typename T>
Vertices_1D Vertices_1D::generate_ex_vertices(const T nnode) const {
    return this->get_size() == 0 ? Vertices_1D(0) : Vertices_1D(this->is - (int64_t)nnode, this->get_ie() + (int64_t)nnode);
}
template Vertices_1D Vertices_1D::generate_ex_vertices<int>(const int nnode) const;
template Vertices_1D Vertices_1D::generate_ex_vertices<uint>(const uint nnode) const;
template Vertices_1D Vertices_1D::generate_ex_vertices<int64_t>(const int64_t nnode) const;
template Vertices_1D Vertices_1D::generate_ex_vertices<uint64_t>(const uint64_t nnode) const;

template<typename T>
Vertices_1D Vertices_1D::generate_ex_vertices(const T* nnode) const {
    return this->get_size() == 0 ? Vertices_1D(0) : Vertices_1D(this->is - (int64_t)nnode[0], this->get_ie() + (int64_t)nnode[0]);
}
template Vertices_1D Vertices_1D::generate_ex_vertices<int>(const int* nnode) const;
template Vertices_1D Vertices_1D::generate_ex_vertices<uint>(const uint* nnode) const;
template Vertices_1D Vertices_1D::generate_ex_vertices<int64_t>(const int64_t* nnode) const;
template Vertices_1D Vertices_1D::generate_ex_vertices<uint64_t>(const uint64_t* nnode) const;

void Vertices_1D::show() const {
    std::cout << "is = " << this->is << ", ie = " << this->get_ie() << ", ni = " << this->ni <<std::endl;
}

Vertices_2D::Vertices_2D() : Vertices_1D::Vertices_1D() {
    this->js = 0;
    this->nj = 0;
}

Vertices_2D::Vertices_2D(const Vertices_1D& vertices_1d, const uint64_t nj) : Vertices_1D::Vertices_1D(vertices_1d) {
    if (vertices_1d.get_size() > 0) {
        this->set_j_vertices(0, nj == 0 ? -1 : (int64_t)nj - 1);
    } else {
        this->set_j_vertices(0, -1);
    }
}

Vertices_2D::Vertices_2D(const Vertices_1D& vertices_1d, const int64_t js, const int64_t je) : Vertices_1D::Vertices_1D(vertices_1d) {
    if (vertices_1d.get_size() > 0) {
        this->set_j_vertices(js, je);
    } else {
        this->set_j_vertices(0, -1);
    }
}

Vertices_2D::Vertices_2D(const uint64_t ni, const uint64_t nj) : Vertices_1D::Vertices_1D(ni) {
    this->set_j_vertices(0, nj == 0 ? -1 : (int64_t)nj - 1);
}

Vertices_2D::Vertices_2D(const int64_t is, const int64_t ie,
                         const int64_t js, const int64_t je)
                    : Vertices_1D::Vertices_1D(is, ie) {
    this->set_j_vertices(js, je);
}

Vertices_2D::Vertices_2D(const Vertices_2D& other) : Vertices_1D::Vertices_1D(other) {
    this->js = other.js;
    this->nj = other.nj;
}

Vertices_2D::~Vertices_2D() {}

void Vertices_2D::swap(Vertices_2D& other) {
    this->Vertices_1D::swap(other);
    std::swap(this->js, other.js);
    std::swap(this->nj, other.nj);
    return;
}

void Vertices_2D::set_vertices(const int64_t is, const int64_t ie,
                               const int64_t js, const int64_t je) {
    this->Vertices_1D::set_vertices(is, ie);
    this->set_j_vertices(js, je);
    return;
}

void Vertices_2D::set_vertices(const Vertices_2D& other) {
    this->set_vertices(other.is, other.get_ie(),
                       other.js, other.get_je());
    return;
}

// // void Vertices_2D::set_i_vertices(const int64_t is, const int64_t ie) {
// //     this->Vertices_1D::set_i_vertices(is, ie);
// //     return;
// // }

void Vertices_2D::set_j_vertices(const int64_t js, const int64_t je){
    assert(je - js + 1 >= 0);
    this->js = js;
    this->nj = je - js + 1;
    return;
}

void Vertices_2D::set_nj(const uint64_t nj) {
    this->set_j_vertices(0, nj == 0 ? -1 : (int64_t)nj - 1);
}

Vertices_2D& Vertices_2D::operator=(const Vertices_2D& other) {
    if (this != &other) {
        this->Vertices_1D::operator=(other);
        this->js = other.js;
        this->nj = other.nj;
    }
    return *this;
}

bool Vertices_2D::operator==(const Vertices_2D& other) const {
    return Vertices_1D::operator==(other) && (this->js == other.js)
            && (this->nj == other.nj);
}

bool Vertices_2D::operator!=(const Vertices_2D& other) const {
    return !(*this == other);
}

int64_t Vertices_2D::get_js() const {
    return this->js;
}

int64_t Vertices_2D::get_je() const {
    return this->js + (int64_t) this->nj - 1;
}

uint64_t Vertices_2D::get_nj() const {
    return this->nj;
}

Vertices_2D Vertices_2D::get_vertices() const {
    return *this;
}

Vertices_2D& Vertices_2D::get_vertices() {
    return *this;
}

bool Vertices_2D::contain_point(const int64_t i, const int64_t j) const {
    // return this->Vertices_1D::contain_point(i) && this->contain_j(j);
    return this->contain_i(i) && this->contain_j(j);
}

bool Vertices_2D::contain_i(const int64_t i) const {
    return (i >= this->is) && (i <= this->get_ie());
}

bool Vertices_2D::contain_j(const int64_t j) const {
    return (j >= this->js) && (j <= this->get_je());
}

bool Vertices_2D::is_index_legal(const uint64_t index) const {
    return index < this->get_size();
}

uint64_t Vertices_2D::get_size() const {
    return this->ni * this->nj;
}

/**
 * @brief get the position in x direction start from "is", namely "i", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int the position in x direction start from "is", namely "i".
 */
int64_t Vertices_2D::get_i(const uint64_t index) const {
    assert(this->is_index_legal(index));
    return this->get_i_nocheck(index);
}

/**
 * @brief get the position in x direction start from "is", namely "i", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int the position in x direction start from "is", namely "i".
 */
int64_t Vertices_2D::get_i_nocheck(const uint64_t index) const {
    return this->Vertices_1D::get_i_nocheck(index % this->ni);
}

/**
 * @brief get the position in y direction start from "js", namely "j", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int the position in y direction start from "js", namely "j".
 */
int64_t Vertices_2D::get_j(const uint64_t index) const {
    assert(this->is_index_legal(index));
    return this->get_j_nocheck(index);
}

/**
 * @brief get the position in y direction start from "js", namely "j", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int the position in y direction start from "js", namely "j".
 */
int64_t Vertices_2D::get_j_nocheck(const uint64_t index) const {
    return index / this->ni + this->js;
}

/**
 * @brief get the index, according to the position in x and y direction, namely "i" and "j".
 * 
 * @param i the position in x direction start from "is".
 * @param j the position in y direction start from "js".
 * @return int the index_th position in the column major vertices, start from 0.
 */
uint64_t Vertices_2D::get_index(const int64_t i, const int64_t j) const {
    assert(this->contain_point(i, j));
    return this->get_index_nocheck(i, j);
}

uint64_t Vertices_2D::get_index_DBC(const int64_t i, const int64_t j) const {
    return this->contain_point(i, j) ? (int64_t)this->get_index_nocheck(i, j) + 1 : 0;
}

uint64_t Vertices_2D::get_index_PBC(const int64_t i, const int64_t j) const {
    return this->get_index_nocheck(
           this->contain_i(i) ? i : this->map_i_into_vertices(i),
           this->contain_j(j) ? j : this->map_j_into_vertices(j)
           );
}

/**
 * @brief get the index, according to the position in x and y direction, namely "i" and "j".
 * 
 * @param i the position in x direction start from "is".
 * @param j the position in y direction start from "js".
 * @return int the index_th position in the column major vertices, start from 0.
 */
uint64_t Vertices_2D::get_index_nocheck(const int64_t i, const int64_t j) const {
    // return this->Vertices_1D::get_index_nocheck(i) + (j - this->js) * this->nx;
    return uint64_t(i - this->is) + uint64_t(j - this->js) * (uint64_t)this->ni;
}

int64_t Vertices_2D::map_i_into_vertices(const int64_t i) const {
    // return this->Vertices_1D::map_i_into_vertices(i);
    assert(this->ni != 0);
    const int64_t offset = (i - this->is) % (int64_t)this->ni;
    return this->is + (offset >= 0 ? offset : offset + (int64_t)this->ni);
}

int64_t Vertices_2D::map_j_into_vertices(const int64_t j) const {
    assert(this->nj != 0);
    const int64_t offset = (j - this->js) % (int64_t)this->nj;
    return this->js + (offset >= 0 ? offset : offset + (int64_t)this->nj);
}

bool Vertices_2D::is_ex_vertices(const Vertices_2D& other) const {
    // return this->Vertices_1D::is_exarr(other) && this->js <= other.js && this->get_je() >= other.get_je();
    return (other.get_size() == 0) || (this->is <= other.is && this->get_ie() >= other.get_ie()
           && this->js <= other.js && this->get_je() >= other.get_je());
}

bool Vertices_2D::is_sub_vertices(const Vertices_2D& other) const {
    // return this->Vertices_1D::is_exarr(other) && this->js <= other.js && this->get_je() >= other.get_je();
    return (this->get_size() == 0) || (this->is >= other.is && this->get_ie() <= other.get_ie()
           && this->js >= other.js && this->get_je() <= other.get_je());
}

bool Vertices_2D::is_overlaped(const Vertices_2D& other) const {
    return !((this->get_size() == 0) || (other.get_size() == 0) 
             ||(other.is>this->get_ie() || other.get_ie()< this->is)
             ||(other.js>this->get_je() || other.get_je()< this->js));
}

std::vector<Vertices_2D> Vertices_2D::split(const uint64_t npart_i, const uint64_t npart_j) const {
    assert(npart_i > 0 && npart_i <= this->ni);
    assert(npart_j > 0 && npart_j <= this->nj);
    uint64_t chunksize_i = Linalg::get_chunksize(npart_i, this->ni);
    uint64_t chunksize_j = Linalg::get_chunksize(npart_j, this->nj);
    std::vector<Vertices_2D> result;
    result.reserve(npart_i * npart_j);
    int64_t js = this->js;
    for (uint64_t j = 0; j < npart_j; j++) {
        int64_t je = js + (int64_t)chunksize_j - 1 < this->get_je() ? js + (int64_t)chunksize_j - 1 : this->get_je();
        int64_t is = this->is;
        for (uint64_t i = 0; i < npart_i; i++) {
            int64_t ie = is + (int64_t)chunksize_i - 1 < this->get_ie() ? is + (int64_t)chunksize_i - 1 : this->get_ie();
            result.emplace_back(is, ie, js, je);
            is += chunksize_i;
        }
        js += chunksize_j;
    }
    return result;
}

Vertices_2D Vertices_2D::get_overlap_vertices(const Vertices_2D& other) const {
    if (!this->is_overlaped(other)) {
        return Vertices_2D(0, 0);
    } else {
        return Vertices_2D(this->is >= other.is ? this->is : other.is,
                           this->get_ie() <= other.get_ie() ? this->get_ie() : other.get_ie(),
                           this->js >= other.js ? this->js : other.js,
                           this->get_je() <= other.get_je() ? this->get_je() : other.get_je());
    }
}

Vertices_2D Vertices_2D::get_overlap_vertices(const Vertices_2D& other, const int64_t i_shift, const int64_t j_shift) const {
    Vertices_2D temp(other.is + i_shift, other.get_ie() + i_shift,
                     other.js + j_shift, other.get_je() + j_shift);
    return this->get_overlap_vertices(temp);
}

Vertices_2D Vertices_2D::get_super_vertices(const Vertices_2D& other) const {
    if (this->get_size() == 0) {
        return Vertices_2D(other);
    } else if (other.get_size() == 0) {
        return Vertices_2D(*this);
    } else {
        return Vertices_2D(this->is <= other.is ? this->is : other.is,
                           this->get_ie() >= other.get_ie() ? this->get_ie() : other.get_ie(),
                           this->js <= other.js ? this->js : other.js,
                           this->get_je() >= other.get_je() ? this->get_je() : other.get_je());
    }
}

Vertices_2D Vertices_2D::get_super_vertices(const Vertices_2D& other, const int64_t i_shift, const int64_t j_shift) const {
    Vertices_2D temp(other.is + i_shift, other.get_ie() + i_shift,
                     other.js + j_shift, other.get_je() + j_shift);
    return this->get_super_vertices(temp);
}

template<typename T>
Vertices_2D Vertices_2D::generate_ex_vertices(const T nnode) const {
    return this->get_size() == 0 ? Vertices_2D(0, 0) : Vertices_2D(this->is - (int64_t)nnode, this->get_ie() + (int64_t)nnode,
                                                                   this->js - (int64_t)nnode, this->get_je() + (int64_t)nnode);
}
template Vertices_2D Vertices_2D::generate_ex_vertices<int>(const int nnode) const;
template Vertices_2D Vertices_2D::generate_ex_vertices<uint>(const uint nnode) const;
template Vertices_2D Vertices_2D::generate_ex_vertices<int64_t>(const int64_t nnode) const;
template Vertices_2D Vertices_2D::generate_ex_vertices<uint64_t>(const uint64_t nnode) const;

template<typename T>
Vertices_2D Vertices_2D::generate_ex_vertices(const T* nnode) const {
    return this->get_size() == 0 ? Vertices_2D(0, 0) : Vertices_2D(this->is - (int64_t)nnode[0], this->get_ie() + (int64_t)nnode[0],
                                                                   this->js - (int64_t)nnode[1], this->get_je() + (int64_t)nnode[1]);
}
template Vertices_2D Vertices_2D::generate_ex_vertices<int>(const int* nnode) const;
template Vertices_2D Vertices_2D::generate_ex_vertices<uint>(const uint* nnode) const;
template Vertices_2D Vertices_2D::generate_ex_vertices<int64_t>(const int64_t* nnode) const;
template Vertices_2D Vertices_2D::generate_ex_vertices<uint64_t>(const uint64_t* nnode) const;

void Vertices_2D::show() const {
    this->Vertices_1D::show();
    std::cout << "js = " << this->js << ", je = " << this->get_je() << ", nj = " << this->nj <<std::endl;
}

Vertices_3D::Vertices_3D() : Vertices_2D::Vertices_2D() {
    this->ks = 0;
    this->nk = 0;
}

Vertices_3D::Vertices_3D(const Vertices_2D& vertices_2d, const uint64_t nk) : Vertices_2D::Vertices_2D(vertices_2d) {
    if (vertices_2d.get_size() > 0) {
        this->set_k_vertices(0, nk == 0 ? -1 : (int64_t)nk - 1);
    } else {
        this->set_k_vertices(0, -1);
    }
}

Vertices_3D::Vertices_3D(const Vertices_2D& vertices_2d, const int64_t ks, const int64_t ke) : Vertices_2D::Vertices_2D(vertices_2d) {
    if (vertices_2d.get_size() > 0) {
        this->set_k_vertices(ks, ke);
    } else {
        this->set_k_vertices(0, -1);
    }
}

Vertices_3D::Vertices_3D(const uint64_t ni, const uint64_t nj, const uint64_t nk) : Vertices_2D::Vertices_2D(ni, nj) {
    this->set_k_vertices(0, nk == 0 ? -1 : (int64_t)nk - 1);
}

Vertices_3D::Vertices_3D(const int64_t is, const int64_t ie,
                         const int64_t js, const int64_t je,
                         const int64_t ks, const int64_t ke)
            : Vertices_2D::Vertices_2D(is, ie, js, je) {
    this->set_k_vertices(ks, ke);
}

Vertices_3D::Vertices_3D(const Vertices_3D& other) : Vertices_2D::Vertices_2D(other) {
    this->ks = other.ks;
    this->nk = other.nk;
}

Vertices_3D::~Vertices_3D() {}

void Vertices_3D::swap(Vertices_3D& other) {
    this->Vertices_2D::swap(other);
    std::swap(this->ks, other.ks);
    std::swap(this->nk, other.nk);
    return;
}

void Vertices_3D::set_vertices(const int64_t is, const int64_t ie,
                               const int64_t js, const int64_t je,
                               const int64_t ks, const int64_t ke) {
    this->Vertices_2D::set_vertices(is, ie, js, je);
    this->set_k_vertices(ks, ke);
    return;
}

void Vertices_3D::set_vertices(const Vertices_3D& other) {
    this->set_vertices(other.is, other.get_ie(),
                       other.js, other.get_je(),
                       other.ks, other.get_ke());
    return;
}

// void Vertices_3D::set_i_vertices(const int64_t is, const int64_t ie) {
//     this->Vertices_2D::set_i_vertices(is ,ie);
//     return;
// }

// void Vertices_3D::set_j_vertices(const int64_t js, const int64_t je) {
//     this->Vertices_2D::set_j_vertices(js, je);
//     return;
// }

void Vertices_3D::set_k_vertices(const int64_t ks, const int64_t ke) {
    assert(ke - ks + 1 >= 0);
    this->ks = ks;
    this->nk = ke - ks + 1;
    return;
}

void Vertices_3D::set_nk(const uint64_t nk) {
    this->set_k_vertices(0, nk == 0 ? -1 : (int64_t)nk - 1);
}

Vertices_3D& Vertices_3D::operator=(const Vertices_3D& other) {
    if (this != &other) {
        this->Vertices_2D::operator=(other);
        this->ks = other.ks;
        this->nk = other.nk;
    }
    return *this;
}

bool Vertices_3D::operator==(const Vertices_3D& other) const {
    return Vertices_2D::operator==(other) && (this->ks == other.ks)
            && (this->nk == other.nk);
}

bool Vertices_3D::operator!=(const Vertices_3D& other) const {
    return !(*this == other);
}

int64_t Vertices_3D::get_ks() const {
    return this->ks;
}

int64_t Vertices_3D::get_ke() const {
    return this->ks + (int64_t) this->nk - 1;
}

uint64_t Vertices_3D::get_nk() const {
    return this->nk;
}

Vertices_3D Vertices_3D::get_vertices() const {
    return *this;
}

Vertices_3D& Vertices_3D::get_vertices() {
    return *this;
}

bool Vertices_3D::contain_point(const int64_t i, const int64_t j, const int64_t k) const {
    // return this-> Vertices_2D::contain_point(i, j) && this->contain_k(k);
    return this->contain_i(i) && this->contain_j(j) && this->contain_k(k);
}

bool Vertices_3D::contain_i(const int64_t i) const {
    return (i >= this->is) && (i <= this->get_ie());
}

bool Vertices_3D::contain_j(const int64_t j) const {
    return (j >= this->js) && (j <= this->get_je());
}

bool Vertices_3D::contain_k(const int64_t k) const {
    return (k >= this->ks) && (k <= this->get_ke());
}

bool Vertices_3D::is_index_legal(const uint64_t index) const {
    return index < this->get_size();
}

uint64_t Vertices_3D::get_size() const {
    return this->ni * this->nj * this->nk;
}

/**
 * @brief get the position in x direction start from "is", namely "i", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int the position in x direction start from "is", namely "i".
 */
int64_t Vertices_3D::get_i(const uint64_t index) const {
    assert(this->is_index_legal(index));
    return this->get_i_nocheck(index);
}

/**
 * @brief get the position in x direction start from "is", namely "i", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int the position in x direction start from "is", namely "i".
 */
int64_t Vertices_3D::get_i_nocheck(const uint64_t index) const {
    return this->Vertices_1D::get_i_nocheck(index % this->ni);
}

/**
 * @brief get the position in y direction start from "js", namely "j", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int the position in y direction start from "js", namely "j".
 */
int64_t Vertices_3D::get_j(const uint64_t index) const {
    assert(this->is_index_legal(index));
    return this->get_j_nocheck(index);
}

/**
 * @brief get the position in y direction start from "js", namely "j", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int the position in y direction start from "js", namely "j".
 */
int64_t Vertices_3D::get_j_nocheck(const uint64_t index) const {
    return this->Vertices_2D::get_j_nocheck(index % (this->ni * this->nj));
}

/**
 * @brief get the position in z direction start from "ks", namely "k", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int the position in z direction start from "ks", namely "k".
 */
int64_t Vertices_3D::get_k(const uint64_t index) const {
    assert(this->is_index_legal(index));
    return this->get_k_nocheck(index);
}

/**
 * @brief get the position in z direction start from "ks", namely "k", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int the position in z direction start from "ks", namely "k".
 */
int64_t Vertices_3D::get_k_nocheck(const uint64_t index) const {
    return index / (this->ni * this->nj) + this->ks;
}

/**
 * @brief get the index, according to the position in x, y and z direction, namely "i", "j" and "k".
 * 
 * @param i the position in x direction start from "is".
 * @param j the position in y direction start from "js".
 * @param k the position in z direction start from "ks".
 * @return int the index_th position in the column major vertices, start from 0.
 */
uint64_t Vertices_3D::get_index(const int64_t i, const int64_t j, const int64_t k) const {
    assert(this->contain_point(i, j, k));
    return this->get_index_nocheck(i, j, k);
}

uint64_t Vertices_3D::get_index_DBC(const int64_t i, const int64_t j, const int64_t k) const {
    return this->contain_point(i, j, k) ? this->get_index_nocheck(i, j, k) + 1 : 0;
}

uint64_t Vertices_3D::get_index_PBC(const int64_t i, const int64_t j, const int64_t k) const {
    return this->get_index_nocheck(
           this->contain_i(i) ? i : this->map_i_into_vertices(i),
           this->contain_j(j) ? j : this->map_j_into_vertices(j),
           this->contain_k(k) ? k : this->map_k_into_vertices(k)
           );
}

/**
 * @brief get the index, according to the position in x, y and z direction, namely "i", "j" and "k".
 * 
 * @param i the position in x direction start from "is".
 * @param j the position in y direction start from "js".
 * @param k the position in z direction start from "ks".
 * @return int the index_th position in the column major vertices, start from 0.
 */
uint64_t Vertices_3D::get_index_nocheck(const int64_t i, const int64_t j, const int64_t k) const {
    // return this->Vertices_2D::get_index_nocheck(i, j) + (k - this->ks) * (this->nx * this->ny);
    return uint64_t(i - this->is)
            + (uint64_t(j - this->js)
                + (uint64_t(k - this->ks)
                    * (uint64_t)this->nj))
                        * (uint64_t)this->ni;
}

int64_t Vertices_3D::map_i_into_vertices(const int64_t i) const {
    // return this->Vertices_2D::map_i_into_vertices(i);
    assert(this->ni != 0);
    const int64_t offset = (i - this->is) % (int64_t)this->ni;
    return this->is + (offset >= 0 ? offset : offset + (int64_t)this->ni);
}

int64_t Vertices_3D::map_j_into_vertices(const int64_t j) const {
    // return this->Vertices_2D::map_j_into_vertices(j);
    assert(this->nj != 0);
    const int64_t offset = (j - this->js) % (int64_t)this->nj;
    return this->js + (offset >= 0 ? offset : offset + (int64_t)this->nj);
}

int64_t Vertices_3D::map_k_into_vertices(const int64_t k) const {
    assert(this->nk != 0);
    const int64_t offset = (k - this->ks) % (int64_t)this->nk;
    return this->ks + (offset >= 0 ? offset : offset + (int64_t)this->nk);
}

bool Vertices_3D::is_ex_vertices(const Vertices_3D& other) const {
    // return this->Vertices_2D::is_exarr(other) && this->ks <= other.ks && this->get_ke() >= other.get_ke();
    return (other.get_size() == 0) || (this->is <= other.is && this->get_ie() >= other.get_ie()
           && this->js <= other.js && this->get_je() >= other.get_je()
           && this->ks <= other.ks && this->get_ke() >= other.get_ke());
}

bool Vertices_3D::is_sub_vertices(const Vertices_3D& other) const {
    // return this->Vertices_2D::is_exarr(other) && this->ks <= other.ks && this->get_ke() >= other.get_ke();
    return (this->get_size() == 0) || (this->is >= other.is && this->get_ie() <= other.get_ie()
           && this->js >= other.js && this->get_je() <= other.get_je()
           && this->ks >= other.ks && this->get_ke() <= other.get_ke());
}

bool Vertices_3D::is_overlaped(const Vertices_3D& other) const {
    return !((this->get_size() == 0) || (other.get_size() == 0) 
             ||(other.is>this->get_ie() || other.get_ie()< this->is)
             ||(other.js>this->get_je() || other.get_je()< this->js)
             ||(other.ks>this->get_ke() || other.get_ke()< this->ks));
}

std::vector<Vertices_3D> Vertices_3D::split(const uint64_t npart_i, const uint64_t npart_j, const uint64_t npart_k) const {
    assert(npart_i > 0 && npart_i <= this->ni);
    assert(npart_j > 0 && npart_j <= this->nj);
    assert(npart_k > 0 && npart_k <= this->nk);
    uint64_t chunksize_i = Linalg::get_chunksize(npart_i, this->ni);
    uint64_t chunksize_j = Linalg::get_chunksize(npart_j, this->nj);
    uint64_t chunksize_k = Linalg::get_chunksize(npart_k, this->nk);
    std::vector<Vertices_3D> result;
    result.reserve(npart_i * npart_j * npart_k);
    int64_t ks = this->ks;
    for (uint64_t k = 0; k < npart_k; k++) {
        int64_t ke = ks + (int64_t)chunksize_k - 1 < this->get_ke() ? ks + (int64_t)chunksize_k - 1 : this->get_ke();
        int64_t js = this->js;
        for (uint64_t j = 0; j < npart_j; j++) {
            int64_t je = js + (int64_t)chunksize_j - 1 < this->get_je() ? js + (int64_t)chunksize_j - 1 : this->get_je();
            int64_t is = this->is;
            for (uint64_t i = 0; i < npart_i; i++) {
                int64_t ie = is + (int64_t)chunksize_i - 1 < this->get_ie() ? is + (int64_t)chunksize_i - 1 : this->get_ie();
                result.emplace_back(is, ie, js, je, ks, ke);
                is += chunksize_i;
            }
            js += chunksize_j;
        }
        ks += chunksize_k;
    }
    return result;
}

Vertices_3D Vertices_3D::get_shifed_vertices(const int64_t i_shift, const int64_t j_shift, const int64_t k_shift) const {
    return Vertices_3D(this->is + i_shift, this->get_ie() + i_shift,
                       this->js + j_shift, this->get_je() + j_shift,
                       this->ks + k_shift, this->get_ke() + k_shift);
}

Vertices_3D Vertices_3D::get_overlap_vertices(const Vertices_3D& other) const {
    if (!this->is_overlaped(other)) {
        return Vertices_3D(0, 0, 0);
    } else {
        return Vertices_3D(this->is >= other.is ? this->is : other.is,
                           this->get_ie() <= other.get_ie() ? this->get_ie() : other.get_ie(),
                           this->js >= other.js ? this->js : other.js,
                           this->get_je() <= other.get_je() ? this->get_je() : other.get_je(),
                           this->ks >= other.ks ? this->ks : other.ks,
                           this->get_ke() <= other.get_ke() ? this->get_ke() : other.get_ke());
    }
}

Vertices_3D Vertices_3D::get_overlap_vertices(const Vertices_3D& other, const int64_t i_shift, const int64_t j_shift,
                                                                        const int64_t k_shift) const {
    Vertices_3D temp(other.is + i_shift, other.get_ie() + i_shift,
                     other.js + j_shift, other.get_je() + j_shift,
                     other.ks + k_shift, other.get_ke() + k_shift);
    return this->get_overlap_vertices(temp);
}

Vertices_3D Vertices_3D::get_super_vertices(const Vertices_3D& other) const {
    if (this->get_size() == 0) {
        return Vertices_3D(other);
    } else if (other.get_size() == 0) {
        return Vertices_3D(*this);
    } else {
        return Vertices_3D(this->is <= other.is ? this->is : other.is,
                           this->get_ie() >= other.get_ie() ? this->get_ie() : other.get_ie(),
                           this->js <= other.js ? this->js : other.js,
                           this->get_je() >= other.get_je() ? this->get_je() : other.get_je(),
                           this->ks <= other.ks ? this->ks : other.ks,
                           this->get_ke() >= other.get_ke() ? this->get_ke() : other.get_ke());
    }
}

Vertices_3D Vertices_3D::get_super_vertices(const Vertices_3D& other, const int64_t i_shift, const int64_t j_shift,
                                                                        const int64_t k_shift) const {
    Vertices_3D temp(other.is + i_shift, other.get_ie() + i_shift,
                     other.js + j_shift, other.get_je() + j_shift,
                     other.ks + k_shift, other.get_ke() + k_shift);
    return this->get_super_vertices(temp);
}

template<typename T>
Vertices_3D Vertices_3D::generate_ex_vertices(const T nnode) const {
    return this->get_size() == 0 ? Vertices_3D(0, 0, 0) : Vertices_3D(this->is - (int64_t)nnode, this->get_ie() + (int64_t)nnode,
                                                                      this->js - (int64_t)nnode, this->get_je() + (int64_t)nnode,
                                                                      this->ks - (int64_t)nnode, this->get_ke() + (int64_t)nnode);
}
template Vertices_3D Vertices_3D::generate_ex_vertices<int>(const int nnode) const;
template Vertices_3D Vertices_3D::generate_ex_vertices<uint>(const uint nnode) const;
template Vertices_3D Vertices_3D::generate_ex_vertices<int64_t>(const int64_t nnode) const;
template Vertices_3D Vertices_3D::generate_ex_vertices<uint64_t>(const uint64_t nnode) const;

template<typename T>
Vertices_3D Vertices_3D::generate_ex_vertices(const T* nnode) const {
    return this->get_size() == 0 ? Vertices_3D(0, 0, 0) : Vertices_3D(this->is - (int64_t)nnode[0], this->get_ie() + (int64_t)nnode[0],
                                                                      this->js - (int64_t)nnode[1], this->get_je() + (int64_t)nnode[1],
                                                                      this->ks - (int64_t)nnode[2], this->get_ke() + (int64_t)nnode[2]);
}
template Vertices_3D Vertices_3D::generate_ex_vertices<int>(const int* nnode) const;
template Vertices_3D Vertices_3D::generate_ex_vertices<uint>(const uint* nnode) const;
template Vertices_3D Vertices_3D::generate_ex_vertices<int64_t>(const int64_t* nnode) const;
template Vertices_3D Vertices_3D::generate_ex_vertices<uint64_t>(const uint64_t* nnode) const;

void Vertices_3D::show() const {
    this->Vertices_2D::show();
    std::cout << "ks = " << this->ks << ", ke = " << this->get_ke() << ", nk = " << this->nk <<std::endl;
}

Vertices_4D::Vertices_4D() : Vertices_3D::Vertices_3D() {
    this->bs = 0;
    this->nb = 0;
}

Vertices_4D::Vertices_4D(const Vertices_3D& vertices_3d, const uint64_t nb) : Vertices_3D::Vertices_3D(vertices_3d) {
    if (vertices_3d.get_size() > 0) {
        this->set_b_vertices(0, nb == 0 ? -1 : (int64_t)nb - 1);
    } else {
        this->set_b_vertices(0, -1);
    }
}

Vertices_4D::Vertices_4D(const Vertices_3D& vertices_3d, const int64_t bs, const int64_t be) : Vertices_3D::Vertices_3D(vertices_3d) {
    if (vertices_3d.get_size() > 0) {
        this->set_b_vertices(bs, be);
    } else {
        this->set_b_vertices(0, -1);
    }
}

Vertices_4D::Vertices_4D(const uint64_t ni, const uint64_t nj, const uint64_t nk, const uint64_t nb) : Vertices_3D::Vertices_3D(ni, nj, nk) {
    this->set_b_vertices(0, nb == 0 ? -1 : (int64_t)nb - 1);
}

Vertices_4D::Vertices_4D(const int64_t is, const int64_t ie, const int64_t js, const int64_t je,
                         const int64_t ks, const int64_t ke, const int64_t bs, const int64_t be)
                                  : Vertices_3D::Vertices_3D(is, ie, js, je, ks, ke) {
    this->set_b_vertices(bs, be);
}

Vertices_4D::Vertices_4D(const Vertices_4D& other) : Vertices_3D::Vertices_3D(other) {
    this->bs = other.bs;
    this->nb = other.nb;
}

Vertices_4D::~Vertices_4D() {}

void Vertices_4D::swap(Vertices_4D& other) {
    this->Vertices_3D::swap(other);
    std::swap(this->bs, other.bs);
    std::swap(this->nb, other.nb);
    return;
}

void Vertices_4D::set_vertices(const int64_t is, const int64_t ie, const int64_t js, const int64_t je,
                               const int64_t ks, const int64_t ke, const int64_t bs, const int64_t be) {
    this->Vertices_3D::set_vertices(is, ie, js, je, ks, ke);
    this->set_b_vertices(bs, be);
    return;
}

void Vertices_4D::set_vertices(const Vertices_4D& other) {
    this->set_vertices(other.is, other.get_ie(),
                       other.js, other.get_je(),
                       other.ks, other.get_ke(),
                       other.bs, other.get_be());
    return;
}

// void Vertices_4D::set_i_vertices(const int64_t is, const int64_t ie) {
//     this->Vertices_3D::set_i_vertices(is, ie);
//     return;
// }

// void Vertices_4D::set_j_vertices(const int64_t js, const int64_t je) {
//     this->Vertices_3D::set_j_vertices(js, je);
//     return;
// }

// void Vertices_4D::set_k_vertices(const int64_t ks, const int64_t ke) {
//     this->Vertices_3D::set_k_vertices(ks, ke);
//     return;
// }

void Vertices_4D::set_b_vertices(const int64_t bs, const int64_t be) {
    assert(be - bs + 1 >= 0);
    this->bs = bs;
    this->nb = be - bs + 1;
    return;
}

void Vertices_4D::set_nb(const uint64_t nb) {
    this->set_b_vertices(0, nb == 0 ? -1 : (int64_t)nb - 1);
}

Vertices_4D& Vertices_4D::operator=(const Vertices_4D& other) {
    if (this != &other) {
        this->Vertices_3D::operator=(other);
        this->bs = other.bs;
        this->nb = other.nb;
    }
    return *this;
}

bool Vertices_4D::operator==(const Vertices_4D& other) const {
    return Vertices_3D::operator==(other) && (this->bs == other.bs)
            && (this->nb == other.nb);
}

bool Vertices_4D::operator!=(const Vertices_4D& other) const {
    return !(*this == other);
}

int64_t Vertices_4D::get_bs() const {
    return this->bs;
}

int64_t Vertices_4D::get_be() const {
    return this->bs + (int64_t) this->nb - 1;
}

uint64_t Vertices_4D::get_nb() const {
    return this->nb;
}

Vertices_4D Vertices_4D::get_vertices() const {
    return *this;
}

Vertices_4D& Vertices_4D::get_vertices() {
    return *this;
}

bool Vertices_4D::contain_point(const int64_t i, const int64_t j, const int64_t k, const int64_t b) const {
    // return this-> Vertices_3D::contain_point(i, j, k) && (b >= this->bs) && (b <= this->be);
    return this->contain_i(i) && this->contain_j(j)
        && this->contain_k(k) && this->contain_b(b);
}

bool Vertices_4D::contain_i(const int64_t i) const {
    return (i >= this->is) && (i <= this->get_ie());
}

bool Vertices_4D::contain_j(const int64_t j) const {
    return (j >= this->js) && (j <= this->get_je());
}

bool Vertices_4D::contain_k(const int64_t k) const {
    return (k >= this->ks) && (k <= this->get_ke());
}

bool Vertices_4D::contain_b(const int64_t b) const {
    return (b >= this->bs) && (b <= this->get_be());
}

bool Vertices_4D::is_index_legal(const uint64_t index) const {
    return index < this->get_size();
}

uint64_t Vertices_4D::get_size() const {
    return this->ni * this->nj * this->nk * this->nb;
}

/**
 * @brief get the position in x direction start from "is", namely "i", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int the position in x direction start from "is", namely "i".
 */
int64_t Vertices_4D::get_i(const uint64_t index) const {
    assert(this->is_index_legal(index));
    return this->get_i_nocheck(index);
}

/**
 * @brief get the position in x direction start from "is", namely "i", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int the position in x direction start from "is", namely "i".
 */
int64_t Vertices_4D::get_i_nocheck(const uint64_t index) const {
    return this->Vertices_1D::get_i_nocheck(index % this->ni);
}

/**
 * @brief get the position in y direction start from "js", namely "j", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int the position in y direction start from "js", namely "j".
 */
int64_t Vertices_4D::get_j(const uint64_t index) const {
    assert(this->is_index_legal(index));
    return this->get_j_nocheck(index);
}

/**
 * @brief get the position in y direction start from "js", namely "j", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int the position in y direction start from "js", namely "j".
 */
int64_t Vertices_4D::get_j_nocheck(const uint64_t index) const {
    return this->Vertices_2D::get_j_nocheck(index % (this->ni * this->nj));
}

/**
 * @brief get the position in z direction start from "ks", namely "k", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int the position in z direction start from "ks", namely "k".
 */
int64_t Vertices_4D::get_k(const uint64_t index) const {
    assert(this->is_index_legal(index));
    return this->get_k_nocheck(index);
}

/**
 * @brief get the position in z direction start from "ks", namely "k", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int the position in z direction start from "ks", namely "k".
 */
int64_t Vertices_4D::get_k_nocheck(const uint64_t index) const {
    return this->Vertices_3D::get_k_nocheck(index % (this->ni * this->nj * this->nk));
}

/**
 * @brief get the position in b direction start from "bs", namely "b", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int the position in b direction start from "bs", namely "b".
 */
int64_t Vertices_4D::get_b(const uint64_t index) const {
    assert(this->is_index_legal(index));
    return this->get_b_nocheck(index);
}

/**
 * @brief get the position in b direction start from "bs", namely "b", according to index.
 * 
 * @param index the index_th position in the column major vertices, start from 0.
 * @return int the position in b direction start from "bs", namely "b".
 */
int64_t Vertices_4D::get_b_nocheck(const uint64_t index) const {
    return index / (this->ni * this->nj * this->nk) + this->bs;
}

/**
 * @brief get the index, according to the position in x, y, z and b direction, namely "i", "j", "k" and "b".
 * 
 * @param i the position in x direction start from "is".
 * @param j the position in y direction start from "js".
 * @param k the position in z direction start from "ks".
 * @param b the position in b direction start from "bs".
 * @return int the index_th position in the column major vertices, start from 0.
 */
uint64_t Vertices_4D::get_index(const int64_t i, const int64_t j, const int64_t k, const int64_t b) const {
    assert(this->contain_point(i, j, k, b));
    return this->get_index_nocheck(i, j, k, b);
}

uint64_t Vertices_4D::get_index_DBC(const int64_t i, const int64_t j, const int64_t k, const int64_t b) const {
    return this->contain_point(i, j, k, b) ? this->get_index_nocheck(i, j, k, b) + 1 : 0;
}

uint64_t Vertices_4D::get_index_PBC(const int64_t i, const int64_t j, const int64_t k, const int64_t b) const {
    return this->get_index_nocheck(
           this->contain_i(i) ? i : this->map_i_into_vertices(i),
           this->contain_j(j) ? j : this->map_j_into_vertices(j),
           this->contain_k(k) ? k : this->map_k_into_vertices(k),
           this->contain_b(b) ? b : this->map_b_into_vertices(b)
           );
}

/**
 * @brief get the index, according to the position in x, y, z and b direction, namely "i", "j", "k" and "b".
 * 
 * @param i the position in x direction start from "is".
 * @param j the position in y direction start from "js".
 * @param k the position in z direction start from "ks".
 * @param b the position in b direction start from "bs".
 * @return int the index_th position in the column major vertices, start from 0.
 */
uint64_t Vertices_4D::get_index_nocheck(const int64_t i, const int64_t j, const int64_t k, const int64_t b) const {
    // return this->Vertices_3D::get_index_nocheck(i, j, k) + (b - this->bs) * (this->nx * this->ny * this->nz);
    return uint64_t(i - this->is)
            + (uint64_t(j - this->js)
                + (uint64_t(k - this->ks)
                    + (uint64_t(b - this->bs)
                        * (uint64_t)this->nk))
                            * (uint64_t)this->nj)
                                * (uint64_t)this->ni;
}

uint64_t Vertices_4D::get_index_nocheck_rowmaj(const int64_t i, const int64_t j, const int64_t k, const int64_t b) const {
    // return this->Vertices_3D::get_index_nocheck(i, j, k) + (b - this->bs) * (this->nx * this->ny * this->nz);
    return uint64_t(b - this->bs)
            + (uint64_t(k - this->ks)
                + (uint64_t(j - this->js)
                    + (uint64_t(i - this->is)
                        * (uint64_t)this->nj))
                            * (uint64_t)this->nk)
                                * (uint64_t)this->nb;
}

int64_t Vertices_4D::map_i_into_vertices(const int64_t i) const {
    // return this->Vertices_3D::map_i_into_vertices(i);
    assert(this->ni != 0);
    const int64_t offset = (i - this->is) % (int64_t)this->ni;
    return this->is + (offset >= 0 ? offset : offset + (int64_t)this->ni);
}

int64_t Vertices_4D::map_j_into_vertices(const int64_t j) const {
    // return this->Vertices_3D::map_j_into_vertices(j);
    assert(this->nj != 0);
    const int64_t offset = (j - this->js) % (int64_t)this->nj;
    return this->js + (offset >= 0 ? offset : offset + (int64_t)this->nj);
}

int64_t Vertices_4D::map_k_into_vertices(const int64_t k) const {
    // return this->Vertices_3D::map_k_into_vertices(k);
    assert(this->nk != 0);
    const int64_t offset = (k - this->ks) % (int64_t)this->nk;
    return this->ks + (offset >= 0 ? offset : offset + (int64_t)this->nk);
}

int64_t Vertices_4D::map_b_into_vertices(const int64_t b) const {
    assert(this->nb != 0);
    const int64_t offset = (b - this->bs) % (int64_t)this->nb;
    return this->bs + (offset >= 0 ? offset : offset + (int64_t)this->nb);
}

bool Vertices_4D::is_ex_vertices(const Vertices_4D& other) const {
    // return this->Vertices_3D::is_exarr(other) && this->bs <= other.bs && this->get_be() >= other.get_be();
    return (other.get_size() == 0) || (this->is <= other.is && this->get_ie() >= other.get_ie()
           && this->js <= other.js && this->get_je() >= other.get_je()
           && this->ks <= other.ks && this->get_ke() >= other.get_ke()
           && this->bs <= other.bs && this->get_be() >= other.get_be());
}

bool Vertices_4D::is_sub_vertices(const Vertices_4D& other) const {
    // return this->Vertices_3D::is_exarr(other) && this->bs <= other.bs && this->get_be() >= other.get_be();
    return (this->get_size() == 0) || (this->is >= other.is && this->get_ie() <= other.get_ie()
           && this->js >= other.js && this->get_je() <= other.get_je()
           && this->ks >= other.ks && this->get_ke() <= other.get_ke()
           && this->bs >= other.bs && this->get_be() <= other.get_be());
}

bool Vertices_4D::is_overlaped(const Vertices_4D& other) const {
    return !((this->get_size() == 0) || (other.get_size() == 0) 
             ||(other.is>this->get_ie() || other.get_ie()< this->is)
             ||(other.js>this->get_je() || other.get_je()< this->js)
             ||(other.ks>this->get_ke() || other.get_ke()< this->ks)
             ||(other.bs>this->get_be() || other.get_be()< this->bs));
}

std::vector<Vertices_4D> Vertices_4D::split(const uint64_t npart_i, const uint64_t npart_j,
                                            const uint64_t npart_k, const uint64_t npart_b) const {
    assert(npart_i > 0 && npart_i <= this->ni);
    assert(npart_j > 0 && npart_j <= this->nj);
    assert(npart_k > 0 && npart_k <= this->nk);
    assert(npart_b > 0 && npart_b <= this->nb);
    uint64_t chunksize_i = Linalg::get_chunksize(npart_i, this->ni);
    uint64_t chunksize_j = Linalg::get_chunksize(npart_j, this->nj);
    uint64_t chunksize_k = Linalg::get_chunksize(npart_k, this->nk);
    uint64_t chunksize_b = Linalg::get_chunksize(npart_b, this->nb);
    std::vector<Vertices_4D> result;
    result.reserve(npart_i * npart_j * npart_k * npart_b);
    int64_t bs = this->bs;
    for (uint64_t b = 0; b < npart_b; b++) {
        int64_t be = bs + (int64_t)chunksize_b - 1 < this->get_be() ? bs + (int64_t)chunksize_b - 1 : this->get_be();
        int64_t ks = this->ks;
        for (uint64_t k = 0; k < npart_k; k++) {
            int64_t ke = ks + (int64_t)chunksize_k - 1 < this->get_ke() ? ks + (int64_t)chunksize_k - 1 : this->get_ke();
            int64_t js = this->js;
            for (uint64_t j = 0; j < npart_j; j++) {
                int64_t je = js + (int64_t)chunksize_j - 1 < this->get_je() ? js + (int64_t)chunksize_j - 1 : this->get_je();
                int64_t is = this->is;
                for (uint64_t i = 0; i < npart_i; i++) {
                    int64_t ie = is + (int64_t)chunksize_i - 1 < this->get_ie() ? is + (int64_t)chunksize_i - 1 : this->get_ie();
                    result.emplace_back(is, ie, js, je, ks, ke, bs, be);
                    is += chunksize_i;
                }
                js += chunksize_j;
            }
            ks += chunksize_k;
        }
        bs += chunksize_b;
    }
    

    
    return result;
}

Vertices_4D Vertices_4D::get_overlap_vertices(const Vertices_4D& other) const {
    if (!this->is_overlaped(other)) {
        return Vertices_4D(0, 0, 0, 0);
    } else {
        return Vertices_4D(this->is >= other.is ? this->is : other.is,
                           this->get_ie() <= other.get_ie() ? this->get_ie() : other.get_ie(),
                           this->js >= other.js ? this->js : other.js,
                           this->get_je() <= other.get_je() ? this->get_je() : other.get_je(),
                           this->ks >= other.ks ? this->ks : other.ks,
                           this->get_ke() <= other.get_ke() ? this->get_ke() : other.get_ke(),
                           this->bs >= other.bs ? this->bs : other.bs,
                           this->get_be() <= other.get_be() ? this->get_be() : other.get_be());
    }
}

Vertices_4D Vertices_4D::get_overlap_vertices(const Vertices_4D& other, const int64_t i_shift, const int64_t j_shift,
                                                                        const int64_t k_shift, const int64_t b_shift) const {
    Vertices_4D temp(other.is + i_shift, other.get_ie() + i_shift,
                     other.js + j_shift, other.get_je() + j_shift,
                     other.ks + k_shift, other.get_ke() + k_shift,
                     other.bs + b_shift, other.get_be() + b_shift);
    return this->get_overlap_vertices(temp);
}

Vertices_4D Vertices_4D::get_super_vertices(const Vertices_4D& other) const {
    if (this->get_size() == 0) {
        return Vertices_4D(other);
    } else if (other.get_size() == 0) {
        return Vertices_4D(*this);
    } else {
        return Vertices_4D(this->is <= other.is ? this->is : other.is,
                           this->get_ie() >= other.get_ie() ? this->get_ie() : other.get_ie(),
                           this->js <= other.js ? this->js : other.js,
                           this->get_je() >= other.get_je() ? this->get_je() : other.get_je(),
                           this->ks <= other.ks ? this->ks : other.ks,
                           this->get_ke() >= other.get_ke() ? this->get_ke() : other.get_ke(),
                           this->bs <= other.bs ? this->bs : other.bs,
                           this->get_be() >= other.get_be() ? this->get_be() : other.get_be());
    }
}

Vertices_4D Vertices_4D::get_super_vertices(const Vertices_4D& other, const int64_t i_shift, const int64_t j_shift,
                                                                        const int64_t k_shift, const int64_t b_shift) const {
    Vertices_4D temp(other.is + i_shift, other.get_ie() + i_shift,
                     other.js + j_shift, other.get_je() + j_shift,
                     other.ks + k_shift, other.get_ke() + k_shift,
                     other.bs + b_shift, other.get_be() + b_shift);
    return this->get_super_vertices(temp);
}

template<typename T>
Vertices_4D Vertices_4D::generate_ex_vertices(const T nnode) const {
    return this->get_size() == 0 ? Vertices_4D(0, 0, 0, 0) : Vertices_4D(this->is - (int64_t)nnode, this->get_ie() + (int64_t)nnode,
                                                                         this->js - (int64_t)nnode, this->get_je() + (int64_t)nnode,
                                                                         this->ks - (int64_t)nnode, this->get_ke() + (int64_t)nnode,
                                                                         this->bs - (int64_t)nnode, this->get_be() + (int64_t)nnode);
}
template Vertices_4D Vertices_4D::generate_ex_vertices<int>(const int nnode) const;
template Vertices_4D Vertices_4D::generate_ex_vertices<uint>(const uint nnode) const;
template Vertices_4D Vertices_4D::generate_ex_vertices<int64_t>(const int64_t nnode) const;
template Vertices_4D Vertices_4D::generate_ex_vertices<uint64_t>(const uint64_t nnode) const;

template<typename T>
Vertices_4D Vertices_4D::generate_ex_vertices(const T* nnode) const {
    return this->get_size() == 0 ? Vertices_4D(0, 0, 0, 0) : Vertices_4D(this->is - (int64_t)nnode[0], this->get_ie() + (int64_t)nnode[0],
                                                                         this->js - (int64_t)nnode[1], this->get_je() + (int64_t)nnode[1],
                                                                         this->ks - (int64_t)nnode[2], this->get_ke() + (int64_t)nnode[2],
                                                                         this->bs - (int64_t)nnode[3], this->get_be() + (int64_t)nnode[3]);
}
template Vertices_4D Vertices_4D::generate_ex_vertices<int>(const int* nnode) const;
template Vertices_4D Vertices_4D::generate_ex_vertices<uint>(const uint* nnode) const;
template Vertices_4D Vertices_4D::generate_ex_vertices<int64_t>(const int64_t* nnode) const;
template Vertices_4D Vertices_4D::generate_ex_vertices<uint64_t>(const uint64_t* nnode) const;

void Vertices_4D::show() const {
    this->Vertices_3D::show();
    std::cout << "bs = " << this->bs << ", be = " << this->get_be() << ", nb = " << this->nb <<std::endl;
}

template<typename T> void Vertices_method::fill_vector(T const* const __restrict__ my_data, const Vertices_3D& my_vertices,
                                                       T* const __restrict__ other_data,
                                                       const Vertices_3D& my_fill_region, const uint64_t offset) {
    assert(my_vertices.is_ex_vertices(my_fill_region));
    const int64_t my_index_origin = my_vertices.get_index_nocheck(my_fill_region.is, my_fill_region.js, my_fill_region.ks);
    const int64_t my_ni = my_vertices.ni;
    const int64_t my_ninj = my_vertices.ni * my_vertices.nj;
    const int64_t region_ni = (int64_t)my_fill_region.ni;
    const int64_t region_nj = (int64_t)my_fill_region.nj;
    const int64_t region_nk = (int64_t)my_fill_region.nk;
    const int64_t other_ni = region_ni;
    const int64_t other_ninj = region_ni * region_nj;
    // int other_offset = offset;
    #ifdef USE_MEMFUNS
    const int64_t memcpy_size = sizeof(T) * region_ni;
    #endif //USE_MEMFUNS
    #ifdef USE_OPENMP
    #pragma omp for schedule(static, (region_nk - 1)/omp_get_num_threads() + 1) nowait
    #endif //USE_OPENMP
    for (int64_t k = 0; k < region_nk; ++k) {
        T const* __restrict__ my_data_j = my_data + my_index_origin + k * my_ninj;
        T* __restrict__ other_data_j = other_data + offset + k * other_ninj;
        for (int64_t j = 0; j < region_nj; ++j) {
            T const* __restrict__ my_data_i = my_data_j;
            T* __restrict__ other_data_i = other_data_j;
            #ifdef USE_MEMFUNS
            std::memcpy(other_data_i, my_data_i, memcpy_size);
            #else
            #ifdef USE_OPENMP_SIMD
            #pragma omp simd
            #endif //USE_OPENMP_SIMD
            for (int64_t i = 0; i < region_ni; ++i) {
                *other_data_i++ = *my_data_i++;
            }
            #endif //USE_MEMFUNS
            my_data_j += my_ni;
            other_data_j += other_ni;
        }
    }
    return;
}
template void Vertices_method::fill_vector<int>(int const* const my_data, const Vertices_3D& my_vertices, int* const other_data,
                                                const Vertices_3D& my_fill_region, const uint64_t offset);
template void Vertices_method::fill_vector<float>(float const* const my_data, const Vertices_3D& my_vertices, float* const other_data,
                                                  const Vertices_3D& my_fill_region, const uint64_t offset);
template void Vertices_method::fill_vector<double>(double const* const my_data, const Vertices_3D& my_vertices, double* const other_data,
                                                   const Vertices_3D& my_fill_region, const uint64_t offset);
#ifdef FP16_FLAG
template void Vertices_method::fill_vector<__fp16>(__fp16 const* const my_data, const Vertices_3D& my_vertices, __fp16* const other_data,
                                                   const Vertices_3D& my_fill_region, const uint64_t offset);
#endif //FP16_FLAG

template<typename T> void Vertices_method::fill_vector(T const* const __restrict__ my_data, const Vertices_4D& my_vertices,
                                                       T* const __restrict__ other_data,
                                                       const Vertices_4D& my_fill_region, const uint64_t offset) {
    assert(my_vertices.is_ex_vertices(my_fill_region));
    #ifdef ARRAY_UNFOLD
    if (unlikely(my_vertices.ni == my_fill_region.ni)) {
        if (my_fill_region.get_size() == 0) return;
        Vertices_3D my_vertices_3d(0, (int64_t)(my_vertices.ni * my_vertices.nj) - 1,
                                   my_vertices.ks, my_vertices.get_ke(),
                                   my_vertices.bs, my_vertices.get_be());
        uint64_t my_fill_region_index = my_vertices.get_index_nocheck(my_fill_region.is,
                                                                  my_fill_region.js,
                                                                  my_fill_region.ks,
                                                                  my_fill_region.bs);
        uint64_t my_fill_region_i = my_vertices_3d.get_i_nocheck(my_fill_region_index);
        Vertices_3D my_fill_region_3d(0 + my_fill_region_i,
                                     (int64_t)(my_fill_region.ni * my_fill_region.nj) - 1 + my_fill_region_i,
                                      my_fill_region.ks, my_fill_region.get_ke(),
                                      my_fill_region.bs, my_fill_region.get_be());
        Vertices_method::fill_vector(my_data, my_vertices_3d, other_data, my_fill_region_3d, offset);
    } else {
    #endif //ARRAY_UNFOLD
        const int64_t my_index_origin = my_vertices.get_index_nocheck(my_fill_region.is, my_fill_region.js,
                                                              my_fill_region.ks, my_fill_region.bs);
        const int64_t my_ni = my_vertices.ni;
        const int64_t my_ninj = my_vertices.ni * my_vertices.nj;
        const int64_t my_ninjnk = my_vertices.ni * my_vertices.nj * my_vertices.nk;
        const int64_t region_ni = (int64_t)my_fill_region.ni;
        const int64_t region_nj = (int64_t)my_fill_region.nj;
        const int64_t region_nk = (int64_t)my_fill_region.nk;
        const int64_t region_nb = (int64_t)my_fill_region.nb;
        const int64_t other_ni = region_ni;
        const int64_t other_ninj = region_ni * region_nj;
        const int64_t other_ninjnk = region_ni * region_nj * region_nk;
        #ifdef USE_MEMFUNS
        const int memcpy_size = sizeof(T) * region_ni;
        #endif //USE_MEMFUNS
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region_nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < region_nb; b++) {
            T const* __restrict__ my_data_k = my_data + my_index_origin + b * my_ninjnk;
            T* __restrict__ other_data_k = other_data + offset + b * other_ninjnk;
            for (int64_t k = 0; k < region_nk; ++k) {
                T const* __restrict__ my_data_j = my_data_k;
                T* __restrict__ other_data_j = other_data_k;
                for (int64_t j = 0; j < region_nj; ++j) {
                    T const* __restrict__ my_data_i = my_data_j;
                    T* __restrict__ other_data_i = other_data_j;
                    #ifdef USE_MEMFUNS
                    std::memcpy(other_data_i, my_data_i, memcpy_size);
                    #else
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif //USE_OPENMP_SIMD
                    for (int64_t i = 0; i < region_ni; ++i) {
                        *other_data_i++ = *my_data_i++;
                    }
                    #endif //USE_MEMFUNS
                    my_data_j += my_ni;
                    other_data_j += other_ni;
                }
                my_data_k += my_ninj;
                other_data_k += other_ninj;
            }
        }
    #ifdef ARRAY_UNFOLD
    }
    #endif //ARRAY_UNFOLD
    return;
}
template void Vertices_method::fill_vector<int>(int const* const my_data, const Vertices_4D& my_vertices, int* const other_data,
                                                const Vertices_4D& my_fill_region, const uint64_t offset);
template void Vertices_method::fill_vector<float>(float const* const my_data, const Vertices_4D& my_vertices, float* const other_data,
                                                  const Vertices_4D& my_fill_region, const uint64_t offset);
template void Vertices_method::fill_vector<double>(double const* const my_data, const Vertices_4D& my_vertices, double* const other_data,
                                                   const Vertices_4D& my_fill_region, const uint64_t offset);
#ifdef FP16_FLAG
template void Vertices_method::fill_vector<__fp16>(__fp16 const* const my_data, const Vertices_4D& my_vertices, __fp16* const other_data,
                                                   const Vertices_4D& my_fill_region, const uint64_t offset);
#endif //FP16_FLAG

template<typename T> void Vertices_method::be_filled_vector(T* const __restrict__ my_data, const Vertices_2D& my_vertices,
                                                            T const* const __restrict__ other_data, const Vertices_2D& other_vertices,
                                                            const Vertices_2D& my_filled_region, const uint64_t offset) {
    assert(my_vertices.is_ex_vertices(my_filled_region));
    const int64_t my_index_origin = my_vertices.get_index_nocheck(my_filled_region.is, my_filled_region.js);
    const int64_t my_ni = my_vertices.ni;
    const int64_t region_ni = (int64_t)my_filled_region.ni;
    const int64_t region_nj = (int64_t)my_filled_region.nj;
    const int64_t other_offset_origin = offset;
    const int64_t other_ni = other_vertices.ni;
    #ifdef USE_MEMFUNS
    const int64_t memcpy_size = sizeof(T) * region_ni;
    #endif //USE_MEMFUNS
    #ifdef USE_OPENMP
    #pragma omp for schedule(static, (region_nj - 1)/omp_get_num_threads() + 1) nowait
    #endif //USE_OPENMP
    for (int64_t j = 0; j < region_nj; ++j) {
        T* __restrict__ my_data_i = my_data + my_index_origin + j * my_ni;
        T const* __restrict__ other_data_i = other_data + other_offset_origin + j * other_ni;
        #ifdef USE_MEMFUNS
        std::memcpy(my_data_i, other_data_i, memcpy_size);
        #else
        #ifdef USE_OPENMP_SIMD
        #pragma omp simd
        #endif //USE_OPENMP_SIMD
        for (int64_t i = 0; i < region_ni; ++i) {
            *my_data_i++ = *other_data_i++;
        }
        #endif //USE_MEMFUNS
    }
    return;
}
template void Vertices_method::be_filled_vector<int>(int* const my_data, const Vertices_2D& my_vertices,
                                                     int const* const other_data, const Vertices_2D& other_vertices,
                                                     const Vertices_2D& my_filled_region, const uint64_t offset);
template void Vertices_method::be_filled_vector<float>(float* const my_data, const Vertices_2D& my_vertices,
                                                       float const* const other_data, const Vertices_2D& other_vertices,
                                                       const Vertices_2D& my_filled_region, const uint64_t offset);
template void Vertices_method::be_filled_vector<double>(double* const my_data, const Vertices_2D& my_vertices,
                                                        double const* const other_data, const Vertices_2D& other_vertices,
                                                        const Vertices_2D& my_filled_region, const uint64_t offset);
#ifdef FP16_FLAG
template void Vertices_method::be_filled_vector<__fp16>(__fp16* const my_data, const Vertices_2D& my_vertices,
                                                        __fp16 const* const other_data, const Vertices_2D& other_vertices,
                                                        const Vertices_2D& my_filled_region, const uint64_t offset);
#endif //FP16_FLAG

template<typename T> void Vertices_method::be_filled_vector(T* const __restrict__ my_data, const Vertices_3D& my_vertices,
                                                            T const* const __restrict__ other_data, const Vertices_3D& other_vertices,
                                                            const Vertices_3D& my_filled_region, const uint64_t offset) {
    assert(my_vertices.is_ex_vertices(my_filled_region));
    #ifdef ARRAY_UNFOLD
    if (unlikely(my_vertices.ni == my_filled_region.ni && my_vertices.ni == other_vertices.ni)) {
    // if (false) {
        if (my_filled_region.get_size() == 0) return;
        Vertices_2D my_vertices_2d(0, (int64_t)(my_vertices.ni * my_vertices.nj) - 1,
                                   my_vertices.ks, my_vertices.get_ke());
        uint64_t my_filled_region_index = my_vertices.get_index_nocheck(my_filled_region.is,
                                                                    my_filled_region.js,
                                                                    my_filled_region.ks);
        uint64_t my_filled_region_i = my_vertices_2d.get_i_nocheck(my_filled_region_index);
        Vertices_2D my_filled_region_2d(0 + my_filled_region_i,
                                        (int64_t)(my_filled_region.ni * my_filled_region.nj) - 1 + my_filled_region_i,
                                        my_filled_region.ks, my_filled_region.get_ke());
        Vertices_2D other_vertices_2d(0, (int64_t)(other_vertices.ni * other_vertices.nj) - 1,
                                               other_vertices.ks, other_vertices.get_ke());
        Vertices_method::be_filled_vector(my_data, my_vertices_2d, other_data, other_vertices_2d,
                                          my_filled_region_2d, offset);
    } else {
    #endif //ARRAY_UNFOLD
        const int64_t my_index_origin = my_vertices.get_index_nocheck(my_filled_region.is, my_filled_region.js, my_filled_region.ks);
        const int64_t my_ni = my_vertices.ni;
        const int64_t my_ninj = my_vertices.ni * my_vertices.nj;
        const int64_t region_ni = (int64_t)my_filled_region.ni;
        const int64_t region_nj = (int64_t)my_filled_region.nj;
        const int64_t region_nk = (int64_t)my_filled_region.nk;
        const int64_t other_ni = other_vertices.ni;
        const int64_t other_ninj = other_vertices.ni * other_vertices.nj;
        #ifdef USE_MEMFUNS
        const int64_t memcpy_size = sizeof(T) * region_ni;
        #endif //USE_MEMFUNS
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region_nk - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t k = 0; k < region_nk; ++k) {
            T* __restrict__ my_data_j = my_data + my_index_origin + k * my_ninj;
            T const* __restrict__ other_data_j = other_data + offset + k * other_ninj;
            for (int64_t j = 0; j < region_nj; ++j) {
                T* __restrict__ my_data_i = my_data_j;
                T const* __restrict__ other_data_i = other_data_j;
                #ifdef USE_MEMFUNS
                std::memcpy(my_data_i, other_data_i, memcpy_size);
                #else
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif //USE_OPENMP_SIMD
                for (int64_t i = 0; i < region_ni; ++i) {
                    *my_data_i++ = *other_data_i++;
                }
                #endif //USE_MEMFUNS
                my_data_j += my_ni;
                other_data_j += other_ni;
            }
        }
    #ifdef ARRAY_UNFOLD
    }
    #endif //ARRAY_UNFOLD
    return;
}
template void Vertices_method::be_filled_vector<int>(int* const my_data, const Vertices_3D& my_vertices,
                                                     int const* const other_data, const Vertices_3D& other_vertices,
                                                     const Vertices_3D& my_filled_region, const uint64_t offset);
template void Vertices_method::be_filled_vector<float>(float* const my_data, const Vertices_3D& my_vertices,
                                                       float const* const other_data, const Vertices_3D& other_vertices,
                                                       const Vertices_3D& my_filled_region, const uint64_t offset);
template void Vertices_method::be_filled_vector<double>(double* const my_data, const Vertices_3D& my_vertices,
                                                        double const* const other_data, const Vertices_3D& other_vertices,
                                                        const Vertices_3D& my_filled_region, const uint64_t offset);
#ifdef FP16_FLAG
template void Vertices_method::be_filled_vector<__fp16>(__fp16* const my_data, const Vertices_3D& my_vertices,
                                                        __fp16 const* const other_data, const Vertices_3D& other_vertices,
                                                        const Vertices_3D& my_filled_region, const uint64_t offset);
#endif //FP16_FLAG

template<typename T> void Vertices_method::be_filled_vector(T* const __restrict__ my_data, const Vertices_4D& my_vertices,
                                                            T const* const __restrict__ other_data, const Vertices_4D& other_vertices,
                                                            const Vertices_4D& my_filled_region, const uint64_t offset) {
    assert(my_vertices.is_ex_vertices(my_filled_region));
    #ifdef ARRAY_UNFOLD
    if (unlikely(my_vertices.ni == my_filled_region.ni && my_vertices.ni == other_vertices.ni)) {
    // if (false) {
        if (my_filled_region.get_size() == 0) return;
        Vertices_3D my_vertices_3d(0, (int64_t)(my_vertices.ni * my_vertices.nj) - 1,
                                   my_vertices.ks, my_vertices.get_ke(),
                                   my_vertices.bs, my_vertices.get_be());
        uint64_t my_filled_region_index = my_vertices.get_index_nocheck(my_filled_region.is,
                                                                    my_filled_region.js,
                                                                    my_filled_region.ks,
                                                                    my_filled_region.bs);
        uint64_t my_filled_region_i = my_vertices_3d.get_i_nocheck(my_filled_region_index);
        Vertices_3D my_filled_region_3d(0 + my_filled_region_i,
                                        (int64_t)(my_filled_region.ni * my_filled_region.nj) - 1 + my_filled_region_i,
                                        my_filled_region.ks, my_filled_region.get_ke(),
                                        my_filled_region.bs, my_filled_region.get_be());
        Vertices_3D other_vertices_3d(0, (int64_t)(other_vertices.ni * other_vertices.nj) - 1,
                                      other_vertices.ks, other_vertices.get_ke(),
                                      other_vertices.bs, other_vertices.get_be());
        Vertices_method::be_filled_vector(my_data, my_vertices_3d, other_data, other_vertices_3d,
                                          my_filled_region_3d, offset);
    } else {
    #endif //ARRAY_UNFOLD
        const int64_t my_index_origin = my_vertices.get_index_nocheck(my_filled_region.is, my_filled_region.js,
                                                                my_filled_region.ks, my_filled_region.bs);
        const int64_t my_ni = my_vertices.ni;
        const int64_t my_ninj = my_vertices.ni * my_vertices.nj;
        const int64_t my_ninjnk = my_vertices.ni * my_vertices.nj * my_vertices.nk;
        const int64_t region_ni = (int64_t)my_filled_region.ni;
        const int64_t region_nj = (int64_t)my_filled_region.nj;
        const int64_t region_nk = (int64_t)my_filled_region.nk;
        const int64_t region_nb = (int64_t)my_filled_region.nb;
        const int64_t other_ni = other_vertices.ni;
        const int64_t other_ninj = other_vertices.ni * other_vertices.nj;
        const int64_t other_ninjnk = other_vertices.ni * other_vertices.nj * other_vertices.nk;
        #ifdef USE_MEMFUNS
        const int64_t memcpy_size = sizeof(T) * region_ni;
        #endif //USE_MEMFUNS
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region_nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < region_nb; b++) {
            T* __restrict__ my_data_k = my_data + my_index_origin + b * my_ninjnk;
            T const* __restrict__ other_data_k = other_data + offset + b * other_ninjnk;
            for (int64_t k = 0; k < region_nk; ++k) {
                T* __restrict__ my_data_j = my_data_k;
                T const* __restrict__ other_data_j = other_data_k;
                for (int64_t j = 0; j < region_nj; ++j) {
                    T* __restrict__ my_data_i = my_data_j;
                    T const* __restrict__ other_data_i = other_data_j;
                    #ifdef USE_MEMFUNS
                    std::memcpy(my_data_i, other_data_i, memcpy_size);
                    #else
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif //USE_OPENMP_SIMD
                    for (int64_t i = 0; i < region_ni; ++i) {
                        *my_data_i++ = *other_data_i++;
                    }
                    #endif //USE_MEMFUNS
                    my_data_j += my_ni;
                    other_data_j += other_ni;
                }
                my_data_k += my_ninj;
                other_data_k += other_ninj;
            }
        }
    #ifdef ARRAY_UNFOLD
    }
    #endif //ARRAY_UNFOLD
    return;
}
template void Vertices_method::be_filled_vector<int>(int* const my_data, const Vertices_4D& my_vertices,
                                                     int const* const other_data, const Vertices_4D& other_vertices,
                                                     const Vertices_4D& my_filled_region, const uint64_t offset);
template void Vertices_method::be_filled_vector<float>(float* const my_data, const Vertices_4D& my_vertices,
                                                       float const* const other_data, const Vertices_4D& other_vertices,
                                                       const Vertices_4D& my_filled_region, const uint64_t offset);
template void Vertices_method::be_filled_vector<double>(double* const my_data, const Vertices_4D& my_vertices,
                                                        double const* const other_data, const Vertices_4D& other_vertices,
                                                        const Vertices_4D& my_filled_region, const uint64_t offset);
#ifdef FP16_FLAG
template void Vertices_method::be_filled_vector<__fp16>(__fp16* const my_data, const Vertices_4D& my_vertices,
                                                        __fp16 const* const other_data, const Vertices_4D& other_vertices,
                                                        const Vertices_4D& my_filled_region, const uint64_t offset);
#endif //FP16_FLAG

template<typename T> void Vertices_method::fill_region(T const* const __restrict__ my_data, const Vertices_1D& my_vertices,
                                                       T* const __restrict__ other_data, const Vertices_1D& other_vertices,
                                                       const Vertices_1D& region) {
    assert(my_vertices.is_ex_vertices(region) && other_vertices.is_ex_vertices(region));
    const int64_t my_index_origin = my_vertices.get_index_nocheck(region.is);
    const int64_t other_offset_origin = other_vertices.get_index_nocheck(region.is);
    Linalg::set_value_general(other_data + other_offset_origin, my_data + my_index_origin, region.ni);
    return;
}
template void Vertices_method::fill_region<int>(int const* const my_data, const Vertices_1D& my_vertices,
                                                int* const other_data, const Vertices_1D& other_vertices,
                                                const Vertices_1D& region);
template void Vertices_method::fill_region<float>(float const* const my_data, const Vertices_1D& my_vertices,
                                                  float* const other_data, const Vertices_1D& other_vertices,
                                                  const Vertices_1D& region);
template void Vertices_method::fill_region<double>(double const* const my_data, const Vertices_1D& my_vertices,
                                                   double* const other_data, const Vertices_1D& other_vertices,
                                                   const Vertices_1D& region);
#ifdef FP16_FLAG
template void Vertices_method::fill_region<__fp16>(__fp16 const* const my_data, const Vertices_1D& my_vertices, 
                                                   __fp16* const other_data, const Vertices_1D& other_vertices,
                                                   const Vertices_1D& region);
#endif //FP16_FLAG

template<typename T> void Vertices_method::fill_region(T const* const __restrict__ my_data, const Vertices_2D& my_vertices,
                                                       T* const __restrict__ other_data, const Vertices_2D& other_vertices,
                                                       const Vertices_2D& region) {
    assert(my_vertices.is_ex_vertices(region) && other_vertices.is_ex_vertices(region));
    #ifdef ARRAY_UNFOLD
    if (unlikely(my_vertices.ni == region.ni && my_vertices.ni == other_vertices.ni)) {
    // if (false) {
        if (region.get_size() == 0) return;
        Vertices_1D my_vertices_1d(0, (int64_t)(my_vertices.ni * my_vertices.nj) - 1);
        Vertices_1D region_1d(0, (int64_t)(region.ni * region.nj) - 1);
        Vertices_1D other_vertices_1d(0, (int64_t)(other_vertices.ni * other_vertices.nj) - 1);
        uint64_t region_index_of_my_vertices = my_vertices.get_index_nocheck(region.is,
                                                                         region.js);
        uint64_t region_i_of_my_vertices = my_vertices_1d.get_i_nocheck(region_index_of_my_vertices);
        my_vertices_1d.is = -(int64_t)region_i_of_my_vertices;
        uint64_t region_index_of_other_vertices_1d = other_vertices.get_index_nocheck(region.is,
                                                                                  region.js);
        uint region_i_of_other_vertices_1d = other_vertices_1d.get_i_nocheck(region_index_of_other_vertices_1d);
        other_vertices_1d.is = -(int64_t)region_i_of_other_vertices_1d;
        Vertices_method::fill_region(my_data, my_vertices_1d, other_data, other_vertices_1d, region_1d);
    } else {
    #endif //ARRAY_UNFOLD
        const int64_t my_index_origin = my_vertices.get_index_nocheck(region.is, region.js);
        const int64_t my_ni = my_vertices.ni;
        const int64_t region_ni = (int64_t)region.ni;
        const int64_t region_nj = (int64_t)region.nj;
        const int64_t other_offset_origin = other_vertices.get_index_nocheck(region.is, region.js);
        const int64_t other_ni = other_vertices.ni;
        #ifdef USE_MEMFUNS
        const int64_t memcpy_size = sizeof(T) * region_ni;
        #endif //USE_MEMFUNS
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region_nj - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t j = 0; j < region_nj; ++j) {
            T const* __restrict__ my_data_i = my_data + my_index_origin + j * my_ni;
            T* __restrict__ other_data_i = other_data + other_offset_origin + j * other_ni;
            #ifdef USE_MEMFUNS
            std::memcpy(other_data_i, my_data_i, memcpy_size);
            #else
            #ifdef USE_OPENMP_SIMD
            #pragma omp simd
            #endif //USE_OPENMP_SIMD
            for (int i = 0; i < region_ni; ++i) {
                *other_data_i++ = *my_data_i++;
            }
            #endif //USE_MEMFUNS
        }
    #ifdef ARRAY_UNFOLD
    }
    #endif //ARRAY_UNFOLD
    return;
}
template void Vertices_method::fill_region<int>(int const* const my_data, const Vertices_2D& my_vertices, 
                                                int* const other_data, const Vertices_2D& other_vertices,
                                                const Vertices_2D& region);
template void Vertices_method::fill_region<float>(float const* const my_data, const Vertices_2D& my_vertices, 
                                                  float* const other_data, const Vertices_2D& other_vertices,
                                                  const Vertices_2D& region);
template void Vertices_method::fill_region<double>(double const* const my_data, const Vertices_2D& my_vertices,
                                                   double* const other_data, const Vertices_2D& other_vertices,
                                                   const Vertices_2D& region);
#ifdef FP16_FLAG
template void Vertices_method::fill_region<__fp16>(__fp16 const* const my_data, const Vertices_2D& my_vertices,
                                                   __fp16* const other_data, const Vertices_2D& other_vertices,
                                                   const Vertices_2D& region);
#endif //FP16_FLAG

template<typename T> void Vertices_method::fill_region(T const* const __restrict__ my_data, const Vertices_3D& my_vertices,
                                                       T* const __restrict__ other_data, const Vertices_3D& other_vertices,
                                                       const Vertices_3D& region) {
    assert(my_vertices.is_ex_vertices(region) && other_vertices.is_ex_vertices(region));
    #ifdef ARRAY_UNFOLD
    if (unlikely(my_vertices.ni == region.ni && my_vertices.ni == other_vertices.ni)) {
    // if (false) {
        if (region.get_size() == 0) return;
        Vertices_2D my_vertices_2d(0, (int64_t)(my_vertices.ni * my_vertices.nj) - 1,
                                   my_vertices.ks, my_vertices.get_ke());
        Vertices_2D region_2d(0, (int64_t)(region.ni * region.nj) - 1,
                              region.ks, region.get_ke());
        Vertices_2D other_vertices_2d(0, (int64_t)(other_vertices.ni * other_vertices.nj) - 1,
                                      other_vertices.ks, other_vertices.get_ke());
        uint64_t region_index_of_my_vertices = my_vertices.get_index_nocheck(region.is,
                                                                         region.js,
                                                                         region.ks);
        uint64_t region_i_of_my_vertices = my_vertices_2d.get_i_nocheck(region_index_of_my_vertices);
        my_vertices_2d.is = -(int64_t)region_i_of_my_vertices;
        uint64_t region_index_of_other_vertices_2d = other_vertices.get_index_nocheck(region.is,
                                                                                  region.js,
                                                                                  region.ks);
        uint64_t region_i_of_other_vertices_2d = other_vertices_2d.get_i_nocheck(region_index_of_other_vertices_2d);
        other_vertices_2d.is = -(int64_t)region_i_of_other_vertices_2d;
        Vertices_method::fill_region(my_data, my_vertices_2d, other_data, other_vertices_2d, region_2d);
    } else {
    #endif //ARRAY_UNFOLD
        const int64_t my_index_origin = my_vertices.get_index_nocheck(region.is, region.js, region.ks);
        const int64_t my_ni = my_vertices.ni;
        const int64_t my_ninj = my_vertices.ni * my_vertices.nj;
        const int64_t region_ni = (int64_t)region.ni;
        const int64_t region_nj = (int64_t)region.nj;
        const int64_t region_nk = (int64_t)region.nk;
        const int64_t other_offset_origin = other_vertices.get_index_nocheck(region.is, region.js, region.ks);
        const int64_t other_ni = other_vertices.ni;
        const int64_t other_ninj = other_vertices.ni * other_vertices.nj;
        #ifdef USE_MEMFUNS
        const int64_t memcpy_size = sizeof(T) * region_ni;
        #endif //USE_MEMFUNS
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region_nk - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t k = 0; k < region_nk; ++k) {
            T const* __restrict__ my_data_j = my_data + my_index_origin + k * my_ninj;
            T* __restrict__ other_data_j = other_data + other_offset_origin + k * other_ninj;
            for (int64_t j = 0; j < region_nj; ++j) {
                T const* __restrict__ my_data_i = my_data_j;
                T* __restrict__ other_data_i = other_data_j;
                #ifdef USE_MEMFUNS
                std::memcpy(other_data_i, my_data_i, memcpy_size);
                #else
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif //USE_OPENMP_SIMD
                for (int64_t i = 0; i < region_ni; ++i) {
                    *other_data_i++ = *my_data_i++;
                }
                #endif //USE_MEMFUNS
                my_data_j += my_ni;
                other_data_j += other_ni;
            }
        }
    #ifdef ARRAY_UNFOLD
    }
    #endif //ARRAY_UNFOLD
    return;
}
template void Vertices_method::fill_region<int>(int const* const my_data, const Vertices_3D& my_vertices,
                                                int* const other_data, const Vertices_3D& other_vertices,
                                                const Vertices_3D& region);
template void Vertices_method::fill_region<float>(float const* const my_data, const Vertices_3D& my_vertices,
                                                  float* const other_data, const Vertices_3D& other_vertices,
                                                  const Vertices_3D& region);
template void Vertices_method::fill_region<double>(double const* const my_data, const Vertices_3D& my_vertices,
                                                   double* const other_data, const Vertices_3D& other_vertices,
                                                   const Vertices_3D& region);
#ifdef FP16_FLAG
template void Vertices_method::fill_region<__fp16>(__fp16 const* const my_data, const Vertices_3D& my_vertices, 
                                                   __fp16* const other_data, const Vertices_3D& other_vertices,
                                                   const Vertices_3D& region);
#endif //FP16_FLAG

template<typename T> void Vertices_method::fill_region(T const* const __restrict__ my_data, const Vertices_4D& my_vertices,
                                                       T* const __restrict__ other_data, const Vertices_4D& other_vertices,
                                                       const Vertices_4D& region) {
    assert(my_vertices.is_ex_vertices(region) && other_vertices.is_ex_vertices(region));
    #ifdef ARRAY_UNFOLD
    if (unlikely(my_vertices.ni == region.ni && my_vertices.ni == other_vertices.ni)) {
    // if (false) {
        if (region.get_size() == 0) return;
        Vertices_3D my_vertices_3d(0, (int64_t)(my_vertices.ni * my_vertices.nj) - 1,
                                   my_vertices.ks, my_vertices.get_ke(),
                                   my_vertices.bs, my_vertices.get_be());
        Vertices_3D region_3d(0, (int64_t)(region.ni * region.nj) - 1,
                              region.ks, region.get_ke(),
                              region.bs, region.get_be());
        Vertices_3D other_vertices_3d(0, (int64_t)(other_vertices.ni * other_vertices.nj) - 1,
                                      other_vertices.ks, other_vertices.get_ke(),
                                      other_vertices.bs, other_vertices.get_be());
        uint64_t region_index_of_my_vertices = my_vertices.get_index_nocheck(region.is,
                                                                         region.js,
                                                                         region.ks,
                                                                         region.bs);
        uint64_t region_i_of_my_vertices = my_vertices_3d.get_i_nocheck(region_index_of_my_vertices);
        my_vertices_3d.is = -(int64_t)region_i_of_my_vertices;
        uint64_t region_index_of_other_vertices_3d = other_vertices.get_index_nocheck(region.is,
                                                                                     region.js,
                                                                                     region.ks,
                                                                                     region.bs);
        uint64_t region_i_of_other_vertices_3d = other_vertices_3d.get_i_nocheck(region_index_of_other_vertices_3d);
        other_vertices_3d.is = -(int64_t)region_i_of_other_vertices_3d;
        Vertices_method::fill_region(my_data, my_vertices_3d, other_data, other_vertices_3d, region_3d);
    } else {
    #endif //ARRAY_UNFOLD
        const int64_t my_index_origin = my_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
        const int64_t my_ni = my_vertices.ni;
        const int64_t my_ninj = my_vertices.ni * my_vertices.nj;
        const int64_t my_ninjnk = my_vertices.ni * my_vertices.nj * my_vertices.nk;
        const int64_t region_ni = (int64_t)region.ni;
        const int64_t region_nj = (int64_t)region.nj;
        const int64_t region_nk = (int64_t)region.nk;
        const int64_t region_nb = (int64_t)region.nb;
        const int64_t other_offset_origin = other_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
        const int64_t other_ni = other_vertices.ni;
        const int64_t other_ninj = other_vertices.ni * other_vertices.nj;
        const int64_t other_ninjnk = other_vertices.ni * other_vertices.nj * other_vertices.nk;
        #ifdef USE_MEMFUNS
        const int64_t memcpy_size = sizeof(T) * region_ni;
        #endif //USE_MEMFUNS
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region_nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < region_nb; b++) {
            T const* __restrict__ my_data_k = my_data + my_index_origin + b * my_ninjnk;
            T* __restrict__ other_data_k = other_data + other_offset_origin + b * other_ninjnk;
            for (int64_t k = 0; k < region_nk; ++k) {
                T const* __restrict__ my_data_j = my_data_k;
                T* __restrict__ other_data_j = other_data_k;
                for (int64_t j = 0; j < region_nj; ++j) {
                    T const* __restrict__ my_data_i = my_data_j;
                    T* __restrict__ other_data_i = other_data_j;
                    #ifdef USE_MEMFUNS
                    std::memcpy(other_data_i, my_data_i, memcpy_size);
                    #else
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif //USE_OPENMP_SIMD
                    for (int64_t i = 0; i < region_ni; ++i) {
                        *other_data_i++ = *my_data_i++;
                    }
                    #endif //USE_MEMFUNS
                    my_data_j += my_ni;
                    other_data_j += other_ni;
                }
                my_data_k += my_ninj;
                other_data_k += other_ninj;
            }
        }
    #ifdef ARRAY_UNFOLD
    }
    #endif //ARRAY_UNFOLD
    return;
}
template void Vertices_method::fill_region<int>(int const* const my_data, const Vertices_4D& my_vertices,
                                                int* const other_data, const Vertices_4D& other_vertices,
                                                const Vertices_4D& region);
template void Vertices_method::fill_region<float>(float const* const my_data, const Vertices_4D& my_vertices,
                                                  float* const other_data, const Vertices_4D& other_vertices,
                                                  const Vertices_4D& region);
template void Vertices_method::fill_region<double>(double const* const my_data, const Vertices_4D& my_vertices,
                                                   double* const other_data, const Vertices_4D& other_vertices,
                                                   const Vertices_4D& region);
#ifdef FP16_FLAG
template void Vertices_method::fill_region<__fp16>(__fp16 const* const my_data, const Vertices_4D& my_vertices, 
                                                   __fp16* const other_data, const Vertices_4D& other_vertices,
                                                   const Vertices_4D& region);
#endif //FP16_FLAG

template<typename T> void Vertices_method::accumulate_overlap(T const* const& __restrict__ my_data, const Vertices_3D& my_vertices,
                                                              T* const& __restrict__ other_data, const Vertices_3D& other_vertices,
                                                              const Vertices_3D& region) {
    assert(my_vertices.is_ex_vertices(region) && other_vertices.is_ex_vertices(region));
    const int64_t my_index_origin = my_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t my_ni = my_vertices.ni;
    const int64_t my_ninj = my_vertices.ni * my_vertices.nj;
    const int64_t region_ni = (int64_t)region.ni;
    const int64_t region_nj = (int64_t)region.nj;
    const int64_t region_nk = (int64_t)region.nk;
    const int64_t other_offset_origin = other_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t other_ni = other_vertices.ni;
    const int64_t other_ninj = other_vertices.ni * other_vertices.nj;
    #ifdef USE_OPENMP
    #pragma omp for schedule(static, (region_nk - 1)/omp_get_num_threads() + 1) nowait
    #endif //USE_OPENMP
    for (int64_t k = 0; k < region_nk; ++k) {
        T const* __restrict__ my_data_j = my_data + my_index_origin + k * my_ninj;
        T* __restrict__ other_data_j = other_data + other_offset_origin + k * other_ninj;
        for (int64_t j = 0; j < region_nj; ++j) {
            T const* __restrict__ my_data_i = my_data_j;
            T* __restrict__ other_data_i = other_data_j;
            #ifdef USE_OPENMP_SIMD
            #pragma omp simd
            #endif //USE_OPENMP_SIMD
            for (int64_t i = 0; i < region_ni; ++i) {
                *other_data_i++ += *my_data_i++;
            }
            my_data_j += my_ni;
            other_data_j += other_ni;
        }
    }
    return;
}
template void Vertices_method::accumulate_overlap<int>(int const* const& my_data, const Vertices_3D& my_vertices, 
                                                       int* const& other_data, const Vertices_3D& other_vertices,
                                                       const Vertices_3D& region);
template void Vertices_method::accumulate_overlap<float>(float const* const& my_data, const Vertices_3D& my_vertices, 
                                                         float* const& other_data, const Vertices_3D& other_vertices,
                                                         const Vertices_3D& region);
template void Vertices_method::accumulate_overlap<double>(double const* const& my_data, const Vertices_3D& my_vertices,
                                                          double* const& other_data, const Vertices_3D& other_vertices,
                                                          const Vertices_3D& region);
#ifdef FP16_FLAG
template void Vertices_method::accumulate_overlap<__fp16>(__fp16 const* const& my_data, const Vertices_3D& my_vertices, 
                                                          __fp16* const& other_data, const Vertices_3D& other_vertices,
                                                          const Vertices_3D& region);
#endif //FP16_FLAG

template<typename T> void Vertices_method::accumulate_overlap(T const* const& __restrict__ my_data, const Vertices_4D& my_vertices,
                                                              T* const& __restrict__ other_data, const Vertices_4D& other_vertices,
                                                              const Vertices_4D& region) {
    assert(my_vertices.is_ex_vertices(region) && other_vertices.is_ex_vertices(region));
    const int64_t my_index_origin = my_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t my_ni = my_vertices.ni;
    const int64_t my_ninj = my_vertices.ni * my_vertices.nj;
    const int64_t my_ninjnk = my_vertices.ni * my_vertices.nj * my_vertices.nk;
    const int64_t region_ni = (int64_t)region.ni;
    const int64_t region_nj = (int64_t)region.nj;
    const int64_t region_nk = (int64_t)region.nk;
    const int64_t region_nb = (int64_t)region.nb;
    const int64_t other_offset_origin = other_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t other_ni = other_vertices.ni;
    const int64_t other_ninj = other_vertices.ni * other_vertices.nj;
    const int64_t other_ninjnk = other_vertices.ni * other_vertices.nj * other_vertices.nk;
    #ifdef USE_OPENMP
    #pragma omp for schedule(static, (region_nb - 1)/omp_get_num_threads() + 1) nowait
    #endif //USE_OPENMP
    for (int64_t b = 0; b < region_nb; ++b) {
        T const* __restrict__ my_data_k = my_data + my_index_origin + b * my_ninjnk;
        T* __restrict__ other_data_k = other_data + other_offset_origin + b * other_ninjnk;
        for (int64_t k = 0; k < region_nk; ++k) {
            T const* __restrict__ my_data_j = my_data_k + k * my_ninj;
            T* __restrict__ other_data_j = other_data_k + k * other_ninj;
            for (int64_t j = 0; j < region_nj; ++j) {
                T const* __restrict__ my_data_i = my_data_j;
                T* __restrict__ other_data_i = other_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif //USE_OPENMP_SIMD
                for (int64_t i = 0; i < region_ni; ++i) {
                    *other_data_i++ += *my_data_i++;
                }
                my_data_j += my_ni;
                other_data_j += other_ni;
            }
        }
    }
    return;
}
template void Vertices_method::accumulate_overlap<int>(int const* const& my_data, const Vertices_4D& my_vertices, 
                                                       int* const& other_data, const Vertices_4D& other_vertices,
                                                       const Vertices_4D& region);
template void Vertices_method::accumulate_overlap<float>(float const* const& my_data, const Vertices_4D& my_vertices, 
                                                         float* const& other_data, const Vertices_4D& other_vertices,
                                                         const Vertices_4D& region);
template void Vertices_method::accumulate_overlap<double>(double const* const& my_data, const Vertices_4D& my_vertices,
                                                          double* const& other_data, const Vertices_4D& other_vertices,
                                                          const Vertices_4D& region);
#ifdef FP16_FLAG
template void Vertices_method::accumulate_overlap<__fp16>(__fp16 const* const& my_data, const Vertices_4D& my_vertices, 
                                                          __fp16* const& other_data, const Vertices_4D& other_vertices,
                                                          const Vertices_4D& region);
#endif //FP16_FLAG
