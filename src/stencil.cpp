#include <cstring>
#include <iomanip>
#include "stencil.h"
#include "linalg.h"

template<typename T>
Stencil<T>::Stencil() {
    this->cell_type = 0;
    this->order = 0;
    this->FDn = 0;
}

template<typename T>
Stencil<T>::Stencil(const int& order, const uint& cell_type) {
    this->set_order(order);
    this->set_cell_type(cell_type);
}

template<typename T>
Stencil<T>::Stencil(const Stencil& other) {
    this->deepcopy(other);
}

template<typename T>
Stencil<T>::Stencil(Stencil&& other) {
    this->deepcopy(std::move(other));
}

template<typename T>
Stencil<T>::~Stencil() {
    this->destructor();
}

template<typename T>
Stencil<T>& Stencil<T>::operator=(const Stencil& other) {
    this->deepcopy(other);
    return *this;
}

template<typename T>
Stencil<T>& Stencil<T>::deepcopy(const Stencil& other) {
    if (this != &other) {
        this->cell_type = other.cell_type;
        this->optimization = other.optimization;
        this->FDn = other.FDn;
        this->order = other.order;
        if (other.D1_coeffs_x != nullptr) {
            if (this->D1_coeffs_x != nullptr) {
                ::operator delete[](this->D1_coeffs_x, std::align_val_t(64));
            }
            this->D1_coeffs_x = new (std::align_val_t(64)) T [this->FDn + 1];
            Linalg::set_value_general(this->D1_coeffs_x, other.D1_coeffs_x, this->FDn + 1);
        }
        if (other.D1_coeffs_y != nullptr) {
            if (this->D1_coeffs_y != nullptr) {
                ::operator delete[](this->D1_coeffs_y, std::align_val_t(64));
            }
            this->D1_coeffs_y = new (std::align_val_t(64)) T [this->FDn + 1];
            Linalg::set_value_general(this->D1_coeffs_y, other.D1_coeffs_y, this->FDn + 1);
        }
        if (other.D1_coeffs_z != nullptr) {
            if (this->D1_coeffs_z != nullptr) {
                ::operator delete[](this->D1_coeffs_z, std::align_val_t(64));
            }
            this->D1_coeffs_z = new (std::align_val_t(64)) T [this->FDn + 1];
            Linalg::set_value_general(this->D1_coeffs_z, other.D1_coeffs_z, this->FDn + 1);
        }
        if (other.D2_coeffs_x != nullptr) {
            if (this->D2_coeffs_x != nullptr) {
                ::operator delete[](this->D2_coeffs_x, std::align_val_t(64));
            }
            this->D2_coeffs_x = new (std::align_val_t(64)) T [this->FDn + 1];
            Linalg::set_value_general(this->D2_coeffs_x, other.D2_coeffs_x, this->FDn + 1);
        }
        if (other.D2_coeffs_y != nullptr) {
            if (this->D2_coeffs_y != nullptr) {
                ::operator delete[](this->D2_coeffs_y, std::align_val_t(64));
            }
            this->D2_coeffs_y = new (std::align_val_t(64)) T [this->FDn + 1];
            Linalg::set_value_general(this->D2_coeffs_y, other.D2_coeffs_y, this->FDn + 1);
        }
        if (other.D2_coeffs_z != nullptr) {
            if (this->D2_coeffs_z != nullptr) {
                ::operator delete[](this->D2_coeffs_z, std::align_val_t(64));
            }
            this->D2_coeffs_z = new (std::align_val_t(64)) T [this->FDn + 1];
            Linalg::set_value_general(this->D2_coeffs_z, other.D2_coeffs_z, this->FDn + 1);
        }
        if (other.D2_coeffs_xyz != nullptr) {
            if (this->D2_coeffs_xyz != nullptr) {
                ::operator delete[](this->D2_coeffs_xyz, std::align_val_t(64));
            }
            this->D2_coeffs_xyz = new (std::align_val_t(64)) T [this->get_D2_coeffs_xyz_length()];
            Linalg::set_value_general(this->D2_coeffs_xyz, other.D2_coeffs_xyz, this->get_D2_coeffs_xyz_length());
        }
        // this->D1_coeffs_x = other.D1_coeffs_x;
        // this->D1_coeffs_y = other.D1_coeffs_y;
        // this->D1_coeffs_z = other.D1_coeffs_z;
        // this->D2_coeffs_x = other.D2_coeffs_x;
        // this->D2_coeffs_y = other.D2_coeffs_y;
        // this->D2_coeffs_z = other.D2_coeffs_z;
        // this->D2_coeffs_xyz = other.D2_coeffs_xyz;
    }
    return *this;
}

template<typename T>
Stencil<T>& Stencil<T>::deepcopy_mp(const Stencil& other, Memory_pool<T, Fast_memory>& pool_fast) {
    if (this != &other) {
        this->cell_type = other.cell_type;
        this->optimization = other.optimization;
        this->FDn = other.FDn;
        this->order = other.order;
        if (other.D1_coeffs_x != nullptr) {
            if (this->D1_coeffs_x != nullptr) {
                ::operator delete[](this->D1_coeffs_x, std::align_val_t(64));
            }
            this->D1_coeffs_x = pool_fast.allocate(this->FDn + 1);
            Linalg::set_value_general(this->D1_coeffs_x, other.D1_coeffs_x, this->FDn + 1);
        }
        if (other.D1_coeffs_y != nullptr) {
            if (this->D1_coeffs_y != nullptr) {
                ::operator delete[](this->D1_coeffs_y, std::align_val_t(64));
            }
            this->D1_coeffs_y = pool_fast.allocate(this->FDn + 1);
            Linalg::set_value_general(this->D1_coeffs_y, other.D1_coeffs_y, this->FDn + 1);
        }
        if (other.D1_coeffs_z != nullptr) {
            if (this->D1_coeffs_z != nullptr) {
                ::operator delete[](this->D1_coeffs_z, std::align_val_t(64));
            }
            this->D1_coeffs_z = pool_fast.allocate(this->FDn + 1);
            Linalg::set_value_general(this->D1_coeffs_z, other.D1_coeffs_z, this->FDn + 1);
        }
        if (other.D2_coeffs_x != nullptr) {
            if (this->D2_coeffs_x != nullptr) {
                ::operator delete[](this->D2_coeffs_x, std::align_val_t(64));
            }
            this->D2_coeffs_x = pool_fast.allocate(this->FDn + 1);
            Linalg::set_value_general(this->D2_coeffs_x, other.D2_coeffs_x, this->FDn + 1);
        }
        if (other.D2_coeffs_y != nullptr) {
            if (this->D2_coeffs_y != nullptr) {
                ::operator delete[](this->D2_coeffs_y, std::align_val_t(64));
            }
            this->D2_coeffs_y = pool_fast.allocate(this->FDn + 1);
            Linalg::set_value_general(this->D2_coeffs_y, other.D2_coeffs_y, this->FDn + 1);
        }
        if (other.D2_coeffs_z != nullptr) {
            if (this->D2_coeffs_z != nullptr) {
                ::operator delete[](this->D2_coeffs_z, std::align_val_t(64));
            }
            this->D2_coeffs_z = pool_fast.allocate(this->FDn + 1);
            Linalg::set_value_general(this->D2_coeffs_z, other.D2_coeffs_z, this->FDn + 1);
        }
        if (other.D2_coeffs_xyz != nullptr) {
            if (this->D2_coeffs_xyz != nullptr) {
                ::operator delete[](this->D2_coeffs_xyz, std::align_val_t(64));
            }
            this->D2_coeffs_xyz = pool_fast.allocate(this->get_D2_coeffs_xyz_length());
            Linalg::set_value_general(this->D2_coeffs_xyz, other.D2_coeffs_xyz, this->get_D2_coeffs_xyz_length());
        }
        // this->D1_coeffs_x = other.D1_coeffs_x;
        // this->D1_coeffs_y = other.D1_coeffs_y;
        // this->D1_coeffs_z = other.D1_coeffs_z;
        // this->D2_coeffs_x = other.D2_coeffs_x;
        // this->D2_coeffs_y = other.D2_coeffs_y;
        // this->D2_coeffs_z = other.D2_coeffs_z;
        // this->D2_coeffs_xyz = other.D2_coeffs_xyz;
    }
    return *this;
}

template<typename T>
Stencil<T>& Stencil<T>::deepcopy(Stencil&& other) {
    this->cell_type = other.cell_type;
    this->optimization = other.optimization;
    this->FDn = other.FDn;
    this->order = other.order;
    // this->D1_coeffs_x.swap(other.D1_coeffs_x);
    // this->D1_coeffs_y.swap(other.D1_coeffs_y);
    // this->D1_coeffs_z.swap(other.D1_coeffs_z);
    // this->D2_coeffs_x.swap(other.D2_coeffs_x);
    // this->D2_coeffs_y.swap(other.D2_coeffs_y);
    // this->D2_coeffs_z.swap(other.D2_coeffs_z);
    // this->D2_coeffs_xyz.swap(other.D2_coeffs_xyz);
    std::swap(this->D1_coeffs_x, other.D1_coeffs_x);
    std::swap(this->D1_coeffs_y, other.D1_coeffs_y);
    std::swap(this->D1_coeffs_z, other.D1_coeffs_z);
    std::swap(this->D2_coeffs_x, other.D2_coeffs_x);
    std::swap(this->D2_coeffs_y, other.D2_coeffs_y);
    std::swap(this->D2_coeffs_z, other.D2_coeffs_z);
    std::swap(this->D2_coeffs_xyz, other.D2_coeffs_xyz);
    return *this;
}

template<typename T>
void Stencil<T>::set_cell_type(const uint& cell_type) {
    this->cell_type = cell_type;
    return;
}

template<typename T>
void Stencil<T>::set_optimization(const uint& optimization) {
    this->optimization = optimization;
    return;
}

/**
 * @brief set finite-difference order
 * 
 * @param order finite-difference order
 */
template<typename T>
void Stencil<T>::set_order(const int& order) {
    this->order = order;
    this->FDn = order/2;
    return;
}

/**
 * @brief set_FDn
 * 
 * @param FDn 
 */
template<typename T>
void Stencil<T>::set_FDn(const int& FDn) {
    this->order = FDn * 2;
    this->FDn = FDn;
    return;
}

/**
 * @brief calculate 1st derivative weights including mesh of x,y,z directions
 * 
 * @param deltas mesh spacing in {x,y,z} direction
 */
template<typename T>
void Stencil<T>::set_D1_coeffs(double* deltas) {
    set_D1_coeffs(deltas[0], deltas[1], deltas[2]);
    return;
}

/**
 * @brief calculate 1st derivative weights including mesh of x,y,z directions
 * 
 * @param dx mesh spacing in x direction
 * @param dy mesh spacing in y direction
 * @param dz mesh spacing in z direction
 */
template<typename T>
void Stencil<T>::set_D1_coeffs(double dx, double dy, double dz) {
    if (this->cell_type <= 2) {
        if (this->optimization == 4) {
            // this->D1_coeffs_x.resize(this->FDn + 1);
            this->D1_coeffs_x = new (std::align_val_t(64)) T [this->FDn + 1];
            std::vector<double> FDweights_D1(this->FDn + 1);
            Stencil_method::set_FDweights_D1(this->FDn, FDweights_D1.data());
            Stencil_method::set_D1_coeffs(FDn, dx, FDweights_D1.data(), this->D1_coeffs_x);
        } else {
            // this->D1_coeffs_x.resize(this->FDn + 1);
            // this->D1_coeffs_y.resize(this->FDn + 1);
            // this->D1_coeffs_z.resize(this->FDn + 1);
            this->D1_coeffs_x = new (std::align_val_t(64)) T [this->FDn + 1];
            this->D1_coeffs_y = new (std::align_val_t(64)) T [this->FDn + 1];
            this->D1_coeffs_z = new (std::align_val_t(64)) T [this->FDn + 1];
            std::vector<double> FDweights_D1(this->FDn + 1);
            Stencil_method::set_FDweights_D1(this->FDn, FDweights_D1.data());
            Stencil_method::set_D1_coeffs(FDn, dx, FDweights_D1.data(), this->D1_coeffs_x);
            Stencil_method::set_D1_coeffs(FDn, dy, FDweights_D1.data(), this->D1_coeffs_y);
            Stencil_method::set_D1_coeffs(FDn, dz, FDweights_D1.data(), this->D1_coeffs_z);
        }
    } else {
        assert(this->cell_type <= 2);
    }
    return;
}

/**
 * @brief calculate 2nd derivative weights including mesh of x,y,z directions
 * 
 * @param deltas mesh spacing in {x,y,z} direction
 */
template<typename T>
void Stencil<T>::set_D2_coeffs(double* deltas) {
    set_D2_coeffs(deltas[0], deltas[1], deltas[2]);
    return;
}

/**
 * @brief calculate 2nd derivative weights including mesh of x,y,z directions
 * 
 * @param dx mesh spacing in x direction
 * @param dy mesh spacing in y direction
 * @param dz mesh spacing in z direction
 */
template<typename T>
void Stencil<T>::set_D2_coeffs(double dx, double dy, double dz) {
    if (this->cell_type <= 2) {
        if (this->optimization == 4) {
            // this->D2_coeffs_x.resize(this->FDn + 1);
            this->D2_coeffs_x = new (std::align_val_t(64)) T [this->FDn + 1];
            std::vector<double> FDweights_D2(this->FDn + 1);
            Stencil_method::set_FDweights_D2(this->FDn, FDweights_D2.data());
            // Stencil_method::set_D2_coeffs(FDn, dx, FDweights_D2.data(), this->D2_coeffs_x.data());
            Stencil_method::set_D2_coeffs(FDn, dx, FDweights_D2.data(), this->D2_coeffs_x);
            // this->D2_coeffs_xyz = this->D2_coeffs_x;
            this->D2_coeffs_xyz = new (std::align_val_t(64)) T [this->FDn + 1];
            for (int i = 0; i <= this->FDn; i++) {
                this->D2_coeffs_xyz[i] = this->D2_coeffs_x[i];
            }
            this->D2_coeffs_xyz[0] *= 3.0;
        } else {
            // this->D2_coeffs_x.resize(this->FDn + 1);
            // this->D2_coeffs_y.resize(this->FDn + 1);
            // this->D2_coeffs_z.resize(this->FDn + 1);
            // this->D2_coeffs_xyz.resize(this->FDn * 3 + 3);
            this->D2_coeffs_x = new (std::align_val_t(64)) T [this->FDn + 1];
            this->D2_coeffs_y = new (std::align_val_t(64)) T [this->FDn + 1];
            this->D2_coeffs_z = new (std::align_val_t(64)) T [this->FDn + 1];
            this->D2_coeffs_xyz = new (std::align_val_t(64)) T [this->FDn * 3 + 3];
            std::vector<double> FDweights_D2(this->FDn + 1);
            Stencil_method::set_FDweights_D2(this->FDn, FDweights_D2.data());
            Stencil_method::set_D2_coeffs(FDn, dx, FDweights_D2.data(), this->D2_coeffs_x);
            Stencil_method::set_D2_coeffs(FDn, dy, FDweights_D2.data(), this->D2_coeffs_y);
            Stencil_method::set_D2_coeffs(FDn, dz, FDweights_D2.data(), this->D2_coeffs_z);
            for (int i = 0; i <= this->FDn; i++) {
                this->D2_coeffs_xyz[3 * i]     = this->D2_coeffs_x[i];
                this->D2_coeffs_xyz[3 * i + 1] = this->D2_coeffs_y[i];
                this->D2_coeffs_xyz[3 * i + 2] = this->D2_coeffs_z[i];
            }
        }
    } else {
        assert(this->cell_type <= 2);
    }
    return;
}

template<typename T>
T const* Stencil<T>::get_D1_coeffs_x() const {
    assert(this->cell_type <= 2);
    // return this->D2_coeffs_x.data();
    return this->D1_coeffs_x;
}

template<typename T>
T const* Stencil<T>::get_D1_coeffs_y() const {
    assert(this->cell_type <= 2);
    if (this->optimization == 4) {
        // return this->D2_coeffs_x.data();
        return this->D1_coeffs_x;
    } else {
        // return this->D2_coeffs_y.data();
        return this->D1_coeffs_y;
    }
}

template<typename T>
T const* Stencil<T>::get_D1_coeffs_z() const {
    assert(this->cell_type <= 2);
    if (this->optimization == 4) {
        // return this->D2_coeffs_x.data();
        return this->D1_coeffs_x;
    } else {
        // return this->D2_coeffs_z.data();
        return this->D1_coeffs_z;
    }
}

template<typename T>
T const* Stencil<T>::get_D2_coeffs_x() const {
    assert(this->cell_type <= 2);
    // return this->D2_coeffs_x.data();
    return this->D2_coeffs_x;
}

template<typename T>
T const* Stencil<T>::get_D2_coeffs_y() const {
    assert(this->cell_type <= 2);
    if (this->optimization == 4) {
        // return this->D2_coeffs_x.data();
        return this->D2_coeffs_x;
    } else {
        // return this->D2_coeffs_y.data();
        return this->D2_coeffs_y;
    }
}

template<typename T>
T const* Stencil<T>::get_D2_coeffs_z() const {
    assert(this->cell_type <= 2);
    if (this->optimization == 4) {
        // return this->D2_coeffs_x.data();
        return this->D2_coeffs_x;
    } else {
        // return this->D2_coeffs_z.data();
        return this->D2_coeffs_z;
    }
}

template<typename T>
T const* Stencil<T>::get_D2_coeffs_xyz() const {
    return this->D2_coeffs_xyz;
}

template<typename T>
uint Stencil<T>::get_D2_coeffs_xyz_length() const {
    return this->optimization == 4 ? this->FDn + 1 : 3 * (this->FDn + 1);
}

template<typename T>
T Stencil<T>::get_D2_coef0() const {
    if (this->cell_type <= 2) {
        if (this->optimization == 4) {
            return this->D2_coeffs_xyz[0];
        } else {
            return this->D2_coeffs_xyz[0] + this->D2_coeffs_xyz[1] + this->D2_coeffs_xyz[2];
        }
    } else {
        assert(this->cell_type <= 2);
        return this->D2_coeffs_xyz[0];
    }
}

template<typename T>
Stencil<T>& Stencil<T>::operator*=(const double& ratio) {
    if (this->D1_coeffs_x != nullptr) Linalg::scalar_product_general(this->D1_coeffs_x, (T)ratio, this->FDn + 1);
    if (this->D1_coeffs_y != nullptr) Linalg::scalar_product_general(this->D1_coeffs_y, (T)ratio, this->FDn + 1);
    if (this->D1_coeffs_z != nullptr) Linalg::scalar_product_general(this->D1_coeffs_z, (T)ratio, this->FDn + 1);
    if (this->D2_coeffs_x != nullptr) Linalg::scalar_product_general(this->D2_coeffs_x, (T)ratio, this->FDn + 1);
    if (this->D2_coeffs_y != nullptr) Linalg::scalar_product_general(this->D2_coeffs_y, (T)ratio, this->FDn + 1);
    if (this->D2_coeffs_z != nullptr) Linalg::scalar_product_general(this->D2_coeffs_z, (T)ratio, this->FDn + 1);
    if (this->D2_coeffs_xyz != nullptr) Linalg::scalar_product_general(this->D2_coeffs_xyz, (T)ratio, this->get_D2_coeffs_xyz_length());
    return *this;
}

template<typename T>
Stencil<T> Stencil<T>::operator*(const double& ratio) {
    Stencil<T> __result(*this);
    __result *= ratio;
    return __result;
}

template<typename T>
Stencil<T> Stencil<T>::coeffs_scale(const double& ratio) const {
    Stencil<T> __result(*this);
    __result *= ratio;
    return __result;
}

template<typename T>
Stencil<T> Stencil<T>::coeffs_scale(const double& ratio, const int& dim) const {
    Stencil<T> result(this->order, this->cell_type);
    result.optimization = this->optimization;
    if (dim == 1) {
        assert(this->D1_coeffs_x != nullptr);
        result.D1_coeffs_x = new (std::align_val_t(64)) T [this->FDn + 1];
        Linalg::set_value_general(result.D1_coeffs_x, this->D1_coeffs_x, this->FDn + 1);
        if (this->optimization != 4) {
            assert(this->D1_coeffs_y != nullptr);
            assert(this->D1_coeffs_z != nullptr);
            result.D1_coeffs_y = new (std::align_val_t(64)) T [this->FDn + 1];
            Linalg::set_value_general(result.D1_coeffs_y, this->D1_coeffs_y, this->FDn + 1);
            result.D1_coeffs_z = new (std::align_val_t(64)) T [this->FDn + 1];
            Linalg::set_value_general(result.D1_coeffs_z, this->D1_coeffs_z, this->FDn + 1);
        }
    } else if (dim == 2) {
        assert(this->D2_coeffs_x != nullptr);
        assert(this->D2_coeffs_xyz != nullptr);
        result.D2_coeffs_x = new (std::align_val_t(64)) T [this->FDn + 1];
        Linalg::set_value_general(result.D2_coeffs_x, this->D2_coeffs_x, this->FDn + 1);
        if (this->optimization != 4) {
            assert(this->D2_coeffs_y != nullptr);
            assert(this->D2_coeffs_z != nullptr);
            result.D2_coeffs_y = new (std::align_val_t(64)) T [this->FDn + 1];
            Linalg::set_value_general(result.D2_coeffs_y, this->D2_coeffs_y, this->FDn + 1);
            result.D2_coeffs_z = new (std::align_val_t(64)) T [this->FDn + 1];
            Linalg::set_value_general(result.D2_coeffs_z, this->D2_coeffs_z, this->FDn + 1);
        }
        result.D2_coeffs_xyz = new (std::align_val_t(64)) T [this->get_D2_coeffs_xyz_length()];
        Linalg::set_value_general(result.D2_coeffs_xyz, this->D2_coeffs_xyz, this->get_D2_coeffs_xyz_length());
    }
    result*=ratio;
    return result;
}

template<typename T>
Stencil<T>& Stencil<T>::coeffs_scale_self(const double& ratio) {
    *this*=ratio;
    return *this;
}

template<typename T>
Stencil<T>& Stencil<T>::shift_D2_coeffs(const T& c) {
    if (this->cell_type <= 2) {
        this->D2_coeffs_xyz[0] += c;
    } else {
        assert(this->cell_type <= 2);
    }
    return *this;
}

template<typename T>
void Stencil<T>::init(const uint& cell_type, const int& order, const double& dx, const double& dy, const double& dz) {
    this->set_cell_type(cell_type);
    this->set_order(order);
    if (cell_type <= 2) {
        double abs_dxy = fabs(dx - dy);
        double abs_dxz = fabs(dx - dz);
        double abs_dyz = fabs(dy - dz);
        if (abs_dxy < 1e-12 && abs_dxz < 1e-12 && abs_dyz < 1e-12) {
            this->set_optimization(4);
        } else if (abs_dxy < 1e-12) {
            this->set_optimization(1);
        } else if (abs_dxz < 1e-12) {
            this->set_optimization(2);
        } else if (abs_dyz < 1e-12) {
            this->set_optimization(3);
        } else {
            this->set_optimization(0);
        }
        this->set_D1_coeffs(dx, dy, dz);
        this->set_D2_coeffs(dx, dy, dz);
    } else {
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}

template<typename T>
template<typename T2>
void Stencil<T>::init(const Stencil<T2>& stencil) {
    this->set_cell_type(stencil.cell_type);
    this->set_order(stencil.order);
    this->set_optimization(stencil.optimization);
    assert(false);
    // if (stencil.D1_coeffs_x.size() > 0) {
    //     this->D1_coeffs_x.resize(stencil.D1_coeffs_x.size());
    //     Linalg::convert_type(this->D1_coeffs_x.data(), stencil.D1_coeffs_x.data(), this->D1_coeffs_x.size());
    // }
    // if (stencil.D1_coeffs_y.size() > 0) {
    //     this->D1_coeffs_y.resize(stencil.D1_coeffs_y.size());
    //     Linalg::convert_type(this->D1_coeffs_y.data(), stencil.D1_coeffs_y.data(), this->D1_coeffs_y.size());
    // }
    // if (stencil.D1_coeffs_z.size() > 0) {
    //     this->D1_coeffs_z.resize(stencil.D1_coeffs_z.size());
    //     Linalg::convert_type(this->D1_coeffs_z.data(), stencil.D1_coeffs_z.data(), this->D1_coeffs_z.size());
    // }
    // if (stencil.D2_coeffs_x.size() > 0) {
    //     this->D2_coeffs_x.resize(stencil.D2_coeffs_x.size());
    //     Linalg::convert_type(this->D2_coeffs_x.data(), stencil.D2_coeffs_x.data(), this->D2_coeffs_x.size());
    // }
    // if (stencil.D2_coeffs_y.size() > 0) {
    //     this->D2_coeffs_y.resize(stencil.D2_coeffs_y.size());
    //     Linalg::convert_type(this->D2_coeffs_y.data(), stencil.D2_coeffs_y.data(), this->D2_coeffs_y.size());
    // }
    // if (stencil.D2_coeffs_z.size() > 0) {
    //     this->D2_coeffs_z.resize(stencil.D2_coeffs_z.size());
    //     Linalg::convert_type(this->D2_coeffs_z.data(), stencil.D2_coeffs_z.data(), this->D2_coeffs_z.size());
    // }
    // if (stencil.D2_coeffs_xyz.size() > 0) {
    //     this->D2_coeffs_xyz.resize(stencil.D2_coeffs_xyz.size());
    //     Linalg::convert_type(this->D2_coeffs_xyz.data(), stencil.D2_coeffs_xyz.data(), this->D2_coeffs_xyz.size());
    // }
    return;
}
template void Stencil<float>::init(const Stencil<float>& stencil);
template void Stencil<double>::init(const Stencil<double>& stencil);
template void Stencil<float>::init(const Stencil<double>& stencil);
template void Stencil<double>::init(const Stencil<float>& stencil);

template<typename T>
void Stencil<T>::destructor() {
    // std::vector<T>().swap(this->D1_coeffs_x);
    // std::vector<T>().swap(this->D1_coeffs_y);
    // std::vector<T>().swap(this->D1_coeffs_z);
    // std::vector<T>().swap(this->D2_coeffs_x);
    // std::vector<T>().swap(this->D2_coeffs_y);
    // std::vector<T>().swap(this->D2_coeffs_z);
    // std::vector<T>().swap(this->D2_coeffs_xyz);
    if (this->D1_coeffs_x != nullptr) {
        ::operator delete[](this->D1_coeffs_x, std::align_val_t(64));
        this->D1_coeffs_x = nullptr;
    }
    if (this->D1_coeffs_y != nullptr) {
        ::operator delete[](this->D1_coeffs_y, std::align_val_t(64));
        this->D1_coeffs_y = nullptr;
    }
    if (this->D1_coeffs_z != nullptr) {
        ::operator delete[](this->D1_coeffs_z, std::align_val_t(64));
        this->D1_coeffs_z = nullptr;
    }
    //
    if (this->D2_coeffs_x != nullptr) {
        ::operator delete[](this->D2_coeffs_x, std::align_val_t(64));
        this->D2_coeffs_x = nullptr;
    }
    if (this->D2_coeffs_y != nullptr) {
        ::operator delete[](this->D2_coeffs_y, std::align_val_t(64));
        this->D2_coeffs_y = nullptr;
    }
    if (this->D2_coeffs_z != nullptr) {
        ::operator delete[](this->D2_coeffs_z, std::align_val_t(64));
        this->D2_coeffs_z = nullptr;
    }
    if (this->D2_coeffs_xyz != nullptr) {
        ::operator delete[](this->D2_coeffs_xyz, std::align_val_t(64));
        this->D2_coeffs_xyz = nullptr;
    }
    return;
}

template<typename T>
void Stencil<T>::destructor_mp() {
    // std::vector<T>().swap(this->D1_coeffs_x);
    // std::vector<T>().swap(this->D1_coeffs_y);
    // std::vector<T>().swap(this->D1_coeffs_z);
    // std::vector<T>().swap(this->D2_coeffs_x);
    // std::vector<T>().swap(this->D2_coeffs_y);
    // std::vector<T>().swap(this->D2_coeffs_z);
    // std::vector<T>().swap(this->D2_coeffs_xyz);
    this->D1_coeffs_x = nullptr;
    this->D1_coeffs_y = nullptr;
    this->D1_coeffs_z = nullptr;
    this->D2_coeffs_x = nullptr;
    this->D2_coeffs_y = nullptr;
    this->D2_coeffs_z = nullptr;
    this->D2_coeffs_xyz = nullptr;
    return;
}

/**
 * @brief print the information of Stencil object
 * 
 */
template<typename T>
void Stencil<T>::show() const {
    int width = 18;
    int precision = 10;
    std::cout << "cell_type = " << this->cell_type << ", order = " << this->order << ", FDn = "  << this->FDn << std::endl;
    std::cout << "optimization = " << this->optimization << std::endl;
    if (this->FDn !=0 ) std::cout << std::setw(2) << std::right << "p";
    // if (this->FDweights_D1 != nullptr) std::cout << std::setw(width) << std::right << "FDweights_D1";
    // if (this->FDweights_D2 != nullptr) std::cout << std::setw(width) << std::right << "FDweights_D2";
    if (this->D1_coeffs_x != nullptr) std::cout << std::setw(width) << std::right << "D1_coeffs_x";
    if (this->D1_coeffs_y != nullptr) std::cout << std::setw(width) << std::right << "D1_coeffs_y";
    if (this->D1_coeffs_z != nullptr) std::cout << std::setw(width) << std::right << "D1_coeffs_z";
    if (this->D2_coeffs_x != nullptr) std::cout << std::setw(width) << std::right << "D2_coeffs_x";
    if (this->D2_coeffs_y != nullptr) std::cout << std::setw(width) << std::right << "D2_coeffs_y";
    if (this->D2_coeffs_z != nullptr) std::cout << std::setw(width) << std::right << "D2_coeffs_z";
    std::cout << std::endl;
    for (int p = 0; p < (int)this->FDn + 1; p++)
    {
        if (this->FDn !=0 ) std::cout << std::setw(2) << p;
        // if (this->FDweights_D1 != nullptr) std::cout << std::setw(width) << std::setprecision(precision) << std::fixed << std::right << this->FDweights_D1[p];
        // if (this->FDweights_D2 != nullptr) std::cout << std::setw(width) << std::setprecision(precision) << std::fixed << std::right << this->FDweights_D2[p];
        if (this->D1_coeffs_x != nullptr) std::cout << std::setw(width) << std::setprecision(precision) << std::fixed << std::right << this->D1_coeffs_x[p];
        if (this->D1_coeffs_y != nullptr) std::cout << std::setw(width) << std::setprecision(precision) << std::fixed << std::right << this->D1_coeffs_y[p];
        if (this->D1_coeffs_z != nullptr) std::cout << std::setw(width) << std::setprecision(precision) << std::fixed << std::right << this->D1_coeffs_z[p];
        if (this->D2_coeffs_x != nullptr) std::cout << std::setw(width) << std::setprecision(precision) << std::fixed << std::right << this->D2_coeffs_x[p];
        if (this->D2_coeffs_y != nullptr) std::cout << std::setw(width) << std::setprecision(precision) << std::fixed << std::right << this->D2_coeffs_y[p];
        if (this->D2_coeffs_z != nullptr) std::cout << std::setw(width) << std::setprecision(precision) << std::fixed << std::right << this->D2_coeffs_z[p];
        std::cout << std::endl;
    }
    if (this->D2_coeffs_xyz != nullptr) std::cout << "D2_coeffs_xyz" << std::endl;
    for (std::size_t p = 0; p < this->get_D2_coeffs_xyz_length(); p++) {
        std::cout << std::setw(width) << std::setprecision(precision) << std::fixed << std::right << this->D2_coeffs_xyz[p]
                                                       << std::endl;
    }
    return;
}

template class Stencil<float>;
template class Stencil<double>;



namespace Stencil_method {

// /**
//  * @brief Calculates the terms involving factorials in finite difference 
//  *          weights calculation.
//  * 
//  * @param n FDn
//  * @param k p
//  * @return double 
//  */
double fract(const int& n, const int& k) {
    int i;
    double Nr=1.0, Dr=1.0, val;
    for(i=n-k+1; i<=n; i++)
        Nr*=i;
    for(i=n+1; i<=n+k; i++)
        Dr*=i;
    val = Nr/Dr;
    return (val);
}

void set_FDweights_D1(const int& FDn, double* FDweights_D1) {
    FDweights_D1[0] = 0.0;
    for (int p = 1; p <= FDn; p++) {
        FDweights_D1[p] = (2*(p%2)-1) * fract(FDn, p) / p;
    }
    return;
}

template<typename T>
void set_D1_coeffs(const int& FDn, const double& delta, const double* FDweights_D1, T* D1_coeffs) {
    double delta_inv = 1.0 / delta;
    D1_coeffs[0] = (T) 0.0;
    for (int p = 1; p <= FDn; p++) {
        D1_coeffs[p] = FDweights_D1[p] * delta_inv;
    }
    return;
}

void set_FDweights_D2(const int& FDn, double* FDweights_D2) {
    FDweights_D2[0] = 0.0;
    for (int p = 1; p <= FDn; p++)
    {
        FDweights_D2[0] -= (2.0/(p*p));
        FDweights_D2[p] = (2*(p%2)-1) * 2 * fract(FDn,p) / (p*p);
    }
    return;
}

template<typename T>
void set_D2_coeffs(const int& FDn, const double& delta, const double* FDweights_D2, T* D2_coeffs) {
    double delta2_inv = 1.0 / (delta * delta);
    for (int p = 0; p <= FDn; p++) {
        D2_coeffs[p] = FDweights_D2[p] * delta2_inv;
    }
    return;
}

/**
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/electrostatics.c#L1682
 */
template<typename T>
void jacobi_preconditioner(const Stencil<T>& stencil, const int& N, const T c, const T *r, T *f) {
    T m_inv;
    m_inv = stencil.get_D2_coef0() + c;
    // m_inv = stencil.D2_coeffs_xyz[0] + stencil.D2_coeffs_xyz[1] + stencil.D2_coeffs_xyz[2] + c;
    if (fabs(m_inv) < 1e-14) {
        m_inv = 1.0;
    }
    m_inv = - 1.0 / m_inv;
    Linalg::scalar_product_general(f, r, m_inv, N);
    return;
}
template void jacobi_preconditioner<float>(const Stencil<float>& stencil, const int& N, const float c, const float *r, float *f);
template void jacobi_preconditioner<double>(const Stencil<double>& stencil, const int& N, const double c, const double *r, double *f);

/**
 * @brief Initialize DST-I preconditioner data for Dirichlet boundary conditions
 * Precomputes the eigenvalues of the Dirichlet discrete Laplacian using DST-I basis
 * and creates FFTW plans for efficient repeated application.
 *
 * Eigenvalue formula (high-order FD):
 *   λ_k = w0 + 2·Σ_{p=1}^{FDn} wp·cos(π·k·p/(N+1))
 * where k = 1, 2, ..., N (1-indexed, no zero mode for Dirichlet)
 */
template<typename T>
void DST_Preconditioner_Data<T>::init(
    const Vertices_3D& local_vertices,
    const Stencil<T>& stencil
    )
{
    const T* w2_x = stencil.get_D2_coeffs_x();
    const T* w2_y = stencil.get_D2_coeffs_y();
    const T* w2_z = stencil.get_D2_coeffs_z();
    int FDn = stencil.FDn;

    this->N1 = local_vertices.get_ni();
    this->N2 = local_vertices.get_nj();
    this->N3 = local_vertices.get_nk();
    this->N = local_vertices.get_size();

    if (N == 0) {
        return;
    }

    // Precompute 1D Laplacian eigenvalues for each direction
    // λ_k = w0 + 2·Σ wp·cos(π·k·p/(N+1)) for k = 1, ..., N
    // Note: These are eigenvalues of L (Laplacian), which are NEGATIVE since w0 < 0
    // For N_d=1 there is one mode (k=1); eigenvalue formula is still valid.
    // N1,N2,N3 >= 1 is guaranteed by the N==0 early-return above.
    std::vector<T> lambda_x(N1), lambda_y(N2), lambda_z(N3);

    // X direction eigenvalues
    for (uint k = 0; k < N1; k++) {
        lambda_x[k] = w2_x[0];
        for (int p = 1; p <= FDn; p++) {
            lambda_x[k] += T(2.0) * w2_x[p] * std::cos(M_PI * (k + 1) * p / (double)(N1 + 1));
        }
    }

    // Y direction eigenvalues
    for (uint k = 0; k < N2; k++) {
        lambda_y[k] = w2_y[0];
        for (int p = 1; p <= FDn; p++) {
            lambda_y[k] += T(2.0) * w2_y[p] * std::cos(M_PI * (k + 1) * p / (double)(N2 + 1));
        }
    }

    // Z direction eigenvalues
    for (uint k = 0; k < N3; k++) {
        lambda_z[k] = w2_z[0];
        for (int p = 1; p <= FDn; p++) {
            lambda_z[k] += T(2.0) * w2_z[p] * std::cos(M_PI * (k + 1) * p / (double)(N3 + 1));
        }
    }

    // Compute 3D eigenvalues of -L_3D (negative Laplacian, for solving -∇²φ = f)
    // λ(-L) = -(λ_k1^x + λ_k2^y + λ_k3^z) where λ^x,y,z are Laplacian eigenvalues (negative)
    // This gives positive eigenvalues since L eigenvalues are negative
    d_hat.resize(N);
    uint count = 0;
    for (uint k3 = 0; k3 < N3; k3++) {
        for (uint k2 = 0; k2 < N2; k2++) {
            for (uint k1 = 0; k1 < N1; k1++) {
                d_hat[count] = -(lambda_x[k1] + lambda_y[k2] + lambda_z[k3]);
                count++;
            }
        }
    }

    work.resize(N);

    #if defined(USE_FFTW) || defined(USE_MKL) || defined(USE_KML)
    // Create 1D DST-I plans only for directions with N_d >= 2.
    // FFTW_RODFT00 requires N >= 2; for N_d=1 the plan stays nullptr and
    // dst_poisson_preconditioner() applies a ×2 multiply instead (the exact
    // 1-element DST-I value under FFTW's RODFT00 convention).

    uint nthread = 1;
    #ifdef USE_OPENMP
    #pragma omp parallel
    nthread = omp_get_num_threads();
    #endif
    this->dst_plans_x.resize(nthread, nullptr);
    this->dst_plans_y.resize(nthread, nullptr);
    this->dst_plans_z.resize(nthread, nullptr);
    if (this->dst_plans_x[0]) {
        std::cerr << "DST: Warning: FFTW plans already initialized, skipping plan creation" << std::endl;
    }

    // Temporary arrays for plan creation (FFTW may overwrite during planning)
    std::vector<T> tmp_x(std::max(N1, 2u)), tmp_y(std::max(N2, 2u)), tmp_z(std::max(N3, 2u));

    if constexpr (std::is_same_v<T, double>) {
        // if (N1 >= 2) dst_plan_x = fftw_plan_r2r_1d(N1, tmp_x.data(), tmp_x.data(), FFTW_RODFT00, FFTW_ESTIMATE);
        // if (N2 >= 2) dst_plan_y = fftw_plan_r2r_1d(N2, tmp_y.data(), tmp_y.data(), FFTW_RODFT00, FFTW_ESTIMATE);
        // if (N3 >= 2) dst_plan_z = fftw_plan_r2r_1d(N3, tmp_z.data(), tmp_z.data(), FFTW_RODFT00, FFTW_ESTIMATE);
        for (uint ithread = 0; ithread < nthread; ithread++) {
             if (N1 >= 2) this->dst_plans_x[ithread] = fftw_plan_r2r_1d(N1, tmp_x.data(), tmp_x.data(), FFTW_RODFT00, FFTW_ESTIMATE);
             if (N2 >= 2) this->dst_plans_y[ithread] = fftw_plan_r2r_1d(N2, tmp_y.data(), tmp_y.data(), FFTW_RODFT00, FFTW_ESTIMATE);
             if (N3 >= 2) this->dst_plans_z[ithread] = fftw_plan_r2r_1d(N3, tmp_z.data(), tmp_z.data(), FFTW_RODFT00, FFTW_ESTIMATE);
        }
    } else {
        // if (N1 >= 2) dst_plan_x = fftwf_plan_r2r_1d(N1, tmp_x.data(), tmp_x.data(), FFTW_RODFT00, FFTW_ESTIMATE);
        // if (N2 >= 2) dst_plan_y = fftwf_plan_r2r_1d(N2, tmp_y.data(), tmp_y.data(), FFTW_RODFT00, FFTW_ESTIMATE);
        // if (N3 >= 2) dst_plan_z = fftwf_plan_r2r_1d(N3, tmp_z.data(), tmp_z.data(), FFTW_RODFT00, FFTW_ESTIMATE);
        for (uint ithread = 0; ithread < nthread; ithread++) {
            if (N1 >= 2) this->dst_plans_x[ithread] = (fftw_plan)fftwf_plan_r2r_1d(N1, tmp_x.data(), tmp_x.data(), FFTW_RODFT00, FFTW_ESTIMATE);
            if (N2 >= 2) this->dst_plans_y[ithread] = (fftw_plan)fftwf_plan_r2r_1d(N2, tmp_y.data(), tmp_y.data(), FFTW_RODFT00, FFTW_ESTIMATE);
            if (N3 >= 2) this->dst_plans_z[ithread] = (fftw_plan)fftwf_plan_r2r_1d(N3, tmp_z.data(), tmp_z.data(), FFTW_RODFT00, FFTW_ESTIMATE);
        }
    }

    // if ((N1 >= 2 && !dst_plan_x) || (N2 >= 2 && !dst_plan_y) || (N3 >= 2 && !dst_plan_z)) {
    if ((N1 >= 2 && !dst_plans_x[0]) || (N2 >= 2 && !dst_plans_y[0]) || (N3 >= 2 && !dst_plans_z[0])) {
        std::cerr << "DST: Failed to create FFTW plans" << std::endl;
    }
    #endif
    return;
}

template<typename T>
template<typename T2>
void DST_Preconditioner_Data<T>::init(const DST_Preconditioner_Data<T2>& DST_preconditioner_data) {
    this->N1 = DST_preconditioner_data.N1;
    this->N2 = DST_preconditioner_data.N2;
    this->N3 = DST_preconditioner_data.N3;
    this->N = DST_preconditioner_data.N;
    this->d_hat.resize(DST_preconditioner_data.d_hat.size());
    Linalg::convert_type(this->d_hat.data(), DST_preconditioner_data.d_hat.data(), this->d_hat.size());
    this->work.resize(DST_preconditioner_data.work.size());
    Linalg::convert_type(this->work.data(), DST_preconditioner_data.work.data(), this->work.size());
    #if defined(USE_FFTW) || defined(USE_MKL) || defined(USE_KML)
    // Create 1D DST-I plans only for directions with N_d >= 2.
    // FFTW_RODFT00 requires N >= 2; for N_d=1 the plan stays nullptr and
    // dst_poisson_preconditioner() applies a ×2 multiply instead (the exact
    // 1-element DST-I value under FFTW's RODFT00 convention).

    uint nthread = 1;
    #ifdef USE_OPENMP
    #pragma omp parallel
    nthread = omp_get_num_threads();
    #endif
    this->dst_plans_x.resize(nthread, nullptr);
    this->dst_plans_y.resize(nthread, nullptr);
    this->dst_plans_z.resize(nthread, nullptr);

    // Temporary arrays for plan creation (FFTW may overwrite during planning)
    std::vector<T> tmp_x(std::max(N1, 2u)), tmp_y(std::max(N2, 2u)), tmp_z(std::max(N3, 2u));

    if constexpr (std::is_same_v<T, double>) {
        // if (N1 >= 2) dst_plan_x = fftw_plan_r2r_1d(N1, tmp_x.data(), tmp_x.data(), FFTW_RODFT00, FFTW_ESTIMATE);
        // if (N2 >= 2) dst_plan_y = fftw_plan_r2r_1d(N2, tmp_y.data(), tmp_y.data(), FFTW_RODFT00, FFTW_ESTIMATE);
        // if (N3 >= 2) dst_plan_z = fftw_plan_r2r_1d(N3, tmp_z.data(), tmp_z.data(), FFTW_RODFT00, FFTW_ESTIMATE);
        for (uint ithread = 0; ithread < nthread; ithread++) {
             if (N1 >= 2) this->dst_plans_x[ithread] = fftw_plan_r2r_1d(N1, tmp_x.data(), tmp_x.data(), FFTW_RODFT00, FFTW_ESTIMATE);
             if (N2 >= 2) this->dst_plans_y[ithread] = fftw_plan_r2r_1d(N2, tmp_y.data(), tmp_y.data(), FFTW_RODFT00, FFTW_ESTIMATE);
             if (N3 >= 2) this->dst_plans_z[ithread] = fftw_plan_r2r_1d(N3, tmp_z.data(), tmp_z.data(), FFTW_RODFT00, FFTW_ESTIMATE);
        }
    } else {
        // if (N1 >= 2) dst_plan_x = fftwf_plan_r2r_1d(N1, tmp_x.data(), tmp_x.data(), FFTW_RODFT00, FFTW_ESTIMATE);
        // if (N2 >= 2) dst_plan_y = fftwf_plan_r2r_1d(N2, tmp_y.data(), tmp_y.data(), FFTW_RODFT00, FFTW_ESTIMATE);
        // if (N3 >= 2) dst_plan_z = fftwf_plan_r2r_1d(N3, tmp_z.data(), tmp_z.data(), FFTW_RODFT00, FFTW_ESTIMATE);
        for (uint ithread = 0; ithread < nthread; ithread++) {
            if (N1 >= 2) this->dst_plans_x[ithread] = (fftw_plan)fftwf_plan_r2r_1d(N1, tmp_x.data(), tmp_x.data(), FFTW_RODFT00, FFTW_ESTIMATE);
            if (N2 >= 2) this->dst_plans_y[ithread] = (fftw_plan)fftwf_plan_r2r_1d(N2, tmp_y.data(), tmp_y.data(), FFTW_RODFT00, FFTW_ESTIMATE);
            if (N3 >= 2) this->dst_plans_z[ithread] = (fftw_plan)fftwf_plan_r2r_1d(N3, tmp_z.data(), tmp_z.data(), FFTW_RODFT00, FFTW_ESTIMATE);
        }
    }

    // if ((N1 >= 2 && !dst_plan_x) || (N2 >= 2 && !dst_plan_y) || (N3 >= 2 && !dst_plan_z)) {
    if ((N1 >= 2 && !dst_plans_x[0]) || (N2 >= 2 && !dst_plans_y[0]) || (N3 >= 2 && !dst_plans_z[0])) {
        std::cerr << "DST: Failed to create FFTW plans" << std::endl;
    }
    #endif
    return;
}
template void DST_Preconditioner_Data<float>::init(const DST_Preconditioner_Data<float>& dst_data);
template void DST_Preconditioner_Data<double>::init(const DST_Preconditioner_Data<double>& dst_data);
template void DST_Preconditioner_Data<float>::init(const DST_Preconditioner_Data<double>& dst_data);
template void DST_Preconditioner_Data<double>::init(const DST_Preconditioner_Data<float>& dst_data);

template<typename T>
void DST_Preconditioner_Data<T>::destructor() {
    #if defined(USE_FFTW) || defined(USE_MKL) || defined(USE_KML)
    uint n_plans_x = dst_plans_x.size();
    uint n_plans_y = dst_plans_y.size();
    uint n_plans_z = dst_plans_z.size();
    if constexpr (std::is_same_v<T, double>) {
        // if (dst_plan_x) { fftw_destroy_plan(dst_plan_x); dst_plan_x = nullptr; }
        // if (dst_plan_y) { fftw_destroy_plan(dst_plan_y); dst_plan_y = nullptr; }
        // if (dst_plan_z) { fftw_destroy_plan(dst_plan_z); dst_plan_z = nullptr; }
        for (uint ix = 0; ix < n_plans_x; ix++) {
            if (dst_plans_x[ix]) { fftw_destroy_plan(dst_plans_x[ix]); dst_plans_x[ix] = nullptr; }
        }
        for (uint iy = 0; iy < n_plans_y; iy++) {
            if (dst_plans_y[iy]) { fftw_destroy_plan(dst_plans_y[iy]); dst_plans_y[iy] = nullptr; }
        }
        for (uint iz = 0; iz < n_plans_z; iz++) {
            if (dst_plans_z[iz]) { fftw_destroy_plan(dst_plans_z[iz]); dst_plans_z[iz] = nullptr; }
        }
    } else {
        // if (dst_plan_x) { fftwf_destroy_plan((fftwf_plan)dst_plan_x); dst_plan_x = nullptr; }
        // if (dst_plan_y) { fftwf_destroy_plan((fftwf_plan)dst_plan_y); dst_plan_y = nullptr; }
        // if (dst_plan_z) { fftwf_destroy_plan((fftwf_plan)dst_plan_z); dst_plan_z = nullptr; }
        for (uint ix = 0; ix < n_plans_x; ix++) {
            if (dst_plans_x[ix]) { fftwf_destroy_plan((fftwf_plan)dst_plans_x[ix]); dst_plans_x[ix] = nullptr; }
        }
        for (uint iy = 0; iy < n_plans_y; iy++) {
            if (dst_plans_y[iy]) { fftwf_destroy_plan((fftwf_plan)dst_plans_y[iy]); dst_plans_y[iy] = nullptr; }
        }
        for (uint iz = 0; iz < n_plans_z; iz++) {
            if (dst_plans_z[iz]) { fftwf_destroy_plan((fftwf_plan)dst_plans_z[iz]); dst_plans_z[iz] = nullptr; }
        }
    }
    std::vector<fftw_plan>().swap(this->dst_plans_x);
    std::vector<fftw_plan>().swap(this->dst_plans_y);
    std::vector<fftw_plan>().swap(this->dst_plans_z);
    #endif
    std::vector<T>().swap(this->d_hat);
    std::vector<T>().swap(this->work);
    this->N1 = 0;
    this->N2 = 0;
    this->N3 = 0;
    this->N = 0;
    return;
}

template<typename T>
void DST_Preconditioner_Data<T>::show() const {
    std::cout << "DST_Preconditioner_Data: N1=" << this->N1 << ", N2=" << this->N2 << ", N3=" << this->N3 << ", N=" << this->N << std::endl;
    if (this->d_hat.size() > 0) {
        std::cout << "First few eigenvalues: ";
        for (size_t i = 0; i < std::min(this->d_hat.size(), size_t(5)); i++) {
            std::cout << this->d_hat[i] << " ";
        }
        std::cout << std::endl;
    }
}

// Explicit instantiations for DST_Preconditioner_Data
template struct DST_Preconditioner_Data<float>;
template struct DST_Preconditioner_Data<double>;

/**
 * @brief DST-I based Poisson preconditioner for Dirichlet boundary conditions
 * Applies inv(L_local) * r where L_local is the Dirichlet discrete Laplacian
 * on the local subdomain. Uses DST-I for O(N log N) complexity.
 *
 * Algorithm:
 *   1. Apply 3D DST-I (as separable 1D DSTs along each axis)
 *   2. Divide by eigenvalues
 *   3. Apply inverse 3D DST-I (DST-I is self-inverse up to normalization)
 *   4. Normalize by 1/(8*(N1+1)*(N2+1)*(N3+1))
 */
template<typename T>
void dst_poisson_preconditioner(DST_Preconditioner_Data<T>& data,
                                 const int& N, const T* r, T* f, const Stencil<T>& stencil)
{
    #if defined(USE_FFTW) || defined(USE_MKL) || defined(USE_KML)
    // Fallback to Jacobi when there is no data (N=0), or when N1=N2=N3=1 (N=1).
    // For N=1 the DST eigenvalue uses antisymmetric BCs, which differs from the
    // zero-ghost BC diagonal for FDn > 1; Jacobi is exact in that degenerate case.
    if (data.N == 0 || data.N == 1) {
        Stencil_method::jacobi_preconditioner(stencil, N, (T)0.0, r, f);
    } else {

        
        const uint N1 = data.N1;
        const uint N2 = data.N2;
        const uint N3 = data.N3;

        // Helper: apply a 1D DST-I along a contiguous slice, or ×2 if plan is null (N_d=1).
        // For N_d=1, FFTW_RODFT00 would produce Y_0 = 2*x_0*sin(π/2) = 2*x_0, so
        // multiplying by 2 is the analytically correct substitute.
        
        auto apply_dst = [](fftw_plan plan, T* buf, uint n) {
            if (plan) {
                if constexpr (std::is_same_v<T, double>)
                    fftw_execute_r2r(plan, buf, buf);
                else
                    fftwf_execute_r2r((fftwf_plan)plan, buf, buf);
            } else {
                for (uint i = 0; i < n; i++) buf[i] *= T(2);
            }
        };

        uint tid = 0;
        // uint nthread = 1;
        #ifdef USE_OPENMP
        tid = omp_get_thread_num();
        // nthread = omp_get_num_threads();
        #endif
        std::vector<T> tmp(std::max(N2, N3));
        T* const& __restrict__ tmp_y = tmp.data();
        T* const& __restrict__ tmp_z = tmp.data();
        T* const& __restrict__ work_data = data.work.data();

        // 1. Copy r to work array
        // for (int i = 0; i < N; i++) {
        //     data.work[i] = r[i];
        // }
        Linalg::set_value_general(work_data, r, N);
        #pragma omp barrier

        // 2. Apply 3D DST-I as separable 1D DSTs
        // DST along x (contiguous, x varies fastest in column-major)
        #pragma omp for collapse(2)
        for (uint k3 = 0; k3 < N3; k3++) {
            for (uint k2 = 0; k2 < N2; k2++) {
                T* slice = work_data + k3 * N1 * N2 + k2 * N1;
                apply_dst(data.dst_plans_x[tid], slice, N1);
            }
        }

        // DST along y (gather/scatter for non-contiguous)
        #pragma omp for collapse(2)
        for (uint k3 = 0; k3 < N3; k3++) {
            for (uint k1 = 0; k1 < N1; k1++) {
                for (uint k2 = 0; k2 < N2; k2++) {
                    tmp_y[k2] = work_data[k3 * N1 * N2 + k2 * N1 + k1];
                }
                apply_dst(data.dst_plans_y[tid], tmp_y, N2);
                for (uint k2 = 0; k2 < N2; k2++) {
                    work_data[k3 * N1 * N2 + k2 * N1 + k1] = tmp_y[k2];
                }
            }
        }

        // DST along z (gather/scatter for non-contiguous)
        #pragma omp for collapse(2)
        for (uint k2 = 0; k2 < N2; k2++) {
            for (uint k1 = 0; k1 < N1; k1++) {
                for (uint k3 = 0; k3 < N3; k3++)
                    tmp_z[k3] = work_data[k3 * N1 * N2 + k2 * N1 + k1];
                apply_dst(data.dst_plans_z[tid], tmp_z, N3);
                for (uint k3 = 0; k3 < N3; k3++)
                    work_data[k3 * N1 * N2 + k2 * N1 + k1] = tmp_z[k3];
            }
        }

        // 3. Divide by eigenvalues (no zero mode for Dirichlet)
        // for (int i = 0; i < N; i++) {
        //     data.work[i] /= data.d_hat[i];
        // }
        Linalg::hadamard_divide_general(work_data, data.d_hat.data(), N);
        #pragma omp barrier

        // 4. Apply inverse 3D DST-I (DST-I is self-inverse; apply same transforms again)
        // DST along x
        #pragma omp for collapse(2)
        for (uint k3 = 0; k3 < N3; k3++) {
            for (uint k2 = 0; k2 < N2; k2++) {
                T* slice = work_data + k3 * N1 * N2 + k2 * N1;
                apply_dst(data.dst_plans_x[tid], slice, N1);
            }
        }

        // DST along y
        #pragma omp for collapse(2)
        for (uint k3 = 0; k3 < N3; k3++) {
            for (uint k1 = 0; k1 < N1; k1++) {
                for (uint k2 = 0; k2 < N2; k2++)
                    tmp_y[k2] = work_data[k3 * N1 * N2 + k2 * N1 + k1];
                apply_dst(data.dst_plans_y[tid], tmp_y, N2);
                for (uint k2 = 0; k2 < N2; k2++)
                    work_data[k3 * N1 * N2 + k2 * N1 + k1] = tmp_y[k2];
            }
        }

        // DST along z
        #pragma omp for collapse(2)
        for (uint k2 = 0; k2 < N2; k2++) {
            for (uint k1 = 0; k1 < N1; k1++) {
                for (uint k3 = 0; k3 < N3; k3++)
                    tmp_z[k3] = work_data[k3 * N1 * N2 + k2 * N1 + k1];
                apply_dst(data.dst_plans_z[tid], tmp_z, N3);
                for (uint k3 = 0; k3 < N3; k3++)
                    work_data[k3 * N1 * N2 + k2 * N1 + k1] = tmp_z[k3];
            }
        }

        // 5. Normalize: each DST-I forward+inverse pair contributes 2*(N_d+1) per dimension.
        // This holds for both FFTW plans (N_d>=2) and the ×2 fallback (N_d=1, 2*(1+1)=4).
        // Total normalization: 1/(8*(N1+1)*(N2+1)*(N3+1))
        T norm = T(1) / (T(8) * (N1 + 1) * (N2 + 1) * (N3 + 1));
        // for (int i = 0; i < N; i++) {
        //     f[i] = data.work[i] * norm;
        // }
        Linalg::scalar_product_general(f, work_data, norm, N);
        #pragma omp barrier
    }
    return;
    #else
    // Fallback when FFTW/MKL/KML is not available: identity preconditioner
    (void) data;
    // (void) stencil;
    // for (int i = 0; i < N; i++) {
    //     f[i] = r[i];
    // }
    std::cout << RED << "WARNING:: No FFTW libs included, using jacobi_preconditioner instead!" << RESET << std::endl;
    Stencil_method::jacobi_preconditioner(stencil, N, (T)0.0, r, f);
    #endif
}
template void dst_poisson_preconditioner<float>(DST_Preconditioner_Data<float>& data,
                                                 const int& N, const float* r, float* f, const Stencil<float>& stencil);
template void dst_poisson_preconditioner<double>(DST_Preconditioner_Data<double>& data,
                                                  const int& N, const double* r, double* f, const Stencil<double>& stencil);

int64_t generate_gradient_stride(const Vertices_3D& ex_vertices, const uint& dir) {
    switch (dir)
    {
    case 0:
        return 1;
        break;
    case 1:
        return (int64_t)ex_vertices.ni;
        break;
    case 2:
        return (int64_t)(ex_vertices.ni * ex_vertices.nj);
        break;
    default:
        assert(!"ERROR:: only x,y,z dir is supported~");
        break;
    }
}

template<typename T>
const T* generate_gradient_stencil_coeffs(const Stencil<T>& stencil, const uint dir) {
    switch (stencil.cell_type)
    {
    case 0:
    case 1:
    case 2:
        switch (stencil.optimization)
        {
        case 0:
        case 1:
        case 2:
        case 3:
            switch (dir)
            {
            case 0:
                return stencil.get_D1_coeffs_x();
                break;
            case 1:
                return stencil.get_D1_coeffs_y();
                break;
            case 2:
                return stencil.get_D1_coeffs_z();
                break;
            default:
                assert(!"ERROR:: only optimization x,y,z dir are supported~");
                return stencil.get_D1_coeffs_x();
                break;
            }
            break;
        case 4:
            return stencil.get_D1_coeffs_x();
            break;
        
        default:
            assert(!"ERROR:: only optimization 0,1,2,3,4 are supported~");
            return stencil.get_D1_coeffs_x();
            break;
        }
        break;
    
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
        return stencil.get_D1_coeffs_x();
        break;
    }
}
template const float* generate_gradient_stencil_coeffs<float>(const Stencil<float>& stencil, const uint dir);
template const double* generate_gradient_stencil_coeffs<double>(const Stencil<double>& stencil, const uint dir);

template<typename T1, typename T2>
void calc_gradient(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                    const uint dir, const Stencil<T2>& stencil, const Vertices_3D& region,
                    T1* const ptr, const Vertices_3D& vertices) {
    switch (stencil.cell_type)
    {
    case 0:
    case 1:
    case 2:
        {
            const int64_t stride = generate_gradient_stride(ex_vertices, dir);
            T2 const* stencil_coeffs = generate_gradient_stencil_coeffs(stencil, dir);
            calc_gradient_d3_c2<T1, T2>(ex_ptr, ex_vertices, stride, stencil_coeffs, stencil.FDn,
                                                   region, ptr, vertices);
        }
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
        break;
    }
    return;
}
template void calc_gradient<float, float>(float const* const ex_ptr, const Vertices_3D& ex_vertices,
                                        const uint dir, const Stencil<float>& stencil, const Vertices_3D& region,
                                        float* const ptr, const Vertices_3D& vertices);
template void calc_gradient<double, double>(double const* const ex_ptr, const Vertices_3D& ex_vertices,
                                        const uint dir, const Stencil<double>& stencil, const Vertices_3D& region,
                                        double* const ptr, const Vertices_3D& vertices);
template void calc_gradient<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices,
                                                        const uint dir, const Stencil<float>& stencil, const Vertices_3D& region,
                                                        std::complex<float>* const ptr, const Vertices_3D& vertices);
template void calc_gradient<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices,
                                                            const uint dir, const Stencil<double>& stencil, const Vertices_3D& region,
                                                            std::complex<double>* const ptr, const Vertices_3D& vertices);

template<typename T1, typename T2>
void calc_gradient_d3_c2(T1 const* const __restrict__ ex_ptr, const Vertices_3D& ex_vertices, const int64_t stride,
                        T2 const* const __restrict__ stencil_coeffs, const int FDn,
                        const Vertices_3D& region, T1* const __restrict__ ptr, const Vertices_3D& vertices) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;

    switch (FDn)
    {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nk - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    T1 temp = static_cast<T1>(0);
                    for (int64_t p = 1; p <= 6; p++)
                    {
                        const int64_t stride_p = p * stride;
                        temp += (*(this_data_i+stride_p) - *(this_data_i-stride_p)) * stencil_coeffs[p];
                    }
                    *result_data_i++ = temp;
                    ++this_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nk - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    T1 temp = static_cast<T1>(0);
                    for (int p = 1; p <= FDn; p++)
                    {
                        const int64_t stride_p = p * stride;
                        temp += (*(this_data_i+stride_p) - *(this_data_i-stride_p)) * stencil_coeffs[p];
                    }
                    *result_data_i++ = temp;
                    ++this_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
            }
        }
        break;
    }
    return;
}
template void calc_gradient_d3_c2<float, float>(float const* const ex_ptr, const Vertices_3D& ex_vertices, const int64_t stride,
                                                float const* const stencil_coeffs, const int FDn,
                                                const Vertices_3D& region, float* const ptr, const Vertices_3D& vertices);
template void calc_gradient_d3_c2<double, double>(double const* const ex_ptr, const Vertices_3D& ex_vertices, const int64_t stride,
                                                    double const* const stencil_coeffs, const int FDn,
                                                    const Vertices_3D& region, double* const ptr, const Vertices_3D& vertices);
template void calc_gradient_d3_c2<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const int64_t stride,
                                                            float const* const stencil_coeffs, const int FDn,
                                                            const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices);
template void calc_gradient_d3_c2<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const int64_t stride,
                                                            double const* const stencil_coeffs, const int FDn,
                                                            const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices);

template<typename T1, typename T2>
void calc_gradient_d4(T1 const* const ex_ptr, const Vertices_4D& ex_vertices,
                    const uint dir, const Stencil<T2>& stencil, const Vertices_4D& region,
                    T1* const ptr, const Vertices_4D& vertices) {
    switch (stencil.cell_type)
    {
    case 0:
    case 1:
    case 2:
        {
            const int64_t stride = generate_gradient_stride(ex_vertices, dir);
            T2 const* stencil_coeffs = generate_gradient_stencil_coeffs(stencil, dir);
            calc_gradient_d4_c2<T1, T2>(ex_ptr, ex_vertices, stride, stencil_coeffs, stencil.FDn,
                                                   region, ptr, vertices);
        }
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
        break;
    }
    return;
}
template void calc_gradient_d4<float, float>(float const* const ex_ptr, const Vertices_4D& ex_vertices,
                                            const uint dir, const Stencil<float>& stencil, const Vertices_4D& region,
                                            float* const ptr, const Vertices_4D& vertices);
template void calc_gradient_d4<double, double>(double const* const ex_ptr, const Vertices_4D& ex_vertices,
                                                const uint dir, const Stencil<double>& stencil, const Vertices_4D& region,
                                                double* const ptr, const Vertices_4D& vertices);
template void calc_gradient_d4<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices,
                                                            const uint dir, const Stencil<float>& stencil, const Vertices_4D& region,
                                                            std::complex<float>* const ptr, const Vertices_4D& vertices);
template void calc_gradient_d4<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices,
                                                                const uint dir, const Stencil<double>& stencil, const Vertices_4D& region,
                                                                std::complex<double>* const ptr, const Vertices_4D& vertices);


template<typename T1, typename T2>
void calc_gradient_d4_c2(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const int64_t stride,
                        T2 const* const __restrict__ stencil_coeffs, const int FDn,
                        const Vertices_4D& region, T1* const __restrict__ ptr, const Vertices_4D& vertices) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int64_t this_ninjnk = ex_vertices.ni * ex_vertices.nj * ex_vertices.nk;
    const int64_t result_ninjnk = vertices.ni * vertices.nj * vertices.nk;
    const int64_t region_ni = (int64_t)region.ni;
    const int64_t region_nj = (int64_t)region.nj;
    const int64_t region_nk = (int64_t)region.nk;
    const int64_t region_nb = (int64_t)region.nb;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;

    switch (FDn)
    {
    case 6:
    {
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region_nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < region_nb; b++) {
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            // int this_offset_k = index_this_origin + this_ninjnk * b;
            // int result_offset_k = index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < region_nk; k++) {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                // int this_offset_j = this_offset_k;
                // int result_offset_j = result_offset_k;
                for (int64_t j = 0; j < region_nj; j++) {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    // int this_offset_i = this_offset_j;
                    // int result_offset_i = result_offset_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < region_ni; i++) {
                        T1 temp = static_cast<T1>(0);
                        for (int p = 1; p <= 6; p++) {
                            const int64_t stride_p = p * stride;
                            temp += (*(this_data_i+stride_p) - *(this_data_i-stride_p)) * stencil_coeffs[p];
                            // temp += (this_data[this_offset_i + stride_p] - this_data[this_offset_i - stride_p])
                            //       * stencil_coeffs[p];
                        }
                        *result_data_i++ = temp;
                        ++this_data_i;
                        // ++this_offset_i;
                        // result_data[result_offset_i++] = temp;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                    // this_offset_j += this_ni;
                    // result_offset_j += result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
                // this_offset_k += this_ninj;
                // result_offset_k += result_ninj;
            }
        }
        break;
    }
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region_nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < region_nb; b++) {
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < region_nk; k++) {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                for (int64_t j = 0; j < region_nj; j++) {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < region_ni; i++) {
                        T1 temp = static_cast<T1>(0);
                        for (int p = 1; p <= FDn; p++) {
                            const int64_t stride_p = p * stride;
                            temp += (*(this_data_i+stride_p) - *(this_data_i-stride_p)) * stencil_coeffs[p];
                        }
                        *result_data_i++ = temp;
                        ++this_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
            }
        }
        break;
    }
    return;
}
template void calc_gradient_d4_c2<float, float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const int64_t stride,
                                                float const* const stencil_coeffs, const int FDn,
                                                const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices);
template void calc_gradient_d4_c2<double, double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const int64_t stride,
                                                double const* const stencil_coeffs, const int FDn,
                                                const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices);
template void calc_gradient_d4_c2<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const int64_t stride,
                                                float const* const stencil_coeffs, const int FDn,
                                                const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices);
template void calc_gradient_d4_c2<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const int64_t stride,
                                                double const* const stencil_coeffs, const int FDn,
                                                const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices);

template<typename T1, typename T2>
void calc_laplacian(T1 const* const __restrict__ ex_ptr, const Vertices_3D& ex_vertices, const Stencil<T2>& stencil, const Vertices_3D& region,
    T1* const __restrict__ ptr, const Vertices_3D& vertices) {
    switch(stencil.cell_type) {
    case 0:
    case 1:
    case 2:
        switch(stencil.optimization) {
        case 4:
            calc_laplacian_d3_c2_o4(ex_ptr, ex_vertices, stencil, region, ptr, vertices);
            break;
        default:
            calc_laplacian_d3_c2_o0(ex_ptr, ex_vertices, stencil, region, ptr, vertices);
        }
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}
template void calc_laplacian<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                    const Vertices_3D& region, float* const ptr, const Vertices_3D& vertices);
template void calc_laplacian<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                    const Vertices_3D& region, double* const ptr, const Vertices_3D& vertices);
template void calc_laplacian<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                        const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices);
template void calc_laplacian<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                        const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices);

template<typename T1, typename T2, typename T3>
void calc_laplacian(T1 const* const __restrict__ ex_ptr, const Vertices_3D& ex_vertices, const Stencil<T2>& stencil, const Vertices_3D& region,
    T1* const __restrict__ ptr, const Vertices_3D& vertices, T3 const* const __restrict__ ptr_A) {
    switch(stencil.cell_type) {
    case 0:
    case 1:
    case 2:
        switch(stencil.optimization) {
        case 4:
            calc_laplacian_d3_c2_o4(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                    ptr_A);
            break;
        default:
            calc_laplacian_d3_c2_o0(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                    ptr_A);
        }
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}
template void calc_laplacian<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                    const Vertices_3D& region, float* const ptr, const Vertices_3D& vertices,
                                    float const* const ptr_A);
template void calc_laplacian<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                    const Vertices_3D& region, double* const ptr, const Vertices_3D& vertices,
                                    double const* const ptr_A);
template void calc_laplacian<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                        const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices,
                                        std::complex<float> const* const ptr_A);
template void calc_laplacian<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                        const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices,
                                        std::complex<double> const* const ptr_A);
template void calc_laplacian<std::complex<float>, float, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                        const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices,
                                        float const* const ptr_A);
template void calc_laplacian<std::complex<double>, double, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                        const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices,
                                        double const* const ptr_A);                                      

template<typename T1, typename T2>
void calc_laplacian(T1 const* const __restrict__ ex_ptr, const Vertices_3D& ex_vertices, const Stencil<T2>& stencil, const Vertices_3D& region,
    T1* const __restrict__ ptr, const Vertices_3D& vertices, T1 const* const __restrict__ ptr_A, const T2 alpha) {
    switch(stencil.cell_type) {
    case 0:
    case 1:
    case 2:
        switch(stencil.optimization) {
        case 4:
            calc_laplacian_d3_c2_o4(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                    ptr_A, alpha);
            break;
        default:
            calc_laplacian_d3_c2_o0(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                    ptr_A, alpha);
        }
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}
template void calc_laplacian<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                    const Vertices_3D& region, float* const ptr, const Vertices_3D& vertices,
                                    float const* const ptr_A, const float alpha);
template void calc_laplacian<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                    const Vertices_3D& region, double* const ptr, const Vertices_3D& vertices,
                                    double const* const ptr_A, const double alpha);
template void calc_laplacian<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                        const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices,
                                        std::complex<float> const* const ptr_A, const float alpha);
template void calc_laplacian<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                        const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices,
                                        std::complex<double> const* const ptr_A, const double alpha);

template<typename T1, typename T2>
void calc_laplacian(T1 const* const __restrict__ ex_ptr, const Vertices_3D& ex_vertices, const Stencil<T2>& stencil, const Vertices_3D& region,
    T1* const __restrict__ ptr, const Vertices_3D& vertices, T1 const* const __restrict__ ptr_A, const T2 alpha,
    T1 const* const __restrict__ ptr_B, const T2 beta) {
    switch(stencil.cell_type) {
    case 0:
    case 1:
    case 2:
        switch(stencil.optimization) {
        case 4:
            calc_laplacian_d3_c2_o4(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                    ptr_A, alpha, ptr_B, beta);
            break;
        default:
            calc_laplacian_d3_c2_o0(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                    ptr_A, alpha, ptr_B, beta);
        }
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}
template void calc_laplacian<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                    const Vertices_3D& region, float* const ptr, const Vertices_3D& vertices,
                                    float const* const ptr_A, const float alpha,
                                    float const* const ptr_B, const float beta);
template void calc_laplacian<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                    const Vertices_3D& region, double* const ptr, const Vertices_3D& vertices,
                                    double const* const ptr_A, const double alpha,
                                    double const* const ptr_B, const double beta);
template void calc_laplacian<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                        const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices,
                                        std::complex<float> const* const ptr_A, const float alpha,
                                        std::complex<float> const* const ptr_B, const float beta);
template void calc_laplacian<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                        const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices,
                                        std::complex<double> const* const ptr_A, const double alpha,
                                        std::complex<double> const* const ptr_B, const double beta);

template<typename T1, typename T2>
void calc_laplacian(T1 const* const __restrict__ ex_ptr, const Vertices_3D& ex_vertices, const Stencil<T2>& stencil, const Vertices_3D& region,
    T1* const __restrict__ ptr, const Vertices_3D& vertices, T1 const* const __restrict__ ptr_A, const T2 alpha,
    T1 const* const __restrict__ ptr_B, const T2 beta, const T2 gamma) {
    switch(stencil.cell_type) {
    case 0:
    case 1:
    case 2:
        switch(stencil.optimization) {
        case 4:
            calc_laplacian_d3_c2_o4(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                    ptr_A, alpha, ptr_B, beta, gamma);
            break;
        default:
            calc_laplacian_d3_c2_o0(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                    ptr_A, alpha, ptr_B, beta, gamma);
        }
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}
template void calc_laplacian<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                    const Vertices_3D& region, float* const ptr, const Vertices_3D& vertices,
                                    float const* const ptr_A, const float alpha,
                                    float const* const ptr_B, const float beta,
                                    const float gamma);
template void calc_laplacian<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                    const Vertices_3D& region, double* const ptr, const Vertices_3D& vertices,
                                    double const* const ptr_A, const double alpha,
                                    double const* const ptr_B, const double beta,
                                    const double gamma);
template void calc_laplacian<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                        const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices,
                                        std::complex<float> const* const ptr_A, const float alpha,
                                        std::complex<float> const* const ptr_B, const float beta,
                                        const float gamma);
template void calc_laplacian<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                        const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices,
                                        std::complex<double> const* const ptr_A, const double alpha,
                                        std::complex<double> const* const ptr_B, const double beta,
                                        const double gamma);

template<typename T1, typename T2>
void calc_laplacian_d3_c2_o0(T1 const* const __restrict__ ex_ptr, const Vertices_3D& ex_vertices, const Stencil<T2>& stencil, const Vertices_3D& region,
    T1* const __restrict__ ptr, const Vertices_3D& vertices) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0] + D2_coeffs_xyz[1] + D2_coeffs_xyz[2];

    switch (FDn)
    {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nk - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = coef_0  * *this_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                    for (int p = 1; p <= 6; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                        T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                        T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res_x + res_y + res_z;
                    }
                    ++this_data_i;
                    ++result_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nk - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = coef_0  * *this_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                    for (int p = 1; p <= FDn; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                        T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                        T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res_x + res_y + res_z;
                    }
                    ++this_data_i;
                    ++result_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
            }
        }
        break;
    }
    return;
}
template void calc_laplacian_d3_c2_o0<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_3D& region, float* const ptr, const Vertices_3D& vertices);
template void calc_laplacian_d3_c2_o0<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_3D& region, double* const ptr, const Vertices_3D& vertices);
template void calc_laplacian_d3_c2_o0<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                                const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices);
template void calc_laplacian_d3_c2_o0<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                                const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices);

template<typename T1, typename T2, typename T3>
void calc_laplacian_d3_c2_o0(T1 const* const __restrict__ ex_ptr, const Vertices_3D& ex_vertices, const Stencil<T2>& stencil, const Vertices_3D& region,
    T1* const __restrict__ ptr, const Vertices_3D& vertices, T3 const* const __restrict__ ptr_A) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0] + D2_coeffs_xyz[1] + D2_coeffs_xyz[2];

    #ifdef USE_OPENMP

    switch (FDn)
    {
    case 6:
        #pragma omp for schedule(static, (region.nk * region.nj - 1)/omp_get_num_threads() + 1) collapse(2) nowait
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                const int64_t result_offset = index_result_origin + result_ninj * k + result_ni * j;
                T1 const* __restrict__ this_data_i = ex_ptr + index_this_origin + this_ninj * k + this_ni * j;
                T1* __restrict__ result_data_i = ptr + result_offset;
                T3 const* __restrict__ ptrA_data_i = ptr_A + result_offset;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + *ptrA_data_i) * *this_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                    for (int64_t p = 1; p <= 6; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                        T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                        T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res_x + res_y + res_z;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                }
            }
        }
        break;
    default:
        #pragma omp for schedule(static, (region.nk * region.nj - 1)/omp_get_num_threads() + 1) collapse(2) nowait
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                const int64_t result_offset = index_result_origin + result_ninj * k + result_ni * j;
                T1 const* __restrict__ this_data_i = ex_ptr + index_this_origin + this_ninj * k + this_ni * j;
                T1* __restrict__ result_data_i = ptr + result_offset;
                T3 const* __restrict__ ptrA_data_i = ptr_A + result_offset;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + *ptrA_data_i) * *this_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                    for (int p = 1; p <= FDn; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                        T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                        T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res_x + res_y + res_z;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                }
            }
        }
        break;
    }

    #else//USE_OPENMP

    switch (FDn)
    {
    case 6:
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            T3 const* __restrict__ ptrA_data_j = ptr_A + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                T3 const* __restrict__ ptrA_data_i = ptrA_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + *ptrA_data_i) * *this_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                    for (int p = 1; p <= 6; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                        T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                        T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res_x + res_y + res_z;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
                ptrA_data_j+=result_ni;
            }
        }
        break;
    default:
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            T3 const* __restrict__ ptrA_data_j = ptr_A + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                T3 const* __restrict__ ptrA_data_i = ptrA_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + *ptrA_data_i) * *this_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                    for (int p = 1; p <= FDn; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                        T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                        T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res_x + res_y + res_z;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
                ptrA_data_j+=result_ni;
            }
        }
        break;
    }

    #endif //USE_OPENMP
    return;
}
template void calc_laplacian_d3_c2_o0<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_3D& region, float* const ptr, const Vertices_3D& vertices,
                                            float const* const ptr_A);
template void calc_laplacian_d3_c2_o0<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_3D& region, double* const ptr, const Vertices_3D& vertices,
                                            double const* const ptr_A);
template void calc_laplacian_d3_c2_o0<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                                const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices,
                                                std::complex<float> const* const ptr_A);
template void calc_laplacian_d3_c2_o0<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                                const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices,
                                                std::complex<double> const* const ptr_A);
template void calc_laplacian_d3_c2_o0<std::complex<float>, float, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                                const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices,
                                                float const* const ptr_A);
template void calc_laplacian_d3_c2_o0<std::complex<double>, double, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                                const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices,
                                                double const* const ptr_A);

template<typename T1, typename T2>
void calc_laplacian_d3_c2_o0(T1 const* const __restrict__ ex_ptr, const Vertices_3D& ex_vertices, const Stencil<T2>& stencil, const Vertices_3D& region,
    T1* const __restrict__ ptr, const Vertices_3D& vertices, T1 const* const __restrict__ ptr_A, const T2 alpha) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0] + D2_coeffs_xyz[1] + D2_coeffs_xyz[2];

    switch (FDn)
    {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nk - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrA_data_j = ptr_A + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                    for (int p = 1; p <= 6; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                        T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                        T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res_x + res_y + res_z;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
                ptrA_data_j+=result_ni;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nk - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrA_data_j = ptr_A + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                    for (int p = 1; p <= FDn; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                        T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                        T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res_x + res_y + res_z;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
                ptrA_data_j+=result_ni;
            }
        }
        break;
    }
    return;
}
template void calc_laplacian_d3_c2_o0<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_3D& region, float* const ptr, const Vertices_3D& vertices,
                                            float const* const ptr_A, const float alpha);
template void calc_laplacian_d3_c2_o0<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_3D& region, double* const ptr, const Vertices_3D& vertices,
                                            double const* const ptr_A, const double alpha);
template void calc_laplacian_d3_c2_o0<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                                const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices,
                                                std::complex<float> const* const ptr_A, const float alpha);
template void calc_laplacian_d3_c2_o0<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                                const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices,
                                                std::complex<double> const* const ptr_A, const double alpha);

template<typename T1, typename T2>
void calc_laplacian_d3_c2_o0(T1 const* const __restrict__ ex_ptr, const Vertices_3D& ex_vertices, const Stencil<T2>& stencil, const Vertices_3D& region,
    T1* const __restrict__ ptr, const Vertices_3D& vertices, T1 const* const __restrict__ ptr_A, const T2 alpha,
    T1 const* const __restrict__ ptr_B, const T2 beta) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0] + D2_coeffs_xyz[1] + D2_coeffs_xyz[2];

    #ifdef USE_OPENMP

    switch (FDn)
    {
    case 6:
        #pragma omp for schedule(static, (region.nk * region.nj - 1)/omp_get_num_threads() + 1) collapse(2) nowait
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                const int64_t result_offset = index_result_origin + result_ninj * k + result_ni * j;
                T1 const* __restrict__ this_data_i = ex_ptr + index_this_origin + this_ninj * k + this_ni * j;
                T1* __restrict__ result_data_i = ptr + result_offset;
                T1 const* __restrict__ ptrA_data_i = ptr_A + result_offset;
                T1 const* __restrict__ ptrB_data_i = ptr_B + result_offset;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                   + beta * *ptrB_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                    for (int p = 1; p <= 6; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                        T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                        T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res_x + res_y + res_z;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                    ++ptrB_data_i;
                }
            }
        }
        break;
    default:
        #pragma omp for schedule(static, (region.nk * region.nj - 1)/omp_get_num_threads() + 1) collapse(2) nowait
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                const int64_t result_offset = index_result_origin + result_ninj * k + result_ni * j;
                T1 const* __restrict__ this_data_i = ex_ptr + index_this_origin + this_ninj * k + this_ni * j;
                T1* __restrict__ result_data_i = ptr + result_offset;
                T1 const* __restrict__ ptrA_data_i = ptr_A + result_offset;
                T1 const* __restrict__ ptrB_data_i = ptr_B + result_offset;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                   + beta * *ptrB_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                    for (int p = 1; p <= FDn; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                        T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                        T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res_x + res_y + res_z;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                    ++ptrB_data_i;
                }
            }
        }
        break;
    }

    #else //USE_OPENMP

    switch (FDn)
    {
    case 6:
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrA_data_j = ptr_A + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrB_data_j = ptr_B + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                T1 const* __restrict__ ptrB_data_i = ptrB_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                   + beta * *ptrB_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                    for (int p = 1; p <= 6; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                        T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                        T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res_x + res_y + res_z;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                    ++ptrB_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
                ptrA_data_j+=result_ni;
                ptrB_data_j+=result_ni;
            }
        }
        break;
    default:
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrA_data_j = ptr_A + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrB_data_j = ptr_B + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                T1 const* __restrict__ ptrB_data_i = ptrB_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                   + beta * *ptrB_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                    for (int p = 1; p <= FDn; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                        T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                        T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res_x + res_y + res_z;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                    ++ptrB_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
                ptrA_data_j+=result_ni;
                ptrB_data_j+=result_ni;
            }
        }
        break;
    }

    #endif //USE_OPENMP
    return;
}
template void calc_laplacian_d3_c2_o0<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_3D& region, float* const ptr, const Vertices_3D& vertices,
                                            float const* const ptr_A, const float alpha,
                                            float const* const ptr_B, const float beta);
template void calc_laplacian_d3_c2_o0<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_3D& region, double* const ptr, const Vertices_3D& vertices,
                                            double const* const ptr_A, const double alpha,
                                            double const* const ptr_B, const double beta);
template void calc_laplacian_d3_c2_o0<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                                const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices,
                                                std::complex<float> const* const ptr_A, const float alpha,
                                                std::complex<float> const* const ptr_B, const float beta);
template void calc_laplacian_d3_c2_o0<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                                const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices,
                                                std::complex<double> const* const ptr_A, const double alpha,
                                                std::complex<double> const* const ptr_B, const double beta);

template<typename T1, typename T2>
void calc_laplacian_d3_c2_o0(T1 const* const __restrict__ ex_ptr, const Vertices_3D& ex_vertices, const Stencil<T2>& stencil, const Vertices_3D& region,
    T1* const __restrict__ ptr, const Vertices_3D& vertices, T1 const* const __restrict__ ptr_A, const T2 alpha,
    T1 const* const __restrict__ ptr_B, const T2 beta, const T2 gamma) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0] + D2_coeffs_xyz[1] + D2_coeffs_xyz[2];

    switch (FDn)
    {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nk - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrA_data_j = ptr_A + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrB_data_j = ptr_B + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                T1 const* __restrict__ ptrB_data_i = ptrB_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                   + beta * *ptrB_data_i + gamma;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                    for (int p = 1; p <= 6; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                        T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                        T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res_x + res_y + res_z;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                    ++ptrB_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
                ptrA_data_j+=result_ni;
                ptrB_data_j+=result_ni;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nk - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrA_data_j = ptr_A + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrB_data_j = ptr_B + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                T1 const* __restrict__ ptrB_data_i = ptrB_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                   + beta * *ptrB_data_i + gamma;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                    for (int p = 1; p <= FDn; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                        T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                        T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res_x + res_y + res_z;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                    ++ptrB_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
                ptrA_data_j+=result_ni;
                ptrB_data_j+=result_ni;
            }
        }
        break;
    }
    return;
}
template void calc_laplacian_d3_c2_o0<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_3D& region, float* const ptr, const Vertices_3D& vertices,
                                            float const* const ptr_A, const float alpha,
                                            float const* const ptr_B, const float beta,
                                            const float gamma);
template void calc_laplacian_d3_c2_o0<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_3D& region, double* const ptr, const Vertices_3D& vertices,
                                            double const* const ptr_A, const double alpha,
                                            double const* const ptr_B, const double beta,
                                            const double gamma);
template void calc_laplacian_d3_c2_o0<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                                const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices,
                                                std::complex<float> const* const ptr_A, const float alpha,
                                                std::complex<float> const* const ptr_B, const float beta,
                                                const float gamma);
template void calc_laplacian_d3_c2_o0<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                                const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices,
                                                std::complex<double> const* const ptr_A, const double alpha,
                                                std::complex<double> const* const ptr_B, const double beta,
                                                const double gamma);

template<typename T1, typename T2>
void calc_laplacian_d3_c2_o4(T1 const* const __restrict__ ex_ptr, const Vertices_3D& ex_vertices, const Stencil<T2>& stencil, const Vertices_3D& region,
                                T1* const __restrict__ ptr, const Vertices_3D& vertices) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0];

    switch (FDn)
    {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nk - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = coef_0  * *this_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                    for (int p = 1; p <= 6; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res;
                    }
                    ++this_data_i;
                    ++result_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nk - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = coef_0  * *this_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                    for (int p = 1; p <= FDn; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res;
                    }
                    ++this_data_i;
                    ++result_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
            }
        }
        break;
    }
    return;
}
template void calc_laplacian_d3_c2_o4<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_3D& region, float* const ptr, const Vertices_3D& vertices);
template void calc_laplacian_d3_c2_o4<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_3D& region, double* const ptr, const Vertices_3D& vertices);
template void calc_laplacian_d3_c2_o4<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                                const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices);
template void calc_laplacian_d3_c2_o4<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                                const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices);

template<typename T1, typename T2, typename T3>
void calc_laplacian_d3_c2_o4(T1 const* const __restrict__ ex_ptr, const Vertices_3D& ex_vertices, const Stencil<T2>& stencil, const Vertices_3D& region,
        T1* const __restrict__ ptr, const Vertices_3D& vertices, T3 const* const __restrict__ ptr_A) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0];
    #ifdef USE_OPENMP

    switch (FDn)
    {
    case 6:
        #pragma omp for schedule(static, (region.nk * region.nj - 1)/omp_get_num_threads() + 1) collapse(2) nowait
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                const int64_t result_offset = index_result_origin + result_ninj * k + result_ni * j;
                T1 const* __restrict__ this_data_i = ex_ptr + index_this_origin + this_ninj * k + this_ni * j;
                T1* __restrict__ result_data_i = ptr + result_offset;
                T3 const* __restrict__ ptrA_data_i = ptr_A + result_offset;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + *ptrA_data_i) * *this_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                    for (int p = 1; p <= 6; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                }
            }
        }
        break;
    default:
        #pragma omp for schedule(static, (region.nk * region.nj - 1)/omp_get_num_threads() + 1) collapse(2) nowait
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                const int64_t result_offset = index_result_origin + result_ninj * k + result_ni * j;
                T1 const* __restrict__ this_data_i = ex_ptr + index_this_origin + this_ninj * k + this_ni * j;
                T1* __restrict__ result_data_i = ptr + result_offset;
                T3 const* __restrict__ ptrA_data_i = ptr_A + result_offset;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + *ptrA_data_i) * *this_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                    for (int p = 1; p <= FDn; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                }
            }
        }
        break;
    }

    #else//USE_OPENMP

    switch (FDn)
    {
    case 6:
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            T3 const* __restrict__ ptrA_data_j = ptr_A + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                T3 const* __restrict__ ptrA_data_i = ptrA_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + *ptrA_data_i) * *this_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                    for (int p = 1; p <= 6; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
                ptrA_data_j+=result_ni;
            }
        }
        break;
    default:
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            T3 const* __restrict__ ptrA_data_j = ptr_A + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                T3 const* __restrict__ ptrA_data_i = ptrA_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + *ptrA_data_i) * *this_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                    for (int p = 1; p <= FDn; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
                ptrA_data_j+=result_ni;
            }
        }
        break;
    }

    #endif //USE_OPENMP
    return;
}
template void calc_laplacian_d3_c2_o4<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_3D& region, float* const ptr, const Vertices_3D& vertices,
                                            float const* const ptr_A);
template void calc_laplacian_d3_c2_o4<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_3D& region, double* const ptr, const Vertices_3D& vertices,
                                            double const* const ptr_A);
template void calc_laplacian_d3_c2_o4<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                                const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices,
                                                std::complex<float> const* const ptr_A);
template void calc_laplacian_d3_c2_o4<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                                const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices,
                                                std::complex<double> const* const ptr_A);
template void calc_laplacian_d3_c2_o4<std::complex<float>, float, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                                const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices,
                                                float const* const ptr_A);
template void calc_laplacian_d3_c2_o4<std::complex<double>, double, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                                const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices,
                                                double const* const ptr_A);

template<typename T1, typename T2>
void calc_laplacian_d3_c2_o4(T1 const* const __restrict__ ex_ptr, const Vertices_3D& ex_vertices, const Stencil<T2>& stencil, const Vertices_3D& region,
    T1* const __restrict__ ptr, const Vertices_3D& vertices, T1 const* const __restrict__ ptr_A, const T2 alpha) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0];

    switch (FDn)
    {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nk - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrA_data_j = ptr_A + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                    for (int p = 1; p <= 6; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
                ptrA_data_j+=result_ni;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nk - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrA_data_j = ptr_A + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                    for (int p = 1; p <= FDn; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
                ptrA_data_j+=result_ni;
            }
        }
        break;
    }
    return;
}
template void calc_laplacian_d3_c2_o4<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_3D& region, float* const ptr, const Vertices_3D& vertices,
                                            float const* const ptr_A, const float alpha);
template void calc_laplacian_d3_c2_o4<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_3D& region, double* const ptr, const Vertices_3D& vertices,
                                            double const* const ptr_A, const double alpha);
template void calc_laplacian_d3_c2_o4<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                                const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices,
                                                std::complex<float> const* const ptr_A, const float alpha);
template void calc_laplacian_d3_c2_o4<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                                const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices,
                                                std::complex<double> const* const ptr_A, const double alpha);

template<typename T1, typename T2>
void calc_laplacian_d3_c2_o4(T1 const* const __restrict__ ex_ptr, const Vertices_3D& ex_vertices, const Stencil<T2>& stencil, const Vertices_3D& region,
    T1* const __restrict__ ptr, const Vertices_3D& vertices, T1 const* const __restrict__ ptr_A, const T2 alpha,
    T1 const* const __restrict__ ptr_B, const T2 beta) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0];

    #ifdef USE_OPENMP

    switch (FDn)
    {
    case 6:
        #pragma omp for schedule(static, (region.nk * region.nj - 1)/omp_get_num_threads() + 1) collapse(2) nowait
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                const int64_t result_offset = index_result_origin + result_ninj * k + result_ni * j;
                T1 const* __restrict__ this_data_i = ex_ptr + index_this_origin + this_ninj * k + this_ni * j;
                T1* __restrict__ result_data_i = ptr + result_offset;
                T1 const* __restrict__ ptrA_data_i = ptr_A + result_offset;
                T1 const* __restrict__ ptrB_data_i = ptr_B + result_offset;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                   + beta * *ptrB_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                    for (int p = 1; p <= 6; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                    ++ptrB_data_i;
                }
            }
        }
        break;
    default:
        #pragma omp for schedule(static, (region.nk * region.nj - 1)/omp_get_num_threads() + 1) collapse(2) nowait
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                const int64_t result_offset = index_result_origin + result_ninj * k + result_ni * j;
                T1 const* __restrict__ this_data_i = ex_ptr + index_this_origin + this_ninj * k + this_ni * j;
                T1* __restrict__ result_data_i = ptr + result_offset;
                T1 const* __restrict__ ptrA_data_i = ptr_A + result_offset;
                T1 const* __restrict__ ptrB_data_i = ptr_B + result_offset;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                   + beta * *ptrB_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                    for (int p = 1; p <= FDn; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                    ++ptrB_data_i;
                }
            }
        }
        break;
    }

    #else //USE_OPENMP

    switch (FDn)
    {
    case 6:
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrA_data_j = ptr_A + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrB_data_j = ptr_B + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                T1 const* __restrict__ ptrB_data_i = ptrB_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                   + beta * *ptrB_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                    for (int p = 1; p <= 6; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                    ++ptrB_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
                ptrA_data_j+=result_ni;
                ptrB_data_j+=result_ni;
            }
        }
        break;
    default:
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrA_data_j = ptr_A + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrB_data_j = ptr_B + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                T1 const* __restrict__ ptrB_data_i = ptrB_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                   + beta * *ptrB_data_i;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                    for (int p = 1; p <= FDn; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                    ++ptrB_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
                ptrA_data_j+=result_ni;
                ptrB_data_j+=result_ni;
            }
        }
        break;
    }

    #endif //USE_OPENMP
    return;
}
template void calc_laplacian_d3_c2_o4<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_3D& region, float* const ptr, const Vertices_3D& vertices,
                                            float const* const ptr_A, const float alpha,
                                            float const* const ptr_B, const float beta);
template void calc_laplacian_d3_c2_o4<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_3D& region, double* const ptr, const Vertices_3D& vertices,
                                            double const* const ptr_A, const double alpha,
                                            double const* const ptr_B, const double beta);
template void calc_laplacian_d3_c2_o4<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                                const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices,
                                                std::complex<float> const* const ptr_A, const float alpha,
                                                std::complex<float> const* const ptr_B, const float beta);
template void calc_laplacian_d3_c2_o4<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                                const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices,
                                                std::complex<double> const* const ptr_A, const double alpha,
                                                std::complex<double> const* const ptr_B, const double beta);

template<typename T1, typename T2>
void calc_laplacian_d3_c2_o4(T1 const* const __restrict__ ex_ptr, const Vertices_3D& ex_vertices, const Stencil<T2>& stencil, const Vertices_3D& region,
    T1* const __restrict__ ptr, const Vertices_3D& vertices, T1 const* const __restrict__ ptr_A, const T2 alpha,
    T1 const* const __restrict__ ptr_B, const T2 beta, const T2 gamma) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0];

    switch (FDn)
    {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nk - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrA_data_j = ptr_A + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrB_data_j = ptr_B + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                T1 const* __restrict__ ptrB_data_i = ptrB_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                   + beta * *ptrB_data_i + gamma;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                    for (int p = 1; p <= 6; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                    ++ptrB_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
                ptrA_data_j+=result_ni;
                ptrB_data_j+=result_ni;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nk - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t k = 0; k < (int64_t)region.nk; k++)
        {
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_ninj * k;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrA_data_j = ptr_A + index_result_origin + result_ninj * k;
            T1 const* __restrict__ ptrB_data_j = ptr_B + index_result_origin + result_ninj * k;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_i = this_data_j;
                T1* __restrict__ result_data_i = result_data_j;
                T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                T1 const* __restrict__ ptrB_data_i = ptrB_data_j;
                #ifdef USE_OPENMP_SIMD
                #pragma omp simd
                #endif
                for (int64_t i = 0; i < (int64_t)region.ni; i++)
                {
                    *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                   + beta * *ptrB_data_i + gamma;
                    T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                    for (int p = 1; p <= FDn; p++)
                    {
                        const int64_t stride_r_y = p * this_ni;
                        const int64_t stride_r_z = p * this_ninj;
                        T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                        *result_data_i += res;
                    }
                    ++this_data_i;
                    ++result_data_i;
                    ++ptrA_data_i;
                    ++ptrB_data_i;
                }
                this_data_j+=this_ni;
                result_data_j+=result_ni;
                ptrA_data_j+=result_ni;
                ptrB_data_j+=result_ni;
            }
        }
        break;
    }
    return;
}
template void calc_laplacian_d3_c2_o4<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_3D& region, float* const ptr, const Vertices_3D& vertices,
                                            float const* const ptr_A, const float alpha,
                                            float const* const ptr_B, const float beta,
                                            const float gamma);
template void calc_laplacian_d3_c2_o4<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_3D& region, double* const ptr, const Vertices_3D& vertices,
                                            double const* const ptr_A, const double alpha,
                                            double const* const ptr_B, const double beta,
                                            const double gamma);
template void calc_laplacian_d3_c2_o4<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<float>& stencil,
                                                const Vertices_3D& region, std::complex<float>* const ptr, const Vertices_3D& vertices,
                                                std::complex<float> const* const ptr_A, const float alpha,
                                                std::complex<float> const* const ptr_B, const float beta,
                                                const float gamma);
template void calc_laplacian_d3_c2_o4<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_3D& ex_vertices, const Stencil<double>& stencil,  
                                                const Vertices_3D& region, std::complex<double>* const ptr, const Vertices_3D& vertices,
                                                std::complex<double> const* const ptr_A, const double alpha,
                                                std::complex<double> const* const ptr_B, const double beta,
                                                const double gamma);

template<typename T1, typename T2>
void calc_laplacian_d4(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
    T1* const __restrict__ ptr, const Vertices_4D& vertices) {
    switch(stencil.cell_type) {
    case 0:
    case 1:
    case 2:
        switch(stencil.optimization) {
        case 4:
            calc_laplacian_d4_c2_o4(ex_ptr, ex_vertices, stencil, region, ptr, vertices);
            break;
        default:
            calc_laplacian_d4_c2_o0(ex_ptr, ex_vertices, stencil, region, ptr, vertices);
        }
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}
template void calc_laplacian_d4<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                        const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices);
template void calc_laplacian_d4<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                        const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices);
template void calc_laplacian_d4<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                        const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices);
template void calc_laplacian_d4<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                        const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices);

template<typename T1, typename T2>
void calc_laplacian_d4(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
    T1* const __restrict__ ptr, const Vertices_4D& vertices, T1 const* const __restrict__ ptr_A) {
    switch(stencil.cell_type) {
    case 0:
    case 1:
    case 2:
        switch(stencil.optimization) {
        case 4:
            calc_laplacian_d4_c2_o4(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                    ptr_A);
            break;
        default:
            calc_laplacian_d4_c2_o0(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                    ptr_A);
        }
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}
template void calc_laplacian_d4<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                        const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices,
                                        float const* const ptr_A);
template void calc_laplacian_d4<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                        const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices,
                                        double const* const ptr_A);
template void calc_laplacian_d4<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices,
                                            std::complex<float> const* const ptr_A);
template void calc_laplacian_d4<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                        const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices,
                                        std::complex<double> const* const ptr_A);

template<typename T1, typename T2>
void calc_laplacian_d4(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
    T1* const __restrict__ ptr, const Vertices_4D& vertices, T1 const* const __restrict__ ptr_A, const T2 alpha) {
    switch(stencil.cell_type) {
    case 0:
    case 1:
    case 2:
        switch(stencil.optimization) {
        case 4:
            calc_laplacian_d4_c2_o4(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                    ptr_A, alpha);
            break;
        default:
            calc_laplacian_d4_c2_o0(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                    ptr_A, alpha);
        }
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}
template void calc_laplacian_d4<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                        const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices,
                                        float const* const ptr_A, const float alpha);
template void calc_laplacian_d4<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                        const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices,
                                        double const* const ptr_A, const double alpha);
template void calc_laplacian_d4<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                        const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices,
                                        std::complex<float> const* const ptr_A, const float alpha);
template void calc_laplacian_d4<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,    
                                        const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices,
                                        std::complex<double> const* const ptr_A, const double alpha);

template<typename T1, typename T2>
void calc_laplacian_d4(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
                        T1* const __restrict__ ptr, const Vertices_4D& vertices, T1 const* const __restrict__ ptr_A, const T2 alpha,
                        T1 const* const __restrict__ ptr_B, const T2 beta) {
    switch(stencil.cell_type) {
    case 0:
    case 1:
    case 2:
        switch(stencil.optimization) {
        case 4:
            calc_laplacian_d4_c2_o4(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                    ptr_A, alpha, ptr_B, beta);
            break;
        default:
            calc_laplacian_d4_c2_o0(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                    ptr_A, alpha, ptr_B, beta);
        }
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}
template void calc_laplacian_d4<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                        const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices,
                                        float const* const ptr_A, const float alpha,
                                        float const* const ptr_B, const float beta);
template void calc_laplacian_d4<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                        const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices,
                                        double const* const ptr_A, const double alpha,
                                        double const* const ptr_B, const double beta);
template void calc_laplacian_d4<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices,
                                            std::complex<float> const* const ptr_A, const float alpha,
                                            std::complex<float> const* const ptr_B, const float beta);
template void calc_laplacian_d4<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices,
                                            std::complex<double> const* const ptr_A, const double alpha,
                                            std::complex<double> const* const ptr_B, const double beta);

template<typename T1, typename T2>
void calc_laplacian_d4(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
                    T1* const __restrict__ ptr, const Vertices_4D& vertices, T1 const* const __restrict__ ptr_A, const T2 alpha,
                    T1 const* const __restrict__ ptr_B, const T2 beta,
                    const T2 gamma) {
    switch(stencil.cell_type) {
    case 0:
    case 1:
    case 2:
        switch(stencil.optimization) {
        case 4:
            calc_laplacian_d4_c2_o4(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                    ptr_A, alpha, ptr_B, beta, gamma);
            break;
        default:
            calc_laplacian_d4_c2_o0(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                    ptr_A, alpha, ptr_B, beta, gamma);
        }
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}
template void calc_laplacian_d4<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                        const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices,
                                        float const* const ptr_A, const float alpha,
                                        float const* const ptr_B, const float beta,
                                        const float gamma);
template void calc_laplacian_d4<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                        const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices,
                                        double const* const ptr_A, const double alpha,
                                        double const* const ptr_B, const double beta,
                                        const double gamma);
template void calc_laplacian_d4<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices,
                                            std::complex<float> const* const ptr_A, const float alpha,
                                            std::complex<float> const* const ptr_B, const float beta,
                                            const float gamma);
template void calc_laplacian_d4<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices,
                                            std::complex<double> const* const ptr_A, const double alpha,
                                            std::complex<double> const* const ptr_B, const double beta,
                                            const double gamma);

template<typename T1, typename T2>
void calc_laplacian_d4_c2_o0(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
    T1* const __restrict__ ptr, const Vertices_4D& vertices) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int64_t this_ninjnk = ex_vertices.ni * ex_vertices.nj * ex_vertices.nk;
    const int64_t result_ninjnk = vertices.ni * vertices.nj * vertices.nk;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0] + D2_coeffs_xyz[1] + D2_coeffs_xyz[2];

    switch (FDn)
    {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = coef_0  * *this_data_i;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                        for (int p = 1; p <= 6; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                            T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                            T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res_x + res_y + res_z;
                        }
                        ++this_data_i;
                        ++result_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = coef_0  * *this_data_i;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                        for (int p = 1; p <= FDn; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                            T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                            T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res_x + res_y + res_z;
                        }
                        ++this_data_i;
                        ++result_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
            }
        }
        break;
    }
    return;
}
template void calc_laplacian_d4_c2_o0<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices);
template void calc_laplacian_d4_c2_o0<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices);
template void calc_laplacian_d4_c2_o0<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                                const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices);
template void calc_laplacian_d4_c2_o0<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                                const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices);

template<typename T1, typename T2>
void calc_laplacian_d4_c2_o0(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
    T1* const __restrict__ ptr, const Vertices_4D& vertices, T1 const* const __restrict__ ptr_A) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int64_t this_ninjnk = ex_vertices.ni * ex_vertices.nj * ex_vertices.nk;
    const int64_t result_ninjnk = vertices.ni * vertices.nj * vertices.nk;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0] + D2_coeffs_xyz[1] + D2_coeffs_xyz[2];

    switch (FDn)
    {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrA_data_k = ptr_A + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                T1 const* __restrict__ ptrA_data_j = ptrA_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = (coef_0 + *ptrA_data_i) * *this_data_i;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                        for (int p = 1; p <= 6; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                            T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                            T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res_x + res_y + res_z;
                        }
                        ++this_data_i;
                        ++result_data_i;
                        ++ptrA_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                    ptrA_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
                ptrA_data_k+=result_ninj;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrA_data_k = ptr_A + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                T1 const* __restrict__ ptrA_data_j = ptrA_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = (coef_0 + *ptrA_data_i) * *this_data_i;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                        for (int p = 1; p <= FDn; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                            T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                            T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res_x + res_y + res_z;
                        }
                        ++this_data_i;
                        ++result_data_i;
                        ++ptrA_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                    ptrA_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
                ptrA_data_k+=result_ninj;
            }
        }
        break;
    }
    return;
}
template void calc_laplacian_d4_c2_o0<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices,
                                            float const* const ptr_A);
template void calc_laplacian_d4_c2_o0<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices,
                                            double const* const ptr_A);
template void calc_laplacian_d4_c2_o0<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                        const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices,
                                        std::complex<float> const* const ptr_A);
template void calc_laplacian_d4_c2_o0<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                        const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices,
                                        std::complex<double> const* const ptr_A);

template<typename T1, typename T2>
void calc_laplacian_d4_c2_o0(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
    T1* const __restrict__ ptr, const Vertices_4D& vertices, T1 const* const __restrict__ ptr_A, const T2 alpha) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int64_t this_ninjnk = ex_vertices.ni * ex_vertices.nj * ex_vertices.nk;
    const int64_t result_ninjnk = vertices.ni * vertices.nj * vertices.nk;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0] + D2_coeffs_xyz[1] + D2_coeffs_xyz[2];

    switch (FDn)
    {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrA_data_k = ptr_A + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                T1 const* __restrict__ ptrA_data_j = ptrA_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                        for (int p = 1; p <= 6; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                            T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                            T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res_x + res_y + res_z;
                        }
                        ++this_data_i;
                        ++result_data_i;
                        ++ptrA_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                    ptrA_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
                ptrA_data_k+=result_ninj;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrA_data_k = ptr_A + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                T1 const* __restrict__ ptrA_data_j = ptrA_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                        for (int p = 1; p <= FDn; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                            T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                            T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res_x + res_y + res_z;
                        }
                        ++this_data_i;
                        ++result_data_i;
                        ++ptrA_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                    ptrA_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
                ptrA_data_k+=result_ninj;
            }
        }
        break;
    }
    return;
}
template void calc_laplacian_d4_c2_o0<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices,
                                            float const* const ptr_A, const float alpha);
template void calc_laplacian_d4_c2_o0<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices,
                                            double const* const ptr_A, const double alpha);
template void calc_laplacian_d4_c2_o0<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                                                const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices,
                                                                std::complex<float> const* const ptr_A, const float alpha);
template void calc_laplacian_d4_c2_o0<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                                                const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices,
                                                                std::complex<double> const* const ptr_A, const double alpha);

template<typename T1, typename T2>
void calc_laplacian_d4_c2_o0(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
    T1* const __restrict__ ptr, const Vertices_4D& vertices, T1 const* const __restrict__ ptr_A, const T2 alpha,
    T1 const* const __restrict__ ptr_B, const T2 beta) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int64_t this_ninjnk = ex_vertices.ni * ex_vertices.nj * ex_vertices.nk;
    const int64_t result_ninjnk = vertices.ni * vertices.nj * vertices.nk;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0] + D2_coeffs_xyz[1] + D2_coeffs_xyz[2];

    switch (FDn)
    {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrA_data_k = ptr_A + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrB_data_k = ptr_B + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                T1 const* __restrict__ ptrA_data_j = ptrA_data_k;
                T1 const* __restrict__ ptrB_data_j = ptrB_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                    T1 const* __restrict__ ptrB_data_i = ptrB_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                       + beta * *ptrB_data_i;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                        for (int p = 1; p <= 6; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                            T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                            T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res_x + res_y + res_z;
                        }
                        ++this_data_i;
                        ++result_data_i;
                        ++ptrA_data_i;
                        ++ptrB_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                    ptrA_data_j+=result_ni;
                    ptrB_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
                ptrA_data_k+=result_ninj;
                ptrB_data_k+=result_ninj;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrA_data_k = ptr_A + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrB_data_k = ptr_B + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                T1 const* __restrict__ ptrA_data_j = ptrA_data_k;
                T1 const* __restrict__ ptrB_data_j = ptrB_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                    T1 const* __restrict__ ptrB_data_i = ptrB_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                       + beta * *ptrB_data_i;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                        for (int p = 1; p <= FDn; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                            T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                            T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res_x + res_y + res_z;
                        }
                        ++this_data_i;
                        ++result_data_i;
                        ++ptrA_data_i;
                        ++ptrB_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                    ptrA_data_j+=result_ni;
                    ptrB_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
                ptrA_data_k+=result_ninj;
                ptrB_data_k+=result_ninj;
            }
        }
        break;
    }
    return;
}
template void calc_laplacian_d4_c2_o0<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices,
                                            float const* const ptr_A, const float alpha,
                                            float const* const ptr_B, const float beta);
template void calc_laplacian_d4_c2_o0<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices,
                                            double const* const ptr_A, const double alpha,
                                            double const* const ptr_B, const double beta);
template void calc_laplacian_d4_c2_o0<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                                const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices,
                                                std::complex<float> const* const ptr_A, const float alpha,
                                                std::complex<float> const* const ptr_B, const float beta);
template void calc_laplacian_d4_c2_o0<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                                const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices,
                                                std::complex<double> const* const ptr_A, const double alpha,
                                                std::complex<double> const* const ptr_B, const double beta);

template<typename T1, typename T2>
void calc_laplacian_d4_c2_o0(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
        T1* const __restrict__ ptr, const Vertices_4D& vertices, T1 const* const __restrict__ ptr_A, const T2 alpha,
        T1 const* const __restrict__ ptr_B, const T2 beta,
        const T2 gamma) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int64_t this_ninjnk = ex_vertices.ni * ex_vertices.nj * ex_vertices.nk;
    const int64_t result_ninjnk = vertices.ni * vertices.nj * vertices.nk;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0] + D2_coeffs_xyz[1] + D2_coeffs_xyz[2];

    switch (FDn)
    {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrA_data_k = ptr_A + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrB_data_k = ptr_B + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                T1 const* __restrict__ ptrA_data_j = ptrA_data_k;
                T1 const* __restrict__ ptrB_data_j = ptrB_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                    T1 const* __restrict__ ptrB_data_i = ptrB_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                       + beta * *ptrB_data_i + gamma;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                        for (int p = 1; p <= 6; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                            T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                            T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res_x + res_y + res_z;
                        }
                        ++this_data_i;
                        ++result_data_i;
                        ++ptrA_data_i;
                        ++ptrB_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                    ptrA_data_j+=result_ni;
                    ptrB_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
                ptrA_data_k+=result_ninj;
                ptrB_data_k+=result_ninj;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrA_data_k = ptr_A + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrB_data_k = ptr_B + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                T1 const* __restrict__ ptrA_data_j = ptrA_data_k;
                T1 const* __restrict__ ptrB_data_j = ptrB_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                    T1 const* __restrict__ ptrB_data_i = ptrB_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                       + beta * *ptrB_data_i + gamma;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                        for (int p = 1; p <= FDn; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                            T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                            T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res_x + res_y + res_z;
                        }
                        ++this_data_i;
                        ++result_data_i;
                        ++ptrA_data_i;
                        ++ptrB_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                    ptrA_data_j+=result_ni;
                    ptrB_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
                ptrA_data_k+=result_ninj;
                ptrB_data_k+=result_ninj;
            }
        }
        break;
    }
    return;
}
template void calc_laplacian_d4_c2_o0<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices,
                                            float const* const ptr_A, const float alpha,
                                            float const* const ptr_B, const float beta,
                                            const float gamma);
template void calc_laplacian_d4_c2_o0<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices,
                                            double const* const ptr_A, const double alpha,
                                            double const* const ptr_B, const double beta,
                                            const double gamma);
template void calc_laplacian_d4_c2_o0<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                                                const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices,
                                                                std::complex<float> const* const ptr_A, const float alpha,
                                                                std::complex<float> const* const ptr_B, const float beta,
                                                                const float gamma);
template void calc_laplacian_d4_c2_o0<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                                                const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices,
                                                                std::complex<double> const* const ptr_A, const double alpha,
                                                                std::complex<double> const* const ptr_B, const double beta,
                                                                const double gamma);

template<typename T1, typename T2>
void calc_laplacian_d4_c2_o4(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
    T1* const __restrict__ ptr, const Vertices_4D& vertices) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int64_t this_ninjnk = ex_vertices.ni * ex_vertices.nj * ex_vertices.nk;
    const int64_t result_ninjnk = vertices.ni * vertices.nj * vertices.nk;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0];

    switch (FDn)
    {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = coef_0  * *this_data_i;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                        for (int p = 1; p <= 6; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                                + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                                + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res;
                        }
                        ++this_data_i;
                        ++result_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = coef_0  * *this_data_i;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                        for (int p = 1; p <= FDn; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                                + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                                + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res;
                        }
                        ++this_data_i;
                        ++result_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
            }
        }
        break;
    }
    return;
}
template void calc_laplacian_d4_c2_o4<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices);
template void calc_laplacian_d4_c2_o4<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices);
template void calc_laplacian_d4_c2_o4<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
    const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices);
template void calc_laplacian_d4_c2_o4<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
    const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices);

template<typename T1, typename T2>
void calc_laplacian_d4_c2_o4(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
    T1* const __restrict__ ptr, const Vertices_4D& vertices, T1 const* const __restrict__ ptr_A) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int64_t this_ninjnk = ex_vertices.ni * ex_vertices.nj * ex_vertices.nk;
    const int64_t result_ninjnk = vertices.ni * vertices.nj * vertices.nk;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0];

    switch (FDn)
    {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrA_data_k = ptr_A + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                T1 const* __restrict__ ptrA_data_j = ptrA_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = (coef_0 + *ptrA_data_i) * *this_data_i;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                        for (int p = 1; p <= 6; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res;
                        }
                        ++this_data_i;
                        ++result_data_i;
                        ++ptrA_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                    ptrA_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
                ptrA_data_k+=result_ninj;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrA_data_k = ptr_A + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                T1 const* __restrict__ ptrA_data_j = ptrA_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = (coef_0 + *ptrA_data_i) * *this_data_i;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                        for (int p = 1; p <= FDn; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res;
                        }
                        ++this_data_i;
                        ++result_data_i;
                        ++ptrA_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                    ptrA_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
                ptrA_data_k+=result_ninj;
            }
        }
        break;
    }
    return;
}
template void calc_laplacian_d4_c2_o4<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices,
                                            float const* const ptr_A);
template void calc_laplacian_d4_c2_o4<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices,
                                            double const* const ptr_A);
template void calc_laplacian_d4_c2_o4<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                                const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices,
                                                std::complex<float> const* const ptr_A);
template void calc_laplacian_d4_c2_o4<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                                const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices,
                                                std::complex<double> const* const ptr_A);

template<typename T1, typename T2>
void calc_laplacian_d4_c2_o4(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
    T1* const __restrict__ ptr, const Vertices_4D& vertices, T1 const* const __restrict__ ptr_A, const T2 alpha) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int64_t this_ninjnk = ex_vertices.ni * ex_vertices.nj * ex_vertices.nk;
    const int64_t result_ninjnk = vertices.ni * vertices.nj * vertices.nk;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0];

    switch (FDn)
    {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrA_data_k = ptr_A + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                T1 const* __restrict__ ptrA_data_j = ptrA_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                        for (int p = 1; p <= 6; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res;
                        }
                        ++this_data_i;
                        ++result_data_i;
                        ++ptrA_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                    ptrA_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
                ptrA_data_k+=result_ninj;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrA_data_k = ptr_A + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                T1 const* __restrict__ ptrA_data_j = ptrA_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                        for (int p = 1; p <= FDn; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res;
                        }
                        ++this_data_i;
                        ++result_data_i;
                        ++ptrA_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                    ptrA_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
                ptrA_data_k+=result_ninj;
            }
        }
        break;
    }
    return;
}
template void calc_laplacian_d4_c2_o4<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices,
                                            float const* const ptr_A, const float alpha);
template void calc_laplacian_d4_c2_o4<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices,
                                            double const* const ptr_A, const double alpha);
template void calc_laplacian_d4_c2_o4<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                        const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices,
                                        std::complex<float> const* const ptr_A, const float alpha);
template void calc_laplacian_d4_c2_o4<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                        const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices,
                                        std::complex<double> const* const ptr_A, const double alpha);

template<typename T1, typename T2>
void calc_laplacian_d4_c2_o4(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
    T1* const __restrict__ ptr, const Vertices_4D& vertices, T1 const* const __restrict__ ptr_A, const T2 alpha,
    T1 const* const __restrict__ ptr_B, const T2 beta) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int64_t this_ninjnk = ex_vertices.ni * ex_vertices.nj * ex_vertices.nk;
    const int64_t result_ninjnk = vertices.ni * vertices.nj * vertices.nk;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0];

    switch (FDn)
    {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrA_data_k = ptr_A + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrB_data_k = ptr_B + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                T1 const* __restrict__ ptrA_data_j = ptrA_data_k;
                T1 const* __restrict__ ptrB_data_j = ptrB_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                    T1 const* __restrict__ ptrB_data_i = ptrB_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                       + beta * *ptrB_data_i;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                        for (int p = 1; p <= 6; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res;
                        }
                        ++this_data_i;
                        ++result_data_i;
                        ++ptrA_data_i;
                        ++ptrB_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                    ptrA_data_j+=result_ni;
                    ptrB_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
                ptrA_data_k+=result_ninj;
                ptrB_data_k+=result_ninj;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrA_data_k = ptr_A + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrB_data_k = ptr_B + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                T1 const* __restrict__ ptrA_data_j = ptrA_data_k;
                T1 const* __restrict__ ptrB_data_j = ptrB_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                    T1 const* __restrict__ ptrB_data_i = ptrB_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                       + beta * *ptrB_data_i;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                        for (int p = 1; p <= FDn; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res;
                        }
                        ++this_data_i;
                        ++result_data_i;
                        ++ptrA_data_i;
                        ++ptrB_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                    ptrA_data_j+=result_ni;
                    ptrB_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
                ptrA_data_k+=result_ninj;
                ptrB_data_k+=result_ninj;
            }
        }
        break;
    }
    return;
}
template void calc_laplacian_d4_c2_o4<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                            const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices,
                                            float const* const ptr_A, const float alpha,
                                            float const* const ptr_B, const float beta);
template void calc_laplacian_d4_c2_o4<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices,
                                            double const* const ptr_A, const double alpha,
                                            double const* const ptr_B, const double beta);
template void calc_laplacian_d4_c2_o4<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                                                const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices,
                                                                std::complex<float> const* const ptr_A, const float alpha,
                                                                std::complex<float> const* const ptr_B, const float beta);
template void calc_laplacian_d4_c2_o4<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                                                const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices,
                                                                std::complex<double> const* const ptr_A, const double alpha,
                                                                std::complex<double> const* const ptr_B, const double beta);

template<typename T1, typename T2>
void calc_laplacian_d4_c2_o4(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
                            T1* const __restrict__ ptr, const Vertices_4D& vertices, T1 const* const __restrict__ ptr_A, const T2 alpha,
                            T1 const* const __restrict__ ptr_B, const T2 beta, const T2 gamma) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int64_t this_ninjnk = ex_vertices.ni * ex_vertices.nj * ex_vertices.nk;
    const int64_t result_ninjnk = vertices.ni * vertices.nj * vertices.nk;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0];

    switch (FDn)
    {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrA_data_k = ptr_A + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrB_data_k = ptr_B + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                T1 const* __restrict__ ptrA_data_j = ptrA_data_k;
                T1 const* __restrict__ ptrB_data_j = ptrB_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                    T1 const* __restrict__ ptrB_data_i = ptrB_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                       + beta * *ptrB_data_i + gamma;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                        for (int p = 1; p <= 6; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res;
                        }
                        ++this_data_i;
                        ++result_data_i;
                        ++ptrA_data_i;
                        ++ptrB_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                    ptrA_data_j+=result_ni;
                    ptrB_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
                ptrA_data_k+=result_ninj;
                ptrB_data_k+=result_ninj;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            T1 const* __restrict__ this_data_k = ex_ptr + index_this_origin + this_ninjnk * b;
            T1* __restrict__ result_data_k = ptr + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrA_data_k = ptr_A + index_result_origin + result_ninjnk * b;
            T1 const* __restrict__ ptrB_data_k = ptr_B + index_result_origin + result_ninjnk * b;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                T1 const* __restrict__ this_data_j = this_data_k;
                T1* __restrict__ result_data_j = result_data_k;
                T1 const* __restrict__ ptrA_data_j = ptrA_data_k;
                T1 const* __restrict__ ptrB_data_j = ptrB_data_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    T1 const* __restrict__ this_data_i = this_data_j;
                    T1* __restrict__ result_data_i = result_data_j;
                    T1 const* __restrict__ ptrA_data_i = ptrA_data_j;
                    T1 const* __restrict__ ptrB_data_i = ptrB_data_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        *result_data_i = (coef_0 + alpha * *ptrA_data_i) * *this_data_i
                                       + beta * *ptrB_data_i + gamma;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                        for (int p = 1; p <= FDn; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res = (*(this_data_i+p)          + *(this_data_i-p)
                               + *(this_data_i+stride_r_y) + *(this_data_i-stride_r_y)
                               + *(this_data_i+stride_r_z) + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_i += res;
                        }
                        ++this_data_i;
                        ++result_data_i;
                        ++ptrA_data_i;
                        ++ptrB_data_i;
                    }
                    this_data_j+=this_ni;
                    result_data_j+=result_ni;
                    ptrA_data_j+=result_ni;
                    ptrB_data_j+=result_ni;
                }
                this_data_k+=this_ninj;
                result_data_k+=result_ninj;
                ptrA_data_k+=result_ninj;
                ptrB_data_k+=result_ninj;
            }
        }
        break;
    }
    return;
}
template void calc_laplacian_d4_c2_o4<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                                const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices,
                                                float const* const ptr_A, const float alpha,
                                                float const* const ptr_B, const float beta,
                                                const float gamma);
template void calc_laplacian_d4_c2_o4<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                            const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices,
                                            double const* const ptr_A, const double alpha,
                                            double const* const ptr_B, const double beta,
                                            const double gamma);
template void calc_laplacian_d4_c2_o4<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                                                const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices,
                                                                std::complex<float> const* const ptr_A, const float alpha,
                                                                std::complex<float> const* const ptr_B, const float beta,
                                                                const float gamma);      
template void calc_laplacian_d4_c2_o4<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                                                 const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices,
                                                                 std::complex<double> const* const ptr_A, const double alpha,
                                                                 std::complex<double> const* const ptr_B, const double beta,
                                                                 const double gamma);                                                     
template<typename T1, typename T2>
void Special::calc_laplacian_d4(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
    T1* const __restrict__ ptr, const Vertices_4D& vertices, T1 const* const __restrict__ ptr_A) {
    switch(stencil.cell_type) {
    case 0:
    case 1:
    case 2:
        switch(stencil.optimization) {
        case 4:
            Special::calc_laplacian_d4_c2_o4(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                             ptr_A);
            break;
        default:
            Special::calc_laplacian_d4_c2_o0(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                             ptr_A);
        }
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}
template void Special::calc_laplacian_d4<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                                const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices,
                                                float const* const ptr_A);
template void Special::calc_laplacian_d4<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                                const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices,
                                                double const* const ptr_A);
template void Special::calc_laplacian_d4<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                                    const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices,
                                                    std::complex<float> const* const ptr_A);
template void Special::calc_laplacian_d4<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                                    const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices,
                                                    std::complex<double> const* const ptr_A);

template<typename T1, typename T2>
void Special::calc_laplacian_d4_c2_o0(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
    T1* const __restrict__ ptr, const Vertices_4D& vertices, T1 const* const __restrict__ ptr_A) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int64_t this_ninjnk = ex_vertices.ni * ex_vertices.nj * ex_vertices.nk;
    const int64_t result_ninjnk = vertices.ni * vertices.nj * vertices.nk;
    const int FDn = stencil.FDn;
    T1 const* const __restrict__ this_data = ex_ptr;
    T1* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0] + D2_coeffs_xyz[1] + D2_coeffs_xyz[2];

    switch(FDn) {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            int64_t this_offset_b = index_this_origin + this_ninjnk * b;
            int64_t result_offset_b = index_result_origin + result_ninjnk * b;
            int64_t this_offset_k = this_offset_b;
            int64_t result_offset_k = result_offset_b;
            int64_t ptr_A_offset_k = index_result_origin;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                int64_t this_offset_j = this_offset_k;
                int64_t result_offset_j = result_offset_k;
                int64_t ptr_A_offset_j = ptr_A_offset_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    int64_t this_offset_i = this_offset_j;
                    int64_t result_offset_i = result_offset_j;
                    int64_t ptr_A_offset_i = ptr_A_offset_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        result_data[result_offset_i] = (coef_0 + ptr_A[ptr_A_offset_i]) * this_data[this_offset_i];
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                        for (int p = 1; p <= 6; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res_x = (this_data[this_offset_i+p]          + this_data[this_offset_i-p])          * *(D2_coeffs_xyz_p3++);
                            T1 res_y = (this_data[this_offset_i+stride_r_y] + this_data[this_offset_i-stride_r_y]) * *(D2_coeffs_xyz_p3++);
                            T1 res_z = (this_data[this_offset_i+stride_r_z] + this_data[this_offset_i-stride_r_z]) * *(D2_coeffs_xyz_p3++);
                            result_data[result_offset_i] += res_x + res_y + res_z;
                        }
                        this_offset_i++;
                        result_offset_i++;
                        ptr_A_offset_i++;
                    }
                    this_offset_j += this_ni;
                    result_offset_j += result_ni;
                    ptr_A_offset_j += result_ni;
                }
                this_offset_k += this_ninj;
                result_offset_k += result_ninj;
                ptr_A_offset_k += result_ninj;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            int64_t this_offset_b = index_this_origin + this_ninjnk * b;
            int64_t result_offset_b = index_result_origin + result_ninjnk * b;
            int64_t this_offset_k = this_offset_b;
            int64_t result_offset_k = result_offset_b;
            int64_t ptr_A_offset_k = index_result_origin;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                int64_t this_offset_j = this_offset_k;
                int64_t result_offset_j = result_offset_k;
                int64_t ptr_A_offset_j = ptr_A_offset_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    int64_t this_offset_i = this_offset_j;
                    int64_t result_offset_i = result_offset_j;
                    int64_t ptr_A_offset_i = ptr_A_offset_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        result_data[result_offset_i] = (coef_0 + ptr_A[ptr_A_offset_i]) * this_data[this_offset_i];
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                        for (int p = 1; p <= FDn; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res_x = (this_data[this_offset_i+p]          + this_data[this_offset_i-p])          * *(D2_coeffs_xyz_p3++);
                            T1 res_y = (this_data[this_offset_i+stride_r_y] + this_data[this_offset_i-stride_r_y]) * *(D2_coeffs_xyz_p3++);
                            T1 res_z = (this_data[this_offset_i+stride_r_z] + this_data[this_offset_i-stride_r_z]) * *(D2_coeffs_xyz_p3++);
                            result_data[result_offset_i] += res_x + res_y + res_z;
                        }
                        this_offset_i++;
                        result_offset_i++;
                        ptr_A_offset_i++;
                    }
                    this_offset_j += this_ni;
                    result_offset_j += result_ni;
                    ptr_A_offset_j += result_ni;
                }
                this_offset_k += this_ninj;
                result_offset_k += result_ninj;
                ptr_A_offset_k += result_ninj;
            }
        }
    }
    return;
}
template void Special::calc_laplacian_d4_c2_o0<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                                    const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices,
                                                    float const* const ptr_A);
template void Special::calc_laplacian_d4_c2_o0<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                                    const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices,
                                                    double const* const ptr_A);
template void Special::calc_laplacian_d4_c2_o0<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                                        const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices,
                                                        std::complex<float> const* const ptr_A);
template void Special::calc_laplacian_d4_c2_o0<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                                        const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices,
                                                        std::complex<double> const* const ptr_A);

template<typename T1, typename T2>
void Special::calc_laplacian_d4_c2_o4(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
    T1* const __restrict__ ptr, const Vertices_4D& vertices, T1 const* const __restrict__ ptr_A) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks, region.bs);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int64_t this_ninjnk = ex_vertices.ni * ex_vertices.nj * ex_vertices.nk;
    const int64_t result_ninjnk = vertices.ni * vertices.nj * vertices.nk;
    const int FDn = stencil.FDn;
    T1 const* const __restrict__ this_data = ex_ptr;
    T1* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0];

    switch(FDn) {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            int64_t this_offset_b = index_this_origin + this_ninjnk * b;
            int64_t result_offset_b = index_result_origin + result_ninjnk * b;
            int64_t this_offset_k = this_offset_b;
            int64_t result_offset_k = result_offset_b;
            int64_t ptr_A_offset_k = index_result_origin;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                int64_t this_offset_j = this_offset_k;
                int64_t result_offset_j = result_offset_k;
                int64_t ptr_A_offset_j = ptr_A_offset_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    int64_t this_offset_i = this_offset_j;
                    int64_t result_offset_i = result_offset_j;
                    int64_t ptr_A_offset_i = ptr_A_offset_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        result_data[result_offset_i] = (coef_0 + ptr_A[ptr_A_offset_i]) * this_data[this_offset_i];
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                        for (int p = 1; p <= 6; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res = (this_data[this_offset_i+p]          + this_data[this_offset_i-p]
                                        + this_data[this_offset_i+stride_r_y] + this_data[this_offset_i-stride_r_y]
                                        + this_data[this_offset_i+stride_r_z] + this_data[this_offset_i-stride_r_z]) * *(D2_coeffs_xyz_p3++);
                            result_data[result_offset_i] += res;
                        }
                        this_offset_i++;
                        result_offset_i++;
                        ptr_A_offset_i++;
                    }
                    this_offset_j += this_ni;
                    result_offset_j += result_ni;
                    ptr_A_offset_j += result_ni;
                }
                this_offset_k += this_ninj;
                result_offset_k += result_ninj;
                ptr_A_offset_k += result_ninj;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.nb - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t b = 0; b < (int64_t)region.nb; b++){
            int64_t this_offset_b = index_this_origin + this_ninjnk * b;
            int64_t result_offset_b = index_result_origin + result_ninjnk * b;
            int64_t this_offset_k = this_offset_b;
            int64_t result_offset_k = result_offset_b;
            int64_t ptr_A_offset_k = index_result_origin;
            for (int64_t k = 0; k < (int64_t)region.nk; k++)
            {
                int64_t this_offset_j = this_offset_k;
                int64_t result_offset_j = result_offset_k;
                int64_t ptr_A_offset_j = ptr_A_offset_k;
                for (int64_t j = 0; j < (int64_t)region.nj; j++)
                {
                    int64_t this_offset_i = this_offset_j;
                    int64_t result_offset_i = result_offset_j;
                    int64_t ptr_A_offset_i = ptr_A_offset_j;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t i = 0; i < (int64_t)region.ni; i++)
                    {
                        result_data[result_offset_i] = (coef_0 + ptr_A[ptr_A_offset_i]) * this_data[this_offset_i];
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                        for (int p = 1; p <= FDn; p++)
                        {
                            const int64_t stride_r_y = p * this_ni;
                            const int64_t stride_r_z = p * this_ninj;
                            T1 res = (this_data[this_offset_i+p]          + this_data[this_offset_i-p]
                                        + this_data[this_offset_i+stride_r_y] + this_data[this_offset_i-stride_r_y]
                                        + this_data[this_offset_i+stride_r_z] + this_data[this_offset_i-stride_r_z]) * *(D2_coeffs_xyz_p3++);
                            result_data[result_offset_i] += res;
                        }
                        this_offset_i++;
                        result_offset_i++;
                        ptr_A_offset_i++;
                    }
                    this_offset_j += this_ni;
                    result_offset_j += result_ni;
                    ptr_A_offset_j += result_ni;
                }
                this_offset_k += this_ninj;
                result_offset_k += result_ninj;
                ptr_A_offset_k += result_ninj;
            }
        }
    }
    return;
}
template void Special::calc_laplacian_d4_c2_o4<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                                    const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices,
                                                    float const* const ptr_A);
template void Special::calc_laplacian_d4_c2_o4<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                                    const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices,
                                                    double const* const ptr_A);
template void Special::calc_laplacian_d4_c2_o4<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                                        const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices,
                                                        std::complex<float> const* const ptr_A);
template void Special::calc_laplacian_d4_c2_o4<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                                        const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices,
                                                        std::complex<double> const* const ptr_A);

template<typename T1, typename T2>
void Special::calc_laplacian_d4_c2_o4_rowmaj(T1 const* const __restrict__ ex_ptr, const Vertices_4D& ex_vertices, const Stencil<T2>& stencil, const Vertices_4D& region,
    T1* const __restrict__ ptr, const Vertices_4D& vertices, T1 const* const __restrict__ ptr_A) {
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck_rowmaj(region.is, region.js, region.ks, region.bs);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck_rowmaj(region.is, region.js, region.ks, region.bs);
    const int64_t this_nb = ex_vertices.nb;
    const int64_t result_nb = vertices.nb;
    const int64_t this_nknb = ex_vertices.nk * ex_vertices.nb;
    const int64_t result_nknb = vertices.nk * vertices.nb;
    const int64_t this_njnknb = ex_vertices.nj * ex_vertices.nk * ex_vertices.nb;
    const int64_t result_njnknb = vertices.nj * vertices.nk * vertices.nb;
    const int FDn = stencil.FDn;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0];

    switch(FDn) {
    case 6:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.ni - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t i = 0; i < (int64_t)region.ni; i++){
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_njnknb * i;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_njnknb * i;
            T1 const* __restrict__ ptrA_data_j = ptr_A + index_result_origin + result_njnknb * i;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_k = this_data_j;
                T1* __restrict__ result_data_k = result_data_j;
                T1 const* __restrict__ ptrA_data_k = ptrA_data_j;
                // #ifdef USE_OPENMP_SIMD
                // #pragma omp simd
                // #endif
                for (int64_t k = 0; k < (int64_t)region.nk; k++)
                {
                    T1 const* __restrict__ this_data_b = this_data_k;
                    T1* __restrict__ result_data_b = result_data_k;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t b = 0; b < (int64_t)region.nb; b++)
                    {
                        *result_data_b = (coef_0 + *ptrA_data_k) * *this_data_b;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                        for (int p = 1; p <= 6; p++)
                        {
                            const int64_t stride_r_x = p * this_njnknb;
                            // const int64_t stride_r_x = p * this_nknb;
                            const int64_t stride_r_y = p * this_nknb;
                            // const int64_t stride_r_y = p * this_nb;
                            const int64_t stride_r_z = p * this_nb;
                            // const int64_t stride_r_z = p;
                            T1 res = (*(this_data_b+stride_r_x) + *(this_data_b-stride_r_x)
                                   + *(this_data_b+stride_r_y) + *(this_data_b-stride_r_y)
                                   + *(this_data_b+stride_r_z) + *(this_data_b-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_b += res;
                        }
                        this_data_b++;
                        result_data_b++;
                    }
                    this_data_k += this_nb;
                    result_data_k += result_nb;
                    ptrA_data_k++;
                }
                this_data_j += this_nknb;
                result_data_j += result_nknb;
                ptrA_data_j += result_nknb;
            }
        }
        break;
    default:
        #ifdef USE_OPENMP
        #pragma omp for schedule(static, (region.ni - 1)/omp_get_num_threads() + 1) nowait
        #endif //USE_OPENMP
        for (int64_t i = 0; i < (int64_t)region.ni; i++){
            T1 const* __restrict__ this_data_j = ex_ptr + index_this_origin + this_njnknb * i;
            T1* __restrict__ result_data_j = ptr + index_result_origin + result_njnknb * i;
            T1 const* __restrict__ ptrA_data_j = ptr_A + index_result_origin + result_njnknb * i;
            for (int64_t j = 0; j < (int64_t)region.nj; j++)
            {
                T1 const* __restrict__ this_data_k = this_data_j;
                T1* __restrict__ result_data_k = result_data_j;
                T1 const* __restrict__ ptrA_data_k = ptrA_data_j;
                // #ifdef USE_OPENMP_SIMD
                // #pragma omp simd
                // #endif
                for (int64_t k = 0; k < (int64_t)region.nk; k++)
                {
                    T1 const* __restrict__ this_data_b = this_data_k;
                    T1* __restrict__ result_data_b = result_data_k;
                    #ifdef USE_OPENMP_SIMD
                    #pragma omp simd
                    #endif
                    for (int64_t b = 0; b < (int64_t)region.nb; b++)
                    {
                        *result_data_b = (coef_0 + *ptrA_data_k) * *this_data_b;
                        T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                        for (int p = 1; p <= FDn; p++)
                        {
                            const int64_t stride_r_x = p * this_njnknb;
                            // const int stride_r_x = p * this_nknb;
                            const int64_t stride_r_y = p * this_nknb;
                            // const int stride_r_y = p * this_nb;
                            const int64_t stride_r_z = p * this_nb;
                            // const int stride_r_z = p;
                            T1 res = (*(this_data_b+stride_r_x) + *(this_data_b-stride_r_x)
                                   + *(this_data_b+stride_r_y) + *(this_data_b-stride_r_y)
                                   + *(this_data_b+stride_r_z) + *(this_data_b-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                            *result_data_b += res;
                        }
                        this_data_b++;
                        result_data_b++;
                    }
                    this_data_k += this_nb;
                    result_data_k += result_nb;
                    ptrA_data_k ++;
                }
                this_data_j += this_nknb;
                result_data_j += result_nknb;
                ptrA_data_j += result_nknb;
            }
        }
    }
    return;
}
template void Special::calc_laplacian_d4_c2_o4_rowmaj<float>(float const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                                            const Vertices_4D& region, float* const ptr, const Vertices_4D& vertices,
                                                            float const* const ptr_A);
template void Special::calc_laplacian_d4_c2_o4_rowmaj<double>(double const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                                            const Vertices_4D& region, double* const ptr, const Vertices_4D& vertices,
                                                            double const* const ptr_A);
template void Special::calc_laplacian_d4_c2_o4_rowmaj<std::complex<float>, float>(std::complex<float> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<float>& stencil,
                                                                const Vertices_4D& region, std::complex<float>* const ptr, const Vertices_4D& vertices,
                                                                std::complex<float> const* const ptr_A);
template void Special::calc_laplacian_d4_c2_o4_rowmaj<std::complex<double>, double>(std::complex<double> const* const ex_ptr, const Vertices_4D& ex_vertices, const Stencil<double>& stencil,
                                                                const Vertices_4D& region, std::complex<double>* const ptr, const Vertices_4D& vertices,
                                                                std::complex<double> const* const ptr_A);

namespace Boundary_safe {

template<typename T1, typename T2>
void calc_laplacian_boundary_safe(T1 const* const& ex_ptr, const Vertices_3D& ex_vertices,
                        const Stencil<T2>& stencil, const Vertices_3D& region,
                        T1* const& ptr, const Vertices_3D& vertices,
                        T1 const* const& ptr_A) {
    switch(stencil.cell_type) {
    case 0:
    case 1:
    case 2:
        switch(stencil.optimization) {
        case 4:
            calc_laplacian_boundary_safe_d3_c2_o4(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                    ptr_A);
            break;
        default:
            calc_laplacian_boundary_safe_d3_c2_o0(ex_ptr, ex_vertices, stencil, region, ptr, vertices,
                                                    ptr_A);
        }
        break;
    default:
        assert(!"ERROR:: only Orthorhombi is supported~");
    }
    return;
}
template void calc_laplacian_boundary_safe<double>(double const* const& ex_ptr, const Vertices_3D& ex_vertices,
                                                    const Stencil<double>& stencil, const Vertices_3D& region,
                                                    double* const& ptr, const Vertices_3D& vertices,
                                                    double const* const& ptr_A);
template void calc_laplacian_boundary_safe<float>(float const* const& ex_ptr, const Vertices_3D& ex_vertices,
                                                    const Stencil<float>& stencil, const Vertices_3D& region,
                                                    float* const& ptr, const Vertices_3D& vertices,
                                                    float const* const& ptr_A);

template<typename T1, typename T2>
void calc_laplacian_boundary_safe_d3_c2_o0(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                        const Stencil<T2>& stencil, const Vertices_3D& region,
                        T1* const ptr, const Vertices_3D& vertices,
                        T1 const* const ptr_A) {
    // (void) ex_ptr;
    // (void) ex_vertices;
    // (void) stencil;
    // (void) region;
    // (void) ptr;
    // (void) vertices;
    // (void) ptr_A;
    // assert(0 && "calc_laplacian_boundary_safe_d3_c2_o0 is not support");
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0] + D2_coeffs_xyz[1] + D2_coeffs_xyz[2];

    #ifdef USE_OPENMP
    #pragma omp for schedule(static, (region.nk * region.nj - 1)/omp_get_num_threads() + 1) collapse(2) nowait
    #endif
    for (int64_t k = 0; k < (int64_t)region.nk; k++)
    {
        for (int64_t j = 0; j < (int64_t)region.nj; j++)
        {
            const int64_t result_offset = index_result_origin + result_ninj * k + result_ni * j;
            T1 const* __restrict__ this_data_i = ex_ptr + index_this_origin + this_ninj * k + this_ni * j;
            T1* __restrict__ result_data_i = ptr + result_offset;
            T1 const* __restrict__ ptrA_data_i = ptr_A + result_offset;
            #ifdef USE_OPENMP_SIMD
            #pragma omp simd
            #endif
            for (int64_t i = 0; i < (int64_t)region.ni; i++)
            {
                *result_data_i = (coef_0 + *ptrA_data_i) * *this_data_i;
                T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 3;
                for (int p = 1; p <= FDn; p++)
                {
                    const int64_t stride_r_y = p * this_ni;
                    const int64_t stride_r_z = p * this_ninj;
                    T1 res_x = ((likely(int64_t(i + region.is + p) < int64_t(ex_vertices.is + ex_vertices.ni)) ? *(this_data_i+p) : T1(0))
                             + (likely(int64_t(i + region.is - p) >= int64_t(ex_vertices.is)) ? *(this_data_i-p) : T1(0)))
                             * *(D2_coeffs_xyz_p3++);
                    T1 res_y = ((likely(int64_t(j + region.js + p) < int64_t(ex_vertices.js + ex_vertices.nj)) ? *(this_data_i+stride_r_y) : T1(0))
                             + (likely(int64_t(j + region.js - p) >= int64_t(ex_vertices.js)) ? *(this_data_i-stride_r_y) : T1(0)))
                             * *(D2_coeffs_xyz_p3++);
                    T1 res_z = ((likely(int64_t(k + region.ks + p) < int64_t(ex_vertices.ks + ex_vertices.nk)) ? *(this_data_i+stride_r_z) : T1(0))
                             + (likely(int64_t(k + region.ks - p) >= int64_t(ex_vertices.ks)) ? *(this_data_i-stride_r_z) : T1(0)))
                             * *(D2_coeffs_xyz_p3++);

                    // T1 res_x = (*(this_data_i+p)           + *(this_data_i-p))          * *(D2_coeffs_xyz_p3++);
                    // T1 res_y = (*(this_data_i+stride_r_y)  + *(this_data_i-stride_r_y)) * *(D2_coeffs_xyz_p3++);
                    // T1 res_z = (*(this_data_i+stride_r_z)  + *(this_data_i-stride_r_z)) * *(D2_coeffs_xyz_p3++);
                    *result_data_i += res_x + res_y + res_z;
                }
                ++this_data_i;
                ++result_data_i;
                ++ptrA_data_i;
            }
        }
    }
    return;
}
template void calc_laplacian_boundary_safe_d3_c2_o0<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices,
                                                            const Stencil<double>& stencil, const Vertices_3D& region,
                                                            double* const ptr, const Vertices_3D& vertices,
                                                            double const* const ptr_A);
template void calc_laplacian_boundary_safe_d3_c2_o0<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices,
                                                            const Stencil<float>& stencil, const Vertices_3D& region,
                                                            float* const ptr, const Vertices_3D& vertices,
                                                            float const* const ptr_A);

template<typename T1, typename T2>
void calc_laplacian_boundary_safe_d3_c2_o4(T1 const* const ex_ptr, const Vertices_3D& ex_vertices,
                        const Stencil<T2>& stencil, const Vertices_3D& region,
                        T1* const ptr, const Vertices_3D& vertices,
                        T1 const* const ptr_A) {
    // (void) ex_ptr;
    // (void) ex_vertices;
    // (void) stencil;
    // (void) region;
    // (void) ptr;
    // (void) vertices;
    // (void) ptr_A;
    // assert(0 && "calc_laplacian_boundary_safe_d3_c2_o4 is not support");
    const int64_t index_this_origin = (int64_t)ex_vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t index_result_origin = (int64_t)vertices.get_index_nocheck(region.is, region.js, region.ks);
    const int64_t this_ni = ex_vertices.ni;
    const int64_t result_ni = vertices.ni;
    const int64_t this_ninj = ex_vertices.ni * ex_vertices.nj;
    const int64_t result_ninj = vertices.ni * vertices.nj;
    const int FDn = stencil.FDn;
    // T const* const __restrict__ this_data = ex_ptr;
    // T* const __restrict__ result_data = ptr;
    T2 const* const __restrict__ D2_coeffs_xyz = stencil.get_D2_coeffs_xyz();
    const T2 coef_0 = D2_coeffs_xyz[0];

    // printf("HAHHA\n");
    // region.show();
    // ex_vertices.show();
    #ifdef USE_OPENMP
    #pragma omp for schedule(static, (region.nk * region.nj - 1)/omp_get_num_threads() + 1) collapse(2) nowait
    #endif
    for (int64_t k = 0; k < (int64_t)region.nk; k++)
    {
        for (int64_t j = 0; j < (int64_t)region.nj; j++)
        {
            const int result_offset = index_result_origin + result_ninj * k + result_ni * j;
            T1 const* __restrict__ this_data_i = ex_ptr + index_this_origin + this_ninj * k + this_ni * j;
            T1* __restrict__ result_data_i = ptr + result_offset;
            T1 const* __restrict__ ptrA_data_i = ptr_A + result_offset;
            #ifdef USE_OPENMP_SIMD
            #pragma omp simd
            #endif
            for (int64_t i = 0; i < (int64_t)region.ni; i++)
            {
                *result_data_i = (coef_0 + *ptrA_data_i) * *this_data_i;
                T2 const* __restrict__ D2_coeffs_xyz_p3 = D2_coeffs_xyz + 1;
                for (int64_t p = 1; p <= FDn; p++)
                {
                    const int64_t stride_r_y = p * this_ni;
                    const int64_t stride_r_z = p * this_ninj;
                    T1 res = ((likely(int(i + region.is + p) < int(ex_vertices.is + ex_vertices.ni)) ? *(this_data_i+p) : T1(0))
                            + (likely(int(i + region.is - p) >= int(ex_vertices.is)) ? *(this_data_i-p) : T1(0))
                            + (likely(int(j + region.js + p) < int(ex_vertices.js + ex_vertices.nj)) ? *(this_data_i+stride_r_y) : T1(0))
                            + (likely(int(j + region.js - p) >= int(ex_vertices.js)) ? *(this_data_i-stride_r_y) : T1(0))
                            + (likely(int(k + region.ks + p) < int(ex_vertices.ks + ex_vertices.nk)) ? *(this_data_i+stride_r_z) : T1(0))
                            + (likely(int(k + region.ks - p) >= int(ex_vertices.ks)) ? *(this_data_i-stride_r_z) : T1(0)))
                            * *(D2_coeffs_xyz_p3++);
                    *result_data_i += res;
                }
                ++this_data_i;
                ++result_data_i;
                ++ptrA_data_i;
            }
        }
    }
    return;
}
template void calc_laplacian_boundary_safe_d3_c2_o4<double>(double const* const ex_ptr, const Vertices_3D& ex_vertices,
                                                            const Stencil<double>& stencil, const Vertices_3D& region,
                                                            double* const ptr, const Vertices_3D& vertices,
                                                            double const* const ptr_A);
template void calc_laplacian_boundary_safe_d3_c2_o4<float>(float const* const ex_ptr, const Vertices_3D& ex_vertices,
                                                            const Stencil<float>& stencil, const Vertices_3D& region,
                                                            float* const ptr, const Vertices_3D& vertices,
                                                            float const* const ptr_A);

} // Boundary_safe

}
