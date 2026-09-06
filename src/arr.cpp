#include <cstring>
#include <stdio.h>
#include <cassert>
#include <cmath>
#include <iomanip>
#include "arr.h"

#ifdef USE_HUGEPAGE_HBM
void* hbm_alloc(std::size_t& size, unsigned long hbm_node, bool use_hugepage) {
    // 页大小：大页固定 2MB，普通页用系统页大小
    const std::size_t actual_page_size = use_hugepage ? (2 * 1024 * 1024) : sysconf(_SC_PAGESIZE);
    std::size_t bytes = align_size(actual_page_size, size);

    // ----- 大页可用性检查与自动缩容 -----
    if (use_hugepage) {
        std::string sysfs_path = "/sys/devices/system/node/node" + std::to_string(hbm_node) +
                                 "/hugepages/hugepages-2048kB/free_hugepages";
        std::ifstream ifs(sysfs_path);
        if (!ifs.is_open()) {
            throw std::runtime_error(
                "hbm_alloc: FATAL. Cannot read hugepage stats for NUMA node " + std::to_string(hbm_node) +
                ". Path does not exist: " + sysfs_path + ". (Is the NUMA node ID correct?)");
        }
        size_t free_pages = 0;
        ifs >> free_pages;
        size_t free_bytes = free_pages * actual_page_size;

        if (free_bytes < bytes) {
            const size_t reserved_pages = 8;   // 保留 8 个页 (16MB) 作为安全余量
            if (free_pages <= reserved_pages) {
                throw std::runtime_error(
                    "hbm_alloc: SIGBUS PREVENTED. Insufficient hugepages on HBM NUMA node " +
                    std::to_string(hbm_node) + " even for fallback allocation.\n  Available : " +
                    std::to_string(free_bytes / 1024 / 1024) + " MB");
            }
            size_t safe_pages = free_pages - reserved_pages;
            bytes = safe_pages * actual_page_size;
            std::cerr << "[WARNING] hbm_alloc: Requested HBM pool size exceeds available hugepages.\n"
                      << "  Auto-shrinking pool size to : " << (bytes / 1024 / 1024) << " MB\n"
                      << "  Reserved safety margin    : " << (reserved_pages * actual_page_size / 1024 / 1024) << " MB\n";
        }
    }

    // 将实际分配的字节数回传给调用者
    size = bytes;

    // ----- mmap 分配虚拟内存 -----
    int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (use_hugepage) {
        mmap_flags |= MAP_HUGETLB;
    }
    void* ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, mmap_flags, -1, 0);
    if (ptr == MAP_FAILED) {
        throw std::system_error(errno, std::generic_category(), "hbm_alloc: mmap failed");
    }

    // ----- 检查节点 ID 有效性 -----
    if (hbm_node >= static_cast<unsigned long>(numa_num_configured_nodes())) {
        munmap(ptr, bytes);
        throw std::runtime_error("hbm_alloc: HBM node " + std::to_string(hbm_node) +
                                 " exceeds system NUMA nodes (" + std::to_string(numa_num_configured_nodes()) + ")");
    }

    // ----- mbind 绑定到指定节点 -----
    // 注意：此处使用 unsigned long 作为掩码，要求节点数 <= 64。若系统节点数更多，需改用 numa_bitmask_alloc()
    unsigned long nodemask = 1UL << hbm_node;
    unsigned long max_node_count = sizeof(nodemask) * 8;
    if (mbind(ptr, bytes, MPOL_BIND, &nodemask, max_node_count, 0) != 0) {
        munmap(ptr, bytes);
        throw std::system_error(errno, std::generic_category(), "hbm_alloc: mbind failed");
    }

    // ----- 强制触发缺页，使物理内存落在绑定节点上 -----
    std::memset(ptr, 0, bytes);

    return ptr;
}

void hbm_free(void* ptr, std::size_t size, bool use_hugepage) {
    if (ptr == nullptr) return;
    const std::size_t actual_page_size = use_hugepage ? (2 * 1024 * 1024) : sysconf(_SC_PAGESIZE);
    const std::size_t bytes = align_size(actual_page_size, size);
    if (munmap(ptr, bytes) != 0) {
        throw std::system_error(errno, std::generic_category(), "hbm_free: munmap failed");
    }
}
#endif

//Array_0D<T>
template<typename T>
Array_0D<T>::Array_0D() {
    this->set_is_Col_Maj(true);
    this->constructor(0);
}

template<typename T>
Array_0D<T>::Array_0D(const uint64_t length) {
    this->set_is_Col_Maj(true);
    this->constructor(length);
}

template<typename T>
Array_0D<T>::Array_0D(const uint64_t length, const uint) {
    this->set_is_Col_Maj(true);
    this->constructor(length, 0);
}

template<typename T>
Array_0D<T>::Array_0D(const Array_0D<T>& other) {
    this->set_is_Col_Maj(other.get_is_Col_Maj());
    this->constructor(other.length);
    // std::memcpy(this->data, other.data, this->length * sizeof(T));
    Linalg::set_value_general(this->data, other.data, this->length);
}

template<typename T>
Array_0D<T>::Array_0D(Array_0D&& other) {
    this->set_is_Col_Maj(other.get_is_Col_Maj());
    this->length = other.length;
    this->data = other.data;
    other.length = 0;
    other.data = nullptr;
}

template<typename T>
Array_0D<T>::~Array_0D() {
    this->destructor();
}

template<typename T>
void Array_0D<T>::destructor() {
    if (this->data != nullptr) {
        // delete [] this->data;
        ::operator delete[](this->data, std::align_val_t(64));
        this->data = nullptr;
    }
    this->length = 0;
    return;
}

template<typename T>
void Array_0D<T>::constructor(const uint64_t length) {
    this->set_is_Col_Maj(true);
    this->length = length;
    if (length != 0) this->data = new (std::align_val_t(64)) T [this->length]; //c++17 aligned new
    // if (length != 0) this->data = new T [this->length];
    return;
}

template<typename T>
void Array_0D<T>::constructor(const uint64_t length, const uint) {
    this->set_is_Col_Maj(true);
    this->length = length;
    if (length != 0) this->data = new (std::align_val_t(64)) T [this->length](); //c++17 aligned new
    // if (length != 0) this->data = new T [this->length]();
    return;
}

template<typename T>
void Array_0D<T>::reconstructor(const uint64_t length) {
    this->destructor();
    this->constructor(length);
}

template<typename T>
void Array_0D<T>::reconstructor(const uint64_t length, const uint) {
    this->destructor();
    this->constructor(length, 0);
}

#ifdef USE_HBM
#ifdef USE_HUGEPAGE_HBM
    template<typename T>
    void Array_0D<T>::constructor_hbm(const uint64_t length) {
        this->set_is_Col_Maj(true);
        this->length = length;
        // if (length != 0) this->data = static_cast<T*>(hbw_malloc(this->length * sizeof(T)));
        // if (length != 0) this->data = new T [this->length]();
        int cpu = sched_getcpu();
        unsigned long nodeid = numa_node_of_cpu(cpu);
        unsigned long hbm_node = nodeid + (numa_num_configured_nodes() / 2);
        if (length != 0) {
            unsigned long size = this->length * sizeof(T);
            this->data = static_cast<T*>(hbm_alloc(size, hbm_node, false));
        }
        return;
    }

    template<typename T>
    void Array_0D<T>::destructor_hbm() {
        if (this->data != nullptr) {
            hbm_free(this->data, this->length * sizeof(T), false);
            this->data = nullptr;
        }
        this->length = 0;
        return;
    }

    template<typename T>
    void Array_0D<T>::reconstructor_hbm(const uint64_t length) {
        this->destructor_hbm();
        this->constructor_hbm(length);
    }

    template<typename T>
    void Array_0D<T>::constructor_hp_hbm(const uint64_t length) {
        this->set_is_Col_Maj(true);
        this->length = length;
        // if (length != 0) this->data = static_cast<T*>(hbw_malloc(this->length * sizeof(T)));
        // if (length != 0) this->data = new T [this->length]();
        int cpu = sched_getcpu();
        unsigned long nodeid = numa_node_of_cpu(cpu);
        unsigned long hbm_node = nodeid + (numa_num_configured_nodes() / 2);
        if (length != 0) {
            unsigned long size = this->length * sizeof(T);
            this->data = static_cast<T*>(hbm_alloc(size, hbm_node, true));
        }
        return;
    }

    template<typename T>
    void Array_0D<T>::destructor_hp_hbm() {
        if (this->data != nullptr) {
            hbm_free(this->data, this->length * sizeof(T), true);
            this->data = nullptr;
        }
        this->length = 0;
        return;
    }

    template<typename T>
    void Array_0D<T>::reconstructor_hp_hbm(const uint64_t length) {
        this->destructor_hp_hbm();
        this->constructor_hp_hbm(length);
    }
#else
    template<typename T>
    void Array_0D<T>::constructor_hbm(const uint64_t length) {
        this->set_is_Col_Maj(true);
        this->length = length;
        if (length != 0) this->data = static_cast<T*>(hbw_malloc(this->length * sizeof(T)));
        // if (length != 0) this->data = new T [this->length]();
        return;
    }

    template<typename T>
    void Array_0D<T>::destructor_hbm() {
        this->length = 0;
        if (this->data != nullptr) {
            hbw_free(this->data);
            this->data = nullptr;
        }
        return;
    }

    template<typename T>
    void Array_0D<T>::reconstructor_hbm(const uint64_t length) {
        this->destructor_hbm();
        this->constructor_hbm(length);
    }
#endif
#endif

template<typename T>
Array_0D<T>& Array_0D<T>::operator=(const Array_0D<T>& other) {
    assert(this->length == other.length);
    this->deepcopy(other);
    return *this;
}

template<typename T>
Array_0D<T>& Array_0D<T>::operator=(Array_0D<T>&& other) {
    assert(this->length == other.length);
    this->deepcopy(std::move(other));
    return *this;
}

template<typename T>
Array_0D<T>& Array_0D<T>::operator=(const T num) {
    Linalg::set_value_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_0D<T>& Array_0D<T>::deepcopy(const Array_0D<T>& other) {
    if (this != &other) {
        if (this->length != other.length) {
            this->reconstructor(other.length);
        }
        // std::memcpy(this->data, other.data, this->length * sizeof(T));
        Linalg::set_value_general(this->data, other.data, this->length);
    }
    return *this;
}

template<typename T>
Array_0D<T>& Array_0D<T>::deepcopy(Array_0D&& other) {
    std::swap(this->length, other.length);
    std::swap(this->data, other.data);
    return *this;
}

template<typename T>
void Array_0D<T>::swap(Array_0D<T>& other) {
    std::swap(this->length, other.length);
    std::swap(this->data, other.data);
    return;
}

template<typename T>
template<typename T2>
Array_0D<T2>
Array_0D<T>::as_type(const T2*) const {
    Array_0D<T2> result(this->length);
    Linalg::convert_type<T2, T>(result.data, this->data, this->length);
    return result;
}
template Array_0D<double> Array_0D<double>::as_type(const double*) const;
template Array_0D<float> Array_0D<float>::as_type(const float*) const;
template Array_0D<int> Array_0D<int>::as_type(const int*) const;
template Array_0D<double> Array_0D<float>::as_type(const double*) const;
template Array_0D<float> Array_0D<double>::as_type(const float*) const;
template Array_0D<int> Array_0D<float>::as_type(const int*) const;
template Array_0D<float> Array_0D<int>::as_type(const float*) const;
template Array_0D<int> Array_0D<double>::as_type(const int*) const;
template Array_0D<double> Array_0D<int>::as_type(const double*) const;
#ifdef FP16_FLAG
template Array_0D<__fp16> Array_0D<__fp16>::as_type(const __fp16*) const;
template Array_0D<__fp16> Array_0D<float>::as_type(const __fp16*) const;
template Array_0D<__fp16> Array_0D<double>::as_type(const __fp16*) const;
template Array_0D<float> Array_0D<__fp16>::as_type(const float*) const;
template Array_0D<double> Array_0D<__fp16>::as_type(const double*) const;
#endif //FP16_FLAG

template<typename T>
void Array_0D<T>::set_is_Col_Maj(const bool is_Col_Maj) {
    this->is_Col_Maj = is_Col_Maj;
    return;
}

template<typename T>
bool Array_0D<T>::get_is_Col_Maj() const {
    return this->is_Col_Maj;
}

// template<typename T>
// template<typename T2>
// void Array_0D<T>::convert_type(const Array_0D<T2>& array_A) {
//     assert(this->length == array_A.length);
//     Linalg::convert_type(this->data, array_A.data, array_A.length);
//     return;
// }
// template void Array_0D<T>::convert_type<int>(const Array_0D<int>& array_A);

template<typename T>
Array_0D<T>& Array_0D<T>::operator+=(const Array_0D& other) {
    assert(this->length == other.length);
    Linalg::hadamard_plus_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_0D<T>& Array_0D<T>::operator-=(const Array_0D& other) {
    assert(this->length == other.length);
    Linalg::hadamard_minus_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_0D<T>& Array_0D<T>::operator*=(const Array_0D& other) {
    assert(this->length == other.length);
    Linalg::hadamard_product_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_0D<T>& Array_0D<T>::operator/=(const Array_0D& other) {
    assert(this->length == other.length);
    Linalg::hadamard_divide_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_0D<T>& Array_0D<T>::operator+=(const T num) {
    Linalg::scalar_plus_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_0D<T>& Array_0D<T>::operator-=(const T num) {
    Linalg::scalar_minus_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_0D<T>& Array_0D<T>::operator*=(const T num) {
    Linalg::scalar_product_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_0D<T>& Array_0D<T>::operator/=(const T num) {
    Linalg::scalar_divide_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_0D<T> Array_0D<T>::operator+(const Array_0D& other) const {
    assert(this->length == other.length);
    Array_0D<T> __result(this->length);
    Linalg::hadamard_plus_general(__result.data, this->data, other.data, this->length);
    return __result;
}

template<typename T>
Array_0D<T> Array_0D<T>::operator-(const Array_0D& other) const {
    assert(this->length == other.length);
    Array_0D<T> __result(this->length);
    Linalg::hadamard_minus_general(__result.data, this->data, other.data, this->length);
    return __result;
}

template<typename T>
Array_0D<T> Array_0D<T>::operator*(const Array_0D& other) const {
    assert(this->length == other.length);
    Array_0D<T> __result(this->length);
    Linalg::hadamard_product_general(__result.data, this->data, other.data, this->length);
    return __result;
}

template<typename T>
Array_0D<T> Array_0D<T>::operator/(const Array_0D& other) const {
    assert(this->length == other.length);
    Array_0D<T> __result(this->length);
    Linalg::hadamard_divide_general(__result.data, this->data, other.data, this->length);
    return __result;
}

template<typename T>
Array_0D<T> Array_0D<T>::operator+(const T num) {
    Array_0D<T> __result(this->length);
    Linalg::scalar_plus_general(__result.data, this->data, num, this->length);
    return __result;
}

template<typename T>
Array_0D<T> Array_0D<T>::operator-(const T num) {
    Array_0D<T> __result(this->length);
    Linalg::scalar_minus_general(__result.data, this->data, num, this->length);
    return __result;
}

template<typename T>
Array_0D<T> Array_0D<T>::operator*(const T num) {
    Array_0D<T> __result(this->length);
    Linalg::scalar_product_general(__result.data, this->data, num, this->length);
    return __result;
}

template<typename T>
Array_0D<T> Array_0D<T>::operator/(const T num) {
    Array_0D<T> __result(this->length);
    Linalg::scalar_divide_general(__result.data, this->data, num, this->length);
    return __result;
}

template<typename T>
T& Array_0D<T>::operator[](const int index) {
    return this->data[index];
}

template<typename T>
T Array_0D<T>::operator[](const int index) const {
    return this->data[index];
}

template<typename T>
uint Array_0D<T>::get_length() const {
    return this->length;
}

template<typename T>
T*& Array_0D<T>::get_data() {
    return this->data;
}

template<typename T>
T* Array_0D<T>::get_data() const {
    return this->data;
}

template<typename T>
std::size_t Array_0D<T>::get_bytes() const {
    return (this->length) * sizeof(T);
}

template<typename T>
int Array_0D<T>::get_type_id() const {
    return Linalg::get_type_id<T>();
    // return this->get_type_id(this->data);
}

template<typename T>
bool Array_0D<T>::is_type_double() const {
    return Linalg::get_type_id<T>() == 2;
}

template<typename T>
bool Array_0D<T>::is_type_float() const {
    return Linalg::get_type_id<T>() == 1;
}

template<typename T>
bool Array_0D<T>::is_type_int() const {
    return Linalg::get_type_id<T>() == 0;
}

template<typename T>
Array_0D<T> Array_0D<T>::sub_arr(const int index_s, const int index_e) {
    assert(index_s >= 0 && index_e >= index_s && index_e < (int)this->length);
    uint length = index_e - index_s + 1;
    Array_0D<T> __result(length);
    // std::memcpy(__result.data, this->data + index_s, length * sizeof(T));
    Linalg::set_value_general(__result.data, this->data + index_s, length);
    return __result;
}

template<typename T>
void Array_0D<T>::show() const {
    int block_size = 20;
    int precision = 13;
    int width = 18;
    switch(this->get_type_id()) {
        case 0 :
            std::cout << std::right << std::setw(block_size) << "this T" << " is" << " int" << std::endl;
            break;
        case 1 :
            std::cout << std::right << std::setw(block_size) << "this T" << " is" << " float" << std::endl;
            break;
        case 2 :
            std::cout << std::right << std::setw(block_size) << "this T" << " is" << " double" << std::endl;
            break;
        default :
            std::cout << std::right << std::setw(block_size) << "this T" << " is" << " unknown" << std::endl;
    }
    std::cout << std::right << std::setw(block_size) << "this->length" << " = "
              << std::left << this->length << std::endl;
    std::cout << std::right << std::setw(block_size) << "this->get_bytes()" << " = "
              << std::left << this->get_bytes() << std::endl;
    std::cout << std::right << std::setw(block_size) << "this->vector_sum()" << " = "
              << std::left << std::setprecision(precision) << std::setw(width) << std::fixed << this->vector_sum() << std::endl;
    std::cout << std::right << std::setw(block_size) << "this->vector_square_sum()" << " = "
              << std::left << std::setprecision(precision) << std::setw(width) << std::fixed << this->vector_norm_square_sum() << std::endl;
    std::cout << std::right << std::setw(block_size) << "this->vector_2norm()" << " = "
              << std::left << std::setprecision(precision) << std::setw(width) << std::fixed << this->vector_2norm() << std::endl;
    return;
}

template<typename T>
void Array_0D<T>::print() const {
    this->print_h();
    return;
}

template<typename T>
void Array_0D<T>::printX() const {
    this->printX_h();
    return;
}

template<typename T>
void Array_0D<T>::print_Col_Maj() const {
    this->print_Col_Maj_h();
    return;
}

template<typename T>
void Array_0D<T>::print_Row_Maj() const {
    this->print_Row_Maj_h();
    return;
}

template<typename T>
void Array_0D<T>::print(const std::string& fname) const {
    std::ofstream outfile(fname);
    this->print_h(outfile);
    outfile.close();
    return;
}

template<typename T>
void Array_0D<T>::printX(const std::string& fname) const {
    std::ofstream outfile(fname);
    this->printX_h(outfile);
    outfile.close();
    return;
}

template<typename T>
void Array_0D<T>::print_Col_Maj(const std::string& fname) const {
    std::ofstream outfile(fname);
    this->print_Col_Maj_h(outfile);
    outfile.close();
    return;
}

template<typename T>
void Array_0D<T>::print_Row_Maj(const std::string& fname) const {
    std::ofstream outfile(fname);
    this->print_Row_Maj_h(outfile);
    outfile.close();
    return;
}

template<typename T>
void Array_0D<T>::printX_h(std::ostream& output) const {
    if (sizeof(T) == 8) {
        long long int* dataX = (long long int*)this->data;
        for (uint index = 0; index < this->length; index++) {
            output << std::hex << std::setw(18) << dataX[index];
            if (index % 6 == 5)
                output << std::endl;
        }
        output << std::endl;
    } else if (sizeof(T) == 4) {
        int* dataX = (int*)this->data;
        for (uint index = 0; index < this->length; index++) {
            output << std::hex << std::setw(18) << dataX[index];
            if (index % 6 == 5)
                output << std::endl;
        }
        output << std::endl;
    }
    return;
}

// template<typename T>
// void Array_0D<T>::print(std::ostream& output) const {
//     this->print_Col_Maj(output);
//     return;
// }

// template<typename T>
// void Array_0D<T>::print_Col_Maj(std::ostream& output) const {
//     for (uint index = 0; index < this->length; index++) {
//         output << std::setw(14) << std::setprecision(6) << std::scientific << std::uppercase << this->data[index];
//         if (index % 6 == 5)
//             output << std::endl;
//     }
//     output << std::endl;
//     return;
// }

// template<typename T>
// void Array_0D<T>::print_Row_Maj(std::ostream& output) const {
//     this->print_Col_Maj(output);
//     return;
// }

//Vertor functions
template<typename T>
void Array_0D<T>::seededrand(const int seed, const T rand_min, const T rand_max) {
    Linalg::seededrand(this->data, this->length, rand_min, rand_max, seed);
    return;
}

template<typename T>
T Array_0D<T>::vector_sum() const{
    return Linalg::vector_sum(this->data, this->length);
}

template<typename T>
T Array_0D<T>::vector_norm_square_sum(const MPI_Comm comm) const{
    return Linalg::vector_norm_square_sum(this->data, this->length, comm);
}

template<typename T>
T Array_0D<T>::vector_2norm(const MPI_Comm comm) const{
    return std::sqrt(this->vector_norm_square_sum(comm));
}

template<typename T>
void Array_0D<T>::hadamard_plus_general(const Array_0D& array_A) {
    assert(this->length == array_A.length);
    Linalg::hadamard_plus_general(this->data, array_A.data, this->length);
    return;
}

template<typename T>
void Array_0D<T>::hadamard_minus_general(const Array_0D& array_A) {
    assert(this->length == array_A.length);
    Linalg::hadamard_minus_general(this->data, array_A.data, this->length);
    return;
}

template<typename T>
void Array_0D<T>::hadamard_product_general(const Array_0D& array_A) {
    assert(this->length == array_A.length);
    Linalg::hadamard_product_general(this->data, array_A.data, this->length);
    return;
}

template<typename T>
void Array_0D<T>::hadamard_divide_general(const Array_0D& array_A) {
    assert(this->length == array_A.length);
    Linalg::hadamard_divide_general(this->data, array_A.data, this->length);
    return;
}

template<typename T>
void Array_0D<T>::hadamard_product_general(const Array_0D& array_A, T const alpha) {
    assert(this->length == array_A.length);
    Linalg::hadamard_product_general(this->data, array_A.data, this->length, alpha);
    return;
}

template<typename T>
void Array_0D<T>::hadamard_product_general(const Array_0D& array_A, T const alpha, T const beta) {
    assert(this->length == array_A.length);
    Linalg::hadamard_product_general(this->data, array_A.data, this->length, alpha, beta);
    return;
}

template<typename T>
void Array_0D<T>::hadamard_product_general(const Array_0D& array_A, T const alpha, T const beta, T const gamma) {
    assert(this->length == array_A.length);
    Linalg::hadamard_product_general(this->data, array_A.data, this->length, alpha, beta, gamma);
    return;
}

template<typename T>
void Array_0D<T>::hadamard_plus_general(const Array_0D& array_A, const Array_0D& array_B) {
    assert(this->length == array_A.length);
    Linalg::hadamard_plus_general(this->data, array_A.data, array_B.data, this->length);
    return;
}

template<typename T>
void Array_0D<T>::hadamard_minus_general(const Array_0D& array_A, const Array_0D& array_B) {
    assert(this->length == array_A.length);
    Linalg::hadamard_minus_general(this->data, array_A.data, array_B.data, this->length);
    return;
}

template<typename T>
void Array_0D<T>::hadamard_product_general(const Array_0D& array_A, const Array_0D& array_B) {
    assert(this->length == array_A.length);
    Linalg::hadamard_product_general(this->data, array_A.data, array_B.data, this->length);
    return;
}

template<typename T>
void Array_0D<T>::hadamard_divide_general(const Array_0D& array_A, const Array_0D& array_B) {
    assert(this->length == array_A.length);
    Linalg::hadamard_divide_general(this->data, array_A.data, array_B.data, this->length);
    return;
}

template<typename T>
void Array_0D<T>::hadamard_product_general(const Array_0D& array_A, const Array_0D& array_B,
                                           T const alpha) {
    assert(this->length == array_A.length);
    Linalg::hadamard_product_general(this->data, array_A.data, array_B.data, this->length,
                                     alpha);
    return;
}

template<typename T>
void Array_0D<T>::hadamard_product_general(const Array_0D& array_A, const Array_0D& array_B,
                                           T const alpha, T const beta) {
    assert(this->length == array_A.length);
    Linalg::hadamard_product_general(this->data, array_A.data, array_B.data, this->length,
                                     alpha, beta);
    return;
}

template<typename T>
void Array_0D<T>::hadamard_product_general(const Array_0D& array_A, const Array_0D& array_B,
                                           T const alpha, T const beta, T const gamma) {
    assert(this->length == array_A.length);
    Linalg::hadamard_product_general(this->data, array_A.data, array_B.data, this->length,
                                     alpha, beta, gamma);
    return;
}

template<typename T>
void Array_0D<T>::accumulate_hadamard_plus_general(const Array_0D& array_A, const Array_0D& array_B) {
    assert(this->length == array_A.length);
    Linalg::accumulate_hadamard_plus_general(this->data, array_A.data, array_B.data, this->length);
    return;
}

template<typename T>
void Array_0D<T>::accumulate_hadamard_minus_general(const Array_0D& array_A, const Array_0D& array_B) {
    assert(this->length == array_A.length);
    Linalg::accumulate_hadamard_minus_general(this->data, array_A.data, array_B.data, this->length);
    return;
}

template<typename T>
void Array_0D<T>::accumulate_hadamard_product_general(const Array_0D& array_A, const Array_0D& array_B) {
    assert(this->length == array_A.length);
    Linalg::accumulate_hadamard_product_general(this->data, array_A.data, array_B.data, this->length);
    return;
}

template<typename T>
void Array_0D<T>::accumulate_hadamard_divide_general(const Array_0D& array_A, const Array_0D& array_B) {
    assert(this->length == array_A.length);
    Linalg::accumulate_hadamard_divide_general(this->data, array_A.data, array_B.data, this->length);
    return;
}

template<typename T>
void Array_0D<T>::accumulate_hadamard_product_general(const Array_0D& array_A, const Array_0D& array_B,
                                           T const alpha) {
    assert(this->length == array_A.length);
    Linalg::accumulate_hadamard_product_general(this->data, array_A.data, array_B.data, this->length,
                                     alpha);
    return;
}

template<typename T>
void Array_0D<T>::accumulate_hadamard_product_general(const Array_0D& array_A, const Array_0D& array_B,
                                           T const alpha, T const beta) {
    assert(this->length == array_A.length);
    Linalg::accumulate_hadamard_product_general(this->data, array_A.data, array_B.data, this->length,
                                     alpha, beta);
    return;
}

template<typename T>
void Array_0D<T>::accumulate_hadamard_product_general(const Array_0D& array_A, const Array_0D& array_B,
                                           T const alpha, T const beta, T const gamma) {
    assert(this->length == array_A.length);
    Linalg::accumulate_hadamard_product_general(this->data, array_A.data, array_B.data, this->length,
                                     alpha, beta, gamma);
    return;
}

template<typename T>
void Array_0D<T>::scalar_plus_general(const T num) {
    Linalg::scalar_plus_general(this->data, num, this->length);
    return;
}

template<typename T>
void Array_0D<T>::scalar_minus_general(const T num) {
    Linalg::scalar_minus_general(this->data, num, this->length);
    return;
}

template<typename T>
void Array_0D<T>::scalar_product_general(const T num) {
    Linalg::scalar_product_general(this->data, num, this->length);
    return;
}

template<typename T>
void Array_0D<T>::scalar_divide_general(const T num) {
    Linalg::scalar_divide_general(this->data, num, this->length);
    return;
}

template<typename T>
void Array_0D<T>::scalar_product_general(const T num, const Array_0D& array_B) {
    Linalg::scalar_product_general(this->data, num, this->length, array_B.data);
    return;
}

template<typename T>
void Array_0D<T>::scalar_product_general(const T num, const Array_0D& array_B, T const alpha) {
    Linalg::scalar_product_general(this->data, num, this->length, array_B.data, alpha);
    return;
}

template<typename T>
void Array_0D<T>::scalar_product_general(const T num, const Array_0D& array_B, T const alpha, T const gamma) {
    Linalg::scalar_product_general(this->data, num, this->length, array_B.data, alpha, gamma);
    return;
}

template<typename T>
void Array_0D<T>::scalar_plus_general(const Array_0D& array_A, const T num) {
    Linalg::scalar_plus_general(this->data, array_A.data, num, this->length);
    return;
}

template<typename T>
void Array_0D<T>::scalar_minus_general(const Array_0D& array_A, const T num) {
    Linalg::scalar_minus_general(this->data, array_A.data, num, this->length);
    return;
}

template<typename T>
void Array_0D<T>::scalar_product_general(const Array_0D& array_A, const T num) {
    Linalg::scalar_product_general(this->data, array_A.data, num, this->length);
    return;
}

template<typename T>
void Array_0D<T>::scalar_divide_general(const Array_0D& array_A, const T num) {
    Linalg::scalar_divide_general(this->data, array_A.data, num, this->length);
    return;
}

template<typename T>
void Array_0D<T>::scalar_product_general(const Array_0D& array_A, const T num, const Array_0D& array_B) {
    Linalg::scalar_product_general(this->data, array_A.data, num, this->length, array_B.data);
    return;
}

template<typename T>
void Array_0D<T>::scalar_product_general(const Array_0D& array_A, const T num, const Array_0D& array_B, T const alpha) {
    Linalg::scalar_product_general(this->data, array_A.data, num, this->length, array_B.data, alpha);
    return;
}

template<typename T>
void Array_0D<T>::scalar_product_general(const Array_0D& array_A, const T num, const Array_0D& array_B, T const alpha, T const gamma) {
    Linalg::scalar_product_general(this->data, array_A.data, num, this->length, array_B.data, alpha, gamma);
    return;
}

template<typename T>
void Array_0D<T>::accumulate_scalar_plus_general(const Array_0D& array_A, const T num) {
    Linalg::accumulate_scalar_plus_general(this->data, array_A.data, num, this->length);
    return;
}

template<typename T>
void Array_0D<T>::accumulate_scalar_minus_general(const Array_0D& array_A, const T num) {
    Linalg::accumulate_scalar_minus_general(this->data, array_A.data, num, this->length);
    return;
}

template<typename T>
void Array_0D<T>::accumulate_scalar_product_general(const Array_0D& array_A, const T num) {
    Linalg::accumulate_scalar_product_general(this->data, array_A.data, num, this->length);
    return;
}

template<typename T>
void Array_0D<T>::accumulate_scalar_divide_general(const Array_0D& array_A, const T num) {
    Linalg::accumulate_scalar_divide_general(this->data, array_A.data, num, this->length);
    return;
}

template<typename T>
void Array_0D<T>::accumulate_scalar_product_general(const Array_0D& array_A, const T num, const Array_0D& array_B) {
    Linalg::accumulate_scalar_product_general(this->data, array_A.data, num, this->length, array_B.data);
    return;
}

template<typename T>
void Array_0D<T>::accumulate_scalar_product_general(const Array_0D& array_A, const T num, const Array_0D& array_B, T const alpha) {
    Linalg::accumulate_scalar_product_general(this->data, array_A.data, num, this->length, array_B.data, alpha);
    return;
}

template<typename T>
void Array_0D<T>::accumulate_scalar_product_general(const Array_0D& array_A, const T num, const Array_0D& array_B, T const alpha, T const gamma) {
    Linalg::accumulate_scalar_product_general(this->data, array_A.data, num, this->length, array_B.data, alpha, gamma);
    return;
}

//Array_1D<T>
template<typename T>
Array_1D<T>::Array_1D() : Array_0D<T>::Array_0D(), Vertices_1D::Vertices_1D() {
}

template<typename T>
Array_1D<T>::Array_1D(const uint64_t length) : Array_0D<T>::Array_0D(length), Vertices_1D::Vertices_1D(length) {
}

template<typename T>
Array_1D<T>::Array_1D(const uint64_t length, const uint) : Array_0D<T>::Array_0D(length, 0), Vertices_1D::Vertices_1D(length) {
}

template<typename T>
Array_1D<T>::Array_1D(const Vertices_1D& other) : Array_0D<T>::Array_0D(other.get_size()), Vertices_1D::Vertices_1D(other) {
}

template<typename T>
Array_1D<T>::Array_1D(const Vertices_1D& other, const uint) : Array_0D<T>::Array_0D(other.get_size(), 0), Vertices_1D::Vertices_1D(other) {
}

template<typename T>
Array_1D<T>::Array_1D(const Array_1D& other) : Array_0D<T>::Array_0D(other), Vertices_1D::Vertices_1D(other) {
}

template<typename T>
Array_1D<T>::Array_1D(Array_1D&& other) : Array_0D<T>::Array_0D(std::move(other)), Vertices_1D::Vertices_1D(other) {
}

template<typename T>
Array_1D<T>::~Array_1D() {
    this->destructor();
}

template<typename T>
void Array_1D<T>::destructor() {
    this->Array_0D<T>::destructor();
    this->Vertices_1D::set_ni(0);
    return;
}

template<typename T>
void Array_1D<T>::constructor(const uint64_t length) {
    this->Array_0D<T>::constructor(length);
    this->Vertices_1D::set_ni(length);
    return;
}

template<typename T>
void Array_1D<T>::constructor(const Vertices_1D& other) {
    this->Array_0D<T>::constructor(other.get_size());
    this->Vertices_1D::set_vertices(other);
    return;
}

template<typename T>
void Array_1D<T>::constructor(const Vertices_1D& other, const uint) {
    this->Array_0D<T>::constructor(other.get_size(), 0);
    this->Vertices_1D::set_vertices(other);
    return;
}

template<typename T>
void Array_1D<T>::reconstructor(const uint64_t length) {
    this->destructor();
    this->constructor(length);
    return;
}

template<typename T>
void Array_1D<T>::reconstructor(const Vertices_1D& other) {
    this->destructor();
    this->constructor(other);
    return;
}

template<typename T>
void Array_1D<T>::reconstructor(const Vertices_1D& other, const uint) {
    this->destructor();
    this->constructor(other, 0);
    return;
}

#ifdef USE_HBM
#ifdef USE_HUGEPAGE_HBM
template<typename T>
void Array_1D<T>::constructor_hbm(const uint64_t length) {
    this->Array_0D<T>::constructor_hbm(length);
    this->Vertices_1D::set_ni(length);
    return;
}

template<typename T>
void Array_1D<T>::constructor_hbm(const Vertices_1D& other) {
    this->Array_0D<T>::constructor_hbm(other.get_size());
    this->Vertices_1D::set_vertices(other);
    return;
}

template<typename T>
void Array_1D<T>::destructor_hbm() {
    this->Array_0D<T>::destructor_hbm();
    this->Vertices_1D::set_ni(0);
    return;
}

template<typename T>
void Array_1D<T>::reconstructor_hbm(const uint64_t length) {
    this->destructor_hbm();
    this->constructor_hbm(length);
    return;
}

template<typename T>
void Array_1D<T>::reconstructor_hbm(const Vertices_1D& other) {
    this->destructor_hbm();
    this->constructor_hbm(other);
    return;
}

//hugepage
template<typename T>
void Array_1D<T>::constructor_hp_hbm(const uint64_t length) {
    this->Array_0D<T>::constructor_hp_hbm(length);
    this->Vertices_1D::set_ni(length);
    return;
}

template<typename T>
void Array_1D<T>::constructor_hp_hbm(const Vertices_1D& other) {
    this->Array_0D<T>::constructor_hp_hbm(other.get_size());
    this->Vertices_1D::set_vertices(other);
    return;
}

template<typename T>
void Array_1D<T>::destructor_hp_hbm() {
    this->Array_0D<T>::destructor_hp_hbm();
    this->Vertices_1D::set_ni(0);
    return;
}

template<typename T>
void Array_1D<T>::reconstructor_hp_hbm(const uint64_t length) {
    this->destructor_hp_hbm();
    this->constructor_hp_hbm(length);
    return;
}

template<typename T>
void Array_1D<T>::reconstructor_hp_hbm(const Vertices_1D& other) {
    this->destructor_hp_hbm();
    this->constructor_hp_hbm(other);
    return;
}
#else
template<typename T>
void Array_1D<T>::constructor_hbm(const uint64_t length) {
    this->Array_0D<T>::constructor_hbm(length);
    this->Vertices_1D::set_ni(length);
    return;
}

template<typename T>
void Array_1D<T>::constructor_hbm(const Vertices_1D& other) {
    this->Array_0D<T>::constructor_hbm(other.get_size());
    this->Vertices_1D::set_vertices(other);
    return;
}

template<typename T>
void Array_1D<T>::destructor_hbm() {
    this->Array_0D<T>::destructor_hbm();
    this->Vertices_1D::set_ni(0);
    return;
}

template<typename T>
void Array_1D<T>::reconstructor_hbm(const uint64_t length) {
    this->destructor_hbm();
    this->constructor_hbm(length);
    return;
}

template<typename T>
void Array_1D<T>::reconstructor_hbm(const Vertices_1D& other) {
    this->destructor_hbm();
    this->constructor_hbm(other);
    return;
}
#endif
#endif

template<typename T>
void Array_1D<T>::swap(Array_1D<T>& other) {
    this->Vertices_1D::swap(other);
    this->Array_0D<T>::swap(other);
    std::swap(this->is_Col_Maj, other.is_Col_Maj);
    return;
}

template<typename T>
template<typename T2>
Array_1D<T2>
Array_1D<T>::as_type(const T2*) const {
    Array_1D<T2> result(this->get_vertices());
    Linalg::convert_type<T2, T>(result.data, this->data, this->length);
    return result;
}
template Array_1D<double> Array_1D<double>::as_type(const double*) const;
template Array_1D<float> Array_1D<float>::as_type(const float*) const;
template Array_1D<int> Array_1D<int>::as_type(const int*) const;
template Array_1D<double> Array_1D<float>::as_type(const double*) const;
template Array_1D<float> Array_1D<double>::as_type(const float*) const;
template Array_1D<int> Array_1D<float>::as_type(const int*) const;
template Array_1D<float> Array_1D<int>::as_type(const float*) const;
template Array_1D<int> Array_1D<double>::as_type(const int*) const;
template Array_1D<double> Array_1D<int>::as_type(const double*) const;
#ifdef FP16_FLAG
template Array_1D<__fp16> Array_1D<__fp16>::as_type(const __fp16*) const;
template Array_1D<__fp16> Array_1D<float>::as_type(const __fp16*) const;
template Array_1D<__fp16> Array_1D<double>::as_type(const __fp16*) const;
template Array_1D<float> Array_1D<__fp16>::as_type(const float*) const;
template Array_1D<double> Array_1D<__fp16>::as_type(const double*) const;
#endif //FP16_FLAG

template<typename T>
uint Array_1D<T>::get_leading_dimension() const {
    return 1;
}

template<typename T>
uint Array_1D<T>::get_second_dimension() const {
    return this->ni;
}

template<typename T>
Array_1D<T>& Array_1D<T>::operator=(const Array_1D<T>& other) {
    assert(this->get_vertices() == other.get_vertices());
    this->deepcopy(other);
    return *this;
}

template<typename T>
Array_1D<T>& Array_1D<T>::operator=(Array_1D<T>&& other) {
    assert(this->get_vertices() == other.get_vertices());
    this->deepcopy(std::move(other));
    return *this;
}

template<typename T>
Array_1D<T>& Array_1D<T>::operator=(const T num) {
    for (uint i = 0; i < this->length; i++) {
        this->data[i] = num;
    }
    return *this;
}

template<typename T>
Array_1D<T>& Array_1D<T>::deepcopy(const Array_1D<T>& other) {
    if (this != &other) {
        if (this->length != other.length) {
            this->reconstructor(other.get_vertices());
        } else if (this->get_vertices() != other.get_vertices()) {
            this->set_vertices(other.get_vertices());
        }
        // std::memcpy(this->data, other.data, this->length * sizeof(T));
        Linalg::set_value_general(this->data, other.data, this->length);
        this->is_Col_Maj = other.is_Col_Maj;
    }
    return *this;
}

template<typename T>
Array_1D<T>& Array_1D<T>::deepcopy(Array_1D&& other) {
    if (this->get_vertices() != other.get_vertices()) {
        this->set_vertices(other.get_vertices());
    }
    std::swap(this->length, other.length);
    std::swap(this->data, other.data);
    this->is_Col_Maj = other.is_Col_Maj;
    return *this;
}

template<typename T>
Array_1D<T>& Array_1D<T>::operator+=(const Array_1D& other) {
    assert(this->get_vertices() == other.get_vertices());
    Linalg::hadamard_plus_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_1D<T>& Array_1D<T>::operator-=(const Array_1D& other) {
    assert(this->get_vertices() == other.get_vertices());
    Linalg::hadamard_minus_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_1D<T>& Array_1D<T>::operator*=(const Array_1D& other) {
    assert(this->get_vertices() == other.get_vertices());
    Linalg::hadamard_product_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_1D<T>& Array_1D<T>::operator/=(const Array_1D& other) {
    assert(this->get_vertices() == other.get_vertices());
    Linalg::hadamard_divide_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_1D<T>& Array_1D<T>::operator+=(const T num) {
    Linalg::scalar_plus_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_1D<T>& Array_1D<T>::operator-=(const T num) {
    Linalg::scalar_minus_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_1D<T>& Array_1D<T>::operator*=(const T num) {
    Linalg::scalar_product_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_1D<T>& Array_1D<T>::operator/=(const T num) {
    Linalg::scalar_divide_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_1D<T> Array_1D<T>::operator+(const Array_1D& other) const {
    assert(this->get_vertices() == other.get_vertices());
    Array_1D<T> __result(*this);
    __result += other;
    return __result;
}

template<typename T>
Array_1D<T> Array_1D<T>::operator-(const Array_1D& other) const {
    assert(this->get_vertices() == other.get_vertices());
    Array_1D<T> __result(*this);
    __result -= other;
    return __result;
}

template<typename T>
Array_1D<T> Array_1D<T>::operator*(const Array_1D& other) const {
    assert(this->get_vertices() == other.get_vertices());
    Array_1D<T> __result(*this);
    __result *= other;
    return __result;
}

template<typename T>
Array_1D<T> Array_1D<T>::operator/(const Array_1D& other) const {
    assert(this->get_vertices() == other.get_vertices());
    Array_1D<T> __result(*this);
    __result /= other;
    return __result;
}

template<typename T>
Array_1D<T> Array_1D<T>::operator+(const T num) const {
    Array_1D<T> __result(*this);
    __result += num;
    return __result;
}

template<typename T>
Array_1D<T> Array_1D<T>::operator-(const T num) const {
    Array_1D<T> __result(*this);
    __result -= num;
    return __result;
}

template<typename T>
Array_1D<T> Array_1D<T>::operator*(const T num) const {
    Array_1D<T> __result(*this);
    __result *= num;
    return __result;
}

template<typename T>
Array_1D<T> Array_1D<T>::operator/(const T num) const {
    Array_1D<T> __result(*this);
    __result /= num;
    return __result;
}

template<typename T>
Array_1D<T> Array_1D<T>::sub_arr(const Vertices_1D& other) const {
    Array_1D<T> result(other);
    return this->sub_arr(result);
}

template<typename T>
Array_1D<T>& Array_1D<T>::sub_arr(Array_1D& other) const {
    const Vertices_1D& other_vertices = other.get_vertices();
    Vertices_method::fill_region(this->data, this->get_vertices(), other.data,
                                 other_vertices, other_vertices);
    return other;
}

// template<typename T>
// Array_1D<T>& Array_1D<T>::fill_overlap(Array_1D& other) const {
//     Vertices_1D temp = this->get_overlap_vertices(other);
//     return this->fill_overlap(other, temp);
// }

// template<typename T>
// Array_1D<T>& Array_1D<T>::fill_overlap(Array_1D& other, const Vertices_1D& temp) const {
//     assert(this->is_ex_vertices(temp) && other.is_ex_vertices(temp));
//     uint index_this_origin = this->get_index_nocheck(temp.is);
//     uint index_other_origin = other.get_index_nocheck(temp.is);
//     T* const& __restrict__ other_data = other.data;
//     T const* const& __restrict__ this_data = this->data;
//     #ifdef USE_OPENMP_SIMD
//     #pragma omp for schedule(static, Linalg::get_chunksize(omp_get_num_threads(), temp.ni))
//     #endif //USE_OPENMP_SIMD
//     for (uint i = 0; i < temp.ni; ++i)
//     {
//         other_data[index_other_origin + i] = this_data[index_this_origin + i];
//     }
//     return other;
// }

// template<typename T>
// Array_1D<T>& Array_1D<T>::be_filled_overlap(const Array_1D& other) {
//     other.fill_overlap(*this);
//     return *this;
// }

// template<typename T>
// Array_1D<T>& Array_1D<T>::be_filled_overlap(const Array_1D& other, const Vertices_1D& temp) {
//     other.fill_overlap(*this, temp);
//     return *this;
// }

template<typename T>
void Array_1D<T>::show() const {
    this->Vertices_1D::show();
    this->Array_0D<T>::show();
    return;
}

template<typename T>
void Array_1D<T>::print_h(std::ostream& output) const {
    if (this->is_Col_Maj) {
        this->print_Col_Maj_h(output);
    } else {
        this->print_Row_Maj_h(output);
    }
    return;
}

template<typename T>
void Array_1D<T>::print_Col_Maj_h(std::ostream& output) const {
    for (uint i = 0; i < this->ni; i++)
    {
        output << std::right << std::setw(14) << std::setprecision(6)
               << std::scientific << std::uppercase 
               << this->data[i];
        if (i % 6 == 5)
            output << std::endl;
    }
    output << std::endl;
    return;
}

template<typename T>
void Array_1D<T>::print_Row_Maj_h(std::ostream& output) const {
    for (uint i = 0; i < this->ni; i++)
    {
        output << std::setw(14) << std::setprecision(6) 
               << std::scientific << std::uppercase 
               << this->data[i];
        if (i % 6 == 5)
            output << std::endl;
    }
    output << std::endl;
    return;
}

//Array_2D<T>
template<typename T>
Array_2D<T>::Array_2D() : Array_0D<T>::Array_0D(), Vertices_2D::Vertices_2D() {
}

template<typename T>
Array_2D<T>::Array_2D(const uint64_t length) : Array_0D<T>::Array_0D(length), Vertices_2D::Vertices_2D(length, 0) {
}

template<typename T>
Array_2D<T>::Array_2D(const uint64_t length, const uint) : Array_0D<T>::Array_0D(length, 0), Vertices_2D::Vertices_2D(length, 0) {
}

template<typename T>
Array_2D<T>::Array_2D(const Vertices_2D& other) : Array_0D<T>::Array_0D(other.get_size()), Vertices_2D::Vertices_2D(other) {
}

template<typename T>
Array_2D<T>::Array_2D(const Vertices_2D& other, const uint) : Array_0D<T>::Array_0D(other.get_size(), 0), Vertices_2D::Vertices_2D(other) {
}

template<typename T>
Array_2D<T>::Array_2D(const Array_2D& other) : Array_0D<T>::Array_0D(other), Vertices_2D::Vertices_2D(other) {
}

template<typename T>
Array_2D<T>::Array_2D(Array_2D&& other) : Array_0D<T>::Array_0D(std::move(other)), Vertices_2D::Vertices_2D(other) {
}

template<typename T>
Array_2D<T>::~Array_2D() {
    this->destructor();
}

template<typename T>
void Array_2D<T>::destructor() {
    this->Array_0D<T>::destructor();
    this->Vertices_2D::set_ni(0);
    this->Vertices_2D::set_nj(0);
    return;
}

template<typename T>
void Array_2D<T>::constructor(const uint64_t length) {
    this->Array_0D<T>::constructor(length);
    this->Vertices_2D::set_ni(length);
    this->Vertices_2D::set_nj(0);
    return;
}

template<typename T>
void Array_2D<T>::constructor(const uint64_t ni, const uint64_t nj) {
    this->Array_0D<T>::constructor(ni * nj);
    this->Vertices_2D::set_ni(ni);
    this->Vertices_2D::set_nj(nj);
    return;
}

template<typename T>
void Array_2D<T>::constructor(const Vertices_2D& other) {
    this->Array_0D<T>::constructor(other.get_size());
    this->Vertices_2D::set_vertices(other);
    return;
}

template<typename T>
void Array_2D<T>::constructor(const Vertices_2D& other, const uint) {
    this->Array_0D<T>::constructor(other.get_size(), 0);
    this->Vertices_2D::set_vertices(other);
    return;
}

template<typename T>
void Array_2D<T>::reconstructor(const uint64_t length) {
    this->destructor();
    this->constructor(length);
    return;
}

template<typename T>
void Array_2D<T>::reconstructor(const uint64_t ni, const uint64_t nj) {
    this->destructor();
    this->constructor(ni, nj);
    return;
}

template<typename T>
void Array_2D<T>::reconstructor(const Vertices_2D& other) {
    this->destructor();
    this->constructor(other);
    return;
}

template<typename T>
void Array_2D<T>::reconstructor(const Vertices_2D& other, const uint) {
    this->destructor();
    this->constructor(other, 0);
    return;
}

#ifdef USE_HBM
#ifdef USE_HUGEPAGE_HBM
template<typename T>
void Array_2D<T>::constructor_hbm(const uint64_t length) {
    this->Array_0D<T>::constructor_hbm(length);
    this->Vertices_2D::set_ni(length);
    this->Vertices_2D::set_nj(0);
    return;
}

template<typename T>
void Array_2D<T>::constructor_hbm(const uint64_t ni, const uint64_t nj) {
    this->Array_0D<T>::constructor_hbm(ni * nj);
    this->Vertices_2D::set_ni(ni);
    this->Vertices_2D::set_nj(nj);
    return;
}

template<typename T>
void Array_2D<T>::constructor_hbm(const Vertices_2D& other) {
    this->Array_0D<T>::constructor_hbm(other.get_size());
    this->Vertices_2D::set_vertices(other);
    return;
}

template<typename T>
void Array_2D<T>::destructor_hbm() {
    this->Array_0D<T>::destructor_hbm();
    this->Vertices_2D::set_ni(0);
    this->Vertices_2D::set_nj(0);
    return;
}

template<typename T>
void Array_2D<T>::reconstructor_hbm(const uint64_t length) {
    this->destructor_hbm();
    this->constructor_hbm(length);
    return;
}

template<typename T>
void Array_2D<T>::reconstructor_hbm(const uint64_t ni, const uint64_t nj) {
    this->destructor_hbm();
    this->constructor_hbm(ni, nj);
    return;
}

template<typename T>
void Array_2D<T>::reconstructor_hbm(const Vertices_2D& other) {
    this->destructor_hbm();
    this->constructor_hbm(other);
    return;
}

//hugapage
template<typename T>
void Array_2D<T>::constructor_hp_hbm(const uint64_t length) {
    this->Array_0D<T>::constructor_hp_hbm(length);
    this->Vertices_2D::set_ni(length);
    this->Vertices_2D::set_nj(0);
    return;
}

template<typename T>
void Array_2D<T>::constructor_hp_hbm(const uint64_t ni, const uint64_t nj) {
    this->Array_0D<T>::constructor_hp_hbm(ni * nj);
    this->Vertices_2D::set_ni(ni);
    this->Vertices_2D::set_nj(nj);
    return;
}

template<typename T>
void Array_2D<T>::constructor_hp_hbm(const Vertices_2D& other) {
    this->Array_0D<T>::constructor_hp_hbm(other.get_size());
    this->Vertices_2D::set_vertices(other);
    return;
}

template<typename T>
void Array_2D<T>::destructor_hp_hbm() {
    this->Array_0D<T>::destructor_hp_hbm();
    this->Vertices_2D::set_ni(0);
    this->Vertices_2D::set_nj(0);
    return;
}

template<typename T>
void Array_2D<T>::reconstructor_hp_hbm(const uint64_t length) {
    this->destructor_hp_hbm();
    this->constructor_hp_hbm(length);
    return;
}

template<typename T>
void Array_2D<T>::reconstructor_hp_hbm(const uint64_t ni, const uint64_t nj) {
    this->destructor_hp_hbm();
    this->constructor_hp_hbm(ni, nj);
    return;
}

template<typename T>
void Array_2D<T>::reconstructor_hp_hbm(const Vertices_2D& other) {
    this->destructor_hp_hbm();
    this->constructor_hp_hbm(other);
    return;
}
#else
template<typename T>
void Array_2D<T>::constructor_hbm(const uint64_t length) {
    this->Array_0D<T>::constructor_hbm(length);
    this->Vertices_2D::set_ni(length);
    this->Vertices_2D::set_nj(0);
    return;
}

template<typename T>
void Array_2D<T>::constructor_hbm(const uint64_t ni, const uint64_t nj) {
    this->Array_0D<T>::constructor_hbm(ni * nj);
    this->Vertices_2D::set_ni(ni);
    this->Vertices_2D::set_nj(nj);
    return;
}

template<typename T>
void Array_2D<T>::constructor_hbm(const Vertices_2D& other) {
    this->Array_0D<T>::constructor_hbm(other.get_size());
    this->Vertices_2D::set_vertices(other);
    return;
}

template<typename T>
void Array_2D<T>::destructor_hbm() {
    this->Array_0D<T>::destructor_hbm();
    this->Vertices_2D::set_ni(0);
    this->Vertices_2D::set_nj(0);
    return;
}

template<typename T>
void Array_2D<T>::reconstructor_hbm(const uint64_t length) {
    this->destructor_hbm();
    this->constructor_hbm(length);
    return;
}

template<typename T>
void Array_2D<T>::reconstructor_hbm(const uint64_t ni, const uint64_t nj) {
    this->destructor_hbm();
    this->constructor_hbm(ni, nj);
    return;
}

template<typename T>
void Array_2D<T>::reconstructor_hbm(const Vertices_2D& other) {
    this->destructor_hbm();
    this->constructor_hbm(other);
    return;
}
#endif
#endif

template<typename T>
void Array_2D<T>::swap(Array_2D<T>& other) {
    this->Vertices_2D::swap(other);
    this->Array_0D<T>::swap(other);
    std::swap(this->is_Col_Maj, other.is_Col_Maj);
    return;
}

template<typename T>
template<typename T2>
Array_2D<T2>
Array_2D<T>::as_type(const T2*) const {
    Array_2D<T2> result(this->get_vertices());
    Linalg::convert_type<T2, T>(result.data, this->data, this->length);
    return result;
}
template Array_2D<double> Array_2D<double>::as_type(const double*) const;
template Array_2D<float> Array_2D<float>::as_type(const float*) const;
template Array_2D<int> Array_2D<int>::as_type(const int*) const;
template Array_2D<double> Array_2D<float>::as_type(const double*) const;
template Array_2D<float> Array_2D<double>::as_type(const float*) const;
template Array_2D<int> Array_2D<float>::as_type(const int*) const;
template Array_2D<float> Array_2D<int>::as_type(const float*) const;
template Array_2D<int> Array_2D<double>::as_type(const int*) const;
template Array_2D<double> Array_2D<int>::as_type(const double*) const;
#ifdef FP16_FLAG
template Array_2D<__fp16> Array_2D<__fp16>::as_type(const __fp16*) const;
template Array_2D<__fp16> Array_2D<float>::as_type(const __fp16*) const;
template Array_2D<__fp16> Array_2D<double>::as_type(const __fp16*) const;
template Array_2D<float> Array_2D<__fp16>::as_type(const float*) const;
template Array_2D<double> Array_2D<__fp16>::as_type(const double*) const;
#endif //FP16_FLAG

template<typename T>
uint Array_2D<T>::get_leading_dimension() const {
    return this->ni;
}

template<typename T>
uint Array_2D<T>::get_second_dimension() const {
    return this->nj;
}

template<typename T>
Array_2D<T>& Array_2D<T>::operator=(const Array_2D<T>& other) {
    assert(this->get_vertices() == other.get_vertices());
    this->deepcopy(other);
    return *this;
}

template<typename T>
Array_2D<T>& Array_2D<T>::operator=(Array_2D<T>&& other) {
    assert(this->get_vertices() == other.get_vertices());
    this->deepcopy(std::move(other));
    return *this;
}

template<typename T>
Array_2D<T>& Array_2D<T>::operator=(const T num) {
    for (uint i = 0; i < this->length; i++) {
        this->data[i] = num;
    }
    return *this;
}

template<typename T>
Array_2D<T>& Array_2D<T>::deepcopy(const Array_2D<T>& other) {
    if (this != &other) {
        if (this->length != other.length) {
            this->reconstructor(other.get_vertices());
        } else if (this->get_vertices() != other.get_vertices()) {
            this->set_vertices(other.get_vertices());
        }
        // std::memcpy(this->data, other.data, this->length * sizeof(T));
        Linalg::set_value_general(this->data, other.data, this->length);
        this->is_Col_Maj = other.is_Col_Maj;
    }
    return *this;
}

template<typename T>
Array_2D<T>& Array_2D<T>::deepcopy(Array_2D&& other) {
    if (this->get_vertices() != other.get_vertices()) {
        this->set_vertices(other.get_vertices());
    }
    std::swap(this->length, other.length);
    std::swap(this->data, other.data);
    this->is_Col_Maj = other.is_Col_Maj;
    return *this;
}

template<typename T>
Array_2D<T>& Array_2D<T>::operator+=(const Array_2D& other) {
    assert(this->get_vertices() == other.get_vertices());
    Linalg::hadamard_plus_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_2D<T>& Array_2D<T>::operator-=(const Array_2D& other) {
    assert(this->get_vertices() == other.get_vertices());
    Linalg::hadamard_minus_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_2D<T>& Array_2D<T>::operator*=(const Array_2D& other) {
    assert(this->get_vertices() == other.get_vertices());
    Linalg::hadamard_product_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_2D<T>& Array_2D<T>::operator/=(const Array_2D& other) {
    assert(this->get_vertices() == other.get_vertices());
    Linalg::hadamard_divide_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_2D<T>& Array_2D<T>::operator+=(const T num) {
    Linalg::scalar_plus_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_2D<T>& Array_2D<T>::operator-=(const T num) {
    Linalg::scalar_minus_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_2D<T>& Array_2D<T>::operator*=(const T num) {
    Linalg::scalar_product_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_2D<T>& Array_2D<T>::operator/=(const T num) {
    Linalg::scalar_divide_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_2D<T> Array_2D<T>::operator+(const Array_2D& other) const {
    assert(this->get_vertices() == other.get_vertices());
    Array_2D<T> __result(*this);
    __result += other;
    return __result;
}

template<typename T>
Array_2D<T> Array_2D<T>::operator-(const Array_2D& other) const {
    assert(this->get_vertices() == other.get_vertices());
    Array_2D<T> __result(*this);
    __result -= other;
    return __result;
}

template<typename T>
Array_2D<T> Array_2D<T>::operator*(const Array_2D& other) const {
    assert(this->get_vertices() == other.get_vertices());
    Array_2D<T> __result(*this);
    __result *= other;
    return __result;
}

template<typename T>
Array_2D<T> Array_2D<T>::operator/(const Array_2D& other) const {
    assert(this->get_vertices() == other.get_vertices());
    Array_2D<T> __result(*this);
    __result /= other;
    return __result;
}

template<typename T>
Array_2D<T> Array_2D<T>::operator+(const T num) const {
    Array_2D<T> __result(*this);
    __result += num;
    return __result;
}

template<typename T>
Array_2D<T> Array_2D<T>::operator-(const T num) const {
    Array_2D<T> __result(*this);
    __result -= num;
    return __result;
}

template<typename T>
Array_2D<T> Array_2D<T>::operator*(const T num) const {
    Array_2D<T> __result(*this);
    __result *= num;
    return __result;
}

template<typename T>
Array_2D<T> Array_2D<T>::operator/(const T num) const {
    Array_2D<T> __result(*this);
    __result /= num;
    return __result;
}

template<typename T>
Array_2D<T> Array_2D<T>::sub_arr(const Vertices_2D& other) const {
    Array_2D<T> result(other);
    return this->sub_arr(result);
}

template<typename T>
Array_2D<T>& Array_2D<T>::sub_arr(Array_2D& other) const {
    const Vertices_2D& other_vertices = other.get_vertices();
    Vertices_method::fill_region(this->data, this->get_vertices(), other.data,
                                 other_vertices, other_vertices);
    return other;
}

// template<typename T>
// Array_2D<T>& Array_2D<T>::fill_overlap(Array_2D& other) const {
//     Vertices_2D temp = this->get_overlap_vertices(other);
//     return this->fill_overlap(other, temp);
// }

// template<typename T>
// Array_2D<T>& Array_2D<T>::fill_overlap(Array_2D& other, const Vertices_2D& temp) const {
//     assert(this->is_ex_vertices(temp) && other.is_ex_vertices(temp));
//     uint index_this_origin = this->get_index_nocheck(temp.is, temp.js);
//     uint index_other_origin = other.get_index_nocheck(temp.is, temp.js);
//     const uint& this_ni = this->ni;
//     const uint& other_ni = other.ni;
//     T* const& __restrict__ other_data = other.data;
//     T const* const& __restrict__ this_data = this->data;
//     #ifdef USE_OPENMP_SIMD
//     #pragma omp for schedule(static, Linalg::get_chunksize(omp_get_num_threads(), temp.nj))
//     #endif //USE_OPENMP_SIMD
//     for (uint j = 0; j < temp.nj; ++j)
//     {
//         uint other_offset_j = j * other_ni;
//         uint this_offset_j = j * this_ni;
//         for (uint i = 0; i < temp.ni; ++i)
//         {
//             // other[index_other_origin + i + j * other_ni] = this->data[index_this_origin + i + j * this_ni];
//             other_data[index_other_origin + i + other_offset_j] = this_data[index_this_origin + i + this_offset_j];
//         }
//     }
//     return other;
// }

// template<typename T>
// Array_2D<T>& Array_2D<T>::be_filled_overlap(const Array_2D& other) {
//     other.fill_overlap(*this);
//     return *this;
// }

// template<typename T>
// Array_2D<T>& Array_2D<T>::be_filled_overlap(const Array_2D& other, const Vertices_2D& temp) {
//     other.fill_overlap(*this, temp);
//     return *this;
// }


template<typename T>
void Array_2D<T>::show() const {
    this->Vertices_2D::show();
    this->Array_0D<T>::show();
    return;
}

template<typename T>
void Array_2D<T>::print_h(std::ostream& output) const {
    if (this->is_Col_Maj) {
        this->print_Col_Maj_h(output);
    } else {
        this->print_Row_Maj_h(output);
    }
    return;
}

template<typename T>
void Array_2D<T>::print_Col_Maj_h(std::ostream& output) const {
    for (uint j = 0; j < this->nj; j++)
    {
        for (uint i = 0; i < this->ni; i++)
        {
            output << std::right << std::setw(14) << std::setprecision(6)
                   << std::scientific << std::uppercase 
                   << this->data[i + j * this->ni];
            if (i % 6 == 5)
                output << std::endl;
        }
        output << std::endl;
    }
    return;
}

template<typename T>
void Array_2D<T>::print_Row_Maj_h(std::ostream& output) const {
    for (uint i = 0; i < this->ni; i++)
    {
        for (uint j = 0; j < this->nj; j++)
        {
            output << std::setw(14) << std::setprecision(6) 
                   << std::scientific << std::uppercase 
                   << this->data[i + j * this->ni];
            if (j % 6 == 5)
                output << std::endl;
        }
        output << std::endl;
    }
    return;
}


//Array_3D<T>
template<typename T>
Array_3D<T>::Array_3D() : Array_0D<T>::Array_0D(), Vertices_3D::Vertices_3D() {
}

template<typename T>
Array_3D<T>::Array_3D(const uint64_t length) : Array_0D<T>::Array_0D(length), Vertices_3D::Vertices_3D(length, 0, 0) {
}

template<typename T>
Array_3D<T>::Array_3D(const uint64_t length, const uint) : Array_0D<T>::Array_0D(length, 0), Vertices_3D::Vertices_3D(length, 0, 0) {
}

template<typename T>
Array_3D<T>::Array_3D(const Vertices_3D& other) : Array_0D<T>::Array_0D(other.get_size()), Vertices_3D::Vertices_3D(other) {
}

template<typename T>
Array_3D<T>::Array_3D(const Vertices_3D& other, const uint) : Array_0D<T>::Array_0D(other.get_size(), 0), Vertices_3D::Vertices_3D(other) {
}

template<typename T>
Array_3D<T>::Array_3D(const Array_3D& other) : Array_0D<T>::Array_0D(other), Vertices_3D::Vertices_3D(other) {
}

template<typename T>
Array_3D<T>::Array_3D(Array_3D&& other) : Array_0D<T>::Array_0D(std::move(other)), Vertices_3D::Vertices_3D(other) {
}

template<typename T>
Array_3D<T>::~Array_3D() {
    this->destructor();
}

template<typename T>
void Array_3D<T>::destructor() {
    this->Array_0D<T>::destructor();
    this->Vertices_3D::set_ni(0);
    this->Vertices_3D::set_nj(0);
    this->Vertices_3D::set_nk(0);
    return;
}

template<typename T>
void Array_3D<T>::constructor(const uint64_t length) {
    this->Array_0D<T>::constructor(length);
    this->Vertices_3D::set_ni(length);
    this->Vertices_3D::set_nj(0);
    this->Vertices_3D::set_nk(0);
    return;
}

template<typename T>
void Array_3D<T>::constructor(const uint64_t ni, const uint64_t nj, const uint64_t nk) {
    this->Array_0D<T>::constructor(ni * nj * nk);
    this->Vertices_3D::set_ni(ni);
    this->Vertices_3D::set_nj(nj);
    this->Vertices_3D::set_nk(nk);
    return;
}

template<typename T>
void Array_3D<T>::constructor(const Vertices_3D& other) {
    this->Array_0D<T>::constructor(other.get_size());
    this->Vertices_3D::set_vertices(other);
    return;
}

template<typename T>
void Array_3D<T>::constructor(const Vertices_3D& other, const uint) {
    this->Array_0D<T>::constructor(other.get_size(), 0);
    this->Vertices_3D::set_vertices(other);
    return;
}

template<typename T>
void Array_3D<T>::reconstructor(const uint64_t length) {
    this->destructor();
    this->constructor(length);
    return;
}

template<typename T>
void Array_3D<T>::reconstructor(const uint64_t ni, const uint64_t nj, const uint64_t nk) {
    this->destructor();
    this->constructor(ni, nj, nk);
    return;
}

template<typename T>
void Array_3D<T>::reconstructor(const Vertices_3D& other) {
    this->destructor();
    this->constructor(other);
    return;
}

template<typename T>
void Array_3D<T>::reconstructor(const Vertices_3D& other, const uint) {
    this->destructor();
    this->constructor(other, 0);
    return;
}

#ifdef USE_HBM
#ifdef USE_HUGEPAGE_HBM
template<typename T>
void Array_3D<T>::constructor_hbm(const uint64_t length) {
    this->Array_0D<T>::constructor_hbm(length);
    this->Vertices_3D::set_ni(length);
    this->Vertices_3D::set_nj(0);
    this->Vertices_3D::set_nk(0);
    return;
}

template<typename T>
void Array_3D<T>::constructor_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk) {
    this->Array_0D<T>::constructor_hbm(ni * nj * nk);
    this->Vertices_3D::set_ni(ni);
    this->Vertices_3D::set_nj(nj);
    this->Vertices_3D::set_nk(nk);
    return;
}

template<typename T>
void Array_3D<T>::constructor_hbm(const Vertices_3D& other) {
    this->Array_0D<T>::constructor_hbm(other.get_size());
    this->Vertices_3D::set_vertices(other);
    return;
}

template<typename T>
void Array_3D<T>::destructor_hbm() {
    this->Array_0D<T>::destructor_hbm();
    this->Vertices_3D::set_ni(0);
    this->Vertices_3D::set_nj(0);
    this->Vertices_3D::set_nk(0);
    return;
}

template<typename T>
void Array_3D<T>::reconstructor_hbm(const uint64_t length) {
    this->destructor_hbm();
    this->constructor_hbm(length);
    return;
}

template<typename T>
void Array_3D<T>::reconstructor_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk) {
    this->destructor_hbm();
    this->constructor_hbm(ni, nj, nk);
    return;
}

template<typename T>
void Array_3D<T>::reconstructor_hbm(const Vertices_3D& other) {
    this->destructor_hbm();
    this->constructor_hbm(other);
    return;
}

//hugepage
template<typename T>
void Array_3D<T>::constructor_hp_hbm(const uint64_t length) {
    this->Array_0D<T>::constructor_hp_hbm(length);
    this->Vertices_3D::set_ni(length);
    this->Vertices_3D::set_nj(0);
    this->Vertices_3D::set_nk(0);
    return;
}

template<typename T>
void Array_3D<T>::constructor_hp_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk) {
    this->Array_0D<T>::constructor_hp_hbm(ni * nj * nk);
    this->Vertices_3D::set_ni(ni);
    this->Vertices_3D::set_nj(nj);
    this->Vertices_3D::set_nk(nk);
    return;
}

template<typename T>
void Array_3D<T>::constructor_hp_hbm(const Vertices_3D& other) {
    this->Array_0D<T>::constructor_hp_hbm(other.get_size());
    this->Vertices_3D::set_vertices(other);
    return;
}

template<typename T>
void Array_3D<T>::destructor_hp_hbm() {
    this->Array_0D<T>::destructor_hp_hbm();
    this->Vertices_3D::set_ni(0);
    this->Vertices_3D::set_nj(0);
    this->Vertices_3D::set_nk(0);
    return;
}

template<typename T>
void Array_3D<T>::reconstructor_hp_hbm(const uint64_t length) {
    this->destructor_hp_hbm();
    this->constructor_hp_hbm(length);
    return;
}

template<typename T>
void Array_3D<T>::reconstructor_hp_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk) {
    this->destructor_hp_hbm();
    this->constructor_hp_hbm(ni, nj, nk);
    return;
}

template<typename T>
void Array_3D<T>::reconstructor_hp_hbm(const Vertices_3D& other) {
    this->destructor_hp_hbm();
    this->constructor_hp_hbm(other);
    return;
}
#else
template<typename T>
void Array_3D<T>::constructor_hbm(const uint64_t length) {
    this->Array_0D<T>::constructor_hbm(length);
    this->Vertices_3D::set_ni(length);
    this->Vertices_3D::set_nj(0);
    this->Vertices_3D::set_nk(0);
    return;
}

template<typename T>
void Array_3D<T>::constructor_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk) {
    this->Array_0D<T>::constructor_hbm(ni * nj * nk);
    this->Vertices_3D::set_ni(ni);
    this->Vertices_3D::set_nj(nj);
    this->Vertices_3D::set_nk(nk);
    return;
}

template<typename T>
void Array_3D<T>::constructor_hbm(const Vertices_3D& other) {
    this->Array_0D<T>::constructor_hbm(other.get_size());
    this->Vertices_3D::set_vertices(other);
    return;
}

template<typename T>
void Array_3D<T>::destructor_hbm() {
    this->Array_0D<T>::destructor_hbm();
    this->Vertices_3D::set_ni(0);
    this->Vertices_3D::set_nj(0);
    this->Vertices_3D::set_nk(0);
    return;
}

template<typename T>
void Array_3D<T>::reconstructor_hbm(const uint64_t length) {
    this->destructor_hbm();
    this->constructor_hbm(length);
    return;
}

template<typename T>
void Array_3D<T>::reconstructor_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk) {
    this->destructor_hbm();
    this->constructor_hbm(ni, nj, nk);
    return;
}

template<typename T>
void Array_3D<T>::reconstructor_hbm(const Vertices_3D& other) {
    this->destructor_hbm();
    this->constructor_hbm(other);
    return;
}
#endif
#endif

template<typename T>
void Array_3D<T>::swap(Array_3D<T>& other) {
    this->Vertices_3D::swap(other);
    this->Array_0D<T>::swap(other);
    std::swap(this->is_Col_Maj, other.is_Col_Maj);
    return;
}

template<typename T>
template<typename T2>
Array_3D<T2>
Array_3D<T>::as_type(const T2*) const {
    Array_3D<T2> result(this->get_vertices());
    Linalg::convert_type<T2, T>(result.data, this->data, this->length);
    return result;
}
template Array_3D<double> Array_3D<double>::as_type(const double*) const;
template Array_3D<float> Array_3D<float>::as_type(const float*) const;
template Array_3D<int> Array_3D<int>::as_type(const int*) const;
template Array_3D<double> Array_3D<float>::as_type(const double*) const;
template Array_3D<float> Array_3D<double>::as_type(const float*) const;
template Array_3D<int> Array_3D<float>::as_type(const int*) const;
template Array_3D<float> Array_3D<int>::as_type(const float*) const;
template Array_3D<int> Array_3D<double>::as_type(const int*) const;
template Array_3D<double> Array_3D<int>::as_type(const double*) const;
#ifdef FP16_FLAG
template Array_3D<__fp16> Array_3D<__fp16>::as_type(const __fp16*) const;
template Array_3D<__fp16> Array_3D<float>::as_type(const __fp16*) const;
template Array_3D<__fp16> Array_3D<double>::as_type(const __fp16*) const;
template Array_3D<float> Array_3D<__fp16>::as_type(const float*) const;
template Array_3D<double> Array_3D<__fp16>::as_type(const double*) const;
#endif //FP16_FLAG

template<typename T>
uint Array_3D<T>::get_leading_dimension() const {
    return this->ni * this->nj;
}

template<typename T>
uint Array_3D<T>::get_second_dimension() const {
    return this->nk;
}

template<typename T>
Array_3D<T>& Array_3D<T>::operator=(const Array_3D<T>& other) {
    assert(this->get_vertices() == other.get_vertices());
    this->deepcopy(other);
    return *this;
}

template<typename T>
Array_3D<T>& Array_3D<T>::operator=(Array_3D<T>&& other) {
    assert(this->get_vertices() == other.get_vertices());
    this->deepcopy(std::move(other));
    return *this;
}

template<typename T>
Array_3D<T>& Array_3D<T>::operator=(const T num) {
    for (uint i = 0; i < this->length; i++) {
        this->data[i] = num;
    }
    return *this;
}

template<typename T>
Array_3D<T>& Array_3D<T>::deepcopy(const Array_3D<T>& other) {
    if (this != &other) {
        if (this->length != other.length) {
            this->reconstructor(other.get_vertices());
        } else if (this->get_vertices() != other.get_vertices()) {
            this->set_vertices(other.get_vertices());
        }
        // std::memcpy(this->data, other.data, this->length * sizeof(T));
        Linalg::set_value_general(this->data, other.data, this->length);
        this->is_Col_Maj = other.is_Col_Maj;
    }
    return *this;
}

template<typename T>
Array_3D<T>& Array_3D<T>::deepcopy(Array_3D&& other) {
    if (this->get_vertices() != other.get_vertices()) {
        this->set_vertices(other.get_vertices());
    }
    std::swap(this->length, other.length);
    std::swap(this->data, other.data);
    this->is_Col_Maj = other.is_Col_Maj;
    return *this;
}

template<typename T>
Array_3D<T>& Array_3D<T>::operator+=(const Array_3D& other) {
    assert(this->get_vertices() == other.get_vertices());
    Linalg::hadamard_plus_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_3D<T>& Array_3D<T>::operator-=(const Array_3D& other) {
    assert(this->get_vertices() == other.get_vertices());
    Linalg::hadamard_minus_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_3D<T>& Array_3D<T>::operator*=(const Array_3D& other) {
    assert(this->get_vertices() == other.get_vertices());
    Linalg::hadamard_product_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_3D<T>& Array_3D<T>::operator/=(const Array_3D& other) {
    assert(this->get_vertices() == other.get_vertices());
    Linalg::hadamard_divide_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_3D<T>& Array_3D<T>::operator+=(const T num) {
    Linalg::scalar_plus_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_3D<T>& Array_3D<T>::operator-=(const T num) {
    Linalg::scalar_minus_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_3D<T>& Array_3D<T>::operator*=(const T num) {
    Linalg::scalar_product_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_3D<T>& Array_3D<T>::operator/=(const T num) {
    Linalg::scalar_divide_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_3D<T> Array_3D<T>::operator+(const Array_3D& other) const {
    assert(this->get_vertices() == other.get_vertices());
    Array_3D<T> __result(*this);
    __result += other;
    return __result;
}

template<typename T>
Array_3D<T> Array_3D<T>::operator-(const Array_3D& other) const {
    assert(this->get_vertices() == other.get_vertices());
    Array_3D<T> __result(*this);
    __result -= other;
    return __result;
}

template<typename T>
Array_3D<T> Array_3D<T>::operator*(const Array_3D& other) const {
    assert(this->get_vertices() == other.get_vertices());
    Array_3D<T> __result(*this);
    __result *= other;
    return __result;
}

template<typename T>
Array_3D<T> Array_3D<T>::operator/(const Array_3D& other) const {
    assert(this->get_vertices() == other.get_vertices());
    Array_3D<T> __result(*this);
    __result /= other;
    return __result;
}

template<typename T>
Array_3D<T> Array_3D<T>::operator+(const T num) const {
    Array_3D<T> __result(*this);
    __result += num;
    return __result;
}

template<typename T>
Array_3D<T> Array_3D<T>::operator-(const T num) const {
    Array_3D<T> __result(*this);
    __result -= num;
    return __result;
}

template<typename T>
Array_3D<T> Array_3D<T>::operator*(const T num) const {
    Array_3D<T> __result(*this);
    __result *= num;
    return __result;
}

template<typename T>
Array_3D<T> Array_3D<T>::operator/(const T num) const {
    Array_3D<T> __result(*this);
    __result /= num;
    return __result;
}

template<typename T>
Array_3D<T> Array_3D<T>::sub_arr(const Vertices_3D& other) const {
    Array_3D<T> result(other);
    return this->sub_arr(result);
}

template<typename T>
Array_3D<T>& Array_3D<T>::sub_arr(Array_3D& other) const {
    const Vertices_3D& other_vertices = other.get_vertices();
    Vertices_method::fill_region(this->data, this->get_vertices(), other.data,
                                 other_vertices, other_vertices);
    return other;
}

template<typename T>
Array_3D<T>& Array_3D<T>::fill_overlap(Array_3D& other) const {
    Vertices_3D temp = this->get_overlap_vertices(other);
    return this->fill_overlap(other, temp);
}

template<typename T>
Array_3D<T>& Array_3D<T>::fill_overlap(Array_3D& other, const Vertices_3D& temp) const {
    Vertices_method::fill_region(this->data, this->get_vertices(),
                                 other.data, other.get_vertices(), temp);
    return other;
}

template<typename T>
Array_3D<T>& Array_3D<T>::accumulate_overlap(Array_3D& other) const {
    Vertices_3D temp = this->get_overlap_vertices(other);
    return this->accumulate_overlap(other, temp);
}

template<typename T>
Array_3D<T>& Array_3D<T>::accumulate_overlap(Array_3D& other, const Vertices_3D& temp) const {
    Vertices_method::accumulate_overlap(this->data, this->get_vertices(),
                                        other.data, other.get_vertices(), temp);
    return other;
}

template<typename T>
Array_0D<T>& Array_3D<T>::fill_vector(Array_0D<T>& other, const Vertices_3D& temp, const uint& offset) const {
    assert(offset + temp.get_size() <= other.get_length());
    Vertices_method::fill_vector(this->data, this->get_vertices(), other.data, temp, offset);
    return other;
}

template<typename T>
Array_3D<T>& Array_3D<T>::be_filled_overlap(const Array_3D& other) {
    other.fill_overlap(*this);
    return *this;
}

template<typename T>
Array_3D<T>& Array_3D<T>::be_filled_overlap(const Array_3D& other, const Vertices_3D& temp) {
    other.fill_overlap(*this, temp);
    return *this;
}

template<typename T>
Array_3D<T>& Array_3D<T>::be_accumulated_overlap(const Array_3D& other) {
    other.accumulate_overlap(*this);
    return *this;
}

template<typename T>
Array_3D<T>& Array_3D<T>::be_accumulated_overlap(const Array_3D& other, const Vertices_3D& temp) {
    other.accumulate_overlap(*this, temp);
    return *this;
}

template<typename T>
Array_3D<T>& Array_3D<T>::be_filled_vector(const Array_0D<T>& other, const Vertices_3D& temp, uint offset) {
    assert(this->is_ex_vertices(temp) && offset + temp.get_size() <= other.get_length());
    Vertices_method::be_filled_vector(this->data, this->get_vertices(), other.data, temp, temp, offset);
    return *this;
}

template<typename T>
Array_3D<T>& Array_3D<T>::be_filled_vector(const Array_0D<T>& other, const Vertices_3D& s_vertices,
                                           const Vertices_3D& t_vertices, const uint& offset) {
    if (t_vertices.get_size() == 0) return *this;
    assert(this->is_ex_vertices(t_vertices));
    Vertices_method::be_filled_vector(this->data, this->get_vertices(), other.data, s_vertices, t_vertices, offset);
    return *this;
}

template<typename T>
void Array_3D<T>::show() const {
    this->Vertices_3D::show();
    this->Array_0D<T>::show();
    return;
}

template<typename T>
void Array_3D<T>::print_h(std::ostream& output) const {
    if (this->is_Col_Maj) {
        this->print_Col_Maj_h(output);
    } else {
        this->print_Row_Maj_h(output);
    }
    return;
}

template<typename T>
void Array_3D<T>::print_Col_Maj_h(std::ostream& output) const {
    for (uint k = 0; k < this->nk; k++)
    {
        for (uint j = 0; j < this->nj; j++)
        {
            for (uint i = 0; i < this->ni; i++)
            {
                output << std::right << std::setw(14) << std::setprecision(6)
                       << std::scientific << std::uppercase 
                       << this->data[i + (j + k * this->nj) * this->ni];
                if (i % 6 == 5)
                    output << std::endl;
            }
            output << std::endl;
        }
    }
    return;
}

template<typename T>
void Array_3D<T>::print_Row_Maj_h(std::ostream& output) const {
    for (uint i = 0; i < this->ni; i++)
    {
        for (uint j = 0; j < this->nj; j++)
        {
            for (uint k = 0; k < this->nk; k++)
            {
                output << std::setw(14) << std::setprecision(6) 
                       << std::scientific << std::uppercase 
                       << this->data[i + (j + k * this->nj) * this->ni];
                if (k % 6 == 5)
                    output << std::endl;
            }
            output << std::endl;
        }
    }
    return;
}

// template<typename T>
// Array_3D<T> Array_3D<T>::calc_gradient(const Vertices_3D& temp, const Stencil<T>& stencil, char direction) const {
//     if (temp.get_size() == 0) return Array_3D(0);
//     assert(this->is <= temp.is - (int)stencil.FDn && this->get_ie() <= temp.get_ie() + (int)stencil.FDn &&
//            this->js <= temp.js - (int)stencil.FDn && this->get_je() <= temp.get_je() + (int)stencil.FDn &&
//            this->ks <= temp.ks - (int)stencil.FDn && this->get_ke() <= temp.get_ke() + (int)stencil.FDn);
//     Array_3D __result(temp);
//     uint index_this_origin = this->get_index_nocheck(temp.is, temp.js, temp.ks);
//     uint index_result_origin = __result.get_index_nocheck(temp.is, temp.js, temp.ks);
//     uint this_ni = this->ni;
//     uint result_ni = __result.ni;
//     uint this_ninj = this->ni * this->nj;
//     uint result_ninj = __result.ni * __result.nj;
//     uint stride;
//     T* stencil_coefs;
//     if ((direction == 'x') || (direction == 'X')|| (direction == 'i')) {
//         stride = 1;
//         stencil_coefs = stencil.D1_coeffs_x;
//     } else if ((direction == 'y') || (direction == 'Y')|| (direction == 'j')) {
//         stride = this_ni;
//         stencil_coefs = stencil.D1_coeffs_y;
//     } else if ((direction == 'z') || (direction == 'Z')|| (direction == 'k')) {
//         stride = this_ninj;
//         stencil_coefs = stencil.D1_coeffs_z;
//     } else {
//         assert(0 &&" ERROR:: direction should be X, Y or Z!");
//     }
 
//     #ifdef USE_OPENMP_SIMD
//     #pragma omp for schedule(static, Linalg::get_chunksize(omp_get_num_threads(), __result.nk))
//     #endif //USE_OPENMP_SIMD
//     for (uint k = 0; k < __result.nk; ++k)
//     {
//         uint result_offset_k = index_result_origin + k * result_ninj;
//         uint this_offset_k = index_this_origin + k * this_ninj;
//         uint result_offset_j = result_offset_k;
//         uint this_offset_j = this_offset_k;
//         for (uint j = 0; j < __result.nj; ++j)
//         {
//             uint result_offset_i = result_offset_j;
//             uint this_offset_i = this_offset_j;
//             for (uint i = 0; i < __result.ni; ++i)
//             {
//                 __result[result_offset_i] = 0.0;
//                 for (uint p = 1; p <= stencil.FDn; ++p)
//                 {
//                     uint stride_r = p * stride;
//                     __result[result_offset_i] += (this->data[this_offset_i+stride_r] - this->data[this_offset_i-stride_r])
//                                                 * stencil_coefs[p];
//                 }
//                 ++result_offset_i;
//                 ++this_offset_i;
//             }
//             result_offset_j += result_ni;
//             this_offset_j += this_ni;
//         }
//     }
//     return __result;
// }

// template<typename T>
// Array_3D<T> Array_3D<T>::calc_gradient(const Array_3D<T>& other, const Stencil<T>& stencil, char direction) const {
//     if (other.get_vertices().get_size() == 0) return Array_3D(0);
//     assert(this->is <= other.is - (int)stencil.FDn && this->get_ie() <= other.get_ie() + (int)stencil.FDn &&
//            this->js <= other.js - (int)stencil.FDn && this->get_je() <= other.get_je() + (int)stencil.FDn &&
//            this->ks <= other.ks - (int)stencil.FDn && this->get_ke() <= other.get_ke() + (int)stencil.FDn);
//     Array_3D __result(other.get_vertices());
//     uint index_this_origin = this->get_index_nocheck(__result.is, __result.js, __result.ks);
//     uint index_result_origin = __result.get_index_nocheck(__result.is, __result.js, __result.ks);
//     uint this_ni = this->ni;
//     uint result_ni = __result.ni;
//     uint this_ninj = this->ni * this->nj;
//     uint result_ninj = __result.ni * __result.nj;
//     uint stride;
//     T* stencil_coefs;
//     if ((direction == 'x') || (direction == 'X')|| (direction == 'i')) {
//         stride = 1;
//         stencil_coefs = stencil.D1_coeffs_x;
//     } else if ((direction == 'y') || (direction == 'Y')|| (direction == 'j')) {
//         stride = this_ni;
//         stencil_coefs = stencil.D1_coeffs_y;
//     } else if ((direction == 'z') || (direction == 'Z')|| (direction == 'k')) {
//         stride = this_ninj;
//         stencil_coefs = stencil.D1_coeffs_z;
//     } else {
//         assert(0 &&" ERROR:: direction should be X, Y or Z!");
//     }
 
//     #ifdef USE_OPENMP_SIMD
//     #pragma omp for schedule(static, Linalg::get_chunksize(omp_get_num_threads(), __result.nk))
//     #endif //USE_OPENMP_SIMD
//     for (uint k = 0; k < __result.nk; ++k)
//     {
//         uint result_offset_k = index_result_origin + k * result_ninj;
//         uint this_offset_k = index_this_origin + k * this_ninj;
//         uint result_offset_j = result_offset_k;
//         uint this_offset_j = this_offset_k;
//         for (uint j = 0; j < __result.nj; ++j)
//         {
//             uint result_offset_i = result_offset_j;
//             uint this_offset_i = this_offset_j;
//             for (uint i = 0; i < __result.ni; ++i)
//             {
//                 __result[result_offset_i] = other[result_offset_i];
//                 for (uint p = 1; p <= stencil.FDn; ++p)
//                 {
//                     uint stride_r = p * stride;
//                     __result[result_offset_i] += (this->data[this_offset_i+stride_r] - this->data[this_offset_i-stride_r])
//                                                 * stencil_coefs[p];
//                 }
//                 ++result_offset_i;
//                 ++this_offset_i;
//             }
//             result_offset_j += result_ni;
//             this_offset_j += this_ni;
//         }
//     }
//     return __result;
// }

// template<typename T>
// Array_3D<T> Array_3D<T>::calc_gradient(const Array_3D<T>& other, const Stencil<T>& stencil, char direction, const T& beta) const {
//     if (other.get_vertices().get_size() == 0) return Array_3D(0);
//     assert(this->is <= other.is - (int)stencil.FDn && this->get_ie() <= other.get_ie() + (int)stencil.FDn &&
//            this->js <= other.js - (int)stencil.FDn && this->get_je() <= other.get_je() + (int)stencil.FDn &&
//            this->ks <= other.ks - (int)stencil.FDn && this->get_ke() <= other.get_ke() + (int)stencil.FDn);
//     Array_3D __result(other.get_vertices());
//     uint index_this_origin = this->get_index_nocheck(__result.is, __result.js, __result.ks);
//     uint index_result_origin = __result.get_index_nocheck(__result.is, __result.js, __result.ks);
//     uint this_ni = this->ni;
//     uint result_ni = __result.ni;
//     uint this_ninj = this->ni * this->nj;
//     uint result_ninj = __result.ni * __result.nj;
//     uint stride;
//     T* stencil_coefs;
//     if ((direction == 'x') || (direction == 'X')|| (direction == 'i')) {
//         stride = 1;
//         stencil_coefs = stencil.D1_coeffs_x;
//     } else if ((direction == 'y') || (direction == 'Y')|| (direction == 'j')) {
//         stride = this_ni;
//         stencil_coefs = stencil.D1_coeffs_y;
//     } else if ((direction == 'z') || (direction == 'Z')|| (direction == 'k')) {
//         stride = this_ninj;
//         stencil_coefs = stencil.D1_coeffs_z;
//     } else {
//         assert(0 &&" ERROR:: direction should be X, Y or Z!");
//     }
 
//     #ifdef USE_OPENMP_SIMD
//     #pragma omp for schedule(static, Linalg::get_chunksize(omp_get_num_threads(), __result.nk))
//     #endif //USE_OPENMP_SIMD
//     for (uint k = 0; k < __result.nk; ++k)
//     {
//         uint result_offset_k = index_result_origin + k * result_ninj;
//         uint this_offset_k = index_this_origin + k * this_ninj;
//         uint result_offset_j = result_offset_k;
//         uint this_offset_j = this_offset_k;
//         for (uint j = 0; j < __result.nj; ++j)
//         {
//             uint result_offset_i = result_offset_j;
//             uint this_offset_i = this_offset_j;
//             for (uint i = 0; i < __result.ni; ++i)
//             {
//                 __result[result_offset_i] = other[result_offset_i] * beta;
//                 for (int p = 1; p <= stencil.FDn; ++p)
//                 {
//                     uint stride_r = p * stride;
//                     __result[result_offset_i] += (this->data[this_offset_i+stride_r] - this->data[this_offset_i-stride_r])
//                                                 * stencil_coefs[p];
//                 }
//                 ++result_offset_i;
//                 ++this_offset_i;
//             }
//             result_offset_j += result_ni;
//             this_offset_j += this_ni;
//         }
//     }
//     return __result;
// }

// template<typename T>
// Array_3D<T> Array_3D<T>::calc_laplacian(const Vertices_3D& temp, const Stencil<T>& stencil) const {
//     if (temp.get_size() == 0) return Array_3D(0);
//     assert(this->is <= temp.is - (int)stencil.FDn && this->get_ie() <= temp.get_ie() + (int)stencil.FDn &&
//            this->js <= temp.js - (int)stencil.FDn && this->get_je() <= temp.get_je() + (int)stencil.FDn &&
//            this->ks <= temp.ks - (int)stencil.FDn && this->get_ke() <= temp.get_ke() + (int)stencil.FDn);
//     Array_3D result(temp);
//     Stencil_method::calc_laplacian(this->data, this->get_vertices(), stencil, temp, result.data, temp);
//     return result;
// }

// template<typename T>
// Array_3D<T> Array_3D<T>::calc_laplacian(const Vertices_3D& temp, const Stencil<T>& stencil,
//                                         const Array_3D<T>& arr_A, const T& alpha) const {
//     if (temp.get_size() == 0) return Array_3D(0);
//     assert(this->is <= temp.is - (int)stencil.FDn && this->get_ie() <= temp.get_ie() + (int)stencil.FDn &&
//            this->js <= temp.js - (int)stencil.FDn && this->get_je() <= temp.get_je() + (int)stencil.FDn &&
//            this->ks <= temp.ks - (int)stencil.FDn && this->get_ke() <= temp.get_ke() + (int)stencil.FDn);
//     Array_3D result(temp);
//     Stencil_method::calc_laplacian(this->data, this->get_vertices(), stencil, temp, result.data, temp, arr_A.data, alpha);
//     return result;
// }

// template<typename T>
// Array_3D<T> Array_3D<T>::calc_laplacian(const Vertices_3D& temp, const Stencil<T>& stencil,
//                                         const Array_3D<T>& arr_A, const T& alpha,
//                                         const Array_3D<T>& arr_B, const T& beta) const {
//     if (temp.get_size() == 0) return Array_3D(0);
//     assert(this->is <= temp.is - (int)stencil.FDn && this->get_ie() <= temp.get_ie() + (int)stencil.FDn &&
//            this->js <= temp.js - (int)stencil.FDn && this->get_je() <= temp.get_je() + (int)stencil.FDn &&
//            this->ks <= temp.ks - (int)stencil.FDn && this->get_ke() <= temp.get_ke() + (int)stencil.FDn);
//     Array_3D result(temp);
//     Stencil_method::calc_laplacian(this->data, this->get_vertices(), stencil, temp, result.data, temp,
//                                    arr_A.data, alpha, arr_B.data, beta);
//     return result;
// }

// template<typename T>
// Array_3D<T> Array_3D<T>::calc_laplacian(const Vertices_3D& temp, const Stencil<T>& stencil,
//                                         const Array_3D<T>& arr_A, const T& alpha,
//                                         const Array_3D<T>& arr_B, const T& beta,
//                                         const T& gamma) const {
//     if (temp.get_size() == 0) return Array_3D(0);
//     assert(this->is <= temp.is - (int)stencil.FDn && this->get_ie() <= temp.get_ie() + (int)stencil.FDn &&
//            this->js <= temp.js - (int)stencil.FDn && this->get_je() <= temp.get_je() + (int)stencil.FDn &&
//            this->ks <= temp.ks - (int)stencil.FDn && this->get_ke() <= temp.get_ke() + (int)stencil.FDn);
//     Array_3D result(temp);
//     Stencil_method::calc_laplacian(this->data, this->get_vertices(), stencil, temp, result.data, temp,
//                                    arr_A.data, alpha, arr_B.data, beta, gamma);
//     return result;
// }

//Array_4D<T>
template<typename T>
Array_4D<T>::Array_4D() : Array_0D<T>::Array_0D(), Vertices_4D::Vertices_4D() {
}

template<typename T>
Array_4D<T>::Array_4D(const uint64_t length) : Array_0D<T>::Array_0D(length), Vertices_4D::Vertices_4D(length, 0, 0, 0) {
}

template<typename T>
Array_4D<T>::Array_4D(const uint64_t length, const uint) : Array_0D<T>::Array_0D(length, 0), Vertices_4D::Vertices_4D(length, 0, 0, 0) {
}

template<typename T>
Array_4D<T>::Array_4D(const Vertices_4D& other) : Array_0D<T>::Array_0D(other.get_size()), Vertices_4D::Vertices_4D(other) {
}

template<typename T>
Array_4D<T>::Array_4D(const Vertices_4D& other, const uint) : Array_0D<T>::Array_0D(other.get_size(), 0), Vertices_4D::Vertices_4D(other) {
}

template<typename T>
Array_4D<T>::Array_4D(const Array_4D& other) : Array_0D<T>::Array_0D(other), Vertices_4D::Vertices_4D(other) {
}

template<typename T>
Array_4D<T>::Array_4D(Array_4D&& other) : Array_0D<T>::Array_0D(std::move(other)), Vertices_4D::Vertices_4D(other) {
}

template<typename T>
Array_4D<T>::~Array_4D() {
    this->destructor();
}

template<typename T>
void Array_4D<T>::destructor() {
    this->Array_0D<T>::destructor();
    this->Vertices_4D::set_ni(0);
    this->Vertices_4D::set_nj(0);
    this->Vertices_4D::set_nk(0);
    this->Vertices_4D::set_nb(0);
    return;
}

template<typename T>
void Array_4D<T>::constructor(const uint64_t length) {
    this->Array_0D<T>::constructor(length);
    this->Vertices_4D::set_ni(length);
    this->Vertices_4D::set_nj(0);
    this->Vertices_4D::set_nk(0);
    this->Vertices_4D::set_nb(0);
    return;
}

template<typename T>
void Array_4D<T>::constructor(const uint64_t ni, const uint64_t nj, const uint64_t nk, const uint64_t nb) {
    this->Array_0D<T>::constructor(ni * nj * nk * nb);
    this->Vertices_4D::set_ni(ni);
    this->Vertices_4D::set_nj(nj);
    this->Vertices_4D::set_nk(nk);
    this->Vertices_4D::set_nk(nb);
    return;
}

template<typename T>
void Array_4D<T>::constructor(const Vertices_4D& other) {
    this->Array_0D<T>::constructor(other.get_size());
    this->Vertices_4D::set_vertices(other);
    return;
}

template<typename T>
void Array_4D<T>::constructor(const Vertices_4D& other, const uint) {
    this->Array_0D<T>::constructor(other.get_size(), 0);
    this->Vertices_4D::set_vertices(other);
    return;
}

template<typename T>
void Array_4D<T>::reconstructor(const uint64_t length) {
    this->destructor();
    this->constructor(length);
    return;
}

template<typename T>
void Array_4D<T>::reconstructor(const uint64_t ni, const uint64_t nj, const uint64_t nk, const uint64_t nb) {
    this->destructor();
    this->constructor(ni, nj, nk, nb);
    return;
}

template<typename T>
void Array_4D<T>::reconstructor(const Vertices_4D& other) {
    this->destructor();
    this->constructor(other);
    return;
}

template<typename T>
void Array_4D<T>::reconstructor(const Vertices_4D& other, const uint) {
    this->destructor();
    this->constructor(other, 0);
    return;
}

#ifdef USE_HBM
#ifdef USE_HUGEPAGE_HBM
template<typename T>
void Array_4D<T>::constructor_hbm(const uint64_t length) {
    this->Array_0D<T>::constructor_hbm(length);
    this->Vertices_4D::set_ni(length);
    this->Vertices_4D::set_nj(0);
    this->Vertices_4D::set_nk(0);
    this->Vertices_4D::set_nb(0);
    return;
}

template<typename T>
void Array_4D<T>::constructor_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk, const uint64_t nb) {
    this->Array_0D<T>::constructor_hbm(ni * nj * nk * nb);
    this->Vertices_4D::set_ni(ni);
    this->Vertices_4D::set_nj(nj);
    this->Vertices_4D::set_nk(nk);
    this->Vertices_4D::set_nk(nb);
    return;
}

template<typename T>
void Array_4D<T>::constructor_hbm(const Vertices_4D& other) {
    this->Array_0D<T>::constructor_hbm(other.get_size());
    this->Vertices_4D::set_vertices(other);
    return;
}

template<typename T>
void Array_4D<T>::destructor_hbm() {
    this->Array_0D<T>::destructor_hbm();
    this->Vertices_4D::set_ni(0);
    this->Vertices_4D::set_nj(0);
    this->Vertices_4D::set_nk(0);
    this->Vertices_4D::set_nb(0);
    return;
}

template<typename T>
void Array_4D<T>::reconstructor_hbm(const uint64_t length) {
    this->destructor_hbm();
    this->constructor_hbm(length);
    return;
}

template<typename T>
void Array_4D<T>::reconstructor_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk, const uint64_t nb) {
    this->destructor_hbm();
    this->constructor_hbm(ni, nj, nk, nb);
    return;
}

template<typename T>
void Array_4D<T>::reconstructor_hbm(const Vertices_4D& other) {
    this->destructor_hbm();
    this->constructor_hbm(other);
    return;
}

//hugepage
template<typename T>
void Array_4D<T>::constructor_hp_hbm(const uint64_t length) {
    this->Array_0D<T>::constructor_hp_hbm(length);
    this->Vertices_4D::set_ni(length);
    this->Vertices_4D::set_nj(0);
    this->Vertices_4D::set_nk(0);
    this->Vertices_4D::set_nb(0);
    return;
}

template<typename T>
void Array_4D<T>::constructor_hp_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk, const uint64_t nb) {
    this->Array_0D<T>::constructor_hp_hbm(ni * nj * nk * nb);
    this->Vertices_4D::set_ni(ni);
    this->Vertices_4D::set_nj(nj);
    this->Vertices_4D::set_nk(nk);
    this->Vertices_4D::set_nk(nb);
    return;
}

template<typename T>
void Array_4D<T>::constructor_hp_hbm(const Vertices_4D& other) {
    this->Array_0D<T>::constructor_hp_hbm(other.get_size());
    this->Vertices_4D::set_vertices(other);
    return;
}

template<typename T>
void Array_4D<T>::destructor_hp_hbm() {
    this->Array_0D<T>::destructor_hp_hbm();
    this->Vertices_4D::set_ni(0);
    this->Vertices_4D::set_nj(0);
    this->Vertices_4D::set_nk(0);
    this->Vertices_4D::set_nb(0);
    return;
}

template<typename T>
void Array_4D<T>::reconstructor_hp_hbm(const uint64_t length) {
    this->destructor_hp_hbm();
    this->constructor_hp_hbm(length);
    return;
}

template<typename T>
void Array_4D<T>::reconstructor_hp_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk, const uint64_t nb) {
    this->destructor_hp_hbm();
    this->constructor_hp_hbm(ni, nj, nk, nb);
    return;
}

template<typename T>
void Array_4D<T>::reconstructor_hp_hbm(const Vertices_4D& other) {
    this->destructor_hp_hbm();
    this->constructor_hp_hbm(other);
    return;
}
#else
template<typename T>
void Array_4D<T>::constructor_hbm(const uint64_t length) {
    this->Array_0D<T>::constructor_hbm(length);
    this->Vertices_4D::set_ni(length);
    this->Vertices_4D::set_nj(0);
    this->Vertices_4D::set_nk(0);
    this->Vertices_4D::set_nb(0);
    return;
}

template<typename T>
void Array_4D<T>::constructor_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk, const uint64_t nb) {
    this->Array_0D<T>::constructor_hbm(ni * nj * nk * nb);
    this->Vertices_4D::set_ni(ni);
    this->Vertices_4D::set_nj(nj);
    this->Vertices_4D::set_nk(nk);
    this->Vertices_4D::set_nk(nb);
    return;
}

template<typename T>
void Array_4D<T>::constructor_hbm(const Vertices_4D& other) {
    this->Array_0D<T>::constructor_hbm(other.get_size());
    this->Vertices_4D::set_vertices(other);
    return;
}

template<typename T>
void Array_4D<T>::destructor_hbm() {
    this->Array_0D<T>::destructor_hbm();
    this->Vertices_4D::set_ni(0);
    this->Vertices_4D::set_nj(0);
    this->Vertices_4D::set_nk(0);
    this->Vertices_4D::set_nb(0);
    return;
}

template<typename T>
void Array_4D<T>::reconstructor_hbm(const uint64_t length) {
    this->destructor_hbm();
    this->constructor_hbm(length);
    return;
}

template<typename T>
void Array_4D<T>::reconstructor_hbm(const uint64_t ni, const uint64_t nj, const uint64_t nk, const uint64_t nb) {
    this->destructor_hbm();
    this->constructor_hbm(ni, nj, nk, nb);
    return;
}

template<typename T>
void Array_4D<T>::reconstructor_hbm(const Vertices_4D& other) {
    this->destructor_hbm();
    this->constructor_hbm(other);
    return;
}
#endif
#endif

template<typename T>
void Array_4D<T>::swap(Array_4D<T>& other) {
    this->Vertices_4D::swap(other);
    this->Array_0D<T>::swap(other);
    std::swap(this->is_Col_Maj, other.is_Col_Maj);
    return;
}

template<typename T>
template<typename T2>
Array_4D<T2>
Array_4D<T>::as_type(const T2*) const {
    Array_4D<T2> result(this->get_vertices());
    Linalg::convert_type<T2, T>(result.data, this->data, this->length);
    return result;
}
template Array_4D<double> Array_4D<double>::as_type(const double*) const;
template Array_4D<float> Array_4D<float>::as_type(const float*) const;
template Array_4D<int> Array_4D<int>::as_type(const int*) const;
template Array_4D<double> Array_4D<float>::as_type(const double*) const;
template Array_4D<float> Array_4D<double>::as_type(const float*) const;
template Array_4D<int> Array_4D<float>::as_type(const int*) const;
template Array_4D<float> Array_4D<int>::as_type(const float*) const;
template Array_4D<int> Array_4D<double>::as_type(const int*) const;
template Array_4D<double> Array_4D<int>::as_type(const double*) const;
#ifdef FP16_FLAG
template Array_4D<__fp16> Array_4D<__fp16>::as_type(const __fp16*) const;
template Array_4D<__fp16> Array_4D<float>::as_type(const __fp16*) const;
template Array_4D<__fp16> Array_4D<double>::as_type(const __fp16*) const;
template Array_4D<float> Array_4D<__fp16>::as_type(const float*) const;
template Array_4D<double> Array_4D<__fp16>::as_type(const double*) const;
#endif //FP16_FLAG

template<typename T>
uint Array_4D<T>::get_leading_dimension() const {
    return this->ni * this->nj * this->nk;
}

template<typename T>
uint Array_4D<T>::get_second_dimension() const {
    return this->nb;
}

template<typename T>
Array_4D<T>& Array_4D<T>::operator=(const Array_4D<T>& other) {
    assert(this->get_vertices() == other.get_vertices());
    this->deepcopy(other);
    return *this;
}

template<typename T>
Array_4D<T>& Array_4D<T>::operator=(Array_4D<T>&& other) {
    assert(this->get_vertices() == other.get_vertices());
    this->deepcopy(std::move(other));
    return *this;
}

template<typename T>
Array_4D<T>& Array_4D<T>::operator=(const T num) {
    for (uint i = 0; i < this->length; i++) {
        this->data[i] = num;
    }
    return *this;
}

template<typename T>
Array_4D<T>& Array_4D<T>::deepcopy(const Array_4D<T>& other) {
    if (this != &other) {
        if (this->length != other.length) {
            this->reconstructor(other.get_vertices());
        } else if (this->get_vertices() != other.get_vertices()) {
            this->set_vertices(other.get_vertices());
        }
        // std::memcpy(this->data, other.data, this->length * sizeof(T));
        Linalg::set_value_general(this->data, other.data, this->length);
        this->is_Col_Maj = other.is_Col_Maj;
    }
    return *this;
}

template<typename T>
Array_4D<T>& Array_4D<T>::deepcopy(Array_4D&& other) {
    if (this->get_vertices() != other.get_vertices()) {
        this->set_vertices(other.get_vertices());
    }
    std::swap(this->length, other.length);
    std::swap(this->data, other.data);
    this->is_Col_Maj = other.is_Col_Maj;
    return *this;
}

template<typename T>
Array_4D<T>& Array_4D<T>::operator+=(const Array_4D& other) {
    assert(this->get_vertices() == other.get_vertices());
    Linalg::hadamard_plus_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_4D<T>& Array_4D<T>::operator-=(const Array_4D& other) {
    assert(this->get_vertices() == other.get_vertices());
    Linalg::hadamard_minus_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_4D<T>& Array_4D<T>::operator*=(const Array_4D& other) {
    assert(this->get_vertices() == other.get_vertices());
    Linalg::hadamard_product_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_4D<T>& Array_4D<T>::operator/=(const Array_4D& other) {
    assert(this->get_vertices() == other.get_vertices());
    Linalg::hadamard_divide_general(this->data, other.data, this->length);
    return *this;
}

template<typename T>
Array_4D<T>& Array_4D<T>::operator+=(const T num) {
    Linalg::scalar_plus_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_4D<T>& Array_4D<T>::operator-=(const T num) {
    Linalg::scalar_minus_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_4D<T>& Array_4D<T>::operator*=(const T num) {
    Linalg::scalar_product_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_4D<T>& Array_4D<T>::operator/=(const T num) {
    Linalg::scalar_divide_general(this->data, num, this->length);
    return *this;
}

template<typename T>
Array_4D<T> Array_4D<T>::operator+(const Array_4D& other) const {
    assert(this->get_vertices() == other.get_vertices());
    Array_4D<T> __result(*this);
    __result += other;
    return __result;
}

template<typename T>
Array_4D<T> Array_4D<T>::operator-(const Array_4D& other) const {
    assert(this->get_vertices() == other.get_vertices());
    Array_4D<T> __result(*this);
    __result -= other;
    return __result;
}

template<typename T>
Array_4D<T> Array_4D<T>::operator*(const Array_4D& other) const {
    assert(this->get_vertices() == other.get_vertices());
    Array_4D<T> __result(*this);
    __result *= other;
    return __result;
}

template<typename T>
Array_4D<T> Array_4D<T>::operator/(const Array_4D& other) const {
    assert(this->get_vertices() == other.get_vertices());
    Array_4D<T> __result(*this);
    __result /= other;
    return __result;
}

template<typename T>
Array_4D<T> Array_4D<T>::operator+(const T num) const {
    Array_4D<T> __result(*this);
    __result += num;
    return __result;
}

template<typename T>
Array_4D<T> Array_4D<T>::operator-(const T num) const {
    Array_4D<T> __result(*this);
    __result -= num;
    return __result;
}

template<typename T>
Array_4D<T> Array_4D<T>::operator*(const T num) const {
    Array_4D<T> __result(*this);
    __result *= num;
    return __result;
}

template<typename T>
Array_4D<T> Array_4D<T>::operator/(const T num) const {
    Array_4D<T> __result(*this);
    __result /= num;
    return __result;
}

template<typename T>
Array_4D<T> Array_4D<T>::sub_arr(const Vertices_4D& other) const {
    Array_4D<T> __result(other);
    return this->sub_arr(__result);
}

template<typename T>
Array_4D<T>& Array_4D<T>::sub_arr(Array_4D& other) const {
    const Vertices_4D& other_vertices = other.get_vertices();
    Vertices_method::fill_region(this->data, this->get_vertices(), other.data,
                                 other_vertices, other_vertices);
    return other;
}

template<typename T>
Array_4D<T>& Array_4D<T>::fill_overlap(Array_4D& other) const {
    Vertices_4D temp = this->get_overlap_vertices(other);
    return this->fill_overlap(other, temp);
}


template<typename T>
Array_4D<T>& Array_4D<T>::fill_overlap(Array_4D& other, const Vertices_4D& temp) const {
    Vertices_method::fill_region(this->data, this->get_vertices(),
                                 other.data, other.get_vertices(), temp);
    return other;
}

template<typename T>
Array_0D<T>& Array_4D<T>::fill_vector(Array_0D<T>& other, const Vertices_4D& temp, const uint offset) const {
    assert(offset + temp.get_size() <= other.get_length());
    Vertices_method::fill_vector(this->data, this->get_vertices(), other.data, temp, offset);
    return other;
}

// template<typename T>
// Array_4D<T>& Array_4D<T>::be_filled_overlap(const Array_4D& other) {
//     other.fill_overlap(*this);
//     return *this;
// }

// template<typename T>
// Array_4D<T>& Array_4D<T>::be_filled_overlap(const Array_4D& other, const Vertices_4D& temp) {
//     other.fill_overlap(*this, temp);
//     return *this;
// }


template<typename T>
Array_4D<T>& Array_4D<T>::be_filled_vector(const Array_0D<T>& other, const Vertices_4D& temp, uint offset) {
    assert(offset + temp.get_size() <= other.get_length());
    return this->be_filled_vector(other, temp, temp, offset);
}

template<typename T>
Array_4D<T>& Array_4D<T>::be_filled_vector(const Array_0D<T>& other, const Vertices_4D& s_vertices,
                                           const Vertices_4D& t_vertices, const uint offset) {
    if (t_vertices.get_size() == 0) return *this;
    Vertices_method::be_filled_vector(this->data, this->get_vertices(),
                                      other.data, s_vertices, t_vertices, offset);
    return *this;
}

template<typename T>
void Array_4D<T>::show() const {
    this->Vertices_4D::show();
    this->Array_0D<T>::show();
    return;
}

template<typename T>
void Array_4D<T>::print_h(std::ostream& output) const {
    if (this->is_Col_Maj) {
        this->print_Col_Maj_h(output);
    } else {
        this->print_Row_Maj_h(output);
    }
    return;
}

template<typename T>
void Array_4D<T>::print_Col_Maj_h(std::ostream& output) const {
    for (uint b = 0; b < this->nb; b++)
    {
        for (uint k = 0; k < this->nk; k++)
        {
            for (uint j = 0; j < this->nj; j++)
            {
                for (uint i = 0; i < this->ni; i++)
                {
                    output << std::right << std::setw(14) << std::setprecision(6)
                           << std::scientific << std::uppercase 
                           << this->data[i + (j + (k + b * this->nk) * this->nj) * this->ni];
                    if (i % 6 == 5)
                        output << std::endl;
                }
                output << std::endl;
            }
        }
    }
    return;
}

template<typename T>
void Array_4D<T>::print_Row_Maj_h(std::ostream& output) const {
    for (uint i = 0; i < this->ni; i++)
    {
        for (uint j = 0; j < this->nj; j++)
        {
            for (uint k = 0; k < this->nk; k++)
            {
                for (uint b = 0; b < this->nb; b++)
                {
                    output << std::setw(14) << std::setprecision(6) 
                           << std::scientific << std::uppercase 
                           << this->data[i + (j + (k + b * this->nk) * this->nj) * this->ni];
                    if (b % 6 == 5)
                        output << std::endl;
                }
                output << std::endl;
            }
        }
    }
    return;
}

//FD functions
// template<typename T>
// Array_4D<T> Array_4D<T>::calc_gradient(const Vertices_4D& temp, const Stencil<T>& stencil, char direction) const {
//     if (temp.get_size() == 0) return Array_4D(0);
//     assert(this->is <= temp.is - (int)stencil.FDn && this->get_ie() <= temp.get_ie() + (int)stencil.FDn &&
//            this->js <= temp.js - (int)stencil.FDn && this->get_je() <= temp.get_je() + (int)stencil.FDn &&
//            this->ks <= temp.ks - (int)stencil.FDn && this->get_ke() <= temp.get_ke() + (int)stencil.FDn &&
//            this->bs <= temp.bs - (int)stencil.FDn && this->get_be() <= temp.get_be() + (int)stencil.FDn);
//     Array_4D __result(temp);
//     uint index_this_origin = this->get_index_nocheck(__result.is, __result.js, __result.ks, __result.bs);
//     uint index_result_origin = __result.get_index_nocheck(__result.is, __result.js, __result.ks, __result.bs);
//     uint this_ni = this->ni;
//     uint result_ni = __result.ni;
//     uint this_ninj = this->ni * this->nj;
//     uint result_ninj = __result.ni * __result.nj;
//     uint this_ninjnk = this->ni * this->nj * this->nk;
//     uint result_ninjnk = __result.ni * __result.nj * __result.nk;
//     uint stride;
//     T* stencil_coefs;
//     if ((direction == 'x') || (direction == 'X')|| (direction == 'i')) {
//         stride = 1;
//         stencil_coefs = stencil.D1_coeffs_x;
//     } else if ((direction == 'y') || (direction == 'Y')|| (direction == 'j')) {
//         stride = this_ni;
//         stencil_coefs = stencil.D1_coeffs_y;
//     } else if ((direction == 'z') || (direction == 'Z')|| (direction == 'k')) {
//         stride = this_ninj;
//         stencil_coefs = stencil.D1_coeffs_z;
//     } else {
//         assert(0 &&" ERROR:: direction should be X, Y or Z!");
//     }

//     #ifdef USE_OPENMP_SIMD
//     #pragma omp for schedule(static, Linalg::get_chunksize(omp_get_num_threads(), __result.nb))
//     #endif //USE_OPENMP_SIMD
//     for (uint b = 0; b < __result.nb; ++b)
//     {
//         uint result_offset_b = b * result_ninjnk;
//         uint this_offset_b = b * this_ninjnk;
//         for (uint k = 0; k < __result.nk; ++k)
//         {
//             uint result_offset_k = k * result_ninj + result_offset_b;
//             uint this_offset_k = k * this_ninj + this_offset_b;
//             for (uint j = 0; j < __result.nj; ++j)
//             {
//                 uint result_offset_j = j * result_ni + result_offset_k;
//                 uint this_offset_j = j * result_ni + this_offset_k;
//                 for (uint i = 0; i < __result.ni; ++i)
//                 {
//                     uint index_result = index_result_origin + i + result_offset_j;
//                     uint index_this = index_this_origin + i + this_offset_j;
//                     __result[index_result] = 0.0;
//                     for (uint p = 1; p <= stencil.FDn; ++p)
//                     {
//                         uint stride_r = p * stride;
//                         __result[index_result] += (this->data[index_this+stride_r] - this->data[index_this-stride_r])
//                                                     * stencil_coefs[p];
//                     }
//                 }
//             }
//         }
//     }
//     return __result;
// }

// template<typename T>
// Array_4D<T> Array_4D<T>::calc_gradient(const Array_4D<T>& other, const Stencil<T>& stencil, char direction) const {
//     if (other.get_vertices().get_size() == 0) return Array_4D(0);
//     assert(this->is <= other.is - (int)stencil.FDn && this->get_ie() <= other.get_ie() + (int)stencil.FDn &&
//            this->js <= other.js - (int)stencil.FDn && this->get_je() <= other.get_je() + (int)stencil.FDn &&
//            this->ks <= other.ks - (int)stencil.FDn && this->get_ke() <= other.get_ke() + (int)stencil.FDn &&
//            this->bs <= other.bs - (int)stencil.FDn && this->get_be() <= other.get_be() + (int)stencil.FDn);
//     Array_4D __result(other.get_vertices());
//     uint index_this_origin = this->get_index_nocheck(__result.is, __result.js, __result.ks, __result.bs);
//     uint index_result_origin = __result.get_index_nocheck(__result.is, __result.js, __result.ks, __result.bs);
//     uint this_ni = this->ni;
//     uint result_ni = __result.ni;
//     uint this_ninj = this->ni * this->nj;
//     uint result_ninj = __result.ni * __result.nj;
//     uint this_ninjnk = this->ni * this->nj * this->nk;
//     uint result_ninjnk = __result.ni * __result.nj * __result.nk;
//     uint stride;
//     T* stencil_coefs;
//     if ((direction == 'x') || (direction == 'X')|| (direction == 'i')) {
//         stride = 1;
//         stencil_coefs = stencil.D1_coeffs_x;
//     } else if ((direction == 'y') || (direction == 'Y')|| (direction == 'j')) {
//         stride = this_ni;
//         stencil_coefs = stencil.D1_coeffs_y;
//     } else if ((direction == 'z') || (direction == 'Z')|| (direction == 'k')) {
//         stride = this_ninj;
//         stencil_coefs = stencil.D1_coeffs_z;
//     } else {
//         assert(0 &&" ERROR:: direction should be X, Y or Z!");
//     }

//     #ifdef USE_OPENMP_SIMD
//     #pragma omp for schedule(static, Linalg::get_chunksize(omp_get_num_threads(), __result.nb))
//     #endif //USE_OPENMP_SIMD
//     for (uint b = 0; b < __result.nb; ++b)
//     {
//         uint result_offset_b = b * result_ninjnk;
//         uint this_offset_b = b * this_ninjnk;
//         for (uint k = 0; k < __result.nk; ++k)
//         {
//             uint result_offset_k = k * result_ninj + result_offset_b;
//             uint this_offset_k = k * this_ninj + this_offset_b;
//             for (uint j = 0; j < __result.nj; ++j)
//             {
//                 uint result_offset_j = j * result_ni + result_offset_k;
//                 uint this_offset_j = j * result_ni + this_offset_k;
//                 for (uint i = 0; i < __result.ni; ++i)
//                 {
//                     uint index_result = index_result_origin + i + result_offset_j;
//                     uint index_this = index_this_origin + i + this_offset_j;
//                     __result[index_result] = other[index_result];
//                     for (int p = 1; p <= stencil.FDn; ++p)
//                     {
//                         uint stride_r = p * stride;
//                         __result[index_result] += (this->data[index_this+stride_r] - this->data[index_this-stride_r])
//                                                     * stencil_coefs[p];
//                     }
//                 }
//             }
//         }
//     }
//     return __result;
// }

// template<typename T>
// Array_4D<T> Array_4D<T>::calc_gradient(const Array_4D<T>& other, const Stencil<T>& stencil, char direction, const T& beta) const {
//     if (other.get_vertices().get_size() == 0) return Array_4D(0);
//     assert(this->is <= other.is - (int)stencil.FDn && this->get_ie() <= other.get_ie() + (int)stencil.FDn &&
//            this->js <= other.js - (int)stencil.FDn && this->get_je() <= other.get_je() + (int)stencil.FDn &&
//            this->ks <= other.ks - (int)stencil.FDn && this->get_ke() <= other.get_ke() + (int)stencil.FDn &&
//            this->bs <= other.bs - (int)stencil.FDn && this->get_be() <= other.get_be() + (int)stencil.FDn);
//     Array_4D __result(other.get_vertices());
//     uint index_this_origin = this->get_index_nocheck(__result.is, __result.js, __result.ks, __result.bs);
//     uint index_result_origin = __result.get_index_nocheck(__result.is, __result.js, __result.ks, __result.bs);
//     uint this_ni = this->ni;
//     uint result_ni = __result.ni;
//     uint this_ninj = this->ni * this->nj;
//     uint result_ninj = __result.ni * __result.nj;
//     uint this_ninjnk = this->ni * this->nj * this->nk;
//     uint result_ninjnk = __result.ni * __result.nj * __result.nk;
//     uint stride;
//     T* stencil_coefs;
//     if ((direction == 'x') || (direction == 'X')|| (direction == 'i')) {
//         stride = 1;
//         stencil_coefs = stencil.D1_coeffs_x.data();
//     } else if ((direction == 'y') || (direction == 'Y')|| (direction == 'j')) {
//         stride = this_ni;
//         stencil_coefs = stencil.D1_coeffs_y.data();
//     } else if ((direction == 'z') || (direction == 'Z')|| (direction == 'k')) {
//         stride = this_ninj;
//         stencil_coefs = stencil.D1_coeffs_z.data();
//     } else {
//         assert(0 &&" ERROR:: direction should be X, Y or Z!");
//     }

//     #ifdef USE_OPENMP_SIMD
//     #pragma omp for schedule(static, Linalg::get_chunksize(omp_get_num_threads(), __result.nb))
//     #endif //USE_OPENMP_SIMD
//     for (uint b = 0; b < __result.nb; ++b)
//     {
//         uint result_offset_b = b * result_ninjnk;
//         uint this_offset_b = b * this_ninjnk;
//         for (uint k = 0; k < __result.nk; ++k)
//         {
//             uint result_offset_k = k * result_ninj + result_offset_b;
//             uint this_offset_k = k * this_ninj + this_offset_b;
//             for (uint j = 0; j < __result.nj; ++j)
//             {
//                 uint result_offset_j = j * result_ni + result_offset_k;
//                 uint this_offset_j = j * result_ni + this_offset_k;
//                 for (uint i = 0; i < __result.ni; ++i)
//                 {
//                     uint index_result = index_result_origin + i + result_offset_j;
//                     uint index_this = index_this_origin + i + this_offset_j;
//                     __result[index_result] = other[index_result] * beta;
//                     for (int p = 1; p <= stencil.FDn; ++p)
//                     {
//                         uint stride_r = p * stride;
//                         __result[index_result] += (this->data[index_this+stride_r] - this->data[index_this-stride_r])
//                                                     * stencil_coefs[p];
//                     }
//                 }
//             }
//         }
//     }
//     return __result;
// }

// template<typename T>
// Array_4D<T> Array_4D<T>::calc_laplacian(const Vertices_4D& temp, const Stencil<T>& stencil) const {
//     if (temp.get_size() == 0) return Array_4D(0);
//     assert(this->is <= temp.is - (int)stencil.FDn && this->get_ie() <= temp.get_ie() + (int)stencil.FDn &&
//            this->js <= temp.js - (int)stencil.FDn && this->get_je() <= temp.get_je() + (int)stencil.FDn &&
//            this->ks <= temp.ks - (int)stencil.FDn && this->get_ke() <= temp.get_ke() + (int)stencil.FDn &&
//            this->bs <= temp.bs                    && this->get_be() <= temp.get_be());
//     Array_4D result(temp);
//     Stencil_method::calc_laplacian_d4(this->data, this->get_vertices(), stencil, temp, result.data, temp);
//     return result;
// }

// template<typename T>
// Array_4D<T> Array_4D<T>::calc_laplacian(const Vertices_4D& temp, const Stencil<T>& stencil,
//                                         const Array_4D<T>& arr_A, const T& alpha) const {
//     if (temp.get_size() == 0) return Array_4D(0);
//     assert(this->is <= temp.is - (int)stencil.FDn && this->get_ie() <= temp.get_ie() + (int)stencil.FDn &&
//            this->js <= temp.js - (int)stencil.FDn && this->get_je() <= temp.get_je() + (int)stencil.FDn &&
//            this->ks <= temp.ks - (int)stencil.FDn && this->get_ke() <= temp.get_ke() + (int)stencil.FDn &&
//            this->bs <= temp.bs                    && this->get_be() <= temp.get_be());
//     Array_4D result(temp);
//     Stencil_method::calc_laplacian_d4(this->data, this->get_vertices(), stencil, temp, result.data, temp, arr_A.data, alpha);
//     return result;
// }

// template<typename T>
// Array_4D<T> Array_4D<T>::calc_laplacian(const Vertices_4D& temp, const Stencil<T>& stencil,
//                                         const Array_4D<T>& arr_A, const T& alpha,
//                                         const Array_4D<T>& arr_B, const T& beta) const {
//     if (temp.get_size() == 0) return Array_4D(0);
//     assert(this->is <= temp.is - (int)stencil.FDn && this->get_ie() <= temp.get_ie() + (int)stencil.FDn &&
//            this->js <= temp.js - (int)stencil.FDn && this->get_je() <= temp.get_je() + (int)stencil.FDn &&
//            this->ks <= temp.ks - (int)stencil.FDn && this->get_ke() <= temp.get_ke() + (int)stencil.FDn &&
//            this->bs <= temp.bs                    && this->get_be() <= temp.get_be());
//     Array_4D result(temp);
//     Stencil_method::calc_laplacian_d4(this->data, this->get_vertices(), stencil, temp, result.data, temp,
//                                       arr_A.data, alpha, arr_B.data, beta);
//     return result;
// }

// template<typename T>
// Array_4D<T> Array_4D<T>::calc_laplacian(const Vertices_4D& temp, const Stencil<T>& stencil,
//                                         const Array_4D<T>& arr_A, const T& alpha,
//                                         const Array_4D<T>& arr_B, const T& beta,
//                                         const T& gamma) const {
//     if (temp.get_size() == 0) return Array_4D(0);
//     assert(this->is <= temp.is - (int)stencil.FDn && this->get_ie() <= temp.get_ie() + (int)stencil.FDn &&
//            this->js <= temp.js - (int)stencil.FDn && this->get_je() <= temp.get_je() + (int)stencil.FDn &&
//            this->ks <= temp.ks - (int)stencil.FDn && this->get_ke() <= temp.get_ke() + (int)stencil.FDn &&
//            this->bs <= temp.bs                    && this->get_be() <= temp.get_be());
//     Array_4D result(temp);
//     Stencil_method::calc_laplacian_d4(this->data, this->get_vertices(), stencil, temp, result.data, temp,
//                                       arr_A.data, alpha, arr_B.data, beta, gamma);
//     return result;
// }

//why these lines 
//https://isocpp.org/wiki/faq/templates#templates-defn-vs-decl
template class Array_0D<float>;
template class Array_0D<double>;
template class Array_0D<int>;
template class Array_1D<float>;
template class Array_1D<double>;
template class Array_1D<int>;
template class Array_2D<float>;
template class Array_2D<double>;
template class Array_2D<int>;
template class Array_3D<float>;
template class Array_3D<double>;
template class Array_3D<int>;
template class Array_4D<float>;
template class Array_4D<double>;
template class Array_4D<int>;
#ifdef FP16_FLAG
template class Array_0D<__fp16>;
template class Array_1D<__fp16>;
template class Array_2D<__fp16>;
template class Array_3D<__fp16>;
template class Array_4D<__fp16>;
#endif //FP16_FLAG