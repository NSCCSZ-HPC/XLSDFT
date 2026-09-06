#pragma once

#include <cstddef>
#include <cassert>
#include <new>
#include <cstdint>
#include <iostream>
#include <unistd.h>

#ifdef USE_HBM
#include <hbwmalloc.h>
#endif

#ifdef MEMORY_ALIGNMENT
#define MEMORY_ALIGN_BYTES MEMORY_ALIGNMENT
#else
#define MEMORY_ALIGN_BYTES 64
#endif

enum class Memory_location
{
    DDR,
#ifdef USE_HBM
    HBM
#endif
};

// fast memory
#ifdef USE_HBM
    constexpr Memory_location Fast_memory = Memory_location::HBM;
#else
    constexpr Memory_location Fast_memory = Memory_location::DDR;
#endif

// capacity memory
constexpr Memory_location Capacity_memory = Memory_location::DDR;

template<typename T, Memory_location location>
class Memory_pool
{
public:
    using value_type = T;
private:
    T* buffer = nullptr;
    size_t capacity = 0;
    size_t offset = 0;
public:
    Memory_pool() = default;
    explicit Memory_pool(size_t n) {
        allocate_memory(n);
    }
    ~Memory_pool() {
        release_memory();
    }
    Memory_pool(const Memory_pool&) = delete;
    Memory_pool& operator=(const Memory_pool&) = delete;
public:
    void allocate_memory(size_t n)
    {
        assert(buffer == nullptr);
        capacity = n;
        offset = 0;
        if constexpr(location == Memory_location::DDR)
        {
            buffer =
                static_cast<T*>(
                    ::operator new[](
                        sizeof(T)*n,
                        std::align_val_t(MEMORY_ALIGN_BYTES)
                    )
                );
        }
#ifdef USE_HBM
        else if constexpr(location == Memory_location::HBM)
        {
            buffer =
                static_cast<T*>(
                    hbw_malloc(sizeof(T)*n)
                );
        }
#endif
        if(buffer == nullptr) {
            // throw std::bad_alloc();
            char hostname[256];
            gethostname(hostname, sizeof(hostname));
            const char* memory_location = "DDR";
        #ifdef USE_HBM
            if constexpr(location == Memory_location::HBM) {
                memory_location = "HBM";
            }
        #endif
            const size_t bytes = n * sizeof(T);
            const double gib =
                static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
            std::fprintf(stderr,
                        "Memory allocation failed on node %s: "
                        "location = %s, "
                        "requested = %zu elements, %zu bytes (%.3f GiB)\n",
                        hostname,
                        memory_location,
                        n,
                        bytes,
                        gib);
            assert(buffer != nullptr);
        }
    }

    void release_memory()
    {
        if(buffer == nullptr)
            return;
        if constexpr(location == Memory_location::DDR)
        {
            ::operator delete[](
                buffer,
                std::align_val_t(MEMORY_ALIGN_BYTES)
            );
        }
#ifdef USE_HBM
        else if constexpr(location == Memory_location::HBM)
        {
            hbw_free(buffer);
        }
#endif
        buffer = nullptr;
        capacity = 0;
        offset = 0;
    }
    T* allocate(size_t n) {
        assert(buffer != nullptr);
        // assert(n >= 0);
        if (n == 0) return nullptr;
        uintptr_t current =
            reinterpret_cast<uintptr_t>(buffer + offset);
        uintptr_t aligned =
            (current + MEMORY_ALIGN_BYTES - 1)
            &
            ~(MEMORY_ALIGN_BYTES - 1);
        size_t aligned_offset =
            (aligned -
            reinterpret_cast<uintptr_t>(buffer))
            /
            sizeof(T);
        assert(aligned_offset + n <= capacity);
        T* ptr =
            reinterpret_cast<T*>(aligned);
        offset = aligned_offset + n;
        return ptr;
    }
    size_t mark() const
    {
        return offset;
    }
    void release(size_t old_offset)
    {
        assert(old_offset <= offset);
        offset = old_offset;
    }
    void reset()
    {
        offset = 0;
    }
    size_t size() const
    {
        return capacity;
    }
    size_t used() const
    {
        return offset;
    }
    size_t available() const
    {
        return capacity-offset;
    }
    void show(std::ostream &out = std::cout) const {
        out << "Memory pool information:\n";
        out << "  value type      : "
            << typeid(T).name()
            << "\n";
        out << "  location        : ";
        if constexpr(location == Memory_location::DDR)
        {
            out << "DDR\n";
        }
    #ifdef USE_HBM
        else if constexpr(location == Memory_location::HBM)
        {
            out << "HBM\n";
        }
    #endif
        out << "  buffer          : "
            << static_cast<const void*>(buffer)
            << "\n";
        out << "  capacity        : "
            << capacity
            << " elements\n";
        out << "  used            : "
            << offset
            << " elements\n";
        out << "  available       : "
            << capacity - offset
            << " elements\n";
        out << "  total memory    : "
            << static_cast<double>(capacity * sizeof(T))
            / 1024.0 / 1024.0
            << " MB\n";
        out << "  used memory     : "
            << static_cast<double>(offset * sizeof(T))
            / 1024.0 / 1024.0
            << " MB\n";
        out << "  alignment       : "
            << MEMORY_ALIGN_BYTES
            << " bytes\n";
        out << "  utilization     : ";
        if(capacity != 0) {
            out << static_cast<double>(offset)
                / static_cast<double>(capacity)
                * 100.0;
        } else {
            out << 0.0;
        }
        out << " %\n";
    }
};

template<typename Pool>
class Memory_pool_scope
{
private:
    Pool& pool;
    size_t offset;
public:
    explicit Memory_pool_scope(Pool& p)
        :
        pool(p),
        offset(p.mark())
    {}
    ~Memory_pool_scope()
    {
        pool.release(offset);
    }
    Memory_pool_scope(const Memory_pool_scope&) = delete;
    Memory_pool_scope& operator=(const Memory_pool_scope&) = delete;
};
