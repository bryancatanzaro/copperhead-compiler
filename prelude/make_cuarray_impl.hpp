#include "make_cuarray.hpp"
#include "cuarray.hpp"
#include "monotype.hpp"

template<typename T>
struct type_deriver {};


template<>
struct type_deriver<float> {
    static std::shared_ptr<backend::type_t> fun() {
        return backend::float32_mt;
    }
};

template<>
struct type_deriver<double> {
    static std::shared_ptr<backend::type_t> fun() {
        return backend::float64_mt;
    }
};

template<>
struct type_deriver<int> {
    static std::shared_ptr<backend::type_t> fun() {
        return backend::int32_mt;
    }
};

template<>
struct type_deriver<long> {
    static std::shared_ptr<backend::type_t> fun() {
        return backend::int64_mt;
    }
};

template<>
struct type_deriver<bool> {
    static std::shared_ptr<backend::type_t> fun() {
        return backend::bool_mt;
    }
};

template<typename T>
sp_cuarray make_cuarray(size_t s) {
    sp_cuarray r(new cuarray());
    r->m_t = std::make_shared<backend::sequence_t>(type_deriver<T>::fun());
    r->m_l.push_back(s);
    r->m_local.push_back(std::make_shared<chunk<host_alloc> >(host_alloc(), s * sizeof(T)));
#ifdef CUDA_SUPPORT
    r->m_remote.push_back(std::make_shared<chunk<cuda_alloc> >(cuda_alloc(), s * sizeof(T)));
    r->m_clean_local = true;
    r->m_clean_remote = true;
#endif
    return r;
}
