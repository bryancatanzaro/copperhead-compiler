#pragma once
#ifndef CUDA_ARCH
#include <cmath>
template<typename T>
T abs(const T& x) {
    return std::abs(x);
}
#endif


template<typename T>
struct fn_abs {
    typedef T result_type;
    __host__ __device__
    T operator()(const T& x) const {
        return abs(x);
    }
};

template<typename T>
struct fn_exp {
    typedef T result_type;
    __host__ __device__
    T operator()(const T& x) const {
        return exp(x);
    }
};

template<typename T>
struct fn_log {
    typedef T result_type;
    __host__ __device__
    T operator()(const T& x) const {
        return log(x);
    }
};

template<typename T>
struct fn_sqrt {
    typedef T result_type;
    __host__ __device__
    T operator()(const T& x) const {
        return sqrt(x);
    }
};

