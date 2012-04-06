#pragma once
#include <climits>
#include <cfloat>

namespace copperhead {

namespace detail {

//Can't use std::numeric_limits because it doesn't
//work on the GPU.
template<typename T> struct numeric_limits {};

template<>
struct numeric_limits<bool> {
    __host__ __device__
    static bool min() {
        return false;
    }
    __host__ __device__
    static bool max() {
        return true;
    }
};

template<>
struct numeric_limits<int> {
    __host__ __device__
    static int min() {
        return INT_MIN;
    }
    __host__ __device__
    static int max() {
        return INT_MAX;
    }
};

template<>
struct numeric_limits<long> {
    __host__ __device__
    static long min() {
        return LONG_MIN;
    }
    __host__ __device__
    static long max() {
        return LONG_MAX;
    }
};

template<>
struct numeric_limits<float> {
    __host__ __device__
    static float min() {
        return FLT_MIN;
    }
    __host__ __device__
    static float max() {
        return FLT_MAX;
    }
};

template<>
struct numeric_limits<double> {
    __host__ __device__
    static double min() {
        return DBL_MIN;
    }
    __host__ __device__
    static double max() {
        return DBL_MAX;
    }
};

}

template<typename T>
__host__ __device__ T min_bound(const T&) {
    return detail::numeric_limits<T>::min();
}

template<typename T>
struct fnmin_bound {
    typedef T result_type;
    __host__ __device__ T operator()(const T& t) {
        return min_bound(t);
    }
};

template<typename T>
__host__ __device__ T max_bound(const T&) {
    return detail::numeric_limits<T>::max();
}


template<typename T>
struct fnmax_bound {
    typedef T result_type;
    __host__ __device__ T operator()(const T& t) {
        return max_bound(t);
    }
};

}
