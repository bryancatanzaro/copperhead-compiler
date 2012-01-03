#pragma once

template<typename T>
__host__ __device__
int op_int32(const T&x) {
    return int(x);
}

template<typename T>
__host__ __device__
int op_int64(const T&x) {
    return long(x);
}

template<typename T>
__host__ __device__
int op_float32(const T&x) {
    return float(x);
}

template<typename T>
__host__ __device__
int op_float64(const T&x) {
    return double(x);
}

template<typename T, typename U>
__host__ __device__
U cast_to(const T& x, const U& y) {
    return U(x);
}

template<typename T, typename U>
__host__ __device__
typename U::value_type cast_to_el(const T& x, const U& y) {
    return typename U::value_type(x);
}


template<typename T>
class fnop_int32 {
    typedef int return_type;
    __host__ __device__
    int operator()(const T& x) {
        return op_int32(x);
    }
};


template<typename T>
class fnop_int64 {
    typedef long return_type;
    __host__ __device__
    long operator()(const T& x) {
        return op_int64(x);
    }
};

template<typename T>
class fnop_float32 {
    typedef float return_type;
    __host__ __device__
    float operator()(const T& x) {
        return op_int32(x);
    }
};

template<typename T>
class fnop_float64 {
    typedef double return_type;
    __host__ __device__
    double operator()(const T& x) {
        return op_int32(x);
    }
};

template<typename T, typename U>
class fncast_to {
    typedef U return_type;
    __host__ __device__
    U operator()(const T& x, const U& y) {
        return cast_to(x, y);
    }
};

template<typename T, typename U>
class fncast_to_el {
    typedef typename U::value_type return_type;
    __host__ __device__
    U operator()(const T& x, const U& y) {
        return cast_to_el(x, y);
    }
};
