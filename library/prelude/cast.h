#pragma once

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

template<typename T, typename U>
class fncast_to {
    typedef U return_type;
    U operator()(const T& x, const U& y) {
        return cast_to(x, y);
    }
};

template<typename T, typename U>
class fncast_to_el {
    typedef typename U::value_type return_type;
    U operator()(const T& x, const U& y) {
        return cast_to_el(x, y);
    }
};
