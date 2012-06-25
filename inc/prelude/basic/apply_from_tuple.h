#pragma once

#include <thrust/tuple.h>

namespace copperhead {
namespace detail {

template<typename F>
__host__ __device__
typename F::result_type apply_from_tuple(F f, const thrust::tuple<>&) {
    return f();
}

template<typename F, typename A0>
__host__ __device__
typename F::result_type apply_from_tuple(F f,
                                         const thrust::tuple<A0>& a) {
    return f(thrust::get<0>(a));
}

template<typename F, typename A0, typename A1>
__host__ __device__
typename F::result_type apply_from_tuple(F f,
                                         const thrust::tuple<A0, A1>& a) {
    return f(
        thrust::get<0>(a),
        thrust::get<1>(a));
}

template<typename F, typename A0, typename A1, typename A2>
__host__ __device__
typename F::result_type apply_from_tuple(F f,
                                         const thrust::tuple<A0, A1, A2>& a) {
    return f(
        thrust::get<0>(a),
        thrust::get<1>(a),
        thrust::get<2>(a));
}

template<typename F, typename A0, typename A1, typename A2, typename A3>
__host__ __device__
typename F::result_type apply_from_tuple(F f,
                                         const thrust::tuple<A0, A1, A2, A3>& a) {
    return f(
        thrust::get<0>(a),
        thrust::get<1>(a),
        thrust::get<2>(a),
        thrust::get<3>(a));
}

template<typename F,
    typename A0, typename A1, typename A2, typename A3, typename A4>
__host__ __device__
typename F::result_type apply_from_tuple(F f,
                                         const thrust::tuple<A0, A1, A2, A3, A4>& a) {
    return f(
        thrust::get<0>(a),
        thrust::get<1>(a),
        thrust::get<2>(a),
        thrust::get<3>(a),
        thrust::get<4>(a));
}

template<typename F,
    typename A0, typename A1, typename A2, typename A3, typename A4,
    typename A5>
__host__ __device__
typename F::result_type apply_from_tuple(F f,
                                         const thrust::tuple<
                                         A0, A1, A2, A3, A4,
                                         A5>& a) {
    return f(
        thrust::get<0>(a),
        thrust::get<1>(a),
        thrust::get<2>(a),
        thrust::get<3>(a),
        thrust::get<4>(a),
        thrust::get<5>(a));
}

template<typename F,
    typename A0, typename A1, typename A2, typename A3, typename A4,
    typename A5, typename A6>
__host__ __device__
typename F::result_type apply_from_tuple(F f,
                                         const thrust::tuple<
                                         A0, A1, A2, A3, A4,
                                         A5, A6>& a) {
    return f(
        thrust::get<0>(a),
        thrust::get<1>(a),
        thrust::get<2>(a),
        thrust::get<3>(a),
        thrust::get<4>(a),
        thrust::get<5>(a),
        thrust::get<6>(a));
}

template<typename F,
    typename A0, typename A1, typename A2, typename A3, typename A4,
    typename A5, typename A6, typename A7>
__host__ __device__
typename F::result_type apply_from_tuple(F f,
                                         const thrust::tuple<
                                         A0, A1, A2, A3, A4,
                                         A5, A6, A7>& a) {
    return f(
        thrust::get<0>(a),
        thrust::get<1>(a),
        thrust::get<2>(a),
        thrust::get<3>(a),
        thrust::get<4>(a),
        thrust::get<5>(a),
        thrust::get<6>(a),
        thrust::get<7>(a));
}

template<typename F,
    typename A0, typename A1, typename A2, typename A3, typename A4,
    typename A5, typename A6, typename A7, typename A8>
__host__ __device__
typename F::result_type apply_from_tuple(F f,
                                         const thrust::tuple<
                                         A0, A1, A2, A3, A4,
                                         A5, A6, A7, A8>& a) {
    return f(
        thrust::get<0>(a),
        thrust::get<1>(a),
        thrust::get<2>(a),
        thrust::get<3>(a),
        thrust::get<4>(a),
        thrust::get<5>(a),
        thrust::get<6>(a),
        thrust::get<7>(a),
        thrust::get<8>(a));
}

template<typename F,
    typename A0, typename A1, typename A2, typename A3, typename A4,
    typename A5, typename A6, typename A7, typename A8, typename A9>
__host__ __device__
typename F::result_type apply_from_tuple(F f,
                                         const thrust::tuple<
                                         A0, A1, A2, A3, A4,
                                         A5, A6, A7, A8, A9>& a) {
    return f(
        thrust::get<0>(a),
        thrust::get<1>(a),
        thrust::get<2>(a),
        thrust::get<3>(a),
        thrust::get<4>(a),
        thrust::get<5>(a),
        thrust::get<6>(a),
        thrust::get<7>(a),
        thrust::get<8>(a),
        thrust::get<9>(a));
}

}
}
