#pragma once

#include <thrust/tuple.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/static_assert.h>

namespace thrust {
namespace detail {

template<int i, typename T, bool enable>
struct tuple_element_enable{
    typedef thrust::null_type type;
};

template<int i, typename T>
struct tuple_element_enable<i, T, true> {
    typedef typename thrust::tuple_element<i, T>::type type;
};

template<typename T0, typename T1, int i>
struct tuple_cat_type_i {
    static const int T0L = thrust::tuple_size<T0>::value;
    static const int T1L = thrust::tuple_size<T1>::value;
    static const bool in_T0 = i < T0L;
    static const bool in_T1 = (!in_T0) && (i-T0L < T1L);
    typedef typename eval_if<in_T0,
                             tuple_element_enable<i, T0, in_T0>,
                             tuple_element_enable<i-T0L, T1, in_T1> >::type type;
    
};
} //end namespace thrust::detail

template<typename T0, typename T1>
struct tuple_cat_type {
    // ========================================================================
    // X Note to the user: If you've found this line due to a compiler error, X
    // X it's because the concatenated tuple type is too long.                X
    // X Thrust tuples can have 10 elements, maximum.                         X
    // ========================================================================
    THRUST_STATIC_ASSERT(thrust::tuple_size<T0>::value + thrust::tuple_size<T1>::value <= 10);

    typedef thrust::tuple<
        typename detail::tuple_cat_type_i<T0, T1, 0>::type,
        typename detail::tuple_cat_type_i<T0, T1, 1>::type,
        typename detail::tuple_cat_type_i<T0, T1, 2>::type,
        typename detail::tuple_cat_type_i<T0, T1, 3>::type,
        typename detail::tuple_cat_type_i<T0, T1, 4>::type,
        typename detail::tuple_cat_type_i<T0, T1, 5>::type,
        typename detail::tuple_cat_type_i<T0, T1, 6>::type,
        typename detail::tuple_cat_type_i<T0, T1, 7>::type,
        typename detail::tuple_cat_type_i<T0, T1, 8>::type,
        typename detail::tuple_cat_type_i<T0, T1, 9>::type> type;
};


namespace detail {

template<typename T0, typename T1, int i, bool select>
struct tuple_element_select {
    __host__ __device__
    static typename thrust::tuple_element<i, T0>::type fun(const T0& t0,
                                                           const T1& t1) {
        return thrust::get<i>(t0);
    }
};

template<typename T0, typename T1, int i>
struct tuple_element_select<T0, T1, i, false> {
    static const int new_i = i - thrust::tuple_size<T0>::value;
    __host__ __device__
    static typename thrust::tuple_element<new_i, T1>::type fun(const T0& t0,
                                                               const T1& t1) {
        return thrust::get<new_i>(t1);
    }
};

template<typename T0, typename T1, typename S, int i>
struct tuple_cat_impl{};

template<typename T0, typename T1, typename HT, typename TT, int i>
struct tuple_cat_impl<T0, T1, thrust::detail::cons<HT, TT>, i> {
    static const int T0L = thrust::tuple_size<T0>::value;
    static const bool in_T0 = i < T0L;

    __host__ __device__
    static thrust::detail::cons<HT, TT> fun(const T0& t0,
                                            const T1& t1) {
        return thrust::detail::cons<HT, TT>(
            tuple_element_select<T0, T1, i, in_T0>::fun(t0, t1),
            tuple_cat_impl<T0, T1,
                           TT, i+1>::fun(t0, t1));
    }
};

template<typename T0, typename T1, int i>
struct tuple_cat_impl<T0, T1, thrust::null_type, i> {
    __host__ __device__
    static thrust::null_type fun(const T0& t0,
                                 const T1& t1) {
        return thrust::null_type();
    }
};


} //end namespace thrust::detail

template<typename T0, typename T1>
__host__ __device__
typename tuple_cat_type<T0, T1>::type tuple_cat(const T0& t0, const T1& t1) {
    typedef typename tuple_cat_type<T0, T1>::type result_type;
    return detail::tuple_cat_impl<T0, T1,
                                  thrust::detail::cons<typename result_type::head_type,
                                                       typename result_type::tail_type>,
                                  0>::fun(t0, t1);
        
}

//Overload for special case of two empty tuples
__host__ __device__
thrust::tuple<> tuple_cat(const thrust::tuple<>& t0, const thrust::tuple<>& t1) {
    return thrust::tuple<>();
}

} //end namespace thrust
