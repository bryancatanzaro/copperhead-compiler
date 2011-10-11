#pragma once
#include "../cudata/cudata.h"
#include "convert.hpp"

#include <thrust/transform.h>
#include <thrust/detail/type_traits.h>
#include "transformed_sequence.h"
#include "zipped_sequence.h"

template<typename F,
         typename Seq0>
sp_cuarray_var map(const F& fn,
                   Seq0& x0) {
    typedef typename F::result_type T;
    sp_cuarray_var result_ary = make_remote<T>(x0.size());
    stored_sequence<T> result = get_remote_w<T>(result_ary);
    thrust::transform(x0.begin(),
                      x0.end(),
                      result.begin(),
                      fn);
    return result_ary;
}

template<typename F,
         typename Seq0>
transformed_sequence<F, Seq0> map_lazy(const F& fn,
                                        Seq0& x0) {
    return transformed_sequence<F, Seq0>(fn, x0);
}

template<typename F,
         typename Seq0>
sp_cuarray_var map(const F& fn,
                   Seq0& x0,
                   Seq0& x1) {
    typedef typename F::result_type T;
    sp_cuarray_var result_ary = make_remote<T>(x0.size());
    stored_sequence<T> result = get_remote_w<T>(result_ary);
    thrust::transform(x0.begin(),
                      x0.end(),
                      x1.begin(),
                      result.begin(),
                      fn);
    return result_ary;
}



template<typename F>
struct map_adapter {
    F m_fn;
    typedef typename F::result_type T;
    
    __host__ __device__ map_adapter(const F& fn) : m_fn(fn) {}
    template<typename T0,
             typename T1,
             typename T2>
    __host__ __device__
    T operator()(thrust::tuple<T0, T1, T2> in) {
        return m_fn(thrust::get<0>(in),
                    thrust::get<1>(in),
                    thrust::get<2>(in));
    }
    template<typename T0,
             typename T1,
             typename T2,
             typename T3>
    __host__ __device__
    T operator()(thrust::tuple<T0, T1, T2, T3> in) {
        return m_fn(thrust::get<0>(in),
                    thrust::get<1>(in),
                    thrust::get<2>(in),
                    thrust::get<3>(in));
    }
};

template<typename F>
map_adapter<F> make_map_adapter(const F& in) {
   return map_adapter<F>(in);
}

template<typename F,
         typename Seq0,
         typename Seq1,
         typename Seq2>
sp_cuarray_var map(const F& fn,
                   Seq0& x0,
                   Seq1& x1,
                   Seq2& x2) {
    typedef typename F::result_type T;
    sp_cuarray_var result_ary = make_remote<T>(x0.size());
    stored_sequence<T> result = get_remote_w<T>(result_ary);
    
    thrust::transform(thrust::make_zip_iterator(
                          thrust::make_tuple(
                              x0.begin(),
                              x1.begin(),
                              x2.begin())),
                      thrust::make_zip_iterator(
                          thrust::make_tuple(
                              x0.end(),
                              x1.end(),
                              x2.end())),
                      result.begin(),
                      make_map_adapter(fn));
    return result_ary;
}

template<typename F,
         typename Seq0,
         typename Seq1,
         typename Seq2,
         typename Seq3>
sp_cuarray_var map(const F& fn,
                   Seq0& x0,
                   Seq1& x1,
                   Seq2& x2,
                   Seq3& x3) {
    typedef typename F::result_type T;
    sp_cuarray_var result_ary = make_remote<T>(x0.size());
    stored_sequence<T> result = get_remote_w<T>(result_ary);
   
    thrust::transform(thrust::make_zip_iterator(
                          thrust::make_tuple(
                              x0.begin(),
                              x1.begin(),
                              x2.begin(),
                              x3.begin())),
                      thrust::make_zip_iterator(
                          thrust::make_tuple(
                              x0.end(),
                              x1.end(),
                              x2.end(),
                              x3.end())),
                      result.begin(),
                      make_map_adapter(fn));
    return result_ary;
}
