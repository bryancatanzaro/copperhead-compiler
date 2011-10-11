#pragma once
#include "../cudata/cudata.h"
#include "convert.hpp"

#include <thrust/transform.h>
#include <thrust/detail/type_traits.h>
#include "transformed_sequence.h"
#include "zipped_sequence.h"

template<typename F>
struct map_adapter {
    F m_fn;
    typedef typename F::result_type T;
    __host__ __device__ map_adapter(const F& fn) : m_fn(fn) {}

    template<typename T0>
    __host__ __device__
    T operator()(thrust::tuple<T0> in) {
        return m_fn(thrust::get<0>(in));
    }
    
    template<typename T0,
             typename T1>
    __host__ __device__
    T operator()(thrust::tuple<T0, T1> in) {
        return m_fn(thrust::get<0>(in),
                    thrust::get<1>(in));
    }
    
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

    template<typename T0,
             typename T1,
             typename T2,
             typename T3,
             typename T4>
    __host__ __device__
    T operator()(thrust::tuple<T0, T1, T2, T3, T4> in) {
        return m_fn(thrust::get<0>(in),
                    thrust::get<1>(in),
                    thrust::get<2>(in),
                    thrust::get<3>(in),
                    thrust::get<4>(in));
    }

    template<typename T0,
             typename T1,
             typename T2,
             typename T3,
             typename T4,
             typename T5>
    __host__ __device__
    T operator()(thrust::tuple<T0, T1, T2, T3, T4,
                               T5> in) {
        return m_fn(thrust::get<0>(in),
                    thrust::get<1>(in),
                    thrust::get<2>(in),
                    thrust::get<3>(in),
                    thrust::get<4>(in),
                    thrust::get<5>(in));
    }

    template<typename T0,
             typename T1,
             typename T2,
             typename T3,
             typename T4,
             typename T5,
             typename T6>
    __host__ __device__
    T operator()(thrust::tuple<T0, T1, T2, T3, T4,
                               T5, T6> in) {
        return m_fn(thrust::get<0>(in),
                    thrust::get<1>(in),
                    thrust::get<2>(in),
                    thrust::get<3>(in),
                    thrust::get<4>(in),
                    thrust::get<5>(in),
                    thrust::get<6>(in));
    }

    template<typename T0,
             typename T1,
             typename T2,
             typename T3,
             typename T4,
             typename T5,
             typename T6,
             typename T7>
    __host__ __device__
    T operator()(thrust::tuple<T0, T1, T2, T3, T4,
                               T5, T6, T7> in) {
        return m_fn(thrust::get<0>(in),
                    thrust::get<1>(in),
                    thrust::get<2>(in),
                    thrust::get<3>(in),
                    thrust::get<4>(in),
                    thrust::get<5>(in),
                    thrust::get<6>(in),
                    thrust::get<7>(in));
    }
    
    template<typename T0,
             typename T1,
             typename T2,
             typename T3,
             typename T4,
             typename T5,
             typename T6,
             typename T7,
             typename T8>
    __host__ __device__
    T operator()(thrust::tuple<T0, T1, T2, T3, T4,
                               T5, T6, T7, T8> in) {
        return m_fn(thrust::get<0>(in),
                    thrust::get<1>(in),
                    thrust::get<2>(in),
                    thrust::get<3>(in),
                    thrust::get<4>(in),
                    thrust::get<5>(in),
                    thrust::get<6>(in),
                    thrust::get<7>(in),
                    thrust::get<8>(in));
    }
    
    template<typename T0,
             typename T1,
             typename T2,
             typename T3,
             typename T4,
             typename T5,
             typename T6,
             typename T7,
             typename T8,
             typename T9>
    __host__ __device__
    T operator()(thrust::tuple<T0, T1, T2, T3, T4,
                               T5, T6, T7, T8, T9> in) {
        return m_fn(thrust::get<0>(in),
                    thrust::get<1>(in),
                    thrust::get<2>(in),
                    thrust::get<3>(in),
                    thrust::get<4>(in),
                    thrust::get<5>(in),
                    thrust::get<6>(in),
                    thrust::get<7>(in),
                    thrust::get<8>(in),
                    thrust::get<9>(in));
    }
};

template<typename F>
map_adapter<F> make_map_adapter(const F& in) {
   return map_adapter<F>(in);
}

template<typename F,
         typename Seq0>
transformed_sequence<F, Seq0> map_lazy(const F& fn,
                                        Seq0& x0) {
    return transformed_sequence<F, Seq0>(fn, x0);
}

template<typename F,
         typename Seq0>
transformed_sequence<map_adapter<F>, zipped_sequence<thrust::tuple<Seq0> > >
map(const F& fn,
    Seq0& x0) {
    return transformed_sequence<map_adapter<F>, zipped_sequence<thrust::tuple<Seq0> > >(make_map_adapter(fn), make_zipped_sequence(x0));
}

template<typename F,
         typename Seq0,
         typename Seq1>
transformed_sequence<map_adapter<F>, zipped_sequence<thrust::tuple<Seq0, Seq1> > >
map(const F& fn,
    Seq0& x0,
    Seq1& x1) {
    return transformed_sequence<map_adapter<F>, zipped_sequence<thrust::tuple<Seq0, Seq1> > >(make_map_adapter(fn), make_zipped_sequence(x0, x1));
}

template<typename F,
         typename Seq0,
         typename Seq1,
         typename Seq2>
transformed_sequence<map_adapter<F>, zipped_sequence<thrust::tuple<Seq0, Seq1, Seq2> > >
map(const F& fn,
    Seq0& x0,
    Seq1& x1,
    Seq2& x2) {
    return transformed_sequence<map_adapter<F>, zipped_sequence<thrust::tuple<Seq0, Seq1, Seq2> > >(make_map_adapter(fn), make_zipped_sequence(x0, x1, x2));
}

template<typename F,
         typename Seq0,
         typename Seq1,
         typename Seq2,
         typename Seq3>
transformed_sequence<map_adapter<F>, zipped_sequence<thrust::tuple<Seq0, Seq1, Seq2, Seq3> > >
map(const F& fn,
    Seq0& x0,
    Seq1& x1,
    Seq2& x2,
    Seq3& x3) {
    return transformed_sequence<map_adapter<F>, zipped_sequence<thrust::tuple<Seq0, Seq1, Seq2, Seq3> > >(make_map_adapter(fn), make_zipped_sequence(x0, x1, x2, x3));
}
