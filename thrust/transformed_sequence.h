#pragma once
#include <thrust/iterator/transform_iterator.h>
#include "zipped_sequence.h"


template<typename F>
struct map_adapter {
    F m_fn;
    typedef typename F::result_type result_type;
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

template<typename F,
         typename S>
struct transformed_sequence {
    map_adapter<F> m_fn;
    zipped_sequence<S> m_seq;
    typedef typename map_adapter<F>::result_type value_type;
    typedef typename zipped_sequence<S>::iterator_type I;
    typedef typename thrust::transform_iterator<map_adapter<F>, I> iterator_type;
    transformed_sequence(F fn,
                         S seqs)
        : m_fn(map_adapter<F>(fn)), m_seq(seqs) {}
    value_type& operator[](int index) {
        return m_fn(m_seq[index]);
    }
    iterator_type begin() const {
        return iterator_type(m_seq.begin(), m_fn);
    }
    iterator_type end() const {
        return iterator_type(m_seq.end(), m_fn);
    }
    int size() const {
        return m_seq.size();
    }
};
