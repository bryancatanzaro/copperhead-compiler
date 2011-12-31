#pragma once
#include "iterator_sequence.h"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>


template<typename Seq>
class shift_functor {
protected:
    const Seq m_data;
    typedef typename Seq::index_type I;
    const I m_shift;
    typedef typename Seq::value_type T;
    const T m_boundary;
public:
    typedef T result_type;
    __host__ __device__
    shift_functor(const Seq& data,
                  const I& shift,
                  const T& boundary)
        : m_data(data), m_shift(shift), m_boundary(boundary) {}
    
    __host__ __device__    
    T operator()(const I& i) const {
        I new_pos = i + m_shift;
        if ((new_pos < 0) ||
            (new_pos >= m_data.size())) {
            return m_boundary;
        }
        return m_data[new_pos];
    }
};


template<typename Seq>
struct shift_iterator_type {
    typedef typename thrust::transform_iterator<shift_functor<Seq>, thrust::counting_iterator<typename Seq::index_type> > type;
};


template<typename Seq>
__host__ __device__
typename shift_iterator_type<Seq>::type
make_shift_iterator(const Seq& in,
                    const typename Seq::index_type& shift,
                    const typename Seq::value_type& boundary) {
    return typename shift_iterator_type<Seq>::type(
        thrust::counting_iterator<typename Seq::index_type>(0),
        shift_functor<Seq>(in, shift, boundary));
}

template<typename Seq>
struct shifted_sequence
    : public iterator_sequence<typename shift_iterator_type<Seq>::type > {
    typedef typename shift_iterator_type<Seq>::type source_t;
    __host__ __device__
    shifted_sequence(const Seq& in,
                     const typename Seq::index_type& shift,
                     const typename Seq::value_type& boundary)
        : iterator_sequence<source_t>(
            make_shift_iterator(in, shift, boundary),
            in.size()) {}
};
