#pragma once
#include <thrust/iterator/transform_iterator.h>

template<typename F,
         typename S>
struct transformed_sequence {
    F m_fn;
    S m_seq;
    typedef typename F::result_type value_type;
    typedef typename S::iterator_type I;
    typedef typename thrust::transform_iterator<F, I> iterator_type;
    transformed_sequence(F fn,
                         S seq)
        : m_fn(fn), m_seq(seq) {}
    value_type& operator[](int index) {
        return m_fn(m_seq[index]);
    }
    iterator_type begin() {
        return iterator_type(m_seq.begin(), m_fn);
    }
    iterator_type end() {
        return iterator_type(m_seq.end(), m_fn);
    }
    int size() const {
        return m_seq.size();
    }
};
