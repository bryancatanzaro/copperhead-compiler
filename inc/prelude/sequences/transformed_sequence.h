/*
 *   Copyright 2012      NVIDIA Corporation
 * 
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 * 
 *       http://www.apache.org/licenses/LICENSE-2.0
 * 
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 * 
 */

#pragma once

#include <thrust/iterator/transform_iterator.h>
#include <prelude/sequences/zipped_sequence.h>

namespace copperhead {

namespace detail {

template<typename F>
struct map_adapter {
    F m_fn;
    typedef typename F::result_type result_type;
    typedef typename F::result_type T;
    __host__ __device__ map_adapter(const F& fn) : m_fn(fn) {}

    template<typename Tuple>
    __host__ __device__
    T operator()(const Tuple& in) {
        return apply_from_tuple(m_fn, in);
    }

};

}

template<typename F,
         typename S>
struct transformed_sequence {
    detail::map_adapter<F> m_fn;
    zipped_sequence<S> m_seq;
    typedef typename detail::map_adapter<F>::result_type value_type;
    typedef typename zipped_sequence<S>::iterator_type I;
    typedef typename zipped_sequence<S>::tag tag;
    typedef typename thrust::transform_iterator<detail::map_adapter<F>, I> TI;
    typedef typename detail::retagged_iterator_type<TI, tag>::type iterator_type;
    typedef value_type& ref_type;
    typedef typename zipped_sequence<S>::index_type index_type;
    transformed_sequence(F fn,
                         S seqs)
        : m_fn(detail::map_adapter<F>(fn)), m_seq(seqs) {}
    __host__ __device__
    ref_type operator[](int index) {
        return m_fn(m_seq[index]);
    }
    iterator_type begin() const {
        return thrust::retag<tag>(TI(m_seq.begin(), m_fn));
    }
    iterator_type end() const {
        return thrust::retag<tag>(TI(m_seq.end(), m_fn));
    }
    __host__ __device__
    index_type size() const {
        return m_seq.size();
    }
};

}
