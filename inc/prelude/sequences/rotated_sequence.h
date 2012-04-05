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

#include <prelude/sequences/iterator_sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <prelude/basic/detail/signed_index_type.h>

namespace copperhead {

template<typename Seq>
class rotate_functor {
protected:
    Seq m_data;
    typedef typename detail::signed_index_type<typename Seq::index_type>::type I;
    I m_shift;
    typedef typename Seq::value_type T;
public:
    typedef T result_type;
    __host__ __device__
    rotate_functor(const Seq& data,
                   const I& shift)
        : m_data(data), m_shift(shift) {}
    
    __host__ __device__    
    T operator()(const I& i) const {
        I new_pos = i + m_shift;
        //If I knew how mod worked with negative operands, I
        //wouldn't need this
        while (new_pos < 0) {
            new_pos += m_data.size();
        }
        if (typename Seq::index_type(new_pos) >= m_data.size()) {
            new_pos %= m_data.size();
        }
        return m_data[new_pos];
    }
};

template<typename Seq>
struct rotate_iterator_type {
    typedef typename thrust::transform_iterator<
        rotate_functor<Seq>,
        thrust::counting_iterator<typename detail::signed_index_type<typename Seq::index_type>::type> > type;
};

template<typename Seq>
__host__ __device__
typename rotate_iterator_type<Seq>::type
make_rotate_iterator(const Seq& in,
                    const typename detail::signed_index_type<typename Seq::index_type>::type rotate) {
    typedef typename detail::signed_index_type<typename Seq::index_type>::type I;
    return typename rotate_iterator_type<Seq>::type(
        thrust::counting_iterator<I>(0),
        rotate_functor<Seq>(in, rotate));
}

template<typename Seq>
struct rotated_sequence
    : public iterator_sequence<typename Seq::tag, typename rotate_iterator_type<Seq>::type > {
    typedef typename rotate_iterator_type<Seq>::type source_t;
    typedef typename detail::signed_index_type<typename Seq::index_type>::type I;
    __host__ __device__
    rotated_sequence(const Seq& in,
                     const I& rotate)
        : iterator_sequence<typename Seq::tag, source_t>(
            make_rotate_iterator(in, rotate),
            in.size()) {}
};

}
