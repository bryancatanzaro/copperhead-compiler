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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

namespace copperhead {

//forward declarations
template<typename T, int D> struct sequence;
template<typename T, int D> struct uniform_sequence;

template<typename Sequence>
struct sequence_viewer
    : public thrust::unary_function<const size_t&, typename Sequence::ref_type> {
    Sequence m_s;
  public:
    __host__ __device__ sequence_viewer() {}
    __host__ __device__ sequence_viewer(const Sequence& s) : m_s(s) {}
    __host__ __device__ typename Sequence::ref_type operator()(const size_t& i) {
        return m_s[i];
    }
};

template<typename Sequence>
struct sequence_iterator {
    typedef thrust::counting_iterator<typename Sequence::index_type> count_type;
    typedef thrust::transform_iterator<
    sequence_viewer<Sequence>, count_type, typename Sequence::ref_type> type;
    static type make_sequence_iterator_impl(const Sequence& s) {
        return type(count_type(0), sequence_viewer<Sequence>(s));
    }
};

template<typename T>
struct sequence_iterator<sequence<T, 0> > {
    //XXX Change based on system
    typedef thrust::device_ptr<T> type;
    static type make_sequence_iterator_impl(const sequence<T, 0>& s) {
        return type(s.m_d);
    }
};

template<typename T>
struct sequence_iterator<uniform_sequence<T, 0> > {
    //XXX Change based on system
    typedef thrust::device_ptr<T> type;
    static type make_sequence_iterator_impl(const uniform_sequence<T, 0>& s) {
        return type(s.m_d);
    }
};



template<typename T>
struct sequence_iterator<

template<typename Sequence>
__host__
typename sequence_iterator<Sequence>::type make_sequence_iterator(const Sequence& s) {
    return sequence_iterator<Sequence>::make_sequence_iterator_impl(s);
}

}
