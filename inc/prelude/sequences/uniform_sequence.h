/*
 *   Copyright 2011-2012 NVIDIA Corporation
 *   Copyright 2010-2011 University of California
 *
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

#include <prelude/sequences/sequence_iterator.h>
#include <thrust/detail/pointer.h>

namespace copperhead {

template<typename Tag, typename T, int D>
struct uniform_sequence {
    typedef Tag tag;
    typedef uniform_sequence<Tag, T, D-1> el_type;
    typedef el_type ref_type;
    typedef thrust::pointer<T, Tag> ptr_type;
    typedef size_t index_type;
    typedef T value_type;
    static const int nesting_depth = D;
    typedef typename sequence_iterator<uniform_sequence<Tag, T, D> >::type iterator_type;
    size_t m_s;
    size_t m_l;
    uniform_sequence<Tag, T, D-1> m_d;

    __host__ __device__
    uniform_sequence() { }

    __host__ __device__
    uniform_sequence(size_t l,
                     size_t s,
                     uniform_sequence<Tag, T, D-1> d)
        : m_l(l), m_s(s), m_d(d) { }

    __host__ __device__
    size_t size() const { return m_l; }

    __host__ __device__
    uniform_sequence<Tag, T, D-1> operator[](size_t i) {
        return slice(m_d, i * m_s);
    }

    __host__ __device__
    void advance(size_t i) {
        m_d.advance(i);
    }

    __host__ __device__
    bool empty() const {
        return size() <= 0;
    }

    __host__ __device__
    uniform_sequence<Tag, T, D-1> next() {
        uniform_sequence<Tag, T, D-1> x = operator[](0);
        advance(m_s);
        m_l--;
        return x;
    }
};

template<typename Tag, typename T>
struct uniform_sequence<Tag, T, 0>
{
    typedef Tag tag;
    typedef T el_type;
    typedef T& ref_type;
    typedef thrust::pointer<T, Tag> ptr_type;
    typedef size_t index_type;
    typedef T value_type;
    static const int nesting_depth = 0;
    typedef typename sequence_iterator<uniform_sequence<Tag, T, 0> >::type iterator_type;
    size_t m_s;
    size_t m_l;
    size_t m_o;
    T* m_d;

    __host__ __device__
    uniform_sequence() { }

    __host__ __device__
    uniform_sequence(size_t l,
                     size_t s,
                     T* d,
                     size_t o=0)
        : m_l(l), m_s(s), m_o(o), m_d(d) { }

    __host__ __device__
    size_t size() const { return m_l; }

    __host__ __device__
    T& operator[](int i) {
        return m_d[i * m_s + m_o];
    }

    __host__ __device__
    void advance(int i) {
        m_o += i;
    }

    __host__ __device__
    bool empty() const {
        return m_l <= 0;
    }

    __host__ __device__
    T next() {
        T x = operator[](0);
        advance(m_s);
        m_l--;
        return x;
    }
    __host__
    iterator_type begin() const {
        return make_sequence_iterator(*this);
    }
    __host__
    iterator_type end() const {
        return make_sequence_iterator(*this) + size();
    }
};

template<typename Tag, typename T, int D>
__host__ __device__
uniform_sequence<Tag, T, D> slice(uniform_sequence<Tag, T, D> seq, int m_o)
{
    return uniform_sequence<Tag, T, D>(seq.m_l, seq.m_s, slice(seq.m_d, m_o));
}

template<typename Tag, typename T>
__host__ __device__
uniform_sequence<Tag, T, 0> slice(uniform_sequence<Tag, T, 0> seq, int m_o)
{
    return uniform_sequence<Tag, T, 0>(seq.m_l, seq.m_s, seq.m_d, seq.m_o + m_o);
}

}
