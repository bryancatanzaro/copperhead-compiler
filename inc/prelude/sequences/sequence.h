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

#include <prelude/sequences/sequence_iterator.h>
#include <thrust/detail/pointer.h>

namespace copperhead {

template<typename Tag, typename T, int D>
struct sequence;

template<typename Tag, typename T>
struct sequence<Tag, T, 0> {
    typedef Tag tag;
    typedef T el_type;
    typedef T& ref_type;
    typedef thrust::pointer<T, Tag> ptr_type;
    typedef size_t index_type;
    static const int nesting_depth = 0;
    typedef typename sequence_iterator<sequence<Tag, T, 0> >::type iterator_type;
    typedef T value_type;
    
    T* m_d;
    size_t m_l;
    __host__ __device__
    sequence() : m_d(NULL), m_l(0) {}
    __host__ __device__
    sequence(T* d, size_t l) : m_d(d), m_l(l) {}

    __host__ __device__
    T& operator[](const size_t& i) const {
        return m_d[i];
    }
    __host__ __device__
    T& operator[](size_t& i) {
        return m_d[i];
    }
    __host__ __device__
    const size_t& size() const {
        return m_l;
    }
    __host__ __device__
    bool empty() const {
        return size() <= 0;
    }
    __host__ __device__
    T& next() {
        T* source = m_d;
        m_d++;
        m_l--;
        return *source;
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

template<typename Tag, typename T>
__host__ __device__
sequence<Tag, T, 0> slice(sequence<Tag, T, 0> seq, size_t base, size_t len) {
    return sequence<Tag, T, 0>(&seq[base], len);
}


template<typename Tag, typename T, int D=0>
struct sequence {
    typedef Tag tag;
    typedef sequence<Tag, T, D-1> el_type;
    typedef el_type ref_type;
    typedef el_type* ptr_type;
    typedef size_t index_type;
    typedef T value_type;
    static const int nesting_depth = D;
    typedef typename sequence_iterator<sequence<Tag, T, D> >::type iterator_type;
    sequence<Tag, size_t, 0> m_d;

    sequence<Tag, T, D-1> m_s;
    __host__ __device__
    sequence() : m_d(), m_s() {}
    __host__ __device__
    sequence(sequence<Tag, size_t, 0> d,
             sequence<Tag, T, D-1> s) : m_d(d), m_s(s) {}
    
    __host__ __device__
    sequence<Tag, T, D-1> operator[](size_t& i) {
        size_t begin=m_d[i], end=m_d[i+1];
        return slice(m_s, begin, end-begin);
    }
    __host__ __device__
    sequence<Tag, T, D-1> operator[](const size_t& i) const {
        size_t begin=m_d[i], end=m_d[i+1];
        return slice(m_s, begin, end-begin);
    }
     __host__ __device__
    size_t size() const {
        return m_d.size() - 1;
    }
    __host__ __device__
    bool empty() const {
        return size() <= 0;
    }
    __host__ __device__
    sequence<Tag, T, D-1> next() {
        sequence<Tag, T, D-1> x = operator[](0);
        m_d.next();
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
sequence<Tag, T, D> slice(sequence<Tag, T, D> seq, size_t base, size_t len) {
    return sequence<Tag, T, D>(slice(seq.m_d, base, len+1), seq.m_s);
}
    
template<typename Tag, typename T, int D>
__host__ __device__
size_t len(const sequence<Tag, T, D>& seq) {
    return seq.size();
}


template<typename Tag, typename T, int D>
__host__
sequence<Tag, T, D-1> dereference(const sequence<Tag, T, D>& seq,
                                  typename sequence<Tag, T, D>::index_type i) {
    return seq[i];
}

template<typename Tag, typename T>
__host__
thrust::reference<T, thrust::pointer<T, Tag> >
dereference(const sequence<Tag, T, 0>& seq,
            typename sequence<Tag, T, 0>::index_type i) {
    return thrust::reference<T, thrust::pointer<T, Tag> >(
        thrust::pointer<T, Tag>(seq.m_d + i));
}



}

