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

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/detail/tuple_meta_transform.h>

namespace copperhead {

namespace detail {

template<typename Seq>
struct extract_value {
    typedef typename Seq::value_type type;
};

template<typename Seq>
struct extract_reference {
    typedef typename Seq::value_type& type;
};
template<typename Seq>
struct extract_iterator {
    typedef typename Seq::iterator_type type;
};
struct index_sequence {
    int m_index;
    index_sequence(int index) : m_index(index) {}
    //XXX Fix references to zipped sequences!
    template<typename Seq>
    typename Seq::value_type operator()(Seq& in) {
        return in[m_index];
    }
};
struct extract_begin {
    template<typename Seq>
    typename Seq::iterator_type operator()(const Seq& in) {
        return in.begin();
    }
};
struct extract_end {
    template<typename Seq>
    typename Seq::iterator_type operator()(const Seq& in) {
        return in.end();
    }
};
}

template<typename S>
struct zipped_sequence {
    S m_seqs;
    typedef typename thrust::detail::tuple_meta_transform<
        S, detail::extract_value>::type value_type;
    //XXX Fix references to zipped sequences!
    typedef typename thrust::detail::tuple_meta_transform<
        S, detail::extract_value>::type reference_type;
    typedef typename thrust::zip_iterator<
        typename thrust::detail::tuple_meta_transform<
            S, detail::extract_iterator>::type > iterator_type;
    zipped_sequence(S seqs) : m_seqs(seqs) {}
    reference_type operator[](int index) {
        //XXX Fix references to zipped sequences!
        return thrust::detail::tuple_host_transform
            <detail::extract_value, S, detail::index_sequence>(
                m_seqs,
                detail::index_sequence(index));
    }
    iterator_type begin() const {
        return thrust::detail::tuple_host_transform
            <detail::extract_iterator, S, detail::extract_begin>(
                m_seqs,
                detail::extract_begin());
    }
    iterator_type end() const {
        return thrust::detail::tuple_host_transform
            <detail::extract_iterator, S, detail::extract_end>(
                m_seqs,
                detail::extract_end());
    }
    int size() const {
        return thrust::get<0>(m_seqs).size();
    }
};

template<typename T0>
zipped_sequence<thrust::tuple<T0> >
make_zipped_sequence(T0 t0) {
    return zipped_sequence<
        thrust::tuple<T0> >(
            thrust::make_tuple(t0));
}

template<typename T0, typename T1>
zipped_sequence<thrust::tuple<T0, T1> >
make_zipped_sequence(T0 t0, T1 t1) {
    return zipped_sequence<
        thrust::tuple<T0, T1> >(
            thrust::make_tuple(t0, t1));
}

template<typename T0, typename T1, typename T2>
zipped_sequence<thrust::tuple<T0, T1, T2> >
make_zipped_sequence(T0 t0, T1 t1, T2 t2) {
    return zipped_sequence<
        thrust::tuple<T0, T1, T2> >(
            thrust::make_tuple(t0, t1, t2));
}

template<typename T0, typename T1, typename T2, typename T3>
zipped_sequence<thrust::tuple<T0, T1, T2, T3> >
make_zipped_sequence(T0 t0, T1 t1, T2 t2, T3 t3) {
    return zipped_sequence<
        thrust::tuple<T0, T1, T2, T3> >(
            thrust::make_tuple(t0, t1, t2, t3));
}

}
