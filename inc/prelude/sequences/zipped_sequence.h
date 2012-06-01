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
#include <thrust/detail/type_traits.h>
#include <prelude/basic/detail/retagged_iterator_type.h>
#include <thrust/detail/reference.h>
#include <thrust/detail/pointer.h>


namespace copperhead {

namespace detail {

//XXX A lot of this machinery could probably be replaced by using
//thrust::zip_iterator more intelligently

template<typename Seq>
struct extract_value {
    typedef typename Seq::value_type type;
};

template<typename S0,
         typename S1,
         typename S2,
         typename S3,
         typename S4,
         typename S5,
         typename S6,
         typename S7,
         typename S8,
         typename S9>
struct extract_value<
    thrust::tuple<S0, S1, S2, S3, S4,
                  S5, S6, S7, S8, S9> > {
    typedef thrust::tuple<S0, S1, S2, S3, S4,
                          S5, S6, S7, S8, S9> sub_tuple;
    typedef typename thrust::detail::tuple_meta_transform<
        sub_tuple, copperhead::detail::extract_value>::type type;
};

template<>
struct extract_value<
    thrust::null_type> {
    typedef thrust::null_type type;
};

template<typename Seq>
struct extract_reference {
    typedef typename Seq::ref_type type;
};


template<typename S0,
         typename S1,
         typename S2,
         typename S3,
         typename S4,
         typename S5,
         typename S6,
         typename S7,
         typename S8,
         typename S9>
struct extract_reference<
    thrust::tuple<S0, S1, S2, S3, S4,
                  S5, S6, S7, S8, S9> > {
    typedef thrust::tuple<S0, S1, S2, S3, S4,
                          S5, S6, S7, S8, S9> sub_tuple;
    typedef typename thrust::detail::tuple_meta_transform<
        sub_tuple, copperhead::detail::extract_reference>::type type;
};

template<>
struct extract_reference<
    thrust::null_type> {
    typedef thrust::null_type type;
};

template<typename Seq>
struct extract_iterator {
    typedef typename Seq::iterator_type type;
};


template<typename S0,
         typename S1,
         typename S2,
         typename S3,
         typename S4,
         typename S5,
         typename S6,
         typename S7,
         typename S8,
         typename S9>
struct extract_iterator<
    thrust::tuple<S0, S1, S2, S3, S4,
                  S5, S6, S7, S8, S9> > {
    typedef thrust::tuple<S0, S1, S2, S3, S4,
                          S5, S6, S7, S8, S9> sub_tuple;
    typedef typename thrust::detail::tuple_meta_transform<
        sub_tuple, copperhead::detail::extract_iterator>::type type;
};

template<>
struct extract_iterator<
    thrust::null_type> {
    typedef thrust::null_type type;
};

template<typename index_type>
struct index_sequence {
    index_type m_index;
    __host__ __device__
    index_sequence(index_type index) : m_index(index) {}

    template<typename Seq>
    __host__ __device__
    typename Seq::ref_type operator()(Seq in) {
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
    typedef typename thrust::tuple_element<0, S>::type::index_type index_type;
    typedef typename thrust::tuple_element<0, S>::type::tag tag;
    typedef typename thrust::detail::tuple_meta_transform<
        S, detail::extract_value>::type value_type;
    typedef typename thrust::detail::tuple_meta_transform<
        S, detail::extract_reference>::type ref_type;
    typedef typename thrust::zip_iterator<
        typename thrust::detail::tuple_meta_transform<
            S, detail::extract_iterator>::type > ZI;
    typedef typename detail::retagged_iterator_type<ZI, tag>::type iterator_type;
    __host__ __device__
    zipped_sequence(S seqs) : m_seqs(seqs) {}
    __host__ __device__
    ref_type operator[](index_type index) {
        return thrust::detail::tuple_host_device_transform
            <detail::extract_reference, S, detail::index_sequence<index_type> >(
                m_seqs,
                detail::index_sequence<index_type>(index));
    }
    __host__ __device__
    value_type operator[](index_type index) const {
        return thrust::detail::tuple_host_device_transform
            <detail::extract_reference, S, detail::index_sequence<index_type> >(
                m_seqs,
                detail::index_sequence<index_type>(index));
    }
    
    //XXX Can only call begin() from host!!
    iterator_type begin() const {
        return thrust::retag<tag>(ZI(thrust::detail::tuple_host_transform
                                  <detail::extract_iterator, S, detail::extract_begin>(
                                      m_seqs,
                                      detail::extract_begin())));
    }
    //XXX Can only call end() from host!!
    iterator_type end() const {
        return thrust::retag<tag>(ZI(thrust::detail::tuple_host_transform
                                     <detail::extract_iterator, S, detail::extract_end>(
                                         m_seqs,
                                         detail::extract_end())));
    }
    __host__ __device__
    index_type size() const {
        return thrust::get<0>(m_seqs).size();
    }
};

namespace detail {

template<typename Seq>
struct extract_thrust_ref {
    typedef typename Seq::tag tag;
    typedef typename Seq::value_type value;
    typedef thrust::pointer<value, tag> pointer;
    typedef thrust::reference<value, pointer> type;
};


template<typename S0,
         typename S1,
         typename S2,
         typename S3,
         typename S4,
         typename S5,
         typename S6,
         typename S7,
         typename S8,
         typename S9>
struct extract_thrust_ref<
    thrust::tuple<S0, S1, S2, S3, S4,
                  S5, S6, S7, S8, S9> > {
    typedef thrust::tuple<S0, S1, S2, S3, S4,
                          S5, S6, S7, S8, S9> sub_tuple;
    typedef typename thrust::detail::tuple_meta_transform<
        sub_tuple, copperhead::detail::extract_thrust_ref>::type type;
};

template<>
struct extract_thrust_ref<
    thrust::null_type> {
    typedef thrust::null_type type;
};

template<typename S>
struct extract_thrust_ref<zipped_sequence<S> > {
    typedef typename extract_thrust_ref<S>::type type;
};

template<typename I>
struct extract_dereference {
    I idx;
    extract_dereference(const I& i) : idx(i) {}
    
    template<typename Tag, typename T>
    typename extract_thrust_ref<sequence<Tag, T> >::type operator()(const sequence<Tag, T>& in) const {
        return dereference(in, idx);
    }

    template<typename S>
    typename extract_thrust_ref<zipped_sequence<S> >::type operator()(const zipped_sequence<S>& in) const {
        return thrust::detail::tuple_host_transform<
            extract_thrust_ref,
            S,
            extract_dereference<I> >(
                in.m_seqs,
                *this);
        
    }
};

}

template<typename S>
__host__
typename detail::extract_thrust_ref<zipped_sequence<S> >::type
dereference(const zipped_sequence<S>& s,
            typename zipped_sequence<S>::index_type i) {
    return detail::extract_dereference<typename zipped_sequence<S>::index_type>(i)(s);
}




}
