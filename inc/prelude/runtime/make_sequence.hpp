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

#include <vector>
#include <prelude/runtime/chunk.hpp>
#include <prelude/sequences/sequence.h>
#include <prelude/sequences/zipped_sequence.h>
#include <cassert>

namespace copperhead {

namespace detail {

template<typename S>
struct make_seq_impl {};

template<typename Tag, typename T>
struct make_seq_impl<sequence<Tag, T, 0> > {
    static sequence<Tag, T, 0> fun(typename std::vector<boost::shared_ptr<chunk> >::iterator d,
                                   std::vector<size_t>::const_iterator l,
                                   const size_t o=0) {
        return sequence<Tag, T, 0>(reinterpret_cast<T*>((*d)->ptr())+o, *l);
    }
};

template<typename Tag, typename T>
struct make_seq_impl<sequence<Tag, T, 1> > {
    static sequence<Tag, T, 1> fun(typename std::vector<boost::shared_ptr<chunk> >::iterator d,
                                   std::vector<size_t>::const_iterator l,
                                   const size_t o=0) {
        sequence<Tag, size_t, 0> desc = make_seq_impl<sequence<Tag, size_t, 0> >::fun(d, l, o);
        sequence<Tag, T, 0> data = make_seq_impl<sequence<Tag, T, 0> >::fun(d+1, l+1);
        return sequence<Tag, T, 1>(desc, data);
    }
};

template<typename Tag, typename T, int D>
struct make_seq_impl<sequence<Tag, T, D > > {
    static sequence<Tag, T, D> fun(typename std::vector<boost::shared_ptr<chunk> >::iterator d,
                                   std::vector<size_t>::const_iterator l,
                                   const size_t o=0) {
        sequence<Tag, size_t, 0> desc = make_seq_impl<sequence<Tag, size_t, 0> >::fun(d, l, o);
        sequence<Tag, T, D-1> sub = make_seq_impl<sequence<Tag, T, D-1> >::fun(d+1, l+1);
        return sequence<Tag, T, D>(desc, sub);
    }
};

template<typename HT, typename TT>
struct make_seq_impl<thrust::detail::cons<HT, TT> > {
    static thrust::detail::cons<HT, TT> fun(typename std::vector<boost::shared_ptr<chunk> >::iterator& d,
                                            std::vector<size_t>::const_iterator& l,
                                            const size_t o=0) {
        HT head = make_seq_impl<HT>::fun(d, l, o);
        TT tail = make_seq_impl<TT>::fun(++d, l, o);
        return thrust::detail::cons<HT, TT>(head, tail);
    }
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
struct make_seq_impl<zipped_sequence<
                         thrust::tuple<S0, S1, S2, S3, S4, S5, S6, S7, S8, S9> > > {
    typedef thrust::tuple<S0, S1, S2, S3, S4, S5, S6, S7, S8, S9> sequences;
    static zipped_sequence<sequences> fun(typename std::vector<boost::shared_ptr<chunk> >::iterator& d,
                                          std::vector<size_t>::const_iterator& l,
                                          const size_t o=0) {
        sequences s = make_seq_impl<
            thrust::detail::cons<
                typename sequences::head_type,
                typename sequences::tail_type> >::fun(d, l, o);
        return zipped_sequence<sequences>(s);
        
    }
};

template<>
struct make_seq_impl<thrust::null_type> {
    static thrust::null_type fun(typename std::vector<boost::shared_ptr<chunk> >::iterator& d,
                                 std::vector<size_t>::const_iterator& l,
                                 const size_t o=0) {
        --d;
        return thrust::null_type();
    }
};



}

template<typename S>
S make_sequence(const sp_cuarray& in, system_variant t, bool write) {
    cuarray& r = *in;
    std::vector<boost::shared_ptr<chunk> >& chunks = r.get_chunks(t, write);
    typename std::vector<boost::shared_ptr<chunk> >::iterator ci = chunks.begin();
    std::vector<size_t>::const_iterator li = r.m_l.begin();
    return detail::make_seq_impl<S>::fun(ci, li, r.m_o);
}

namespace detail {

template<typename S,
         typename T>
struct make_tuple_seq_impl {
    static S fun(
        const T& in,
        system_variant t,
        bool write) {
        typename S::head_type head = make_tuple_seq_impl<typename S::head_type, typename T::head_type>::fun(in.get_head(), t, write);
        typename S::tail_type tail = make_tuple_seq_impl<typename S::tail_type, typename T::tail_type>::fun(in.get_tail(), t, write);
        return thrust::detail::cons<
            typename S::head_type,
            typename S::tail_type>(head, tail);
    }
};

template<typename S>
struct make_tuple_seq_impl<S,
                           sp_cuarray> {
    static S fun(const sp_cuarray& in,
                 system_variant t,
                 bool write) {
        return make_sequence<S>(in, t, write);
    }
};

template<>
struct make_tuple_seq_impl<thrust::null_type,
                           thrust::null_type> {
    static thrust::null_type fun(const thrust::null_type& in,
                                 system_variant t,
                                 bool write) {
        return thrust::null_type();
    }
};

}


template<typename S,
         typename T0,
         typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename T6,
         typename T7,
         typename T8,
         typename T9>
S make_sequence(thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>& in, system_variant t, bool write) {
    typedef thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> T;
    return detail::make_tuple_seq_impl<
        typename thrust::detail::cons<typename S::head_type, typename S::tail_type>,
        typename thrust::detail::cons<typename T::head_type, typename T::tail_type> >::
        fun(in, t, write);
}
                
         

}
