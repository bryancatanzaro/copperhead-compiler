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
#include <prelude/runtime/make_cuarray.hpp>
#include <prelude/runtime/make_sequence.hpp>
#include <thrust/copy.h>
#include <prelude/runtime/tags.h>
#include <thrust/tuple.h>

#include <prelude/primitives/stored_sequence.h>

namespace copperhead {


template<typename Seq>
boost::shared_ptr<cuarray> phase_boundary(const Seq& in) {
    typedef typename Seq::value_type T;
    typedef typename Seq::tag Tag;
    boost::shared_ptr<cuarray> result_ary = make_cuarray<T>(in.size());
    typedef typename detail::stored_sequence<Tag, T>::type sequence_type;
    sequence_type result =
        make_sequence<sequence_type>(result_ary,
                                     Tag(),
                                     true);
    thrust::copy(in.begin(),
                 in.end(),
                 result.begin());
    return result_ary;
}

namespace detail {
template<typename Seq>
struct pb_result_type {
    typedef sp_cuarray type;
};

template<>
struct pb_result_type<thrust::null_type> {
    typedef thrust::null_type type;
};

template<typename Tag, typename Seq>
struct stored_sequence_type {
    typedef typename Seq::value_type T;
    typedef typename stored_sequence<Tag, T>::type type;
};

template<typename Tag>
struct stored_sequence_type<Tag, thrust::null_type> {
    typedef thrust::null_type type;
};

template<
    typename Tag,
    typename Seq0,
    typename Seq1,
    typename Seq2,
    typename Seq3,
    typename Seq4,
    typename Seq5,
    typename Seq6,
    typename Seq7,
    typename Seq8,
    typename Seq9>
struct stored_sequence_type<Tag, thrust::tuple<Seq0, Seq1, Seq2, Seq3, Seq4, Seq5, Seq6, Seq7, Seq8, Seq9> > {
    typedef thrust::tuple<
        typename stored_sequence_type<Tag, Seq0>::type,
        typename stored_sequence_type<Tag, Seq1>::type,
        typename stored_sequence_type<Tag, Seq2>::type,
        typename stored_sequence_type<Tag, Seq3>::type,
        typename stored_sequence_type<Tag, Seq4>::type,
        typename stored_sequence_type<Tag, Seq5>::type,
        typename stored_sequence_type<Tag, Seq6>::type,
        typename stored_sequence_type<Tag, Seq7>::type,
        typename stored_sequence_type<Tag, Seq8>::type,
        typename stored_sequence_type<Tag, Seq9>::type> type;
};
                
template<typename S, typename T>
struct make_cuarrays {};

template<
    typename SHT, typename STT,
    typename RHT, typename RTT>
struct make_cuarrays<
    thrust::detail::cons<SHT, STT>,
    thrust::detail::cons<RHT, RTT> > {
    static thrust::detail::cons<RHT, RTT> fun(
        const thrust::detail::cons<SHT, STT>& s) {
        const SHT& head = s.get_head();
        sp_cuarray head_ary = make_cuarray<typename SHT::value_type>(head.size());
        return thrust::detail::cons<RHT, RTT>(
            head_ary,
            make_cuarrays<STT, RTT>::fun(s.get_tail()));
            
    }
};

template<>
struct make_cuarrays<thrust::null_type, thrust::null_type> {
    static thrust::null_type fun(const thrust::null_type& x) {
        return thrust::null_type();
    }
};

}

template<
typename Seq0, typename Seq1, typename Seq2, typename Seq3, typename Seq4,
typename Seq5, typename Seq6, typename Seq7, typename Seq8, typename Seq9>
    
    typename thrust::detail::tuple_meta_transform<
    thrust::tuple<Seq0, Seq1, Seq2, Seq3, Seq4, Seq5, Seq6, Seq7, Seq8, Seq9>,
    detail::pb_result_type>::type

    phase_boundary(
        thrust::tuple<Seq0, Seq1, Seq2, Seq3, Seq4,
        Seq5, Seq6, Seq7, Seq8, Seq9>& seqs) {
    
    typedef thrust::tuple<Seq0, Seq1, Seq2, Seq3, Seq4,
        Seq5, Seq6, Seq7, Seq8, Seq9> Seqs;
    typedef typename thrust::detail::tuple_meta_transform<Seqs,
        detail::pb_result_type>::type result_type;
    typedef typename Seq0::tag Tag;
    typedef typename detail::stored_sequence_type<Tag, Seqs>::type stored_type;
    
    result_type arrays = detail::make_cuarrays<
        thrust::detail::cons<
        typename Seqs::head_type,
        typename Seqs::tail_type>,
        thrust::detail::cons<
        typename result_type::head_type,
        typename result_type::tail_type> >::fun(seqs);
    
    stored_type stored_seqs = make_sequence<stored_type>(arrays, Tag(), true);
    zipped_sequence<stored_type> d(stored_seqs);
    zipped_sequence<Seqs> s(seqs);
    thrust::copy(s.begin(),
                 s.end(),
                 d.begin());

    return arrays;
    
}

}
