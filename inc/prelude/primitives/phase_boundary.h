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

namespace copperhead {

namespace detail {

template<typename Tag, typename T>
struct stored_sequence {
    typedef sequence<Tag, T> type;
};

template<typename Tag,
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
struct stored_sequence<Tag,
                       thrust::tuple<T0,
                                     T1,
                                     T2,
                                     T3,
                                     T4,
                                     T5,
                                     T6,
                                     T7,
                                     T8,
                                     T9> > {
    typedef thrust::tuple<T0, T1, T2, T3, T4,
                          T5, T6, T7, T8, T9> sub_type;
    typedef zipped_sequence<
        thrust::tuple<
            typename stored_sequence<Tag, T0>::type,
            typename stored_sequence<Tag, T1>::type,
            typename stored_sequence<Tag, T2>::type,
            typename stored_sequence<Tag, T3>::type,
            typename stored_sequence<Tag, T4>::type,
            typename stored_sequence<Tag, T5>::type,
            typename stored_sequence<Tag, T6>::type,
            typename stored_sequence<Tag, T7>::type,
            typename stored_sequence<Tag, T8>::type,
            typename stored_sequence<Tag, T9>::type > > type;
};

template<typename Tag>
struct stored_sequence<Tag,
                       thrust::null_type> {
    typedef thrust::null_type type;
};

}

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

}
