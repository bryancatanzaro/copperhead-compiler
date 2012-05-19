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

#include <thrust/transform.h>
#include <thrust/detail/type_traits.h>
#include <prelude/sequences/transformed_sequence.h>
#include <prelude/sequences/zipped_sequence.h>

namespace copperhead {

namespace detail {

template<typename SeqX,
         typename SeqI>
struct gather_functor {
    SeqX m_x;
    __host__ __device__
    gather_functor(const SeqX& x) : m_x(x) {}
    typedef typename SeqX::value_type X;
    typedef X result_type;
    __host__ __device__
    X operator()(const typename SeqI::value_type& i) {
        return m_x[i];
    }
};

}

template<typename SeqX,
         typename SeqI>
transformed_sequence<detail::gather_functor<SeqX,
                                            SeqI>,
                     thrust::tuple<SeqI> >
gather(const SeqX& x,
       const SeqI& i) {
    return transformed_sequence<
        detail::gather_functor<SeqX,
                               SeqI>,
        thrust::tuple<SeqI> >(
            detail::gather_functor<SeqX, SeqI>(x),
            thrust::make_tuple(i));
}



}
