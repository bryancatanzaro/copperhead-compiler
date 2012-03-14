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

template<typename F,
         typename Seq0>
transformed_sequence<F, thrust::tuple<Seq0> >
map1(const F& fn,
    Seq0& x0) {
    return transformed_sequence<F, thrust::tuple<Seq0> >(fn, thrust::make_tuple(x0));
}

template<typename F,
         typename Seq0,
         typename Seq1>
transformed_sequence<F, thrust::tuple<Seq0, Seq1> >
map2(const F& fn,
    Seq0& x0,
    Seq1& x1) {
    return transformed_sequence<F, thrust::tuple<Seq0, Seq1> >(fn, thrust::make_tuple(x0, x1));
}

template<typename F,
         typename Seq0,
         typename Seq1,
         typename Seq2>
transformed_sequence<F, thrust::tuple<Seq0, Seq1, Seq2> >
map3(const F& fn,
    Seq0& x0,
    Seq1& x1,
    Seq2& x2) {
    return transformed_sequence<F, thrust::tuple<Seq0, Seq1, Seq2> >(fn, thrust::make_tuple(x0, x1, x2));
}

template<typename F,
         typename Seq0,
         typename Seq1,
         typename Seq2,
         typename Seq3>
transformed_sequence<F, thrust::tuple<Seq0, Seq1, Seq2, Seq3> >
map4(const F& fn,
    Seq0& x0,
    Seq1& x1,
    Seq2& x2,
    Seq3& x3) {
    return transformed_sequence<F, thrust::tuple<Seq0, Seq1, Seq2, Seq3> >(fn, thrust::make_tuple(x0, x1, x2, x3));
}

}
