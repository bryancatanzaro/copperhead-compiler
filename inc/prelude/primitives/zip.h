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
#include <prelude/sequences/zipped_sequence.h>

namespace copperhead {

template<typename Seq0>
zipped_sequence<thrust::tuple<Seq0> >
zip1(Seq0& x0) {
    return zipped_sequence<thrust::tuple<Seq0> >(thrust::make_tuple(x0));
}

template<typename Seq0,
         typename Seq1>
zipped_sequence<thrust::tuple<Seq0,
                              Seq1> >
zip2(Seq0& x0,
     Seq1& x1) {
    return zipped_sequence<thrust::tuple<Seq0,
                                         Seq1> >(
                                             thrust::make_tuple(
                                                 x0,
                                                 x1));
}

template<typename Seq0,
         typename Seq1,
         typename Seq2>
zipped_sequence<thrust::tuple<Seq0,
                              Seq1,
                              Seq2> >
zip3(Seq0& x0,
     Seq1& x1,
     Seq2& x2) {
    return zipped_sequence<thrust::tuple<Seq0,
                                         Seq1,
                                         Seq2> >(
                                             thrust::make_tuple(
                                                 x0,
                                                 x1,
                                                 x2));
}

template<typename Seq0,
         typename Seq1,
         typename Seq2,
         typename Seq3>
zipped_sequence<thrust::tuple<Seq0,
                              Seq1,
                              Seq2,
                              Seq3> >
zip4(Seq0& x0,
     Seq1& x1,
     Seq2& x2,
     Seq3& x3) {
    return zipped_sequence<thrust::tuple<Seq0,
                                         Seq1,
                                         Seq2,
                                         Seq3> >(
                                             thrust::make_tuple(
                                                 x0,
                                                 x1,
                                                 x2,
                                                 x3));
}


}
