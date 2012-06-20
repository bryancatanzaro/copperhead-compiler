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

#include <thrust/scan.h>
#include <prelude/runtime/make_cuarray.hpp>
#include <prelude/runtime/make_sequence.hpp>
#include <prelude/runtime/tags.h>
#include <thrust/iterator/retag.h>
#include <prelude/primitives/stored_sequence.h>
#include <thrust/iterator/permutation_iterator.h>

namespace copperhead {

template<typename SeqX, typename SeqI>
sp_cuarray
permute(const SeqX& x, const SeqI& i) {
    typedef typename SeqX::tag Tag;
    typedef typename SeqX::value_type T;
    typedef typename detail::stored_sequence<Tag, T>::type sequence_type;
    
    boost::shared_ptr<cuarray> result_ary = make_cuarray<T>(x.size());
    sequence_type result =
        make_sequence<sequence_type>(result_ary,
                                     Tag(),
                                     true);
    typedef typename sequence_type::iterator_type ElementIterator;
    typedef typename SeqI::iterator_type IndexIterator;
    thrust::permutation_iterator<ElementIterator,
                                 IndexIterator> pi(
                                     result.begin(),
                                     i.begin());
    thrust::copy(x.begin(), x.end(), pi);
    return result_ary;
}

template<typename SeqX, typename SeqI, typename SeqD>
sp_cuarray
scatter(const SeqX& x, const SeqI& i, const SeqD& d) {
    typedef typename SeqX::tag Tag;
    typedef typename SeqX::value_type T;
    typedef typename detail::stored_sequence<Tag, T>::type sequence_type;
    
    boost::shared_ptr<cuarray> result_ary = make_cuarray<T>(d.size());
    sequence_type result =
        make_sequence<sequence_type>(result_ary,
                                     Tag(),
                                     true);
    //Copy d to preserve value semantics
    thrust::copy(d.begin(), d.end(), result.begin());
    typedef typename sequence_type::iterator_type ElementIterator;
    typedef typename SeqI::iterator_type IndexIterator;
    thrust::permutation_iterator<ElementIterator,
                                 IndexIterator> pi(
                                     result.begin(),
                                     i.begin());
    thrust::copy(x.begin(), x.end(), pi);
    return result_ary;
}


}
