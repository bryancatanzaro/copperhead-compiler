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

#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <prelude/runtime/make_cuarray.hpp>
#include <prelude/runtime/make_sequence.hpp>
#include <prelude/runtime/tags.h>
#include <thrust/iterator/retag.h>
#include <prelude/primitives/stored_sequence.h>
#include <prelude/primitives/map.h>

#include <iostream>

namespace copperhead {

namespace detail {

struct counter {
    typedef int result_type;
    __host__ __device__
    int operator()(const bool& x) const {
        if (x) {
            return 1;
        } else {
            return 0;
        }
    }
};

}

template<typename F, typename Seq>
sp_cuarray
filter(const F& fn, Seq& x) {
    typedef typename Seq::value_type T;
    typedef typename Seq::tag Tag;
    typedef typename detail::stored_sequence<Tag, T>::type sequence_type;
    typedef typename Seq::iterator_type it;
    
    boost::shared_ptr<cuarray> result_ary = make_cuarray<T>(x.size());
    sequence_type result =
        make_sequence<sequence_type>(result_ary,
                                     Tag(),
                                     true);
   
    it result_end = thrust::copy_if(x.begin(),
                                    x.end(),
                                    result.begin(),
                                    fn);

    //XXX Decide if we want to keep this extraneous copy or add a shrink method to cuarray.
    boost::shared_ptr<cuarray> compacted_ary = make_cuarray<T>(result_end - result.begin());
    sequence_type compacted =
        make_sequence<sequence_type>(compacted_ary,
                                     Tag(),
                                     true);
    thrust::copy(result.begin(), result_end, compacted.begin());
    
    return compacted_ary;
}

}
