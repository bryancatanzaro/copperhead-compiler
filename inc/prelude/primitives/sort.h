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

#include <thrust/sort.h>

#include <prelude/basic/functors.h>
#include <prelude/runtime/make_cuarray.hpp>
#include <prelude/runtime/make_sequence.hpp>
#include <prelude/runtime/tags.h>


namespace copperhead {

template<typename F, typename Seq>
sp_cuarray
sort(const F& fn, Seq& x) {
    typedef typename Seq::value_type T;
    typedef typename Seq::tag Tag;
    
    //Copy for value semantics (since thrust sort is "in-place")
    sp_cuarray result_ary = make_cuarray<T>(x.size());
    sequence<Tag, T> result = make_sequence<sequence<Tag, T> >(result_ary,
                                                               detail::real_to_fake_converter(Tag()),
                                                               true);
    thrust::copy(x.begin(),
                 x.end(),
                 result.begin());

    thrust::sort(result.begin(),
                 result.end(),
                 fn);
    return result_ary;
}


template<typename Seq>
sp_cuarray
sort(const fn_cmp_lt<typename Seq::value_type>& fn, Seq& x) {
    typedef typename Seq::value_type T;
    typedef typename Seq::tag Tag;
    
    //Copy for value semantics (since thrust sort is "in-place")
    sp_cuarray result_ary = make_cuarray<T>(x.size());
    sequence<Tag, T> result = make_sequence<sequence<Tag, T> >(result_ary,
                                                               detail::real_to_fake_converter(Tag()),
                                                               true);
    thrust::copy(x.begin(),
                 x.end(),
                 result.begin());

    thrust::sort(result.begin(),
                 result.end(),
                 thrust::less<T>());
    return result_ary;
}

template<typename Seq>
sp_cuarray
sort(const fn_cmp_gt<typename Seq::value_type>& fn, Seq& x) {
    typedef typename Seq::value_type T;
    typedef typename Seq::tag Tag;
    
    //Copy for value semantics (since thrust sort is "in-place")
    sp_cuarray result_ary = make_cuarray<T>(x.size());
    sequence<Tag, T> result = make_sequence<sequence<Tag, T> >(result_ary,
                                                               detail::real_to_fake_converter(Tag()),
                                                               true);
    thrust::copy(x.begin(),
                 x.end(),
                 result.begin());

    thrust::sort(result.begin(),
                 result.end(),
                 thrust::greater<T>());
    return result_ary;
}

}
