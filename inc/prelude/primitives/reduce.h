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

#include <thrust/reduce.h>

namespace copperhead {

template<typename F, typename Seq>
typename F::result_type
reduce(const F& fn, Seq& x, const typename F::result_type& p) {
    return thrust::reduce(x.begin(), x.end(), p, fn);
}

template<typename Seq>
typename Seq::value_type
sum(Seq& x) {
    return thrust::reduce(x.begin(), x.end());
}

}
