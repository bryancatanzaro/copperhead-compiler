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
#include "cudata.h"
#include "stored_sequence.h"
#include <thrust/copy.h>

template<typename Seq>
boost::shared_ptr<cuarray> phase_boundary(const Seq& in) {
    typedef typename Seq::value_type T;
    boost::shared_ptr<cuarray> result_ary = make_remote<T>(in.size());
    stored_sequence<T> result = get_remote_w<T>(result_ary);
    thrust::copy(in.begin(), in.end(), result.begin());
    return result_ary;
}
