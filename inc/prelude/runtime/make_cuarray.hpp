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

#include <prelude/runtime/cuarray.hpp>
#include <prelude/runtime/make_cu_and_c_types.hpp>
#include <prelude/runtime/tags.h>

namespace copperhead {

template<typename T>
sp_cuarray make_cuarray(size_t s) {
    cu_and_c_types* type_holder =
        make_type_holder(T());
    sp_cuarray r(new cuarray(type_holder, 0));
    r->push_back_length(s);    

    r->add_chunk(boost::shared_ptr<chunk>(new chunk(omp_tag(), s * sizeof(T))), true);
#ifdef CUDA_SUPPORT
    r->add_chunk(boost::shared_ptr<chunk>(new chunk(cuda_tag(), s * sizeof(T))), true);
#endif
    
    return r;
}


}
