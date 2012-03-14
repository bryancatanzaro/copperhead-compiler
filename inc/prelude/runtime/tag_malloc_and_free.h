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

//This minor addition to Thrust allows us to use Thrust tags for
//malloc and free. In a future version of Thrust where some similar
//interface is provided, this file can be deleted.

#pragma once

#include <thrust/detail/malloc_and_free_adl_helper.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/memory.h>

namespace thrust {
namespace detail {
template<typename Tag>
void* tag_malloc(Tag, size_t cnt) {
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::malloc;
    return malloc(select_system(Tag()), cnt);
}

template<typename Tag>
void tag_free(Tag, void* p) {
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::free;
    return free(select_system(Tag()), p);
}
}
}
