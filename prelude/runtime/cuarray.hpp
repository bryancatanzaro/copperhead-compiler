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

#include <vector>
#include <iostream>
#include <memory>
#include <prelude/runtime/allocators.hpp>
#include <prelude/runtime/chunk.hpp>
#include <prelude/runtime/type.hpp>
#include <prelude/runtime/ctype.hpp>

namespace copperhead {

struct cuarray {
    std::vector<std::shared_ptr<chunk<host_alloc> > > m_local;
#ifdef CUDA_SUPPORT
    std::vector<std::shared_ptr<chunk<cuda_alloc> > > m_remote;
    bool m_clean_local;
    bool m_clean_remote;
#endif
    std::vector<size_t> m_l;
    std::shared_ptr<backend::type_t> m_t;
    std::shared_ptr<backend::ctype::type_t> m_ct;
    size_t m_o;
};

}
