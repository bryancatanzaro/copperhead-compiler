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

#include <prelude/runtime/make_cuarray.hpp>

namespace copperhead {

/* This file allows the creation of views of cuarray objects,
   even when cuarray objects can't be instantiated by the compiler.

   nvcc cannot yet instantiate cuarray, for two reasons:
   * parts of cuarray use c++11 move semantics
   * parts of cuarray use c++11 std::shared_ptr

   Since the view objects created from cuarray objects can be
   instantiated by all compilers, they can be used in nvcc-compiled
   code.
   
*/

template<typename S>
S make_sequence(sp_cuarray& in, detail::fake_system_tag t, bool write=true);

}
