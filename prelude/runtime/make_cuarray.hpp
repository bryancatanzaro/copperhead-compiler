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

#include <prelude/sequences/sequence.hpp>

//XXX Use std::shared_ptr once nvcc can pass it through
#define BOOST_SP_USE_SPINLOCK
#include <boost/shared_ptr.hpp>

/* This file allows construction of cuarray objects
   even when cuarray can't be instantiated by the compiler.

   nvcc cannot yet instantiate cuarray, for two reasons:
   * parts of cuarray use c++11 move semantics
   * parts of cuarray use c++11 std::shared_ptr

   Separating the interface for cuarray in this manner
   allows code compiled by nvcc to construct cuarray objects
   without needing to instantiate them directly.
*/

namespace copperhead {

//Forward declaration
class cuarray;

typedef boost::shared_ptr<cuarray> sp_cuarray;

template<typename T>
sp_cuarray make_cuarray(size_t s);

}
