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
#include <iostream>


namespace backend {
namespace detail {

struct inspect_impl : boost::static_visitor<> {
    template<typename S>
    void operator()(const S& n) {
        std::cout << typeid(n).name();
    }
};
/*!
  \addtogroup utilities
  @{
*/

//! Inspects a variant and prints out its dynamic type to std::cout
template<typename V>
void inspect(V& n) {
    inspect_impl g;
    boost::apply_visitor(g, n);
    return;
}
/*!
  @}
*/

}
}

