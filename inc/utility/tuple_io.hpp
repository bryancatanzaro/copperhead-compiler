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
#include <tuple>

namespace utility {
namespace detail {
template<class Tuple, std::size_t N>
struct TuplePrinter {
    static void print(std::ostream& o, const Tuple& t) {
        TuplePrinter<Tuple, N-1>::print(o, t);
        o << ", " << std::get<N-1>(t);
    }
};

template<class Tuple>
struct TuplePrinter<Tuple, 1> {
    static void print(std::ostream& o, const Tuple& t) {
        o << std::get<0>(t);
    }
};
}
}

/*!
  \addtogroup utilities
  @{
*/

//! Prints std::tuple types
/*! Since std::tuple does not come along with print functionality,
  this is a simple way of printing them out.
  
  \param o Stream to print to.
  \param t Tuple to be printed.
*/
template<class... Args>
std::ostream& operator<<(std::ostream& o, const std::tuple<Args...>& t) {
    typedef const std::tuple<Args...>& tuple_t;
    static const int num = sizeof...(Args);
    o << "(";
    utility::detail::TuplePrinter<tuple_t, num>::print(o, t);
    o << ")";
    return o;
}

/*!
  @}
*/
