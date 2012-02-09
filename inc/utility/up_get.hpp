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
#include <boost/variant.hpp>


namespace backend {
namespace detail {
/*!
  \addtogroup utilities
  @{
*/


//! Used internally.
template<typename T>
struct type_extractor
    : public boost::static_visitor<const T&>
{
    template<typename U>
    const T& operator()(const U& u) const {
        return (const T&)u;
    }
};

//! Similar to boost::get, but won't fail if you retrieve a base class.
/*! boost::get<T> allows you to extract a T object from a variant
    holding a T.  However, it fails if the variant holds U and you ask
    for T, even if U derives from T. This function succeeds in that case.

  \tparam T The type requested by the programmer.
  \param u The variant being examined.
  
  \return 
*/
template<typename T, typename U>
const T& up_get(const U& u) {
    return boost::apply_visitor(type_extractor<T>(), u);
}

/*!
  @}
*/


}
}
        
