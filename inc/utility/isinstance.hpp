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
#include <boost/type_traits.hpp>
#include <boost/mpl/logical.hpp>


namespace backend {
namespace detail {
/*!
  \addtogroup utilities
  @{
*/

//! Used internally to examine types.
template<typename T>
class type_checker:
        public boost::static_visitor<>
{
public:
    explicit type_checker(bool& res) : m_res(res) {}
    template<typename U>
    typename boost::disable_if<
        boost::is_base_of<T, U> >::type
    operator()(const U& u) const {
        m_res = false;
    }
    template<typename U>
    typename boost::enable_if<
        boost::is_base_of<T, U> >::type
    operator()(const U& t) const {
        m_res = true;
    }
private:
    bool& m_res;
};
//! Checks if dynamic type is an instance of another type.
/*! This procedure examines a variant to discover if the dynamic type
  which the variant currently holds is an instance of another
  type. Will return true if the variant holds a derived type of the
  base type.
  \tparam T Base type.
  \param v Instance being checked.
  
  \return true if v is an instance of T, false otherwise.
*/
template<typename T, typename V>
bool isinstance(const V& v) {
    bool result = false;
    boost::apply_visitor(type_checker<T>(result), v);
    return result;
}

/*!
  @}
*/


}
}

