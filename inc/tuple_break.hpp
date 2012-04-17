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
#include "node.hpp"
#include "type.hpp"
#include "ctype.hpp"
#include "cppnode.hpp"
#include "rewriter.hpp"
#include "utility/name_supply.hpp"
#include "utility/isinstance.hpp"
#include "utility/snippets.hpp"
#include "utility/initializers.hpp"

namespace backend {

/*!
  \addtogroup rewriters
  @{
*/

//! A rewrite pass which makes tuple assembly/disassembly explicit
/*! The input is assumed to have tuples in the form
  \code
  t = x, y
  x, y = t
  \endcode
  Although convenient for the programmer, C++ requires more work to
  deal with tuples.
  \code
  t = x, y
  \endcode
  becomes:
  \code
  t = thrust::make_tuple(x, y)
  \endcode
  Also,
  \code
  x, y = t
  \endcode
  becomes
  \code
  x = thrust::get<0>(t)
  y = thrust::get<1>(t)
  \endcode
  And
  \code
  x, y = z, w
  \endcode
  becomes
  \code
  x = z
  y = w
  \endcode
  
  
  Also, tuples which appear as arguments to a
  function are given a unique id and lowered:
  \code
  def _foo((_a, _b)):
      return _a
  \endcode
  becomes:
  \code
  def _foo(t):
      _a = thrust::get<0>(t)
      _b = thrust::get<1>(t)
      return _a
  \endcode
  
*/
class tuple_break
    : public rewriter
{
public:
    //! Constructor
    tuple_break();
    using rewriter::operator();
    result_type operator()(const bind& n);
    result_type operator()(const procedure& n);
    result_type operator()(const suite& n);
private:
    detail::name_supply m_supply;
};

}
