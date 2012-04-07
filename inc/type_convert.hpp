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
#include "rewriter.hpp"
#include "type.hpp"
#include "ctype.hpp"
#include "type_printer.hpp"

#include <iostream>
#include <cassert>

namespace backend {

namespace detail {

//! Converts Copperhead types to C++ types
/*! Input ASTs typically have their C++ implementation types
  unspecified (defaulting to void). This visitor translates from
  Copperhead type to C++ type.
*/
class cu_to_c
    : public boost::static_visitor<std::shared_ptr<const ctype::type_t> >
{
public:
    result_type operator()(const monotype_t& mt);
    
    result_type operator()(const sequence_t & st);
    
    result_type operator()(const tuple_t& tt);
    
    result_type operator()(const fn_t& ft);
    
    //XXX Need polytypes! This code is probably not right.
    result_type operator()(const polytype_t& p);
};
}
/*! 
\addtogroup rewriters
@{
 */

//! A rewrite pass that converts Copperhead types to C++ types
/*! It does not change the structure of the AST, just
  creates a new AST where the C++ types are freshly derived
  from the Copperhead types embedded in the input AST.
*/
class type_convert
    : public rewriter
{
private:
    detail::cu_to_c m_c;
public:
    //! Constructor
    type_convert();

    using rewriter::operator();
    //! Rewrite rule for \p procedure nodes
    result_type operator()(const procedure &p);
    //! Rewrite rule for \p name nodes
    result_type operator()(const name &p);
    //! Rewrite rule for \p literal nodes
    result_type operator()(const literal &p);
};

/*!
  @}
*/
}
