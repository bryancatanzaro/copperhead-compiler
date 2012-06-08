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

#include <iostream>
#include <cassert>

namespace backend {

/*! 
\addtogroup rewriters
@{
 */

//! A rewrite pass that protects dereferences in the entry point
/*! The entry point may execute in a different memory space than
  the remainder of the computation.  Accordingly, dereferences must be
  protected to ensure that data transfers are performed when necessary
*/
class dereference
    : public rewriter<dereference>
{
private:
    const std::string& m_entry_point;
    bool m_in_entry;
public:
    //! Constructor
    //* @param entry_point Name of the entry point procedure
    dereference(const std::string& entry_point);
    
    using rewriter<dereference>::operator();
    //! Rewrite rule for \p procedure nodes
    result_type operator()(const procedure &p);
    //! Rewrite rule for \p subscript nodes
    result_type operator()(const subscript &s);
};

/*!
  @}
*/
}

