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
#include "utility/isinstance.hpp"
#include "utility/snippets.hpp"
#include "utility/markers.hpp"

#include <iostream>
#include <cassert>

namespace backend {

/*! 
\addtogroup rewriters
@{
 */

//! A rewrite pass that ensures the proper containers are constructed for returns
/*! The entry point must return containers rather than views. These containers do not
 *  always exist, and so must be constructed to ensure they can be returned
*/
class containerize
    : public rewriter<containerize>
{
private:
    const std::string& m_entry_point;
    bool m_in_entry;
    std::shared_ptr<const ctype::type_t> container_type(const ctype::type_t&);
    std::shared_ptr<const expression> container_args(const expression&);
public:
    //! Constructor
    //* @param entry_point Name of the entry point procedure
    containerize(const std::string& entry_point);
    
    using rewriter<containerize>::operator();
    //! Rewrite rule for \p suite nodes
    result_type operator()(const suite &p);
    //! Rewrite rule for \p procedure nodes
    result_type operator()(const procedure &p);
    //! Rewrite rule for \p subscript nodes
    result_type operator()(const bind &n);
};

/*!
  @}
*/
}

