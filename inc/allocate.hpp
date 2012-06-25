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
#include "utility/isinstance.hpp"
#include "utility/markers.hpp"
#include "utility/snippets.hpp"
#include "utility/initializers.hpp"
#include "py_printer.hpp"
#include "cpp_printer.hpp"
#include "rewriter.hpp"
#include "prelude/runtime/tags.h"

/*!
  \file   allocate.hpp
  \brief  The declaration of the \p allocate rewrite pass.
  
  
*/


namespace backend {
/*! 
  \addtogroup rewriters
  @{
 */


//! A rewrite pass that inserts memory allocation.
/*! Temporary variables and results need to have storage explicitly
  allocated. This rewrite pass makes this explicit in the program text.
*/
class allocate
    : public rewriter<allocate>
{
private:
    const copperhead::system_variant& m_target;
    const std::string& m_entry_point;
    bool m_in_entry;
    std::vector<std::shared_ptr<const statement> > m_allocations;
    std::shared_ptr<const ctype::type_t> container_type(const ctype::type_t& t);
public:

/*!   
  \param entry_point The name of the entry point procedure 
*/
    allocate(const copperhead::system_variant&, const std::string& entry_point);
    
    using rewriter<allocate>::operator();

//! Rewrite rule for \p procedure nodes

    result_type operator()(const procedure &n);

//! Rewrite rule for \p bind nodes
    result_type operator()(const bind &n);
};

//! @}

}
