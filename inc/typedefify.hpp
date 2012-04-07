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
#include "rewriter.hpp"

namespace backend {

/*!
  \addtogroup rewriters
  @{
*/

//! A rewrite pass to add typedefs
/*! Because many compilers do not yet support C++11,
  we have to add a unique typename for every identifier in the program.
  For example, the following declaration:
  \verbatim int _a; \endverbatim
  becomes
  \verbatim typedef int T_a;
T_a _a; \endverbatim

  This allows us to use the type of \p _a elsewhere in the program without
  having to know exactly what it was instantiated as.
*/
class typedefify
    : public rewriter
{
private:
    std::shared_ptr<const statement> m_typedef;
public:
    //! Constructor
    typedefify();
    
    using rewriter::operator();
    //! Rewrite rule for \p suite nodes
    result_type operator()(const suite &n);
    //! Rewrite rule for \p bind nodes
    result_type operator()(const bind &n);
    //! Rewrite rule for \p procedure nodes
    result_type operator()(const procedure &n);
        
};

/*!
  @}
*/

}
