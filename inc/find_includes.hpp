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
#include "import/library.hpp"
#include <set>
#include <map>

namespace backend {

/*!
  \addtogroup rewriters
  @{
*/

//! A rewrite pass which finds include files needed by program
/*! Because the compiler assembles code fragments from various places,
    we need to examine the generated code and determine what include
    statements to add to make the program complete.
  
*/
class find_includes
    : public rewriter
{
private:
    const registry& m_reg;
    std::set<std::string> m_includes;
    bool m_outer;
public:
    //! Constructor
    /*! \param reg The registry maintained by the compiler
     */
    find_includes(const registry& reg);
    using rewriter::operator();
    result_type operator()(const suite& n);
    result_type operator()(const apply& n);
};

}
