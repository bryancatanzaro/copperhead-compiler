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
#include "utility/isinstance.hpp"
#include "rewriter.hpp"
#include <map>
/*!
  \file   allocate.hpp
  \brief  The declaration of the \p allocate rewrite pass.
  
  
*/


namespace backend {
/*! 
  \addtogroup rewriters
  @{
 */


//! A rewrite pass that translates identifiers between the frontend and the backend.
/*! The frontend IR defines some identifiers differently than C++. This translates them.
 *  Currently, this is limited to the two boolean constants.
*/
class backend_translate
    : public rewriter<backend_translate>
{
private:
    std::map<std::string, std::string> m_table;
public:
    /*!
      Constructor
    */
    backend_translate();

    using rewriter<backend_translate>::operator();
    
    //! Rewrite rule for \p name nodes

    result_type operator()(const name &n);
};

//! @}

}
