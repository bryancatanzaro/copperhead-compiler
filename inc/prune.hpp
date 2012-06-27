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
#include "environment.hpp"
#include <deque>

namespace backend {

/*!
  \addtogroup rewriters
  @{
*/

//! A rewrite pass which prunes dead code
/*! The compiler may generate code which is never used. This can lead
 *  to C++ compiler warnings when the output is compiled. To prevent
 *  this, we remove dead code with this compiler pass.
*/

class prune
    : public rewriter<prune>
{
private:
    environment<std::string> m_used;
public:
    using rewriter<prune>::operator();
    result_type operator()(const suite& n);
    result_type operator()(const name& n);
    result_type operator()(const bind& n);
};

}
