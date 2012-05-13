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
#include <map>
#include <string>
#include <sstream>

#include "node.hpp"
#include "type.hpp"
#include "ctype.hpp"
#include "rewriter.hpp"
#include "utility/isinstance.hpp"
#include "utility/markers.hpp"
#include "utility/initializers.hpp"
#include "type_printer.hpp"
#include <prelude/runtime/tags.h>

namespace backend {

/*! 
  \addtogroup rewriters
  @{
 */

//! Rewriter for Thrust calls
/*! This rewriter performs all rewrites specific to the Thrust library.
  For example, it makes mapn calls produce a transformed_sequence<>
  C++ implementation type, or indices produce a counting_sequence
  C++ implementation type.
*/
class thrust_rewriter
    : public rewriter {
private:
    const copperhead::system_variant& m_target;
    
    result_type map_rewrite(const bind& n);
    
    result_type indices_rewrite(const bind& n);

    result_type shift_rewrite(const bind& n);

    result_type rotate_rewrite(const bind& n);
    
    result_type replicate_rewrite(const bind& n);

    result_type zip_rewrite(const bind& n);
    
public:
    //! Constructor
    thrust_rewriter(const copperhead::system_variant&);
    
    using rewriter::operator();
    
    result_type operator()(const bind& n);
    
};

/*! 
  @}
 */


}
