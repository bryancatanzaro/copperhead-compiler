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
#include <string>

namespace backend {
namespace detail {

/*!
  \addtogroup utilities
  @{
*/


//! Gets string for make_sequence
const std::string make_sequence();
//! Gets string for wrap
const std::string wrap();
//! Gets string for make_remote
const std::string make_remote();
//! Gets string for boost_python_module
const std::string boost_python_module();
//! Gets string for boost_python_def
const std::string boost_python_def();
//! Gets string for phase_boundary
const std::string phase_boundary();

/*!
  @}
*/


}
}

