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
//! Creates a name for a function object built from this identifier.
std::string fnize_id(const std::string &in);
//! Creates a name for the container which holds an array.
std::string wrap_array_id(const std::string &in);
//! Creates a name for the wrapping procedure.
std::string wrap_proc_id(const std::string &in);
//! Creates a name for the unique type of an identifier.
std::string typify(const std::string &in);
//! Creates a name for the completed identifier after a synchronization point.
std::string complete(const std::string &in);

/*!
  @}
*/


}
}

