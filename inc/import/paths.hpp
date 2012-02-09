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
#include <cstdlib>

#define GCC_VERSION (__GNUC__ * 10000                 \
                     + __GNUC_MINOR__ * 100           \
                     + __GNUC_PATCHLEVEL__)
      
#if GCC_VERSION < 40600
#define nullptr NULL
#endif

namespace backend {
namespace detail {
const char* get_path(const char* env_name);

}
}
