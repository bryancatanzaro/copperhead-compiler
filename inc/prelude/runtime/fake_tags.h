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

namespace copperhead {
namespace detail {

//This exists due to a sad chain of incompatibility dependences.
//Once nvcc can digest std::shared_ptr, used everywhere else in the
//backend, we can compile the rest of the backend with nvcc, and then
//we can use thrust tags in chunk.  But since nvcc can't be exposed to
//std::shared_ptr, and g++ can't be exposed to
//thrust::system::cuda::tag, we have to interface through an enum.

enum fake_system_tag {fake_omp_tag, fake_cuda_tag};

}
}
