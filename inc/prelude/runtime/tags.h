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

//This file collects all Thrust system tags which Copperhead can use,
//and defines a system_variant type which can hold any of these tags.

#pragma once

#include <boost/variant.hpp>

#include <thrust/system/omp/memory.h>
namespace copperhead {
typedef thrust::system::omp::tag omp_tag;
}

#ifdef CUDA_SUPPORT
#include <thrust/system/cuda/memory.h>
namespace copperhead {
typedef thrust::system::cuda::tag cuda_tag;
}
#endif


namespace copperhead {
#ifdef CUDA_SUPPORT
typedef boost::variant<omp_tag, cuda_tag> system_variant;
#else
typedef boost::variant<omp_tag> system_variant;
#endif
}
