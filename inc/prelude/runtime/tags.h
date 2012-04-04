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
#include <functional>
#include <string>

namespace copperhead {

#ifdef CUDA_SUPPORT
#include <thrust/system/cuda/memory.h>

typedef thrust::system::cpp::tag cpp_tag;
typedef thrust::system::omp::tag omp_tag;
typedef thrust::system::cuda::tag cuda_tag;

typedef boost::variant<cpp_tag, omp_tag, cuda_tag> system_variant;

#else

typedef thrust::system::cpp::tag cpp_tag;
typedef thrust::system::omp::tag omp_tag;

typedef boost::variant<cpp_tag, omp_tag> system_variant;

#endif

struct system_variant_less {
    bool operator()(const system_variant& x,
                    const system_variant& y) const;
};

bool system_variant_equal(const system_variant& x,
                          const system_variant& y);

namespace detail {
    
struct system_variant_to_string
    : boost::static_visitor<std::string> {
    std::string operator()(const cpp_tag&) const;
    std::string operator()(const omp_tag&) const;
    #ifdef CUDA_SUPPORT
    std::string operator()(const cuda_tag&) const;
    #endif
};

}

std::string to_string(const system_variant& x);

}
