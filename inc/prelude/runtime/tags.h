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
#include <functional>
#include <string>

#include <thrust/system/cpp/memory.h>

#ifdef OMP_SUPPORT
#include <thrust/system/omp/detail/tag.h>
#endif

#ifdef TBB_SUPPORT
#include <thrust/system/tbb/detail/tag.h>
#endif

#ifdef CUDA_SUPPORT
#include <thrust/system/cuda/detail/tag.h>
#endif

namespace copperhead {
   
typedef thrust::system::cpp::tag cpp_tag;

#ifdef OMP_SUPPORT
typedef thrust::system::omp::tag omp_tag;
#endif

#ifdef TBB_SUPPORT
typedef thrust::system::tbb::tag tbb_tag;
#endif

#ifdef CUDA_SUPPORT
typedef thrust::system::cuda::tag cuda_tag;
#endif



typedef boost::variant<cpp_tag
#ifdef OMP_SUPPORT
    ,omp_tag
#endif
#ifdef TBB_SUPPORT
    ,tbb_tag
#endif
#ifdef CUDA_SUPPORT
    ,cuda_tag
#endif
    > system_variant;

namespace detail {
//Computes the canonical memory space tag
//This is normally an identity
template<typename T>
struct canonical_memory_tag {
    typedef T tag;
};
//Except when thrust tags share a memory space
//In which case we choose one of the tags
//As the canonical tag

#ifdef OMP_SUPPORT
//The OMP tag's canonical memory space is CPP
template<>
struct canonical_memory_tag<omp_tag> {
    typedef cpp_tag tag;
};
#endif

#ifdef TBB_SUPPORT
//The TBB tag's canonical memory space is CPP
template<>
struct canonical_memory_tag<tbb_tag> {
    typedef cpp_tag tag;
};
#endif
}


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
    #ifdef OMP_SUPPORT
    std::string operator()(const omp_tag&) const;
    #endif
    #ifdef TBB_SUPPORT
    std::string operator()(const tbb_tag&) const;
    #endif
    #ifdef CUDA_SUPPORT
    std::string operator()(const cuda_tag&) const;
    #endif
};

}

std::string to_string(const system_variant& x);

system_variant canonical_memory_tag(const system_variant& x);

}
