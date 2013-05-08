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
#include <thrust/system/omp/memory.h>
#endif

#ifdef TBB_SUPPORT
#ifdef __CUDACC__
//Including the tag without the memory.h
//Will define the type, but won't activate the
//overloads provided by ADL. In this case,
//it's what we want, since NVCC can't understand
//TBB headers
//Functionally, if a program compiled by nvcc were
//to attempt to use tbb, it would silently fall back
//to the sequential thrust backend.
#include <thrust/version.h>
#if THRUST_VERSION < 100700
#include <thrust/system/tbb/detail/tag.h>
#else
#include <thrust/system/tbb/detail/execution_policy.h>
#endif

#else
#include <thrust/system/tbb/memory.h>
#endif
#endif

#ifdef CUDA_SUPPORT
#include <thrust/system/cuda/memory.h>
#endif

namespace copperhead {

struct cpp_tag : thrust::system::cpp::detail::execution_policy<cpp_tag>{};

#ifdef OMP_SUPPORT
struct omp_tag : thrust::system::omp::detail::execution_policy<omp_tag>{};
#endif

#ifdef TBB_SUPPORT
struct tbb_tag : thrust::system::tbb::detail::execution_policy<tbb_tag>{};
#endif

#ifdef CUDA_SUPPORT
struct cuda_tag : thrust::system::cuda::detail::execution_policy<cuda_tag>{};
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


//Computes the Thrust tag
template<typename T>
struct thrust_memory_tag {
    typedef typename thrust_memory_tag<
        typename canonical_memory_tag<T>::tag>::tag tag;
};

template<>
struct thrust_memory_tag<cpp_tag> {
    typedef thrust::system::cpp::tag tag;
};

#ifdef CUDA_SUPPORT
template<>
struct thrust_memory_tag<cuda_tag> {
    typedef thrust::system::cuda::tag tag;
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

void* malloc(cpp_tag, size_t);
void free(cpp_tag, void* ptr);
template<typename T>
void free(cpp_tag, T* ptr) {
    return free(cpp_tag(), (void*)ptr);
}

template<typename P>
void free(cpp_tag, P ptr) {
    return free(cpp_tag(), ptr.get());
}

#ifdef CUDA_SUPPORT
void* malloc(cuda_tag, size_t);
void free(cuda_tag, void* ptr);
template<typename T>
void free(cuda_tag, T* ptr) {
    return free(cuda_tag(), (void*)ptr);
}
template<typename P>
void free(cuda_tag, P ptr) {
    return free(cuda_tag(), ptr.get());
}

#endif


}
