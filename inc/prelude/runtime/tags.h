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


//Fake tags exist due to a sad chain of incompatibilities.
//Once nvcc can digest std::shared_ptr, nvcc can be used to compile
//The entire backend, which will render the fake tag scheme unnecessary.
//As it is, since g++ can't be shown thrust::system::cuda::tag, and
//since nvcc can't be shown std::shared_ptr, we are at an impasse,
//and must mediate things through an enum.
#include <prelude/runtime/fake_tags.h>

//Convert from real tag to fake tag
namespace copperhead {
namespace detail {

struct real_to_fake_tag_converter
    : public boost::static_visitor<fake_system_tag> {
    fake_system_tag operator()(omp_tag) const {
        return fake_omp_tag;
    }
#ifdef CUDA_SUPPORT
    fake_system_tag operator()(cuda_tag) const {
        return fake_cuda_tag;
    }
#endif
};

fake_system_tag real_to_fake_tag_convert(copperhead::system_variant real_tag) {
    return boost::apply_visitor(real_to_fake_tag_converter(), real_tag);
}

}
}
        
