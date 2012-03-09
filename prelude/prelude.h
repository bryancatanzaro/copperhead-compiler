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
#define BOOST_SP_USE_SPINLOCK
#include <boost/shared_ptr.hpp>


#ifdef CUDA_SUPPORT
#define THRUST_SYSTEM_DEVICE_CUDA
#else
#define THRUST_SYSTEM_DEVICE_OMP
#endif
#include <thrust/detail/config.h>

#include <prelude/basic/basic.h>
#include <prelude/sequences/sequence.h>
#include <prelude/sequences/uniform_sequence.h>

#include <prelude/primitives/phase_boundary.h>
