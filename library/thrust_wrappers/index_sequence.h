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
#include "iterator_sequence.h"
#include <thrust/iterator/counting_iterator.h>

struct index_sequence : public iterator_sequence<thrust::counting_iterator<long> >
{
  __host__ __device__ index_sequence(long length) :
  iterator_sequence<thrust::counting_iterator<long> >(thrust::counting_iterator<long>(0), length) {}
};
    
