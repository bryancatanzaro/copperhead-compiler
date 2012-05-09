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
#include <prelude/basic/detail/retagged_iterator_type.h>

/* This sequence wraps Thrust iterators into a sequence.
   This allows the use of Fancy Thrust iterators in
   Copperhead generated code.  (such as zipped, constant,
   counting, transformed, etc.)
 */

namespace copperhead {

template<typename Tag, typename I, typename IT=long>
struct iterator_sequence
{
    typedef Tag tag;
    typedef IT index_type;
    typedef typename I::value_type& ref_type;
    typedef typename I::value_type value_type;
    typedef typename I::value_type T;
    typedef typename detail::retagged_iterator_type<I, tag>::type iterator_type;
    
    I data;
    index_type length;
  
    __host__ __device__
    iterator_sequence(I _data, index_type _length) : data(_data), length(_length) {}

    __host__ __device__
    iterator_sequence(I begin, I end) : data(begin), length(end-begin) {}

    //
    // Methods supporting stream interface
    //
    __host__ __device__
    bool empty() const { return length<=0; }

    __host__ __device__
    T next()
    {
        T x = *(data++);
        --length;
        return x;
    }

    //
    // Methods supporting sequence interface
    //
    __host__ __device__
    T        operator[](index_type index)       { return data[index]; }
    __host__ __device__
    const T  operator[](index_type index) const { return data[index]; }

    __host__ __device__
    index_type size() const { return length; }

    __host__ __device__
    iterator_type begin() const {
        return thrust::retag<tag>(data);
    }
  
    __host__ __device__
    iterator_type end() const {
        return thrust::retag<tag>(data + length);
    }
};

}
