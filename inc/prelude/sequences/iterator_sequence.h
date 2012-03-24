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

/* This sequence wraps Thrust iterators into a sequence.
   This allows the use of Fancy Thrust iterators in
   Copperhead generated code.  (such as zipped, constant,
   counting, transformed, etc.)
 */
template<typename Tag, typename I>
struct iterator_sequence
{
    typedef Tag tag;
    typedef typename I::value_type value_type;
    typedef typename I::value_type T;
    typedef I iterator_type;
    
    I data;
    int length;
  
    __host__ __device__
    iterator_sequence(I _data, int _length) : data(_data), length(_length) {}

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
    T        operator[](int index)       { return data[index]; }
    __host__ __device__
    const T  operator[](int index) const { return data[index]; }

    __host__ __device__
    int size() const { return length; }

    __host__ __device__
    I begin() const {
      return data;
    }
  
    __host__ __device__
    I end() const {
      return data + length;
    }
};

