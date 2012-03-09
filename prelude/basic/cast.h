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

template<typename T>
__host__ __device__
int op_int32(const T&x) {
    return int(x);
}

template<typename T>
__host__ __device__
int op_int64(const T&x) {
    return long(x);
}

template<typename T>
__host__ __device__
int op_float32(const T&x) {
    return float(x);
}

template<typename T>
__host__ __device__
int op_float64(const T&x) {
    return double(x);
}

template<typename T, typename U>
__host__ __device__
U cast_to(const T& x, const U& y) {
    return U(x);
}

template<typename T, typename U>
__host__ __device__
typename U::value_type cast_to_el(const T& x, const U& y) {
    return typename U::value_type(x);
}


template<typename T>
class fnop_int32 {
    typedef int return_type;
    __host__ __device__
    int operator()(const T& x) {
        return op_int32(x);
    }
};


template<typename T>
class fnop_int64 {
    typedef long return_type;
    __host__ __device__
    long operator()(const T& x) {
        return op_int64(x);
    }
};

template<typename T>
class fnop_float32 {
    typedef float return_type;
    __host__ __device__
    float operator()(const T& x) {
        return op_int32(x);
    }
};

template<typename T>
class fnop_float64 {
    typedef double return_type;
    __host__ __device__
    double operator()(const T& x) {
        return op_int32(x);
    }
};

template<typename T, typename U>
class fncast_to {
    typedef U return_type;
    __host__ __device__
    U operator()(const T& x, const U& y) {
        return cast_to(x, y);
    }
};

template<typename T, typename U>
class fncast_to_el {
    typedef typename U::value_type return_type;
    __host__ __device__
    U operator()(const T& x, const U& y) {
        return cast_to_el(x, y);
    }
};

}
