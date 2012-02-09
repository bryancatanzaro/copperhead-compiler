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
#include <vector>

#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>
#define BOOST_SP_USE_SPINLOCK
#include <boost/shared_ptr.hpp>

//For dynamically typing the contents of a cuarray
//This makes integration with dynamically typed languages easier.
enum CUTYPE {
    CUVOID=0,
    CUBOOL,
    CUINT32,
    CUINT64,
    CUFLOAT32,
    CUFLOAT64
};

class cuarray {
public:
    void* l_d;
    void* r_d;
    ssize_t n;
    ssize_t e;
    ssize_t s;
    CUTYPE t;
private:
    bool clean_local;
    bool clean_remote;
public:
    cuarray();
    ~cuarray();
    cuarray(ssize_t n, CUTYPE t, bool host=true);
    cuarray(ssize_t n, bool* l);
    cuarray(ssize_t n, int* l);
    cuarray(ssize_t n, long* l);
    cuarray(ssize_t n, float* l);
    cuarray(ssize_t n, double* l);
    std::pair<void*, ssize_t> get_local_r();
    std::pair<void*, ssize_t> get_local_w();
    std::pair<void*, ssize_t> get_remote_r();
    std::pair<void*, ssize_t> get_remote_w();
private:
    cuarray(const cuarray&);
    void retrieve();
    void exile();    
};

typedef boost::shared_ptr<cuarray> sp_cuarray;

template<typename T>
sp_cuarray make_remote(ssize_t in);

template<>
sp_cuarray make_remote<bool>(ssize_t in);

template<>
sp_cuarray make_remote<int>(ssize_t in);

template<>
sp_cuarray make_remote<long>(ssize_t in);

template<>
sp_cuarray make_remote<float>(ssize_t in);

template<>
sp_cuarray make_remote<double>(ssize_t in);
