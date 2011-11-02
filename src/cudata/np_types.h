#pragma once

#include <numpy/arrayobject.h>

template<typename T>
struct np_type {};

template<>
struct np_type<bool> {
    static const NPY_TYPES value = NPY_BOOL;
    static const char* name;
};

template<>
struct np_type<int> {
    static const NPY_TYPES value = NPY_INT;
    static const char* name;
};

template<>
struct np_type<long> {
    static const NPY_TYPES value = NPY_LONG;
    static const char* name;
};

template<>
struct np_type<float> {
    static const NPY_TYPES value = NPY_FLOAT;
    static const char* name;
};

template<>
struct np_type<double> {
    static const NPY_TYPES value = NPY_DOUBLE;
    static const char* name;
};

const char* np_type<bool>::name = "bool";
const char* np_type<int>::name = "int";
const char* np_type<long>::name = "long";
const char* np_type<float>::name = "float";
const char* np_type<double>::name = "double";


template<NPY_TYPES N>
struct c_type{};

template<>
struct c_type<NPY_BOOL> {
    typedef bool type;
};

template<>
struct c_type<NPY_INT> {
    typedef int type;
};

template<>
struct c_type<NPY_LONG> {
    typedef long type;
};

template<>
struct c_type<NPY_FLOAT> {
    typedef float type;
};

template<>
struct c_type<NPY_DOUBLE> {
    typedef double type;
};
