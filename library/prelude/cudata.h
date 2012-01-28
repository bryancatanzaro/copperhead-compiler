#pragma once
#include <vector>

#include <iostream>

#include <boost/variant.hpp>
#include <stdexcept>
#include <string>
#include <sstream>
#include <boost/shared_ptr.hpp>

//The private data members of cuarray have to be hidden
//in order to separate CUDA stuff from the host C++ compiler
//This is necessary when, for example, nvcc can't parse
//The host bindings necessary to bind cuarray objects into
//A host language.  As of NVCC 4.1, this is the case with
//boost::python.  Segregating all the data structures which
//NVCC must see allows this to work.
template<typename T>
class cuarray_impl;

template<typename T>
class cuarray {
  public:
    cuarray_impl<T>* m_impl;
  public:
    cuarray();
    ~cuarray();
    cuarray(ssize_t n, bool host=true);
    cuarray(ssize_t n, T*);
    cuarray(const cuarray<T>& r);
    cuarray& operator=(const cuarray<T>& r);
    T& operator[](const ssize_t& index);
    void swap(cuarray<T>& r);
    T* get_view();
    ssize_t get_size() const;
    
};

typedef boost::variant<cuarray<bool>, cuarray<int>, cuarray<long>, cuarray<float>, cuarray<double> > cuarray_var;

typedef boost::shared_ptr<cuarray_var> sp_cuarray_var;

sp_cuarray_var make_cuarray(ssize_t n, bool* d);
sp_cuarray_var make_cuarray(ssize_t n, int* d);
sp_cuarray_var make_cuarray(ssize_t n, long* d);
sp_cuarray_var make_cuarray(ssize_t n, float* d);
sp_cuarray_var make_cuarray(ssize_t n, double* d);

std::string repr_cuarray(const sp_cuarray_var &in);

