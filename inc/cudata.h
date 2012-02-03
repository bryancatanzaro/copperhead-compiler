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

template<typename T>
boost::shared_ptr<cuarray> make_remote(ssize_t in);

template<>
boost::shared_ptr<cuarray> make_remote<bool>(ssize_t in);

template<>
boost::shared_ptr<cuarray> make_remote<int>(ssize_t in);

template<>
boost::shared_ptr<cuarray> make_remote<long>(ssize_t in);

template<>
boost::shared_ptr<cuarray> make_remote<float>(ssize_t in);

template<>
boost::shared_ptr<cuarray> make_remote<double>(ssize_t in);
