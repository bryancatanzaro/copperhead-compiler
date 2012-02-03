#pragma once
#include <thrust/device_ptr.h>
#include "cudata.h"

template<typename T>
struct stored_sequence 
{
    typedef T value_type;
    typedef T& reference;
    //XXX This should be templated
    typedef int index_type;
    T *data;
    int length;

    __host__ __device__
    stored_sequence() : data(NULL), length(0) {}

    __host__ __device__
    stored_sequence(T *_data, int _length) : data(_data), length(_length) {}

    __host__ __device__
    stored_sequence(T *begin, T *end) : data(begin), length(end-begin) {}

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
    T&       operator[](int index)       { return data[index]; }
    __host__ __device__
    const T& operator[](int index) const { return data[index]; }

    __host__ __device__
    int size() const { return length; }


    typedef thrust::device_ptr<T> iterator_type;
    iterator_type begin() const {
        return iterator_type(data);
    }
    iterator_type end() const {
        return iterator_type(data + length);
    }
};


template<typename T>
__host__ __device__
stored_sequence<T> slice(stored_sequence<T> seq, int base, int len)
{
    return stored_sequence<T>(&seq[base], len);
}


template<typename T>
stored_sequence<T> get_remote_w(boost::shared_ptr<cuarray>& in) {
    std::pair<void*, ssize_t> info = in->get_remote_w();
    return stored_sequence<T>(reinterpret_cast<T*>(info.first),
                              info.second);
}

template<typename T>
stored_sequence<T> get_remote_r(boost::shared_ptr<cuarray>& in) {
    std::pair<void*, ssize_t> info = in->get_remote_r();
    return stored_sequence<T>(reinterpret_cast<T*>(info.first),
                              info.second);
}
