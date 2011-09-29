#pragma once
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/make_constructor.hpp>
#include <vector>

#include "stored_sequence.h"

#include <numpy/arrayobject.h>
#include <iostream>
#include "np_types.h"

#include <boost/variant.hpp>
#include <stdexcept>
#include <string>
#include <sstream>


template<typename T>
class cuarray {
    stored_sequence<T> m_h;
    stored_sequence<T> m_d;
    bool clean_local;
    bool clean_remote;
public:
    cuarray() : m_h(NULL, 0), m_d(NULL, 0),
                clean_local(true), clean_remote(true) {
    }
    cuarray(ssize_t n, bool host=true) {
        clean_local = host;
        clean_remote = !host;
        T* h = new T[n];
        m_h = stored_sequence<T>(h, n);
        T* d;
        cudaMalloc(&d, sizeof(T) * n);
        m_d = stored_sequence<T>(d, n);
    }
    cuarray(ssize_t n, T* h_s) : clean_local(true), clean_remote(false) {
        T* h = new T[n];
        memcpy(h, h_s, sizeof(T) * n);
        m_h = stored_sequence<T>(h, n);
        T* d;
        cudaMalloc(&d, sizeof(T) * n);
        m_d = stored_sequence<T>(d, n);
    }
    
    cuarray(stored_sequence<T> _h,
            stored_sequence<T> _d,
            bool _local,
            bool _remote)
        : m_h(_h), m_d(_d), clean_local(_local), clean_remote(_remote) {}
    
    ~cuarray() {
        if (m_h.data != NULL)
            delete[] m_h.data;
        if (m_d.data != NULL)
            cudaFree(m_d.data);
    }
    void retrieve() {
        if (!clean_local) {
            cudaMemcpy(m_h.data, m_d.data, sizeof(T) * m_h.size(), cudaMemcpyDeviceToHost);
            clean_local = true;
        }
    }
    void exile() {
        if (!clean_remote) {
            cudaMemcpy(m_d.data, m_h.data, sizeof(T) * m_h.size(), cudaMemcpyHostToDevice);
            clean_remote = true;
        }
    }
    stored_sequence<T> get_remote_r() {
        exile();
        return m_d;
    }
    stored_sequence<T> get_remote_w() {
        exile();
        clean_local = false;
        return m_d;
    }
    stored_sequence<T> get_local_r() {
        retrieve();
        return m_h;
    }
    stored_sequence<T> get_local_w() {
        retrieve();
        clean_remote = false;
        return m_d;
    }
};



typedef boost::variant<cuarray<bool>, cuarray<int>, cuarray<long>, cuarray<float>, cuarray<double> > cuarray_var;

typedef boost::shared_ptr<cuarray_var> sp_cuarray_var;

sp_cuarray_var make_cuarray(PyObject* in) {
    if (!(PyArray_Check(in))) {
        throw std::invalid_argument("Input was not a numpy array");
    }
    NPY_TYPES dtype = NPY_TYPES(PyArray_TYPE(in));
    
    PyArrayObject *vecin = (PyArrayObject*)PyArray_ContiguousFromObject(in, dtype, 1, 1);
    if (vecin == NULL) {
        throw std::invalid_argument("Can't create CuArray from this object");
    }
    ssize_t n = vecin->dimensions[0];
    void* d = vecin->data;
    switch (dtype) {
    case NPY_BOOL:
        return sp_cuarray_var(new cuarray_var(*new cuarray<bool>(n, (bool*)d)));
    case NPY_INT:
        return sp_cuarray_var(new cuarray_var(*new cuarray<int>(n, (int*)d)));
    case NPY_LONG:
        return sp_cuarray_var(new cuarray_var(*new cuarray<long>(n, (long*)d)));
    case NPY_FLOAT:
        return sp_cuarray_var(new cuarray_var(*new cuarray<float>(n, (float*)d)));
    case NPY_DOUBLE:
        return sp_cuarray_var(new cuarray_var(*new cuarray<double>(n, (double*)d)));
    default:
        throw std::invalid_argument("Can't create CuArray from this object");
    }
}

struct repr_cuarray_printer :
    public boost::static_visitor<std::string> {
    template<typename T>
    std::string operator()(cuarray<T>& in) {
        std::ostringstream os;
        os << "cuarray<" << np_type<T>::name << ">(";
        stored_sequence<T> m_h = in.get_local_r();
        for(int i = 0; i < m_h.size(); i++) {
            os << m_h[i];
            if (i != (m_h.size() - 1))
                os << ", ";
        }
        os << ")";
        return os.str();
    }
};

std::string repr_cuarray(const sp_cuarray_var &in) {
    repr_cuarray_printer rp;
    return boost::apply_visitor(rp, *in);
}

template<typename T>
stored_sequence<T> get_remote_r(sp_cuarray_var &in) {
    return boost::get<cuarray<T> >(*in).get_remote_r();
};

template<typename T>
stored_sequence<T> get_remote_r(cuarray<T>* in) {
    return in->get_remote_r();
}

template<typename T>
stored_sequence<T> get_remote_w(sp_cuarray_var &in) {
    return boost::get< cuarray<T> >(*in).get_remote_w();
};

template<typename T>
stored_sequence<T> get_remote_w(cuarray<T>* in) {
    return in->get_remote_w();
}

template<typename T>
sp_cuarray_var make_remote(ssize_t size) {
    return sp_cuarray_var(new cuarray_var(
                              *new cuarray<T>(size, false)));
}

template<typename T>
sp_cuarray_var make_local(ssize_t size) {
    return sp_cuarray_var(new cuarray_var(
                              *new cuarray<T>(size, true)));
}
