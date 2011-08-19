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
    cuarray(ssize_t n) : clean_local(true), clean_remote(true) {
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
    const stored_sequence<T> get_remote_r() {
        exile();
        return m_d;
    }
    stored_sequence<T> get_remote_w() {
        exile();
        clean_local = false;
        return m_d;
    }
    const stored_sequence<T> get_local_r() {
        retrieve();
        return m_h;
    }
    stored_sequence<T> get_local_w() {
        retrieve();
        clean_remote = false;
        return m_d;
    }
};



typedef boost::variant<boost::shared_ptr<cuarray<bool> >, boost::shared_ptr<cuarray<int> >, boost::shared_ptr<cuarray<long> >, boost::shared_ptr<cuarray<float> >, boost::shared_ptr<cuarray<double> > > cuarray_var;



boost::shared_ptr<cuarray_var> make_cuarray(PyObject* in) {
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
        return boost::shared_ptr<cuarray_var>(new cuarray_var(boost::shared_ptr<cuarray<bool> >(new cuarray<bool>(n, (bool*)d))));
    case NPY_INT:
        return boost::shared_ptr<cuarray_var>(new cuarray_var(boost::shared_ptr<cuarray<int> >(new cuarray<int>(n, (int*)d))));
    case NPY_LONG:
        return boost::shared_ptr<cuarray_var>(new cuarray_var(boost::shared_ptr<cuarray<long> >(new cuarray<long>(n, (long*)d))));
    case NPY_FLOAT:
        return boost::shared_ptr<cuarray_var>(new cuarray_var(boost::shared_ptr<cuarray<float> >(new cuarray<float>(n, (float*)d))));
    case NPY_DOUBLE:
        return boost::shared_ptr<cuarray_var>(new cuarray_var(boost::shared_ptr<cuarray<double> >(new cuarray<double>(n, (double*)d))));
    default:
        throw std::invalid_argument("Can't create CuArray from this object");
    }
}

struct repr_cuarray_printer :
    public boost::static_visitor<std::string> {
    template<typename T>
    std::string operator()(const boost::shared_ptr<cuarray<T> >& in) {
        std::ostringstream os;
        os << "cuarray<" << np_type<T>::name << ">(";
        stored_sequence<T> m_h = in->get_local_r();
        for(int i = 0; i < m_h.size(); i++) {
            os << m_h[i];
            if (i != (m_h.size() - 1))
                os << ", ";
        }
        os << ")";
        return os.str();
    }
};

std::string repr_cuarray(const boost::shared_ptr<cuarray_var> &in) {
    repr_cuarray_printer rp;
    return boost::apply_visitor(rp, *in);
}

__global__ void test_kernel(stored_sequence<float> a) {
    if (threadIdx.x == 0) {
        a[1] = a[1] + 2.78;
    } else if (threadIdx.x == 1) {
        a[0] = a[0] + 3.14;
    }
}

void test(const boost::shared_ptr<cuarray_var>& in) {
    stored_sequence<float> seq = boost::get<boost::shared_ptr<cuarray<float> > >(*in)->get_remote_w();
    test_kernel<<<1, 2>>>(seq);
}


    
BOOST_PYTHON_MODULE(cudata) {
    import_array();
    using namespace boost::python;
    class_<cuarray_var, boost::shared_ptr<cuarray_var> >("CuArray", no_init)
        .def("__init__", make_constructor(make_cuarray))
        .def("__repr__", repr_cuarray);
    def("test", test);
}
