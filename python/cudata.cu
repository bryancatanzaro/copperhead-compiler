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
// template<typename T>
// static boost::shared_ptr<stored_sequence<T> > make_stored_sequence_l(const list& vals) {
//     boost::python::ssize_t n = len(vals);
//     T* data_store = (T*)malloc(sizeof(T) * n);
    
//     for(boost::python::ssize_t i=0; i<n; i++) {
//         object elem = vals[i];
//         data_store[i] = extract<T>(elem);
//     }
//     T* cuda_store;
//     cudaMalloc(&T, sizeof(T) * n);
//     cudaMemcpy(cuda_store, data_store, sizeof(T) * n, cudaMemcpyDeviceToHost);
//     return boost::shared_ptr<stored_sequence<T> >(new stored_sequence<T>(cuda_store, n));
// }


template<typename T>
struct cuarray {
    ssize_t m_n;
    T* m_h;
    stored_sequence<T> m_d;
    cuarray() : m_n(0), m_h(NULL), m_d(NULL, 0) {
    }
        
    cuarray(ssize_t n, T* h) : m_n(n) {
        m_h = new T[m_n];
        memcpy(m_h, h, sizeof(T) * n);
        T* d;
        cudaMalloc(&d, sizeof(T) * n);
        cudaMemcpy(d, m_h, sizeof(T) * n, cudaMemcpyHostToDevice);
        m_d=stored_sequence<T>(d, n);
    }
    ~cuarray() {
        if (m_h != NULL)
            delete[] m_h;
        if (m_d.data != NULL)
            cudaFree(m_d.data);
    }
};



typedef boost::variant<boost::shared_ptr<cuarray<bool> >, boost::shared_ptr<cuarray<int> >, boost::shared_ptr<cuarray<long> >, boost::shared_ptr<cuarray<float> >, boost::shared_ptr<cuarray<double> > > cuarray_var;



boost::shared_ptr<cuarray_var> make_cuarray(PyObject* in) {
    NPY_TYPES dtype = NPY_TYPES(PyArray_TYPE(in));
    
    PyArrayObject *vecin = (PyArrayObject*)PyArray_ContiguousFromObject(in, dtype, 1, 1);
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
    std::string operator()(const boost::shared_ptr<cuarray<bool> >& in) {
        std::ostringstream os;
        os << "cuarray<bool>, length: " << in->m_n << ", host: " << in->m_h << ", device: " << in->m_d.data;
        return os.str();
    }
    std::string operator()(const boost::shared_ptr<cuarray<int> >& in) {
        std::ostringstream os;
        os << "cuarray<int>, length: " << in->m_n << ", host: " << in->m_h << ", device: " << in->m_d.data;
        return os.str();
    }
    std::string operator()(const boost::shared_ptr<cuarray<long> >& in) {
        std::ostringstream os;
        os << "cuarray<long>, length: " << in->m_n << ", host: " << in->m_h << ", device: " << in->m_d.data;
        return os.str();
    }
    std::string operator()(const boost::shared_ptr<cuarray<float> >& in) {
        std::ostringstream os;
        os << "cuarray<float>, length: " << in->m_n << ", host: " << in->m_h << ", device: " << in->m_d.data;
        return os.str();
    }
    std::string operator()(const boost::shared_ptr<cuarray<double> >& in) {
        std::ostringstream os;
        os << "cuarray<double>, length: " << in->m_n << ", host: " << in->m_h << ", device: " << in->m_d.data;
        return os.str();
    }
};

std::string repr_cuarray(const boost::shared_ptr<cuarray_var> &in) {
    repr_cuarray_printer rp;
    return boost::apply_visitor(rp, *in);
}

__global__ void test_kernel(stored_sequence<float> a) {
    if (threadIdx.x == 0) {
        a[1] = 2.78;
    } else if (threadIdx.x == 1) {
        a[0] = 3.14;
    }
}

void test(const boost::shared_ptr<cuarray_var>& in) {
    stored_sequence<float> seq = boost::get<boost::shared_ptr<cuarray<float> > >(*in)->m_d;
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
