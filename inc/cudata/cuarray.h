#pragma once
#include "../../library/prelude/stored_sequence.h"
//#include "np_types.h"

//This class exists to isolate anything that touches CUDA
//And make sure the host compiler doesn't need to see it.
template<typename T>
class cuarray_impl {
    friend cuarray<T>;
  private:
    stored_sequence<T> m_h;
    stored_sequence<T> m_d;
    bool clean_local;
    bool clean_remote;
  public:
    void retrieve() {
        //Lazy data movement
        if (!clean_local) {
            assert(m_d.data != NULL);
            //Lazy allocation
            if (m_h.data == NULL) {
                int size = m_d.size();
                m_h = stored_sequence<T>(new T[size],
                                         size);
            }
            
            cudaMemcpy(m_h.data, m_d.data, sizeof(T) * m_h.size(), cudaMemcpyDeviceToHost);
            clean_local = true;
        }
    }
    void exile() {
        //Lazy data movement
        if (!clean_remote) {
            assert(m_h.data != NULL);
            //Lazy allocation
            if (m_d.data == NULL) {
                int size = m_h.size();
                T* remote_data;
                cudaMalloc(&remote_data, sizeof(T) * size);
                m_d = stored_sequence<T>(remote_data, size);
            }
            cudaMemcpy(m_d.data, m_h.data, sizeof(T) * m_h.size(), cudaMemcpyHostToDevice);
            clean_remote = true;
        }
    }
    stored_sequence<T> get_local_r() {
        retrieve();
        return m_h;
    }
    stored_sequence<T> get_local_w() {
        retrieve();
        clean_remote = false;
        return m_h;
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
};

template<typename T>
cuarray<T>::cuarray() : m_impl(new cuarray_impl<T>()){
        m_impl->m_h = stored_sequence<T>(NULL, 0);
        m_impl->m_d = stored_sequence<T>(NULL, 0);
        m_impl->clean_local = true;
        m_impl->clean_remote = true;
}

template<typename T>
cuarray<T>::cuarray(ssize_t n, bool host) : m_impl(new cuarray_impl<T>()) {
    m_impl->clean_local = host;
    m_impl->clean_remote = !host;
    //Lazy allocation
    if (m_impl->clean_local) {
        T* h = new T[n];
        m_impl->m_h = stored_sequence<T>(h, n);
        m_impl->m_d = stored_sequence<T>(NULL, 0);
    } else {
        T* d;
        cudaMalloc(&d, sizeof(T) * n);
        m_impl->m_d = stored_sequence<T>(d, n);
        m_impl->m_h = stored_sequence<T>(NULL, 0);    
    }
}

template<typename T>
cuarray<T>::cuarray(ssize_t n, T* h_s) : m_impl(new cuarray_impl<T>()) {
    m_impl->clean_local = true;
    m_impl->clean_remote = false;
    T* h = new T[n];
    memcpy(h, h_s, sizeof(T) * n);
    m_impl->m_h = stored_sequence<T>(h, n);
    m_impl->m_d = stored_sequence<T>(NULL, 0);
}
    
    /* cuarray(stored_sequence<T> _h, */
    /*         stored_sequence<T> _d, */
    /*         bool _local, */
    /*         bool _remote) */
    /*     : m_h(_h), m_d(_d), clean_local(_local), clean_remote(_remote) { */
    /* } */


template<typename T>
cuarray<T>::~cuarray() {
    if (m_impl->m_h.data != NULL)
        delete[] m_impl->m_h.data;
    if (m_impl->m_d.data != NULL)
        cudaFree(m_impl->m_d.data);
}

template<typename T>
cuarray<T>::cuarray(const cuarray<T>& r) : m_impl(new cuarray_impl<T>()) {
    //copy all the state from the other cuarray
    *m_impl = *(r.m_impl);
}

template<typename T>
void cuarray<T>::swap(cuarray<T>& r) {
    m_impl.swap(r.m_impl);    
}

template<typename T>
cuarray<T>& cuarray<T>::operator=(const cuarray<T>& r) {
    cuarray<T> temp(r);
    swap(temp);
    return *this;
}






template class cuarray<bool>;
template class cuarray<int>;
template class cuarray<long>;
template class cuarray<float>;
template class cuarray<double>;



template<typename T>
stored_sequence<T> get_remote_r(sp_cuarray_var &in) {
    return boost::get<cuarray<T> >(*in).m_impl->get_remote_r();
};

template<typename T>
stored_sequence<T> get_remote_w(sp_cuarray_var &in) {
    return boost::get< cuarray<T> >(*in).m_impl->get_remote_w();
};

template<typename T>
stored_sequence<T> get_local_r(sp_cuarray_var &in) {
    return boost::get<cuarray<T> >(*in).m_impl->get_local_r();
};

template<typename T>
stored_sequence<T> get_local_w(sp_cuarray_var &in) {
    return boost::get< cuarray<T> >(*in).m_impl->get_local_w();
};


template<typename T>
stored_sequence<T> get_remote_r(boost::shared_ptr<cuarray<T> > &in) {
    return in->m_impl->get_remote_r();
};

template<typename T>
stored_sequence<T> get_remote_w(boost::shared_ptr<cuarray<T> > &in) {
    return in->m_impl->get_remote_w();
};

template<typename T>
stored_sequence<T> get_local_r(boost::shared_ptr<cuarray<T> > &in) {
    return in->m_impl->get_local_r();
};

template<typename T>
stored_sequence<T> get_local_w(boost::shared_ptr<cuarray<T> > &in) {
    return in->m_impl->get_local_w();
};


const char* bool_n = "bool";
const char* int_n = "int";
const char* long_n = "long";
const char* float_n = "float";
const char* double_n = "double";

const char* name_this_type(bool) {
    return bool_n;
}

const char* name_this_type(int) {
    return int_n;
}

const char* name_this_type(long) {
    return long_n;
}

const char* name_this_type(float) {
    return float_n;
}

const char* name_this_type(double) {
    return double_n;
}


struct repr_cuarray_printer :
    public boost::static_visitor<std::string> {
    template<typename T>
    std::string operator()(cuarray<T>& in) {
        std::ostringstream os;
        T ex = 0;
        os << "cuarray<" << name_this_type(ex) << ">(";
        stored_sequence<T> m_h = in.m_impl->get_local_r();
        for(int i = 0; i < m_h.size(); i++) {
            os << m_h[i];
            if (i != (m_h.size() - 1))
                os << ", ";
        }
        os << ")";
        return os.str();
    }
};


namespace detail {

template<typename T>
class release_deleter{
public:
    release_deleter() : m_released(false) {}
    void release() {m_released = true;}
    void operator()(T* ptr){if(!m_released) delete ptr;}
private:
    bool m_released;
};

template<typename T>
T* release(boost::shared_ptr<T>& in) {
    release_deleter<T>* deleter = boost::get_deleter<
        release_deleter<T> >(in);
    deleter->release();
    return in.get();
}

}

template<typename T>
boost::shared_ptr<cuarray<T> > make_remote(ssize_t size) {
    return boost::shared_ptr<cuarray<T> >(
        new cuarray<T>(size, false),
        detail::release_deleter<cuarray<T> >());
}

template<typename T>
boost::shared_ptr<cuarray<T> > make_local(ssize_t size) {
    return boost::shared_ptr<cuarray<T> >(
        new cuarray<T>(size, true),
        detail::release_deleter<cuarray<T> >());
}

template<typename T>
sp_cuarray_var wrap_cuarray(boost::shared_ptr<cuarray<T> >& in) {
    return sp_cuarray_var(new cuarray_var(
                              *detail::release(in)));
}
