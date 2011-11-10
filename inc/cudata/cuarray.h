#pragma once
#include "../../library/prelude/stored_sequence.h"

template<typename T>
struct cuarray_data {
    stored_sequence<T> m_h;
    stored_sequence<T> m_d;
    bool clean_local;
    bool clean_remote;
};

template<typename T>
cuarray::cuarray() {
        m_data = new cuarray_data<T>();
        m_data->m_h = stored_sequence<T>(NULL, 0);
        m_data->m_d = stored_sequence<T>(NULL, 0);
        m_data->clean_local = true;
        m_data->clean_remote = true;
}

template<typename T>
cuarray::cuarray(ssize_t n, bool host=true) {
    m_data = new cuarray_data<T>();
    m_data->clean_local = host;
    m_data->clean_remote = !host;
    //Lazy allocation
    if (m_data->clean_local) {
        T* h = new T[n];
        m_data->m_h = stored_sequence<T>(h, n);
        m_data->m_d = stored_sequence<T>(NULL, 0);
    } else {
        T* d;
        cudaMalloc(&d, sizeof(T) * n);
        m_data->m_d = stored_sequence<T>(d, n);
        m_data->m_h = stored_sequence<T>(NULL, 0);    
    }
}

template<typename T>
cuarray::cuarray(ssize_t n, T* h_s) {
    m_data = new cuarray_data<T>();
    m_data->clean_local = true;
    m_data->clean_remote = false;
    T* h = new T[n];
    memcpy(h, h_s, sizeof(T) * n);
    m_data->m_h = stored_sequence<T>(h, n);
    m_data->m_d = stored_sequence<T>(NULL, 0);
}
    
    /* cuarray(stored_sequence<T> _h, */
    /*         stored_sequence<T> _d, */
    /*         bool _local, */
    /*         bool _remote) */
    /*     : m_h(_h), m_d(_d), clean_local(_local), clean_remote(_remote) { */
    /* } */

template<typename T>
cuarray::~cuarray() {
    if (m_data->m_h.data != NULL)
        delete[] m_data->m_h.data;
    if (m_data->m_d.data != NULL)
        cudaFree(m_data->m_d.data);
    delete m_data;
}

template<typename T>
void cuarray::retrieve() {
    //Lazy data movement
    if (!m_data->clean_local) {
        assert(m_data->m_d.data != NULL);
        //Lazy allocation
        if (m_data->m_h.data == NULL) {
            int size = m_data->m_d.size();
            m_data->m_h = stored_sequence<T>(new T[size],
                                             size);
        }
        
        cudaMemcpy(m_data->m_h.data, m_data->m_d.data, sizeof(T) * m_data->m_h.size(), cudaMemcpyDeviceToHost);
        m_data->clean_local = true;
    }
}

template<typename T>
void cuarray::exile() {
    //Lazy data movement
    if (!m_data->clean_remote) {
        assert(m_data->m_h.data != NULL);
        //Lazy allocation
        if (m_data->m_d.data == NULL) {
            int size = m_data->m_h.size();
            T* remote_data;
            cudaMalloc(&remote_data, sizeof(T) * size);
            m_data->m_d = stored_sequence<T>(remote_data, size);
        }
        cudaMemcpy(m_data->m_d.data, m_data->m_h.data, sizeof(T) * m_data->m_h.size(), cudaMemcpyHostToDevice);
        m_data->clean_remote = true;
    }
}


extern template class cuarray<bool>;
extern template class cuarray<int>;
extern template class cuarray<long>;
extern template class cuarray<float>;
extern template class cuarray<double>;

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


template<typename T>
stored_sequence<T> get_remote_r(sp_cuarray_var &in) {
    return boost::get<cuarray<T> >(*in).get_remote_r();
};

template<typename T>
stored_sequence<T> get_remote_w(sp_cuarray_var &in) {
    return boost::get< cuarray<T> >(*in).get_remote_w();
};

template<typename T>
stored_sequence<T> get_local_r(sp_cuarray_var &in) {
    return boost::get<cuarray<T> >(*in).get_local_r();
};

template<typename T>
stored_sequence<T> get_local_w(sp_cuarray_var &in) {
    return boost::get< cuarray<T> >(*in).get_local_w();
};


template<typename T>
stored_sequence<T> get_remote_r(boost::shared_ptr<cuarray<T> > &in) {
    return in->get_remote_r();
};

template<typename T>
stored_sequence<T> get_remote_w(boost::shared_ptr<cuarray<T> > &in) {
    return in->get_remote_w();
};

template<typename T>
stored_sequence<T> get_local_r(boost::shared_ptr<cuarray<T> > &in) {
    return in->get_local_r();
};

template<typename T>
stored_sequence<T> get_local_w(boost::shared_ptr<cuarray<T> > &in) {
    return in->get_local_w();
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
