#pragma once
#include <stddef.h>
#include <iostream>

template<typename T, int D>
struct sequence;

template<typename T>
struct sequence<T, 0> {
    typedef T el_type;
    typedef T& ref_type;
    typedef T* ptr_type;
    typedef size_t index_type;
    static const int nesting_depth = 0;
    
    T* m_d;
    size_t m_l;
    sequence() : m_d(NULL), m_l(0) {}
    sequence(T* d, size_t l) : m_d(d), m_l(l) {}
    T& operator[](const size_t& i) const {
        return m_d[i];
    }
    T& operator[](size_t& i) {
        return m_d[i];
    }
    const size_t& size() const {
        return m_l;
    }
};

template<typename T>
sequence<T, 0> slice(sequence<T, 0> seq, size_t base, size_t len) {
    return sequence<T, 0>(&seq[base], len);
}

template<typename T, int D>
struct sequence_base {
    typedef sequence<T, D-1> el_type;
    typedef el_type ref_type;
    typedef el_type* ptr_type;
    typedef size_t index_type;
    typedef T value_type;
    static const int nesting_depth = D;
};

template<typename T, int D=0>
struct sequence
    : public sequence_base<T, D> {
    
    sequence<size_t, 0> m_d;
    sequence<T, D-1> m_s;

    sequence() : m_d(), m_s() {}
    
    sequence(sequence<size_t, 0> d,
             sequence<T, D-1> s) : m_d(d), m_s(s) {}

    sequence<T, D-1> operator[](size_t& i) {
        size_t begin=m_d[i], end=m_d[i+1];
        return slice(m_s, begin, end-begin);
    }

    sequence<T, D-1> operator[](const size_t& i) const {
        size_t begin=m_d[i], end=m_d[i+1];
        return slice(m_s, begin, end-begin);
    }
    
    size_t size() const {
        return m_d.size() - 1;
    }
    

};


template<typename T, int D>
sequence<T, D> slice(sequence<T, D> seq, size_t base, size_t len) {
    return sequence<T, D>(slice(seq.m_d, base, len+1), seq.m_s);
}
    

template<typename T>
struct sequence<T, 1>
    : public sequence_base<T, 1> {
    sequence<size_t, 0> m_d;
    sequence<T, 0> m_s;

    sequence() : m_d(), m_s() {}

    sequence(sequence<size_t, 0> d,
             sequence<T, 0> s) : m_d(d), m_s(s) {}
    sequence<T, 0> operator[](size_t& i) {
        size_t begin=m_d[i], end=m_d[i+1];
        return slice(m_s, begin, end-begin);
    }
    sequence<T, 0> operator[](const size_t& i) const {
        size_t begin=m_d[i], end=m_d[i+1];
        return slice(m_s, begin, end-begin);
    }

    size_t size() const {
        return m_d.size()-1;
    }
};


template<typename T>
std::ostream& operator<<(std::ostream& os, const sequence<T, 0>& in) {
    os << "[";
    for(size_t i = 0; i < in.size(); i++) {
        os << in[i];
        if (i + 1 != in.size()) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

template<typename T, int D>
std::ostream& operator<<(std::ostream& os, const sequence<T, D>& in) {
    os << "[";
    for(size_t i = 0; i < in.size(); i++) {
        sequence<T, D-1> cur = in[i];
        os << cur;
        if (i + 1 != in.size()) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}
