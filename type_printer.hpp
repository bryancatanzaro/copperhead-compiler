#pragma once

#include <stack>

#include "monotype.hpp"
#include "polytype.hpp"
#include "ctype.hpp"

namespace backend
{

class repr_type_printer
    : public boost::static_visitor<>
{
public:
    inline repr_type_printer(std::ostream &os)
        : m_os(os)
        {}
    inline void operator()(const monotype_t &mt) {
        m_os << mt.name();
    }
    inline void operator()(const polytype_t &pt) {
        m_os << "Polytype(";
        for(auto i = pt.begin();
            i != pt.end();
            i++) {
            boost::apply_visitor(*this, *i);
            m_os << ", ";
        }
        boost::apply_visitor(*this, pt.monotype());
        m_os << ")";
    }
    inline void operator()(const sequence_t &st) {
        m_os << st.name();
        open(); boost::apply_visitor(*this, st.sub()); close();
    }
    inline void operator()(const fn_t &ft) {
        m_os << ft.name();
        open();
        boost::apply_visitor(*this, ft.args());
        sep();
        boost::apply_visitor(*this, ft.result());
        close();
    }
    inline void operator()(const tuple_t &tt) {
        m_os << tt.name();
        open();
        for(auto i = tt.begin();
            i != tt.end();
            i++) {
            boost::apply_visitor(*this, *i);
            if (std::next(i) != tt.end())
                sep();
        }
        close();
    }
        
    std::ostream &m_os;
    inline void sep() const {
        m_os << ", ";
    }
protected:
    inline void open() const {
        m_os << "(";
    }
    inline void close() const {
        m_os << ")";
    }
};

namespace ctype {
class ctype_printer
    : public boost::static_visitor<>
{
private:
    std::stack<bool> m_need_space;
public:
    inline ctype_printer(std::ostream &os)
        : m_os(os)
        {
            m_need_space.push(false);
        }
    inline void operator()(const monotype_t &mt) {
        m_os << mt.name();
        m_need_space.top() = false;
    }
    inline void operator()(const sequence_t &st) {
        m_os << st.name() << "<";
        boost::apply_visitor(*this, st.sub());
        m_os << ">";
        m_need_space.top() = true;
    }
    inline void operator()(const cuarray_t &ct) {
        //Because cuarray_t is a variant, we don't want to
        //print out the template definition.
        this->operator()((monotype_t)ct);
        m_need_space.top() = false;
    }
    inline void operator()(const polytype_t &pt) {
    }
    inline void operator()(const templated_t &tt) {
        boost::apply_visitor(*this, tt.base());
        m_os << "<";
        m_need_space.push(false);
        for(auto i = tt.begin();
            i != tt.end();
            i++) {
            boost::apply_visitor(*this, *i);
            if (std::next(i) != tt.end()) {
                m_os << ", ";
            }
        }
        //This can be removed once C++OX support is nvcc
        //It is a workaround to prevent emitting foo<bar<baz>>
        //And instead emit foo<bar<baz> > 
        if (m_need_space.top())
            m_os << " ";
        m_need_space.pop();
        m_need_space.top() = true;
        m_os << ">";
    }
    std::ostream &m_os;
    inline void sep() const {
        m_os << ", ";
    }
protected:
    inline void open() const {
        m_os << "(";
    }
    inline void close() const {
        m_os << ")";
    }
};
}
}
