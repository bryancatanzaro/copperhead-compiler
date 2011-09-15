#pragma once
#include <cassert>
#include <sstream>

#include "py_printer.hpp"


namespace backend
{

class cuda_printer
    : public py_printer
{
private:
    std::string& entry;
public:
    inline cuda_printer(std::string &entry_point,
                        std::ostream &os)
        : py_printer(os), entry(entry_point)
        {}
    
    using backend::py_printer::operator();

    //XXX Why do I need to qualify backend::name?
    //Am I hiding backend::name somewhere?
    //Or a compiler bug?
    inline void operator()(const backend::name &n) {
        m_os << n.id();
    }

    inline void operator()(const number &n) {
        m_os << n.val();
    }

    inline void operator()(const tuple &n) {
        open();
        list(n);
        close();
    }

    inline void operator()(const apply &n) {
        (*this)(n.fn());
        (*this)(n.args());
    }
    inline void operator()(const closure &n) {
    }
    inline void operator()(const conditional &n) {
    }
    inline void operator()(const ret &n) {
        m_os << "return ";
        boost::apply_visitor(*this, n.val());
        m_os << ";";
    }
    inline void operator()(const bind &n) {
    }
    inline void operator()(const procedure &n) {
        const std::string& proc_id = n.id().id();
        bool is_entry = proc_id == entry;
        if (!is_entry) {
            m_os << "__device__ ";
        }
        (*this)(n.id());
        (*this)(n.args());
        m_os << " {" << std::endl;
        indent();
        (*this)(n.stmts());
        dedent();
        indentation();
        m_os << "}" << std::endl;
    }
    inline void operator()(const suite &n) {
        for(auto i = n.begin();
            i != n.end();
            i++) {
            indentation();
            boost::apply_visitor(*this, *i);
            m_os << std::endl;
        }
    }

    inline void operator()(const structure &n) {
        indentation();
        m_os << "struct ";
        (*this)(n.id());
        m_os << " {" << std::endl;
        indent();
        (*this)(n.stmts());
        dedent();
        m_os << "};" << std::endl;
    }
    inline void operator()(const std::string &s) {
        m_os << s;
    }
    template<typename T>
        inline void operator()(const std::vector<T> &v) {
        list(v);
    }
};


}
