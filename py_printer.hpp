#pragma once
#include <cassert>
#include <sstream>

#include "node.hpp"
#include "expression.hpp"
#include "statement.hpp"
#include "cppnode.hpp"

namespace backend
{
namespace detail {
template<typename Visitor,
         typename Iterable>
inline void list(Visitor& v,
                 const Iterable &l) {
    for(auto i = l.begin();
        i != l.end();
        i++) {
        boost::apply_visitor(v, *i);
        if (std::next(i) != l.end())
            v.sep();
    }
}
}

class py_printer
    : public no_op_visitor<>
{
public:
    inline py_printer(std::ostream &os)
        : m_os(os), indent_level(0), indent_atom("    "),
          current_indent("")
        {}
// XXX why do we have to use 'using' to make the base class's overloads visible?
    //using backend::no_op_visitor<>::operator();

    inline void operator()(const name &n) {
        m_os << n.id();
    }

    inline void operator()(const number &n) {
        m_os << n.val();
    }

    inline void operator()(const tuple &n) {
        open();
        detail::list(*this, n);
        close();
    }

    inline void operator()(const apply &n) {
        (*this)(n.fn());
        (*this)(n.args());
    }
    
    inline void operator()(const lambda &n) {
        m_os << "lambda ";
        detail::list(*this, n.args());
        m_os << ": ";
        boost::apply_visitor(*this, n.body());
    }
    inline void operator()(const closure &n) {
    }
    inline void operator()(const conditional &n) {
    }
    inline void operator()(const ret &n) {
        m_os << "return ";
        boost::apply_visitor(*this, n.val());
    }
    inline void operator()(const bind &n) {
    }
    inline void operator()(const procedure &n) {
        m_os << "def ";
        (*this)(n.id());
        (*this)(n.args());
        m_os << ":" << std::endl;
        indent();
        (*this)(n.stmts());
        dedent();
        
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
        m_os << "class ";
        (*this)(n.id());
        m_os << ":" << std::endl;
        indent();
        (*this)(n.stmts());
        dedent();
    }
    inline void operator()(const std::string &s) {
        m_os << s;
    }
    template<typename T>
        inline void operator()(const std::vector<T> &v) {
        detail::list(this, v);
    }

    inline void sep(void) const {
        m_os << ", ";
    }
    inline void open(void) const {
        m_os << "(";
    }
    inline void close(void) const {
        m_os << ")";
    }
    protected:
    template<typename Value>
        inline void name(const std::string &n,
                         const Value &val) {
        m_os << n;
        open();
        (*this)(val);
        close();
    }

    std::ostream &m_os;
    int indent_level;
    std::string indent_atom;
    std::string current_indent;
    inline void indent(int amount=1) {
        indent_level += amount;
        assert(indent_level > -1);
        std::stringstream ss;
        std::fill_n(std::ostream_iterator<std::string>(ss),
                    indent_level,
                    indent_atom);
        current_indent = ss.str();
    }
    inline void dedent() {
        indent(-1);
    }
    inline void indentation() {
        m_os << current_indent;
    }
    
    
};


}
