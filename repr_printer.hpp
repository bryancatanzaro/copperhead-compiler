#pragma once

namespace backend
{

class repr_printer
    : public no_op_visitor<>
{
public:
    inline repr_printer(std::ostream &os)
        : m_os(os)
        {}

    // XXX why do we have to use 'using' to make the base class's overloads visible?
    using backend::no_op_visitor<>::operator();

    inline void operator()(const name &n) {
        name("Name", n.id());
    }

    inline void operator()(const literal &n) {
        name("Literal", n.id());
    }

    inline void operator()(const tuple &n) {
        m_os << "Tuple";
        open();
        list(n);
        close();
    }

    inline void operator()(const apply &n) {
        m_os << "Apply(";
        (*this)(n.fn());
        sep();
        (*this)(n.args());
        m_os << ")";
    }
    
    inline void operator()(const lambda &n) {
        m_os << "Lambda(";
        (*this)(n.args());
        sep();
        boost::apply_visitor(*this, n.body());
        m_os << ")";
    }
   
    inline void operator()(const ret &n) {
        m_os << "Return(";
        boost::apply_visitor(*this, n.val());
        m_os << ")";
    }
    inline void operator()(const procedure &n) {
        m_os << "Procedure(";
        (*this)(n.id());
        sep();
        (*this)(n.args());
        sep();
        (*this)(n.stmts());
        m_os << ")";
        
    }
    inline void operator()(const suite &n) {
        m_os << "Suite(";
        list(n);
        m_os << ")";
    }
    inline void operator()(const include &n) {
        m_os << "Include(";
        boost::apply_visitor(*this, n.id());
        m_os << ")";
    }
        
    inline void operator()(const std::string &s) {
        m_os << s;
    }
    template<typename T>
        inline void operator()(const std::vector<T> &v) {
        list(v);
    }
private:
    inline void sep(void) const {
        m_os << ", ";
    }
    inline void open(void) const {
        m_os << "(";
    }
    inline void close(void) const {
        m_os << ")";
    }
    template<typename Value>
        inline void name(const std::string &n,
                         const Value &val) {
        m_os << n;
        open();
        (*this)(val);
        close();
    }
    template<typename Iterable>
    inline void list(const Iterable &l,
                     const std::string sep = ", ") {
        for(auto i = l.begin();
            i != l.end();
            i++) {
            boost::apply_visitor(*this, *i);
            if (std::next(i) != l.end())
                m_os << sep;
        }
    }
    
    std::ostream &m_os;
};

}

