#include "repr_printer.hpp"

namespace backend
{

repr_printer::repr_printer(std::ostream &os)
    : m_os(os) {}

void repr_printer::operator()(const name &n) {
    print_name("Name", n.id());
}
    
void repr_printer::operator()(const literal &n) {
    print_name("Literal", n.id());
}

void repr_printer::operator()(const tuple &n) {
    m_os << "Tuple";
    open();
    list(n);
    close();
}

void repr_printer::operator()(const bind &n) {
    m_os << "Bind";
    open();
    boost::apply_visitor(*this, n.lhs());
    sep();
    boost::apply_visitor(*this, n.rhs());
    close();
}

void repr_printer::operator()(const apply &n) {
    m_os << "Apply(";
    (*this)(n.fn());
    sep();
    (*this)(n.args());
    m_os << ")";
}
    
void repr_printer::operator()(const lambda &n) {
    m_os << "Lambda(";
    (*this)(n.args());
    sep();
    boost::apply_visitor(*this, n.body());
    m_os << ")";
}
void repr_printer::operator()(const closure &n) {
    m_os << "Closure(";
    (*this)(n.args());
    sep();
    boost::apply_visitor(*this, n.body());
    m_os << ")";
}
    
void repr_printer::operator()(const ret &n) {
    m_os << "Return(";
    boost::apply_visitor(*this, n.val());
    m_os << ")";
}
void repr_printer::operator()(const procedure &n) {
    m_os << "Procedure(";
    (*this)(n.id());
    sep();
    (*this)(n.args());
    sep();
    (*this)(n.stmts());
    m_os << ")";
        
}
void repr_printer::operator()(const suite &n) {
    m_os << "Suite(";
    list(n);
    m_os << ")";
}
void repr_printer::operator()(const include &n) {
    m_os << "Include(";
    boost::apply_visitor(*this, n.id());
    m_os << ")";
}

void repr_printer::operator()(const typedefn &n) {
    m_os << "Typedef()";
}
    
void repr_printer::operator()(const conditional &n) {
    m_os << "Conditional(";
    boost::apply_visitor(*this, n.cond());
    m_os << ", ";
    boost::apply_visitor(*this, n.then());
    m_os << ", ";
    boost::apply_visitor(*this, n.orelse());
    m_os << ")";
}
    
void repr_printer::operator()(const std::string &s) {
    m_os << s;
}

void repr_printer::sep(void) const {
    m_os << ", ";
}
void repr_printer::open(void) const {
    m_os << "(";
}
void repr_printer::close(void) const {
    m_os << ")";
}

}

