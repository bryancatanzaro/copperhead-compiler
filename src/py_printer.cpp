#include "py_printer.hpp"

namespace backend
{

py_printer::py_printer(std::ostream &os)
    : m_os(os), indent_level(0), indent_atom("    "),
      current_indent("") {}

void py_printer::operator()(const literal &n) {
    m_os << n.id();
}

void py_printer::operator()(const tuple &n) {
    open();
    detail::list(*this, n);
    close();
}

void py_printer::operator()(const apply &n) {
    (*this)(n.fn());
    (*this)(n.args());
}
    
void py_printer::operator()(const lambda &n) {
    m_os << "lambda ";
    detail::list(*this, n.args());
    m_os << ": ";
    boost::apply_visitor(*this, n.body());
}
void py_printer::operator()(const closure &n) {
    m_os << "closure([";
    detail::list(*this, n.args());
    m_os << "], ";
    boost::apply_visitor(*this, n.body());
    m_os << ")";
}
void py_printer::operator()(const conditional &n) {
    m_os << "if ";
    boost::apply_visitor(*this, n.cond());
    m_os << ":" << std::endl;
    indent();
    boost::apply_visitor(*this, n.then());
    dedent();
    indentation();
    m_os << "else:" << std::endl;
    indent();
    boost::apply_visitor(*this, n.orelse());
    dedent();
}
void py_printer::operator()(const ret &n) {
    m_os << "return ";
    boost::apply_visitor(*this, n.val());
}
void py_printer::operator()(const bind &n) {
    boost::apply_visitor(*this, n.lhs());
    m_os << " = ";
    boost::apply_visitor(*this, n.rhs());
}
void py_printer::operator()(const call & n) {
    boost::apply_visitor(*this, n.sub());
}
void py_printer::operator()(const procedure &n) {
    m_os << "def ";
    (*this)(n.id());
    (*this)(n.args());
    m_os << ":" << std::endl;
    indent();
    (*this)(n.stmts());
    dedent();
        
}
void py_printer::operator()(const suite &n) {
    for(auto i = n.begin();
        i != n.end();
        i++) {
        indentation();
        boost::apply_visitor(*this, *i);
        m_os << std::endl;
    }
}

void py_printer::operator()(const structure &n) {
    indentation();
    m_os << "class ";
    (*this)(n.id());
    m_os << ":" << std::endl;
    indent();
    (*this)(n.stmts());
    dedent();
}
void py_printer::operator()(const include &n) {
    //No #include statement in python
    assert(false);
}

void py_printer::operator()(const typedefn &n) {
    //No typedef statement in python
    assert(false);
}
    
void py_printer::operator()(const std::string &s) {
    m_os << s;
}

void py_printer::sep(void) const {
    m_os << ", ";
}
void py_printer::open(void) const {
    m_os << "(";
}
void py_printer::close(void) const {
    m_os << ")";
}
void py_printer::indent(int amount) {
    indent_level += amount;
    assert(indent_level > -1);
    std::stringstream ss;
    std::fill_n(std::ostream_iterator<std::string>(ss),
                indent_level,
                indent_atom);
    current_indent = ss.str();
}
void py_printer::dedent() {
    indent(-1);
}
void py_printer::indentation() {
    m_os << current_indent;
}

}
