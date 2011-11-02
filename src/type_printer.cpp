#include "type_printer.hpp"

namespace backend
{

repr_type_printer::repr_type_printer(std::ostream &os)
    : m_os(os) {}

void repr_type_printer::operator()(const monotype_t &mt) {
    m_os << mt.name();
    if(mt.begin() != mt.end()) {
        open();
        for(auto i = mt.begin();
            i != mt.end();
            i++) {
            boost::apply_visitor(*this, *i);
            if (std::next(i) != mt.end()) 
                sep();
        }
        close();
    }
}
void repr_type_printer::operator()(const polytype_t &pt) {
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
void repr_type_printer::sep() const {
    m_os << ", ";
}
void repr_type_printer::open() const {
    m_os << "(";
}
void repr_type_printer::close() const {
    m_os << ")";
}

namespace ctype {
ctype_printer::ctype_printer(std::ostream &os)
    : m_os(os)
{
    m_need_space.push(false);
}
void ctype_printer::operator()(const monotype_t &mt) {
    m_os << mt.name();
    m_need_space.top() = false;
}
void ctype_printer::operator()(const sequence_t &st) {
    m_os << st.name() << "<";
    boost::apply_visitor(*this, st.sub());
    m_os << ">";
    m_need_space.top() = true;
}
void ctype_printer::operator()(const cuarray_t &ct) {
    //Because cuarray_t is a variant, we don't want to
    //print out the template definition.
    this->operator()((monotype_t)ct);
    m_need_space.top() = false;
}
void ctype_printer::operator()(const polytype_t &pt) {
}
void ctype_printer::operator()(const templated_t &tt) {
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
void ctype_printer::sep() const {
    m_os << ", ";
}
void ctype_printer::open() const {
    m_os << "(";
}
void ctype_printer::close() const {
    m_os << ")";
}

}
}
