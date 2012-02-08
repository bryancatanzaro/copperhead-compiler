#include "cppnode.hpp"

using std::shared_ptr;
using std::make_shared;
using std::vector;

namespace backend {

structure::structure(const std::shared_ptr<name> &name,
                     const std::shared_ptr<suite> &stmts,
                     std::vector<shared_ptr<ctype::type_t> >&& typevars)
    : statement(*this),
      m_id(name),
      m_stmts(stmts),
      m_typevars(std::move(typevars)) {}
const name& structure::id(void) const {
    return *m_id;
}
const suite& structure::stmts(void) const {
    return *m_stmts;
}
structure::const_iterator structure::begin(void) const {
    return boost::make_indirect_iterator(m_typevars.cbegin());
}

structure::const_iterator structure::end(void) const {
    return boost::make_indirect_iterator(m_typevars.cend());
}
structure::const_ptr_iterator structure::p_begin(void) const {
    return m_typevars.cbegin();
}

structure::const_ptr_iterator structure::p_end(void) const {
    return m_typevars.cend();
}

templated_name::templated_name(const std::string &id,
                               const std::shared_ptr<ctype::tuple_t> &template_types,
                               const std::shared_ptr<type_t> &type,
                               const std::shared_ptr<ctype::type_t> &ctype)
    : name(*this, id, type, ctype),
      m_template_types(template_types) {}

const ctype::tuple_t& templated_name::template_types() const {
    return *m_template_types;
}

include::include(const std::shared_ptr<literal> &id,
                 const char open,
                 const char close) : statement(*this),
                                     m_id(id),
                                     m_open(open),
                                     m_close(close) {}
const literal& include::id() const {
    return *m_id;
}
const char& include::open() const {
    return m_open;
}
const char& include::close() const {
    return m_close;
}

typedefn::typedefn(const std::shared_ptr<ctype::type_t> origin,
                   const std::shared_ptr<ctype::type_t> rename)
    : statement(*this),
      m_origin(origin),
      m_rename(rename) {}
const ctype::type_t& typedefn::origin() const {
    return *m_origin;
}
const ctype::type_t& typedefn::rename() const {
    return *m_rename;
}

namespace_block::namespace_block(const std::string& name,
                                 const std::shared_ptr<suite>& stmts)
    : statement(*this), m_name(name), m_stmts(stmts) {}

const std::string& namespace_block::name() const {
    return m_name;
}

const suite& namespace_block::stmts() const {
    return *m_stmts;
}


}
