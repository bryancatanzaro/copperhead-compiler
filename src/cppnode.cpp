#include "cppnode.hpp"

namespace backend {

structure::structure(const std::shared_ptr<name> &name,
                     const std::shared_ptr<suite> &stmts)
    : statement(*this),
      m_id(name),
      m_stmts(stmts) {}
const name& structure::id(void) const {
    return *m_id;
}
const suite& structure::stmts(void) const {
    return *m_stmts;
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

}
