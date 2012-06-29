#include "cppnode.hpp"

using std::shared_ptr;
using std::make_shared;
using std::vector;
using std::static_pointer_cast;

namespace backend {

structure::structure(const std::shared_ptr<const name> &name,
                     const std::shared_ptr<const suite> &stmts,
                     std::vector<shared_ptr<const ctype::type_t> >&& typevars)
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

shared_ptr<const structure> structure::ptr() const {
    return static_pointer_cast<const structure>(this->shared_from_this());
}


templated_name::templated_name(const std::string &id,
                               const std::shared_ptr<const ctype::tuple_t> &template_types,
                               const std::shared_ptr<const type_t> &type,
                               const std::shared_ptr<const ctype::type_t> &ctype)
    : name(*this, id, type, ctype),
      m_template_types(template_types) {}

const ctype::tuple_t& templated_name::template_types() const {
    return *m_template_types;
}

shared_ptr<const templated_name> templated_name::ptr() const {
    return static_pointer_cast<const templated_name>(this->shared_from_this());
}


include::include(const std::shared_ptr<const literal> &id,
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

shared_ptr<const include> include::ptr() const {
    return static_pointer_cast<const include>(this->shared_from_this());
}


typedefn::typedefn(const std::shared_ptr<const ctype::type_t> origin,
                   const std::shared_ptr<const ctype::type_t> rename)
    : statement(*this),
      m_origin(origin),
      m_rename(rename) {}
const ctype::type_t& typedefn::origin() const {
    return *m_origin;
}
const ctype::type_t& typedefn::rename() const {
    return *m_rename;
}

shared_ptr<const typedefn> typedefn::ptr() const {
    return static_pointer_cast<const typedefn>(this->shared_from_this());
}


namespace_block::namespace_block(const std::string& name,
                                 const std::shared_ptr<const suite>& stmts)
    : statement(*this), m_name(name), m_stmts(stmts) {}

const std::string& namespace_block::name() const {
    return m_name;
}

const suite& namespace_block::stmts() const {
    return *m_stmts;
}

shared_ptr<const namespace_block> namespace_block::ptr() const {
    return static_pointer_cast<const namespace_block>(this->shared_from_this());
}

while_block::while_block(const std::shared_ptr<const expression>& pred,
                         const std::shared_ptr<const suite>& stmts)
    : statement(*this), m_pred(pred), m_stmts(stmts) {}

const expression& while_block::pred() const {
    return *m_pred;
}

const suite& while_block::stmts() const {
    return *m_stmts;
}

shared_ptr<const while_block> while_block::ptr() const {
    return static_pointer_cast<const while_block>(this->shared_from_this());
}


}
