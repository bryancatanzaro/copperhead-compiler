#include "expression.hpp"

namespace backend {



const type_t& expression::type(void) const {
    return *m_type;
}
const ctype::type_t& expression::ctype(void) const {
    return *m_ctype;
}
literal::literal(const std::string& val,
                 const std::shared_ptr<type_t>& type,
                 const std::shared_ptr<ctype::type_t>& ctype)
    : expression(*this, type, ctype), m_val(val) {}

const std::string& literal::id(void) const {
    return m_val;
}

name::name(const std::string &val,
         const std::shared_ptr<type_t>& type,
         const std::shared_ptr<ctype::type_t>& ctype)
        : literal(*this, val, type, ctype)
        {}

tuple::tuple(std::vector<std::shared_ptr<expression> > &&values,
             const std::shared_ptr<type_t>& type,
             const std::shared_ptr<ctype::type_t>& ctype)
    : expression(*this, type, ctype),
      m_values(std::move(values)) {}

tuple::const_iterator tuple::begin() const {
    return boost::make_indirect_iterator(m_values.cbegin());
}

tuple::const_iterator tuple::end() const {
    return boost::make_indirect_iterator(m_values.cend());
}

int tuple::arity() const {
    return m_values.size();
}

apply::apply(const std::shared_ptr<name> &fn,
             const std::shared_ptr<tuple> &args)
    : expression(*this),
      m_fn(fn), m_args(args) {}

const name& apply::fn(void) const {
    return *m_fn;
}
const tuple& apply::args(void) const {
    return *m_args;
}


lambda::lambda(const std::shared_ptr<tuple> &args,
               const std::shared_ptr<expression> &body,
               const std::shared_ptr<type_t>& type,
               const std::shared_ptr<ctype::type_t>& ctype)
    : expression(*this, type, ctype),
      m_args(args), m_body(body) {}

const tuple& lambda::args(void) const {
    return *m_args;
}
const expression& lambda::body(void) const {
    return *m_body;
}

closure::closure(const std::shared_ptr<tuple> &args,
                 const std::shared_ptr<expression> &body,
                 const std::shared_ptr<type_t>& type,
                 const std::shared_ptr<ctype::type_t>& ctype)
    : expression(*this, type, ctype), m_args(args), m_body(body) {}

const tuple& closure::args(void) const {
    return *m_args;
}

const expression& closure::body(void) const {
    return *m_body;
}


}
    

