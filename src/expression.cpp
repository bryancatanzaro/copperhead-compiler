#include "expression.hpp"

using std::string;
using std::shared_ptr;
using std::move;
using std::vector;
using std::static_pointer_cast;

namespace backend {

shared_ptr<const expression> expression::ptr() const {
    return static_pointer_cast<const expression>(this->shared_from_this());
}

const type_t& expression::type(void) const {
    return *m_type;
}
const ctype::type_t& expression::ctype(void) const {
    return *m_ctype;
}


literal::literal(const std::string& val,
                 const std::shared_ptr<const type_t>& type,
                 const std::shared_ptr<const ctype::type_t>& ctype)
    : expression(*this, type, ctype), m_val(val) {}

const string& literal::id(void) const {
    return m_val;
}

shared_ptr<const literal> literal::ptr() const {
    return static_pointer_cast<const literal>(this->shared_from_this());
}


name::name(const std::string &val,
           const std::shared_ptr<const type_t>& type,
           const std::shared_ptr<const ctype::type_t>& ctype)
    : literal(*this, val, type, ctype)
{}

shared_ptr<const name> name::ptr() const {
    return static_pointer_cast<const name>(this->shared_from_this());
}


tuple::tuple(vector<shared_ptr<const expression> > &&values,
             const shared_ptr<const type_t>& type,
             const shared_ptr<const ctype::type_t>& ctype)
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

shared_ptr<const tuple> tuple::ptr() const {
    return static_pointer_cast<const tuple>(this->shared_from_this());
}


apply::apply(const shared_ptr<const name> &fn,
             const shared_ptr<const tuple> &args)
    : expression(*this),
      m_fn(fn), m_args(args) {}

const name& apply::fn(void) const {
    return *m_fn;
}
const tuple& apply::args(void) const {
    return *m_args;
}

shared_ptr<const apply> apply::ptr() const {
    return static_pointer_cast<const apply>(this->shared_from_this());
}


lambda::lambda(const shared_ptr<const tuple> &args,
               const shared_ptr<const expression> &body,
               const shared_ptr<const type_t>& type,
               const shared_ptr<const ctype::type_t>& ctype)
    : expression(*this, type, ctype),
      m_args(args), m_body(body) {}

const tuple& lambda::args(void) const {
    return *m_args;
}
const expression& lambda::body(void) const {
    return *m_body;
}

shared_ptr<const lambda> lambda::ptr() const {
    return static_pointer_cast<const lambda>(this->shared_from_this());
}


closure::closure(const shared_ptr<const tuple> &args,
                 const shared_ptr<const expression> &body,
                 const shared_ptr<const type_t>& type,
                 const shared_ptr<const ctype::type_t>& ctype)
    : expression(*this, type, ctype), m_args(args), m_body(body) {}

const tuple& closure::args(void) const {
    return *m_args;
}

const expression& closure::body(void) const {
    return *m_body;
}

shared_ptr<const closure> closure::ptr() const {
    return static_pointer_cast<const closure>(this->shared_from_this());
}


subscript::subscript(const shared_ptr<const name> &src,
                     const shared_ptr<const expression> &idx,
                     const shared_ptr<const type_t>& type,
                     const shared_ptr<const ctype::type_t>& ctype)
    : expression(*this, type, ctype), m_src(src), m_idx(idx) {}

const name& subscript::src(void) const {
    return *m_src;
}

const expression& subscript::idx(void) const {
    return *m_idx;
}

shared_ptr<const subscript> subscript::ptr() const {
    return static_pointer_cast<const subscript>(this->shared_from_this());
}


}
    

