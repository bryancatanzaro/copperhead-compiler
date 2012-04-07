#include "expression.hpp"

using std::string;
using std::shared_ptr;
using std::move;
using std::vector;


namespace backend {



const type_t& expression::type(void) const {
    return *m_type;
}
const ctype::type_t& expression::ctype(void) const {
    return *m_ctype;
}

shared_ptr<type_t> expression::p_type(void) const {
    return m_type;
}
shared_ptr<ctype::type_t> expression::p_ctype(void) const {
    return m_ctype;
}

literal::literal(const std::string& val,
                 const std::shared_ptr<type_t>& type,
                 const std::shared_ptr<ctype::type_t>& ctype)
    : expression(*this, type, ctype), m_val(val) {}

const string& literal::id(void) const {
    return m_val;
}

name::name(const std::string &val,
           const std::shared_ptr<type_t>& type,
           const std::shared_ptr<ctype::type_t>& ctype)
    : literal(*this, val, type, ctype)
{}

tuple::tuple(vector<shared_ptr<expression> > &&values,
             const shared_ptr<type_t>& type,
             const shared_ptr<ctype::type_t>& ctype)
    : expression(*this, type, ctype),
      m_values(std::move(values)) {}

tuple::const_iterator tuple::begin() const {
    return boost::make_indirect_iterator(m_values.cbegin());
}

tuple::const_iterator tuple::end() const {
    return boost::make_indirect_iterator(m_values.cend());
}

tuple::const_ptr_iterator tuple::p_begin() const {
    return m_values.cbegin();
}

tuple::const_ptr_iterator tuple::p_end() const {
    return m_values.cend();
}

int tuple::arity() const {
    return m_values.size();
}

apply::apply(const shared_ptr<name> &fn,
             const shared_ptr<tuple> &args)
    : expression(*this),
      m_fn(fn), m_args(args) {}

const name& apply::fn(void) const {
    return *m_fn;
}
const tuple& apply::args(void) const {
    return *m_args;
}

shared_ptr<name> apply::p_fn(void) const {
    return m_fn;
}

shared_ptr<tuple> apply::p_args(void) const {
    return m_args;
}


lambda::lambda(const shared_ptr<tuple> &args,
               const shared_ptr<expression> &body,
               const shared_ptr<type_t>& type,
               const shared_ptr<ctype::type_t>& ctype)
    : expression(*this, type, ctype),
      m_args(args), m_body(body) {}

const tuple& lambda::args(void) const {
    return *m_args;
}
const expression& lambda::body(void) const {
    return *m_body;
}

shared_ptr<tuple> lambda::p_args(void) const {
    return m_args;
}

shared_ptr<expression> lambda::p_body(void) const {
    return m_body;
}

closure::closure(const shared_ptr<tuple> &args,
                 const shared_ptr<expression> &body,
                 const shared_ptr<type_t>& type,
                 const shared_ptr<ctype::type_t>& ctype)
    : expression(*this, type, ctype), m_args(args), m_body(body) {}

const tuple& closure::args(void) const {
    return *m_args;
}

const expression& closure::body(void) const {
    return *m_body;
}

shared_ptr<tuple> closure::p_args(void) const {
    return m_args;
}

shared_ptr<expression> closure::p_body(void) const {
    return m_body;
}

subscript::subscript(const shared_ptr<name> &src,
                     const shared_ptr<expression> &idx,
                     const shared_ptr<type_t>& type,
                     const shared_ptr<ctype::type_t>& ctype)
    : expression(*this, type, ctype), m_src(src), m_idx(idx) {}

const name& subscript::src(void) const {
    return *m_src;
}

const expression& subscript::idx(void) const {
    return *m_idx;
}

shared_ptr<name> subscript::p_src(void) const {
    return m_src;
}

shared_ptr<expression> subscript::p_idx(void) const {
    return m_idx;
}

}
    

