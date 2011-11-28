#include "statement.hpp"

using std::shared_ptr;
using std::string;

namespace backend {

ret::ret(const shared_ptr<expression> &val)
    : statement(*this),
      m_val(val) {}

const expression& ret::val(void) const {
    return *m_val;
}

shared_ptr<expression> ret::p_val(void) const {
    return m_val;
}

bind::bind(const shared_ptr<expression> &lhs,
           const shared_ptr<expression> &rhs)
    : statement(*this),
      m_lhs(lhs), m_rhs(rhs) {}

const expression& bind::lhs(void) const {
    return *m_lhs;
}
const expression& bind::rhs(void) const {
    return *m_rhs;
}

shared_ptr<expression> bind::p_lhs(void) const {
    return m_lhs;
}
shared_ptr<expression> bind::p_rhs(void) const {
    return m_rhs;
}

call::call(const shared_ptr<apply> &n)
    : statement(*this), m_sub(n) {}

const apply& call::sub(void) const {
    return *m_sub;
}

shared_ptr<apply> call::p_sub(void) const {
    return m_sub;
}

procedure::procedure(const shared_ptr<name> &id,
                     const shared_ptr<tuple> &args,
                     const shared_ptr<suite> &stmts,
                     const shared_ptr<type_t> &type,
                     const shared_ptr<ctype::type_t> &ctype,
                     const string &place)
        : statement(*this),
          m_id(id), m_args(args), m_stmts(stmts), m_type(type),
          m_ctype(ctype), m_place(place) {}

const name& procedure::id(void) const {
    return *m_id;
}
const tuple& procedure::args(void) const {
    return *m_args;
}
const suite& procedure::stmts(void) const {
    return *m_stmts;
}

shared_ptr<name> procedure::p_id(void) const {
    return m_id;
}
shared_ptr<tuple> procedure::p_args(void) const {
    return m_args;
}
shared_ptr<suite> procedure::p_stmts(void) const {
    return m_stmts;
}

const type_t& procedure::type(void) const {
    return *m_type;
}
const ctype::type_t& procedure::ctype(void) const {
    return *m_ctype;
}

shared_ptr<type_t> procedure::p_type(void) const {
    return m_type;
}
shared_ptr<ctype::type_t> procedure::p_ctype(void) const {
    return m_ctype;
}

const string& procedure::place(void) const {
    return m_place;
}

conditional::conditional(shared_ptr<expression> cond,
                         shared_ptr<suite> then,
                         shared_ptr<suite> orelse)
    : statement(*this), m_cond(cond),
      m_then(then), m_orelse(orelse) {}

const expression& conditional::cond(void) const {
    return *m_cond;
}
const suite& conditional::then(void) const {
    return *m_then;
}
const suite& conditional::orelse(void) const {
        return *m_orelse;
}

shared_ptr<expression> conditional::p_cond(void) const {
    return m_cond;
}
shared_ptr<suite> conditional::p_then(void) const {
    return m_then;
}
shared_ptr<suite> conditional::p_orelse(void) const {
    return m_orelse;
}

suite::suite(std::vector<shared_ptr<statement> > &&stmts)
    : node(*this),
      m_stmts(std::move(stmts)) {}
suite::const_iterator suite::begin() const {
    return boost::make_indirect_iterator(m_stmts.cbegin());
}
suite::const_iterator suite::end() const {
    return boost::make_indirect_iterator(m_stmts.cend());
}

suite::const_ptr_iterator suite::p_begin() const {
    return m_stmts.cbegin();
}
suite::const_ptr_iterator suite::p_end() const {
    return m_stmts.cend();
}

int suite::size() const {
    return m_stmts.size();
}

}

