#include "statement.hpp"

using std::shared_ptr;
using std::string;
using std::static_pointer_cast;

namespace backend {

shared_ptr<const statement> statement::ptr() const {
    return static_pointer_cast<const statement>(this->shared_from_this());
}

ret::ret(const shared_ptr<const expression> &val)
    : statement(*this),
      m_val(val) {}

const expression& ret::val(void) const {
    return *m_val;
}

shared_ptr<const ret> ret::ptr() const {
    return static_pointer_cast<const ret>(this->shared_from_this());
}


bind::bind(const shared_ptr<const expression> &lhs,
           const shared_ptr<const expression> &rhs)
    : statement(*this),
      m_lhs(lhs), m_rhs(rhs) {}

const expression& bind::lhs(void) const {
    return *m_lhs;
}
const expression& bind::rhs(void) const {
    return *m_rhs;
}

shared_ptr<const bind> bind::ptr() const {
    return static_pointer_cast<const bind>(this->shared_from_this());
}


call::call(const shared_ptr<const apply> &n)
    : statement(*this), m_sub(n) {}

const apply& call::sub(void) const {
    return *m_sub;
}

shared_ptr<const call> call::ptr() const {
    return static_pointer_cast<const call>(this->shared_from_this());
}


procedure::procedure(const shared_ptr<const name> &id,
                     const shared_ptr<const tuple> &args,
                     const shared_ptr<const suite> &stmts,
                     const shared_ptr<const type_t> &type,
                     const shared_ptr<const ctype::type_t> &ctype,
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

const type_t& procedure::type(void) const {
    return *m_type;
}
const ctype::type_t& procedure::ctype(void) const {
    return *m_ctype;
}

const string& procedure::place(void) const {
    return m_place;
}

shared_ptr<const procedure> procedure::ptr() const {
    return static_pointer_cast<const procedure>(this->shared_from_this());
}


conditional::conditional(shared_ptr<const expression> cond,
                         shared_ptr<const suite> then,
                         shared_ptr<const suite> orelse)
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

shared_ptr<const conditional> conditional::ptr() const {
    return static_pointer_cast<const conditional>(this->shared_from_this());
}


suite::suite(std::vector<shared_ptr<const statement> > &&stmts)
    : node(*this),
      m_stmts(std::move(stmts)) {}

suite::const_iterator suite::begin() const {
    return boost::make_indirect_iterator(m_stmts.cbegin());
}

suite::const_iterator suite::end() const {
    return boost::make_indirect_iterator(m_stmts.cend());
}

int suite::size() const {
    return m_stmts.size();
}

shared_ptr<const suite> suite::ptr() const {
    return static_pointer_cast<const suite>(this->shared_from_this());
}


}

