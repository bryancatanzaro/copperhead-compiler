#include "statement.hpp"
namespace backend {

ret::ret(const std::shared_ptr<expression> &val)
    : statement(*this),
      m_val(val) {}

const expression& ret::val(void) const {
    return *m_val;
}

bind::bind(const std::shared_ptr<expression> &lhs,
           const std::shared_ptr<expression> &rhs)
    : statement(*this),
      m_lhs(lhs), m_rhs(rhs) {}

const expression& bind::lhs(void) const {
    return *m_lhs;
}
const expression& bind::rhs(void) const {
    return *m_rhs;
}

call::call(const std::shared_ptr<apply> &n)
    : statement(*this), m_sub(n) {}

const apply& call::sub(void) const {
    return *m_sub;
}
        

procedure::procedure(const std::shared_ptr<name> &id,
                     const std::shared_ptr<tuple> &args,
                     const std::shared_ptr<suite> &stmts,
                     const std::shared_ptr<type_t> &type,
                     const std::shared_ptr<ctype::type_t> &ctype,
                     const std::string &place)
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

std::shared_ptr<type_t> procedure::p_type(void) const {
    return m_type;
}

std::shared_ptr<ctype::type_t> procedure::p_ctype(void) const {
    return m_ctype;
}

const std::string& procedure::place(void) const {
    return m_place;
}

conditional::conditional(std::shared_ptr<expression> cond,
                         std::shared_ptr<suite> then,
                         std::shared_ptr<suite> orelse)
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



suite::suite(std::vector<std::shared_ptr<statement> > &&stmts)
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

}

