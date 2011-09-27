#pragma once
#include "node.hpp"
#include "type.hpp"
#include "ctype.hpp"

namespace backend {

class type_copier
    : public no_op_visitor<std::shared_ptr<type_t> > {
public:
    inline result_type operator()(const monotype_t &mt) {
        return result_type(new monotype_t(mt));
    }
    inline result_type operator()(const polytype_t &pt) {
        return result_type(new polytype_t(pt));
    }
    inline result_type operator()(const sequence_t &st) {
        result_type sub = boost::apply_visitor(*this, st.sub());
        return result_type(new sequence_t(sub));
    }
    inline result_type operator()(const fn_t &ft) {
        std::shared_ptr<tuple_t> args = std::static_pointer_cast<tuple_t>(this->operator()(ft.args()));
        result_type result = boost::apply_visitor(*this, ft.result());
        return result_type(new fn_t(args, result));
    }
    inline result_type operator()(const tuple_t &tt) {
        std::vector<result_type> subs;
        for(auto i = tt.begin();
            i != tt.end();
            i++) {
            subs.push_back(boost::apply_visitor(*this, *i));
        }
        return result_type(new tuple_t(std::move(subs)));
    }
};

namespace ctype {
class ctype_copier
    : public no_op_visitor<std::shared_ptr<type_t> > {
public:
    inline result_type operator()(const monotype_t &mt) {
        return result_type(new monotype_t(mt));
    }
    inline result_type operator()(const polytype_t &pt) {
        return result_type(new polytype_t(pt));
    }
    inline result_type operator()(const sequence_t &st) {
        result_type sub = boost::apply_visitor(*this, st.sub());
        return result_type(new sequence_t(sub));
    }
    inline result_type operator()(const fn_t &ft) {
        std::shared_ptr<tuple_t> args = std::static_pointer_cast<tuple_t>(this->operator()(ft.args()));
        result_type result = boost::apply_visitor(*this, ft.result());
        return result_type(new fn_t(args, result));
    }
    inline result_type operator()(const tuple_t &tt) {
        std::vector<result_type> subs;
        for(auto i = tt.begin();
            i != tt.end();
            i++) {
            subs.push_back(boost::apply_visitor(*this, *i));
        }
        return result_type(new tuple_t(std::move(subs)));
    }
};

}


class copier
    : public no_op_visitor<std::shared_ptr<node> >
{
private:
    type_copier m_tc;
    ctype::ctype_copier m_ctc;
public:
    copier() : m_tc(), m_ctc() {}
    using backend::no_op_visitor<std::shared_ptr<node> >::operator();

    // Although these operator() methods could be declared here as
    // ... operator(..) const
    // I'm leaving out the const because they may be overridden in a sub
    // class with methods that are not const.
    
    virtual result_type operator()(const number& n) {
        return result_type(new number(n));
    }
    virtual result_type operator()(const name &n) {
        return result_type(new name(n));
    }
    virtual result_type operator()(const tuple &n) {
        std::vector<std::shared_ptr<expression> > n_values;
        for(auto i = n.begin(); i != n.end(); i++) {
            auto n_i = std::static_pointer_cast<expression>(boost::apply_visitor(*this, *i));
            n_values.push_back(n_i);
        }
        std::shared_ptr<type_t> t = boost::apply_visitor(m_tc, n.type());
        std::shared_ptr<ctype::type_t> ct = boost::apply_visitor(m_ctc, n.ctype());
        return result_type(new tuple(std::move(n_values), t, ct));
    }
    virtual result_type operator()(const apply &n) {
        auto n_fn = std::static_pointer_cast<name>((*this)(n.fn()));
        auto n_args = std::static_pointer_cast<tuple>((*this)(n.args()));
        return result_type(new apply(n_fn, n_args));
    }
    virtual result_type operator()(const lambda &n) {
        auto n_args = std::static_pointer_cast<tuple>((*this)(n.args()));
        auto n_body = std::static_pointer_cast<expression>(boost::apply_visitor(*this, n.body()));
        std::shared_ptr<type_t> t = boost::apply_visitor(m_tc, n.type());
        std::shared_ptr<ctype::type_t> ct = boost::apply_visitor(m_ctc, n.ctype());
        return result_type(new lambda(n_args, n_body, t, ct));
    }
    virtual result_type operator()(const closure &n) {
        return result_type(new closure());
    }
    virtual result_type operator()(const conditional &n) {
        return result_type(new conditional());
    }
    virtual result_type operator()(const ret &n) {
        auto n_val = std::static_pointer_cast<expression>(boost::apply_visitor(*this, n.val()));
        return result_type(new ret(n_val));
    }
    virtual result_type operator()(const bind &n) {
        auto n_lhs = std::static_pointer_cast<expression>(boost::apply_visitor(*this, n.lhs()));
        auto n_rhs = std::static_pointer_cast<expression>(boost::apply_visitor(*this, n.rhs()));
        return result_type(new bind(n_lhs, n_rhs));
    }
    virtual result_type operator()(const procedure &n) {
        std::cout << "Copying procedure" << std::endl;
        auto n_id = std::static_pointer_cast<name>((*this)(n.id()));
        auto n_args = std::static_pointer_cast<tuple>((*this)(n.args()));
        auto n_stmts = std::static_pointer_cast<suite>((*this)(n.stmts()));
        std::shared_ptr<type_t> t = boost::apply_visitor(m_tc, n.type());
        std::shared_ptr<ctype::type_t> ct = boost::apply_visitor(m_ctc, n.ctype());
        return result_type(new procedure(n_id, n_args, n_stmts, t, ct));
    }
    virtual result_type operator()(const suite &n) {
        std::vector<std::shared_ptr<statement> > n_stmts;
        for(auto i = n.begin(); i != n.end(); i++) {
            auto n_stmt = std::static_pointer_cast<statement>(boost::apply_visitor(*this, *i));
            n_stmts.push_back(n_stmt);
        }
        return result_type(new suite(std::move(n_stmts)));
    }
    virtual result_type operator()(const structure &n) {
        auto n_id = std::static_pointer_cast<name>((*this)(n.id()));
        auto n_stmts = std::static_pointer_cast<suite>((*this)(n.stmts()));
        return result_type(new structure(n_id, n_stmts));
    }

};

}
