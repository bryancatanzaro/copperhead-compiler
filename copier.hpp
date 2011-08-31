#pragma once
#include "node.hpp"

namespace backend {

class copier
    : public no_op_visitor<std::shared_ptr<node> >
{
public:
    // XXX why do we have to use 'using' to make the base class's overloads visible?
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
        return result_type(new tuple(std::move(n_values)));
    }
    virtual result_type operator()(const apply &n) {
        auto n_fn = std::static_pointer_cast<name>((*this)(n.fn()));
        auto n_args = std::static_pointer_cast<tuple>((*this)(n.args()));
        return result_type(new apply(n_fn, n_args));
    }
    virtual result_type operator()(const lambda &n) {
        auto n_args = std::static_pointer_cast<tuple>((*this)(n.args()));
        auto n_body = std::static_pointer_cast<expression>(boost::apply_visitor(*this, n.body()));
        return result_type(new lambda(n_args, n_body));
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
        auto n_id = std::static_pointer_cast<name>((*this)(n.id()));
        auto n_args = std::static_pointer_cast<tuple>((*this)(n.args()));
        auto n_stmts = std::static_pointer_cast<suite>((*this)(n.stmts()));
        return result_type(new procedure(n_id, n_args, n_stmts));
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
