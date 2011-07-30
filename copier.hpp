#pragma once
#include "node.hpp"

namespace backend {

class copier
    : public no_op_visitor<std::shared_ptr<node> >
{
private:
    typedef std::shared_ptr<node> ResultType;
public:
    // XXX why do we have to use 'using' to make the base class's overloads visible?
    using backend::no_op_visitor<std::shared_ptr<node> >::operator();
    
    inline ResultType operator()(const number& n) const {
        return ResultType(new number(n));
    }
    inline ResultType operator()(const name &n) const {
        return ResultType(new name(n));
    }
    inline ResultType operator()(const tuple &n) const {
        std::vector<std::shared_ptr<expression> > n_values;
        for(auto i = n.begin(); i != n.end(); i++) {
            auto n_i = std::static_pointer_cast<expression>(boost::apply_visitor(*this, *i));
            n_values.push_back(n_i);
        }
        return ResultType(new tuple(n_values));
    }
    inline ResultType operator()(const apply &n) const {
        auto n_fn = std::static_pointer_cast<name>((*this)(n.fn()));
        auto n_args = std::static_pointer_cast<tuple>((*this)(n.args()));
        return ResultType(new apply(n_fn, n_args));
    }
    inline ResultType operator()(const lambda &n) const {
        auto n_args = std::static_pointer_cast<tuple>((*this)(n.args()));
        auto n_body = std::static_pointer_cast<expression>(boost::apply_visitor(*this, n.body()));
        return ResultType(new lambda(n_args, n_body));
    }
    inline ResultType operator()(const closure &n) const {
        return ResultType(new closure());
    }
    inline ResultType operator()(const conditional &n) const {
        return ResultType(new conditional());
    }
    inline ResultType operator()(const ret &n) const {
        auto n_val = std::static_pointer_cast<expression>(boost::apply_visitor(*this, n.val()));
        return ResultType(new ret(n_val));
    }
    inline ResultType operator()(const bind &n) const {
        auto n_lhs = std::static_pointer_cast<expression>(boost::apply_visitor(*this, n.lhs()));
        auto n_rhs = std::static_pointer_cast<expression>(boost::apply_visitor(*this, n.rhs()));
        return ResultType(new bind(n_lhs, n_rhs));
    }
    inline ResultType operator()(const procedure &n) const {
        auto n_id = std::static_pointer_cast<name>((*this)(n.id()));
        auto n_args = std::static_pointer_cast<tuple>((*this)(n.args()));
        auto n_stmts = std::static_pointer_cast<suite>((*this)(n.stmts()));
        return ResultType(new procedure(n_id, n_args, n_stmts));
    }
    inline ResultType operator()(const suite &n) const {
        std::vector<std::shared_ptr<statement> > n_stmts;
        for(auto i = n.begin(); i != n.end(); i++) {
            auto n_stmt = std::static_pointer_cast<statement>(boost::apply_visitor(*this, *i));
            n_stmts.push_back(n_stmt);
        }
        return ResultType(new suite(n_stmts));
    }
    inline ResultType operator()(const structure &n) const {
        auto n_id = std::static_pointer_cast<name>((*this)(n.id()));
        auto n_stmts = std::static_pointer_cast<suite>((*this)(n.stmts()));
        return ResultType(new structure(n_id, n_stmts));
    }

};

}
