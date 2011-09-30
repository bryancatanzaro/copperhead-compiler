#pragma once
#include "node.hpp"
#include "type.hpp"
#include "ctype.hpp"

namespace backend {

class copier
    : public no_op_visitor<std::shared_ptr<node> >
{
protected:
    //If you know you're not going to do any rewriting at deeper
    //levels of the AST, just grab the pointer from the node
    template<typename Node>
    result_type get_node_ptr(const Node &n) {
        return std::const_pointer_cast<node>(n.shared_from_this());
    }
    template<typename Type>
    std::shared_ptr<type_t> get_type_ptr(const Type &n) {
        return std::const_pointer_cast<type_t>(n.shared_from_this());
    }
    template<typename Ctype>
    std::shared_ptr<ctype::type_t> get_ctype_ptr(const Ctype &n) {
        return std::const_pointer_cast<ctype::type_t>(n.shared_from_this());
    }
public:
    using backend::no_op_visitor<std::shared_ptr<node> >::operator();
  
    
    virtual result_type operator()(const literal& n) {
        std::shared_ptr<type_t> t = get_type_ptr(n.type());
        std::shared_ptr<ctype::type_t> ct = get_ctype_ptr(n.ctype());
        return result_type(new literal(n.id(), t, ct));
    }
    virtual result_type operator()(const name &n) {
        std::shared_ptr<type_t> t = get_type_ptr(n.type());
        std::shared_ptr<ctype::type_t> ct = get_ctype_ptr(n.ctype());
        return result_type(new name(n.id(), t, ct));
    }
    virtual result_type operator()(const tuple &n) {
        std::vector<std::shared_ptr<expression> > n_values;
        for(auto i = n.begin(); i != n.end(); i++) {
            auto n_i = std::static_pointer_cast<expression>(boost::apply_visitor(*this, *i));
            n_values.push_back(n_i);
        }
        std::shared_ptr<type_t> t = get_type_ptr(n.type());
        std::shared_ptr<ctype::type_t> ct = get_ctype_ptr(n.ctype());
        
        return result_type(new tuple(std::move(n_values), t, ct));
    }
    virtual result_type operator()(const apply &n) {
        auto n_fn = std::static_pointer_cast<name>(boost::apply_visitor(*this, n.fn()));
        auto n_args = std::static_pointer_cast<tuple>((*this)(n.args()));
        return result_type(new apply(n_fn, n_args));
    }
    virtual result_type operator()(const lambda &n) {
        auto n_args = std::static_pointer_cast<tuple>((*this)(n.args()));
        auto n_body = std::static_pointer_cast<expression>(boost::apply_visitor(*this, n.body()));
        std::shared_ptr<type_t> t = get_type_ptr(n.type());
        std::shared_ptr<ctype::type_t> ct = get_ctype_ptr(n.ctype());
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
    virtual result_type operator()(const call &n) {
        auto n_sub = std::static_pointer_cast<apply>(boost::apply_visitor(*this, n.sub()));
        return result_type(new call(n_sub));
    }
    virtual result_type operator()(const procedure &n) {
        auto n_id = std::static_pointer_cast<name>((*this)(n.id()));
        auto n_args = std::static_pointer_cast<tuple>((*this)(n.args()));
        auto n_stmts = std::static_pointer_cast<suite>((*this)(n.stmts()));
        std::shared_ptr<type_t> t = get_type_ptr(n.type());
        std::shared_ptr<ctype::type_t> ct = get_ctype_ptr(n.ctype());

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
    virtual result_type operator()(const templated_name &n) {
        std::shared_ptr<type_t> t = get_type_ptr(n.type());
        std::shared_ptr<ctype::type_t> ct = get_ctype_ptr(n.ctype());
        auto n_args = std::static_pointer_cast<ctype::tuple_t>(
            get_ctype_ptr(n.template_types()));
        return result_type(new templated_name(n.id(), n_args, t, ct));
    }
    virtual result_type operator()(const include &n) {
        std::shared_ptr<name> id = std::static_pointer_cast<name>(
            boost::apply_visitor(*this, n.id()));
        return result_type(new include(id));
    }
};

}
