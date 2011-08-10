#pragma once
#include <string>
#include <set>
#include "copier.hpp"
#include "repr_printer.hpp"
#include "utility/isinstance.hpp"

namespace backend {

class functorize
    : public copier
{
private:
    //typedef std::shared_ptr<node> result_type;
    std::vector<result_type> m_additionals;
    std::set<std::string> m_fns;
public:
    functorize() : m_additionals({}) {}
public:
    // XXX why do we have to use 'using' to make the base class's overloads visible?
    using copier::operator();

    result_type operator()(const apply &n) {
        std::vector<std::shared_ptr<expression> > n_arg_list;
        const tuple& n_args = n.args();
        for(auto n_arg = n_args.begin();
            n_arg != n_args.end();
            ++n_arg) {
            if (!(detail::isinstance<name, expression>(*n_arg)))
                n_arg_list.push_back(std::static_pointer_cast<expression>(boost::apply_visitor(*this, *n_arg)));
            else {
                name* p_name = static_cast<name*>(&*n_arg);
                const std::string id = p_name->id();
                auto found = m_fns.find(id);
                if (found == m_fns.end()) {
                    n_arg_list.push_back(std::static_pointer_cast<expression>(boost::apply_visitor(*this, *n_arg)));
                } else {
                    n_arg_list.push_back(std::shared_ptr<name>(new name(id + std::string("_fn()"))));
                }
            }
        }
        auto n_fn = std::static_pointer_cast<name>(this->copier::operator()(n.fn()));
        auto new_args = std::shared_ptr<tuple>(new tuple(n_arg_list));
        return std::shared_ptr<apply>(new apply(n_fn, new_args));
    }
    
    result_type operator()(const suite &n) {
        std::shared_ptr<suite> result(new suite({}));
        for(auto i = n.begin(); i != n.end(); i++) {
            auto p = std::static_pointer_cast<statement>(boost::apply_visitor(*this, *i));
            result->push_back(p);
            while(m_additionals.size() > 0) {
                auto p = std::static_pointer_cast<statement>(m_additionals.back());
                result->push_back(p);
                m_additionals.pop_back();
            }
        }
        return result;
    }
    result_type operator()(const procedure &n) {
        auto n_proc = std::static_pointer_cast<procedure>(this->copier::operator()(n));
        std::shared_ptr<tuple> forward_args = std::static_pointer_cast<tuple>(this->copier::operator()(n_proc->args()));
        std::shared_ptr<name> forward_name = std::static_pointer_cast<name>(this->copier::operator()(n_proc->id()));
        std::shared_ptr<apply> op_call(new apply(forward_name, forward_args));
        std::shared_ptr<ret> op_ret(new ret(op_call));
        std::vector<std::shared_ptr<statement> > op_body_stmts{op_ret};
        std::shared_ptr<suite> op_body(new suite(op_body_stmts));
        auto op_args = std::static_pointer_cast<tuple>(this->copier::operator()(n.args()));
        std::shared_ptr<name> op_id(new name(std::string("operator()")));
        std::shared_ptr<procedure> op(new procedure(op_id, op_args, op_body));
        std::shared_ptr<suite> st_body(new suite(std::vector<std::shared_ptr<statement> >{op}));
        std::shared_ptr<name> st_id(new name(std::string(n_proc->id().id() + "_fn")));
        std::shared_ptr<structure> st(new structure(st_id, st_body));
        m_additionals.push_back(st);
        m_fns.insert(n_proc->id().id());
        return n_proc;

    }
    
};

}
