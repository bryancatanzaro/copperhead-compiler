#pragma once
#include <string>
#include "copier.hpp"

namespace backend {

class functorize
    : public copier
{
private:
    typedef std::shared_ptr<node> ResultType;
    std::vector<ResultType> m_additionals;
public:
    inline functorize() : m_additionals({}) {}
public:
    // XXX why do we have to use 'using' to make the base class's overloads visible?
    using copier::operator();
    inline ResultType operator()(const suite &n) {
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
    inline ResultType operator()(const procedure &n) {
        auto n_proc = std::static_pointer_cast<procedure>(this->copier::operator()(n));
        auto op_body = std::static_pointer_cast<suite>(this->copier::operator()(n.stmts()));
        auto op_args = std::static_pointer_cast<tuple>(this->copier::operator()(n.args()));
        std::shared_ptr<name> op_id(new name(std::string("operator()")));
        std::shared_ptr<procedure> op(new procedure(op_id, op_args, op_body));
        std::shared_ptr<suite> st_body(new suite(std::vector<std::shared_ptr<statement> >{op}));
        std::shared_ptr<name> st_id(new name(std::string(n_proc->id().id() + "_fn")));
        std::shared_ptr<structure> st(new structure(st_id, st_body));
        m_additionals.push_back(st);
        return n_proc;

    }
    
};

}
