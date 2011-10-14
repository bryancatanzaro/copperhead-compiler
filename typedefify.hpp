#pragma once
#include "node.hpp"
#include "type.hpp"
#include "ctype.hpp"
#include "utility/isinstance.hpp"
#include "utility/markers.hpp"
#include "utility/snippets.hpp"

namespace backend {


class typedefify
    : public copier
{
private:
    std::shared_ptr<statement> m_typedef;
public:
    typedefify() :
        m_typedef() {}
    using copier::operator();
    result_type operator()(const suite &n) {
        std::vector<std::shared_ptr<statement> > stmts;
        for(auto i = n.begin();
            i != n.end();
            i++) {
            std::shared_ptr<statement> s =
                std::static_pointer_cast<statement>(
                    boost::apply_visitor(*this, *i));
            if (m_typedef) {
                stmts.push_back(m_typedef);
                m_typedef = std::shared_ptr<statement>();
            }
            stmts.push_back(s);
        }
        return result_type(
            new suite(std::move(stmts)));
    }
    result_type operator()(const bind &n) {

        //We can only deal with names in the LHS
        assert(detail::isinstance<name>(n.lhs()));

        const name& lhs = boost::get<const name&>(n.lhs());
        
        std::shared_ptr<ctype::type_t> unique_type =
            std::make_shared<ctype::monotype_t>(
                detail::typify(lhs.id()));
        std::shared_ptr<expression> rhs =
            std::static_pointer_cast<expression>(
                boost::apply_visitor(*this, n.rhs()));
        std::shared_ptr<name> new_lhs =
            std::make_shared<name>(lhs.id(),
                              get_type_ptr(lhs.type()),
                              unique_type);
        m_typedef =
            std::make_shared<typedefn>(
                get_ctype_ptr(lhs.ctype()),
                unique_type);
        return result_type(
            new bind(
                new_lhs, rhs));
    }
};

}
