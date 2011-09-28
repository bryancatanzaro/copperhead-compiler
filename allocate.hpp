#pragma once
#include "node.hpp"
#include "type.hpp"
#include "ctype.hpp"
#include "utility/isinstance.hpp"
#include "utility/markers.hpp"
#include "utility/snippets.hpp"
#include "py_printer.hpp"

namespace backend {


class allocate
    : public copier
{
private:
    const std::string& m_entry_point;
    bool m_in_entry;
    std::vector<std::shared_ptr<statement> > m_allocations;
public:
    allocate(const std::string& entry_point) : m_entry_point(entry_point),
                                               m_in_entry(false),
                                               m_allocations{}
        {}
    using copier::operator();
    result_type operator()(const procedure &n) {
        if (n.id().id()  == m_entry_point) {
            m_in_entry = true;
            std::vector<std::shared_ptr<statement> > statements;
            for(auto i = n.stmts().begin();
                i != n.stmts().end();
                i++) {
                auto new_stmt = std::static_pointer_cast<statement>(
                    boost::apply_visitor(*this, *i));
                if (m_allocations.size() > 0) {
                    for(auto j = m_allocations.begin();
                        j != m_allocations.end();
                        j++) {
                        statements.push_back(*j);
                    }
                    m_allocations.clear();
                }
                statements.push_back(new_stmt);
            }
            std::shared_ptr<suite> stmts(
                new suite(std::move(statements)));
            std::shared_ptr<tuple> args =
                std::static_pointer_cast<tuple>(
                    boost::apply_visitor(*this, n.args()));
            auto t = boost::apply_visitor(m_tc, n.type());
            auto ct = boost::apply_visitor(m_ctc, n.ctype());
            std::shared_ptr<name> id =
                std::static_pointer_cast<name>(
                    boost::apply_visitor(*this, n.id()));
            result_type allocated(
                new procedure(id, args, stmts, t, ct));

            m_in_entry = false;
            return allocated;
        } else {
            return this->copier::operator()(n);
        }
    }
    result_type operator()(const bind &n) {
        if (m_in_entry &&
            detail::isinstance<ctype::sequence_t, ctype::type_t>(
                n.lhs().ctype()) &&
            detail::isinstance<apply, node>(n.rhs())) {

            //We can only deal with names on the LHS of a bind
            bool lhs_is_name = detail::isinstance<name, node>(n.lhs());
            assert(lhs_is_name);


            

            //Return convert
            const apply& pre_apply = boost::get<const apply&>(
                n.rhs());
            std::vector<std::shared_ptr<expression> > args;
            const tuple& pre_args = boost::get<const tuple&>(
                pre_apply.args());
            for(auto i = pre_args.begin();
                i != pre_args.end();
                i++) {
                args.push_back(
                    std::static_pointer_cast<expression>(
                        boost::apply_visitor(*this, *i)));
            }
            args.push_back(
                std::static_pointer_cast<expression>(
                    boost::apply_visitor(*this, n.lhs())));
            std::shared_ptr<tuple> converted_args(
                new tuple(std::move(args)));
            std::shared_ptr<name> id =
                std::static_pointer_cast<name>(
                    boost::apply_visitor(*this, pre_apply.fn()));
            std::shared_ptr<apply> converted_apply(
                new apply(id, converted_args));
            std::shared_ptr<call> make_the_call(
                new call(converted_apply));
            return make_the_call;
        } else {
            return this->copier::operator()(n);
        }
    }
};

}
