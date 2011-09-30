#pragma once
#include "node.hpp"
#include "type.hpp"
#include "ctype.hpp"
#include "utility/isinstance.hpp"
#include "utility/markers.hpp"
#include "utility/snippets.hpp"
#include "py_printer.hpp"

#include "cuda_printer.hpp"

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
            auto t = get_type_ptr(n.type());
            auto ct = get_ctype_ptr(n.ctype());
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

            const name& pre_lhs = boost::get<const name&>(n.lhs());
            
            //Construct cuarray for result
            const ctype::sequence_t& pre_lhs_ct =
                boost::get<const ctype::sequence_t&>(pre_lhs.ctype());
            std::shared_ptr<ctype::type_t> sub_lhs_ct =
                get_ctype_ptr(pre_lhs_ct.sub());
            std::shared_ptr<ctype::tuple_t> tuple_sub_lhs_ct(
                new ctype::tuple_t(
                    std::vector<std::shared_ptr<ctype::type_t> >{sub_lhs_ct}));
            std::shared_ptr<ctype::type_t> result_ct(
                new ctype::cuarray_t(sub_lhs_ct));
            std::shared_ptr<type_t> result_t =
                get_type_ptr(pre_lhs.type());
            std::shared_ptr<name> result_name(
                new name(detail::wrap_array_id(pre_lhs.id()),
                         result_t,
                         result_ct));

            std::shared_ptr<expression> new_rhs =
                std::static_pointer_cast<expression>(
                    boost::apply_visitor(*this, n.rhs()));
            
            std::shared_ptr<bind> allocator(
                new bind(result_name, new_rhs));
            m_allocations.push_back(allocator);
            

            std::shared_ptr<name> new_lhs = std::static_pointer_cast<name>(
                boost::apply_visitor(*this, n.lhs()));
            std::shared_ptr<templated_name> getter_name(
                new templated_name(detail::get_remote_w(),
                                   tuple_sub_lhs_ct));
            std::shared_ptr<tuple> getter_args(
                new tuple(
                    std::vector<std::shared_ptr<expression> >{result_name}));
            std::shared_ptr<apply> getter_call(
                new apply(getter_name, getter_args));
            std::shared_ptr<bind> retriever(
                new bind(new_lhs, getter_call));
            return retriever;
        } else {
            return this->copier::operator()(n);
        }
    }
};

}
