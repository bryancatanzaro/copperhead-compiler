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

            const name& pre_lhs = boost::get<const name&>(n.lhs());
            const apply& pre_apply = boost::get<const apply&>(
                n.rhs());
            const tuple& pre_args = boost::get<const tuple&>(
                pre_apply.args());

            //Construct cuarray for result
            const ctype::sequence_t& pre_lhs_ct =
                boost::get<const ctype::sequence_t&>(pre_lhs.ctype());
            std::shared_ptr<ctype::type_t> sub_lhs_ct =
                boost::apply_visitor(m_ctc, pre_lhs_ct.sub());
            std::shared_ptr<ctype::type_t> result_ct(
                new ctype::cuarray_t(sub_lhs_ct));
            std::shared_ptr<type_t> result_t =
                boost::apply_visitor(m_tc, pre_lhs.type());
            std::shared_ptr<name> result_name(
                new name(detail::wrap_array_id(pre_lhs.id()),
                         result_t,
                         result_ct));
            std::shared_ptr<ctype::tuple_t> tuple_sub_lhs_ct(
                new ctype::tuple_t(
                    std::vector<std::shared_ptr<ctype::type_t> >{
                        sub_lhs_ct}));
            std::shared_ptr<templated_name> maker_name(
                        new templated_name(detail::make_remote(),
                                           tuple_sub_lhs_ct));

            //XXX THIS IS A NULLARY SHAPE INFERENCE HACK XXX
            const name& first_arg = boost::get<const name&>(
                *pre_args.begin());
            std::shared_ptr<name> size(
                new name(first_arg.id() + ".size()"));
            std::shared_ptr<tuple> size_tupled(
                new tuple(std::vector<std::shared_ptr<expression> >{size}));
            std::shared_ptr<apply> maker_call(
                new apply(maker_name, size_tupled));
            
            std::shared_ptr<bind> allocator(
                new bind(result_name, maker_call));
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
            m_allocations.push_back(retriever);
            

            //Return convert
            std::vector<std::shared_ptr<expression> > args;

            for(auto i = pre_args.begin();
                i != pre_args.end();
                i++) {
                args.push_back(
                    std::static_pointer_cast<expression>(
                        boost::apply_visitor(*this, *i)));
            }
            args.push_back(new_lhs);
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
