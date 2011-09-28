#pragma once
#include "node.hpp"
#include "type.hpp"
#include "ctype.hpp"
#include "utility/isinstance.hpp"
#include "utility/markers.hpp"
#include "utility/snippets.hpp"
#include "py_printer.hpp"

namespace backend {


class wrap
    : public copier
{
private:
    const std::string& m_entry_point;
public:
    wrap(const std::string& entry_point) : m_entry_point(entry_point) {}
    using copier::operator();
    result_type operator()(const procedure &n) {
        if (n.id().id()  == m_entry_point) {
            std::vector<std::shared_ptr<expression> > wrapped_args;
            std::vector<std::shared_ptr<statement> > statements;
            py_printer pp(std::cout);

            for(auto i = n.args().begin();
                i != n.args().end();
                i++) {
                if (detail::isinstance<ctype::sequence_t, ctype::type_t>(
                        i->ctype())) {
                    //We can only handle names in procedure arguments
                    //at this time
                    bool is_node = detail::isinstance<name, node>(*i);
                    assert(is_node);
                    
                    std::shared_ptr<type_t> t =
                        boost::apply_visitor(m_tc, i->type());
                    const ctype::sequence_t& arg_c_type =
                        boost::get<const ctype::sequence_t&>(i->ctype());
                    std::shared_ptr<ctype::type_t> sub_ct =
                        boost::apply_visitor(m_ctc, arg_c_type.sub());
                    std::shared_ptr<ctype::type_t> ct(
                        new ctype::cuarray_t(sub_ct));
                    const name& arg_name =
                        boost::get<const name&>(*i);
                    std::shared_ptr<name> wrapped_name(
                        new name(detail::wrap_array_id(arg_name.id()),
                                 t, ct));
                    wrapped_args.push_back(wrapped_name);
                    std::shared_ptr<name> original_name =
                        std::static_pointer_cast<name>(boost::apply_visitor(*this, *i));
                    std::shared_ptr<ctype::tuple_t> tuple_sub_ct(
                        new ctype::tuple_t(
                            std::vector<std::shared_ptr<ctype::type_t> >{
                                sub_ct}));
                    std::shared_ptr<templated_name> getter_name(
                        new templated_name(detail::get_remote_r(),
                                           tuple_sub_ct));
                    std::shared_ptr<tuple_t> wrapped_tuple_t(
                        new tuple_t(
                            std::vector<std::shared_ptr<type_t> >{t}));
                    std::shared_ptr<ctype::tuple_t> wrapped_tuple_ct(
                        new ctype::tuple_t(
                            std::vector<std::shared_ptr<ctype::type_t> >{ct}));
                    std::shared_ptr<tuple> wrapped_name_tuple(
                        new tuple(
                            std::vector<std::shared_ptr<expression> >{wrapped_name},
                            wrapped_tuple_t, wrapped_tuple_ct));
                    std::shared_ptr<apply> getter_apply(
                        new apply(getter_name,
                                  wrapped_name_tuple));
                    std::shared_ptr<bind> do_get(
                        new bind(original_name, 
                                 getter_apply));
               
                    statements.push_back(do_get);
                } else {
                    wrapped_args.push_back(
                        std::static_pointer_cast<expression>(boost::apply_visitor(*this, *i)));
                }
            }
            
            for(auto i = n.stmts().begin();
                i != n.stmts().end();
                i++) {
                statements.push_back(
                    std::static_pointer_cast<statement>(boost::apply_visitor(*this, *i)));
            }
            std::shared_ptr<suite> stmts(
                new suite(std::move(statements)));
            //auto stmts = std::static_pointer_cast<suite>(this->operator()(n.stmts()));
            std::shared_ptr<tuple> args(
                new tuple(std::move(wrapped_args)));

            auto id = std::static_pointer_cast<name>(this->operator()(n.id()));
            auto t = boost::apply_visitor(m_tc, n.type());
            auto ct = boost::apply_visitor(m_ctc, n.ctype());
            return result_type(new procedure(id, args, stmts, t, ct));
        } else {
            return this->copier::operator()(n);
        }
        
    }

};
}
