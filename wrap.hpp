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
    bool m_wrapping;
public:
    wrap(const std::string& entry_point) : m_entry_point(entry_point),
                                           m_wrapping(false){}
    using copier::operator();
    result_type operator()(const procedure &n) {
        if (n.id().id()  == m_entry_point) {
            m_wrapping = true;
            std::vector<std::shared_ptr<expression> > wrapped_args;
            std::vector<std::shared_ptr<statement> > statements;
            py_printer pp(std::cout);

            //Wrap input arguments
            for(auto i = n.args().begin();
                i != n.args().end();
                i++) {
                //If they're sequences, wrap them, construct extractors
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
                        std::static_pointer_cast<name>(
                            boost::apply_visitor(*this, *i));
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
                    //Fallback
                    wrapped_args.push_back(
                        std::static_pointer_cast<expression>(boost::apply_visitor(*this, *i)));
                }
            }
            //Concatenate the extractors and the body of the procedure
            for(auto i = n.stmts().begin();
                i != n.stmts().end();
                i++) {
                statements.push_back(
                    std::static_pointer_cast<statement>(boost::apply_visitor(*this, *i)));
            }

            //Derive the new output c type of the procedure
            const ctype::fn_t& previous_c_t =
                boost::get<const ctype::fn_t&>(n.ctype());
            const ctype::type_t& previous_c_res_t =
                previous_c_t.result();
            
            std::shared_ptr<ctype::type_t> new_ct;
            std::vector<std::shared_ptr<ctype::type_t> > arg_cts;
            for(auto i=wrapped_args.begin();
                i != wrapped_args.end();
                i++) {
                arg_cts.push_back(
                    boost::apply_visitor(m_ctc, (*i)->ctype()));
            }
            std::shared_ptr<ctype::tuple_t> new_args_ct(
                new ctype::tuple_t(std::move(arg_cts)));

            if (detail::isinstance<ctype::sequence_t, ctype::type_t>(
                    previous_c_res_t)) {
                const ctype::sequence_t& res_seq_t =
                    boost::get<const ctype::sequence_t&>(previous_c_res_t);
                std::shared_ptr<ctype::type_t> sub_res_t =
                    boost::apply_visitor(m_ctc, res_seq_t.sub());
                
                std::shared_ptr<ctype::type_t> new_res_ct(
                    new ctype::cuarray_t(sub_res_t));
                new_ct = std::shared_ptr<ctype::fn_t>(
                    new ctype::fn_t(new_args_ct, new_res_ct));
                                                      
            } else {
                std::shared_ptr<ctype::type_t> new_res_ct =
                    boost::apply_visitor(m_ctc, previous_c_res_t);
                new_ct = std::shared_ptr<ctype::fn_t>(
                    new ctype::fn_t(new_args_ct, new_res_ct));
            }

            
            std::shared_ptr<suite> stmts(
                new suite(std::move(statements)));
            
            std::shared_ptr<tuple> args(
                new tuple(std::move(wrapped_args)));

            auto id = std::static_pointer_cast<name>(this->operator()(n.id()));
            auto t = boost::apply_visitor(m_tc, n.type());
            result_type completed_wrap(
                new procedure(id, args, stmts, t, new_ct));
            m_wrapping = false;
            return completed_wrap;
        } else {
            return this->copier::operator()(n);
        }
        
    }
    result_type operator()(const ret& n) {
        const name& val =
            boost::get<const name&>(n.val());
        if (m_wrapping && detail::isinstance<ctype::sequence_t,
            ctype::type_t>(val.ctype())) {
            std::shared_ptr<name> array_wrapped(
                new name(
                    detail::wrap_array_id(val.id()),
                    boost::apply_visitor(m_tc, val.type()),
                    boost::apply_visitor(m_ctc, val.ctype())));
            std::shared_ptr<tuple> wrap_call_tuple(
                new tuple(
                    std::vector<std::shared_ptr<expression> >{array_wrapped}));
            std::shared_ptr<apply> wrap_call(
                new apply(
                    std::shared_ptr<name>(
                        new name(detail::wrap())),
                    wrap_call_tuple));
            return result_type(new ret(wrap_call));
        } else {
            return this->copier::operator()(n);
        }
    }
};
}
