#pragma once
#include "node.hpp"
#include "type.hpp"
#include "ctype.hpp"
#include "utility/isinstance.hpp"
#include "utility/markers.hpp"

namespace backend {


class wrap
    : public copier
{
private:
    const std::string& m_entry_point;
    bool m_entry;
public:
    wrap(const std::string& entry_point) : m_entry_point(entry_point),
                                           m_entry(false) {}
    using copier::operator();
    result_type operator()(const procedure &n) {
        if (n.id().id()  == m_entry_point) {
            m_entry = true;
            std::vector<std::shared_ptr<expression> > wrapped_args;
            std::vector<std::shared_ptr<statement> > statements;
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
                    
                    std::shared_ptr<ctype::type_t> sub_ct =
                        boost::apply_visitor(m_ctc, i->ctype().sub());
                    std::shared_ptr<ctype::type_t> ct(
                        new ctype::cuarray_t(sub_ct));
                    std::shared_ptr<name> wrapped_name(
                        new name(detail::wrap_array_id(i->id()),
                                 t, ct));
                    
                    wrapped_args.push_back(wrapped_name);
                    std::shared_ptr<name> original_name =
                        this->copier::operator()(*i);

                    std::shared_ptr<ctype::tuple_t> tuple_sub_ct(
                        new ctype::tuple_t(
                            std::vector<std::shared_ptr<ctype::type_t> >{
                                sub_ct}));
                    std::shared_ptr<templated_name> getter_name(
                        new templated_name(detail::get_remote_r(),
                                           tuple_sub_ct));
                    std::shared_ptr<apply> getter_apply(getter_name,
                                           wrapped_name);
                    std::shared_ptr<bind> do_get(
                        new bind(original_name, 
                                 getter_apply));
                    statements.push_back(do_get);
                }   
            }
            for(auto i = n.stmts().begin();
                i != n.stmts().end();
                i++) {
                statements.push_back(this->operator()(*i));
            }
            
            std::shared_ptr<suite> stmts(
                new suite(std::move(statements)));            
            auto args = std::static_pointer_cast<tuple>(this->operator()(n.args()));

            auto id = std::static_pointer_cast<name>(this->operator()(n.id()));
            auto t = boost::apply_visitor(m_tc, n.type());
            auto ct = boost::apply_visitor(m_ctw, n.ctype());
            m_entry = false;
            return result_type(new procedure(id, args, stmts, t, ct));
        } else {
            return this->copier::operator()(n);
        }
    }
    result_type operator()(const name &n) {
        if (!m_entry) {
            return this->copier::operator()(n);
        } else {
            auto t = boost::apply_visitor(m_tc, n.type());
            auto ct = boost::apply_visitor(m_ctw, n.ctype());
            return result_type(new name(n.id(), t, ct));
        }
    }
};
}
