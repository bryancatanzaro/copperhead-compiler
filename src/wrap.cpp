#include "wrap.hpp"

namespace backend {

wrap::wrap(const std::string& entry_point) : m_entry_point(entry_point),
                                             m_wrapping(false),
                                             m_wrapper(){}


wrap::result_type wrap::operator()(const procedure &n) {
    if (n.id().id()  == m_entry_point) {
        m_wrapping = true;
                        
        std::vector<std::shared_ptr<expression> > wrapper_args;
        std::vector<std::shared_ptr<expression> > getter_args;
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
                    get_type_ptr(i->type());
                    
                const ctype::sequence_t& arg_c_type =
                    boost::get<const ctype::sequence_t&>(i->ctype());
                std::shared_ptr<ctype::type_t> sub_ct =
                    get_ctype_ptr(arg_c_type.sub());
                std::shared_ptr<ctype::type_t> ct(
                    new ctype::cuarray_t(sub_ct));
                const name& arg_name =
                    boost::get<const name&>(*i);
                std::shared_ptr<name> wrapped_name(
                    new name(detail::wrap_array_id(arg_name.id()),
                             t, ct));
                wrapper_args.push_back(wrapped_name);
                    
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
                getter_args.push_back(getter_apply);
            } else {
                //Fallback
                std::shared_ptr<expression> passed =
                    std::static_pointer_cast<expression>(
                        boost::apply_visitor(*this, *i));
                wrapper_args.push_back(passed);
                getter_args.push_back(passed);                    
            }
        }
            

        //Derive the new output c type of the wrapper procedure
        const ctype::fn_t& previous_c_t =
            boost::get<const ctype::fn_t&>(n.ctype());
        const ctype::type_t& previous_c_res_t =
            previous_c_t.result();
            
        std::shared_ptr<ctype::type_t> new_wrap_ct;
        std::shared_ptr<ctype::type_t> new_ct;
        std::shared_ptr<ctype::type_t> new_res_ct;
        std::vector<std::shared_ptr<ctype::type_t> > wrap_arg_cts;
        for(auto i=wrapper_args.begin();
            i != wrapper_args.end();
            i++) {
            wrap_arg_cts.push_back(
                get_ctype_ptr((*i)->ctype()));
        }
        std::shared_ptr<ctype::tuple_t> new_wrap_args_ct =
            std::make_shared<ctype::tuple_t>(std::move(wrap_arg_cts));
        std::shared_ptr<ctype::tuple_t> new_args_ct =
            std::static_pointer_cast<ctype::tuple_t>(
                get_ctype_ptr(previous_c_t.args()));

        if (detail::isinstance<ctype::sequence_t, ctype::type_t>(
                previous_c_res_t)) {
            const ctype::sequence_t& res_seq_t =
                boost::get<const ctype::sequence_t&>(previous_c_res_t);
            std::shared_ptr<ctype::type_t> sub_res_t =
                get_ctype_ptr(res_seq_t.sub());
                
            std::shared_ptr<ctype::type_t> new_wrap_res_ct(
                new ctype::cuarray_t(sub_res_t));
            new_wrap_ct = std::shared_ptr<ctype::fn_t>(
                new ctype::fn_t(new_wrap_args_ct, new_wrap_res_ct));

            new_res_ct =
                std::make_shared<ctype::templated_t>(
                    std::make_shared<ctype::monotype_t>(
                        "boost::shared_ptr"),
                    std::vector<std::shared_ptr<ctype::type_t> >{
                        std::make_shared<ctype::templated_t>(
                            std::make_shared<ctype::monotype_t>("cuarray"),
                            std::vector<std::shared_ptr<ctype::type_t> >{
                                sub_res_t})});
            new_ct = std::make_shared<ctype::fn_t>(
                new_args_ct,
                new_res_ct);
                
        } else {
            std::shared_ptr<ctype::type_t> new_res_ct =
                get_ctype_ptr(previous_c_res_t);
            new_wrap_ct = std::shared_ptr<ctype::fn_t>(
                new ctype::fn_t(new_wrap_args_ct, new_res_ct));
            new_ct = get_ctype_ptr(previous_c_t);
            new_res_ct = get_ctype_ptr(previous_c_res_t);
        }



            
        std::shared_ptr<name> wrapper_proc_id =
            std::make_shared<name>(
                detail::wrap_proc_id(n.id().id()));

        auto t = get_type_ptr(n.type());
        auto res_t = get_type_ptr(
            std::static_pointer_cast<fn_t>(t)->result());
            
        std::shared_ptr<name> result_id =
            std::make_shared<name>(
                "result", res_t, new_res_ct);
        std::shared_ptr<bind> make_the_call =
            std::make_shared<bind>(
                result_id,
                std::make_shared<apply>(
                    std::static_pointer_cast<name>(get_node_ptr(n.id())),
                    std::make_shared<tuple>(
                        std::move(getter_args))));
        std::shared_ptr<ret> dynamize =
            std::make_shared<ret>(
                std::make_shared<apply>(
                    std::make_shared<name>("wrap_cuarray"),
                    std::make_shared<tuple>(
                        std::vector<std::shared_ptr<expression> >{
                            result_id})));
        std::shared_ptr<suite> wrapper_stmts =
            std::make_shared<suite>(
                std::vector<std::shared_ptr<statement> >{
                    make_the_call,
                        dynamize});
        std::shared_ptr<tuple> wrapper_args_tuple(
            new tuple(std::move(wrapper_args)));
        std::shared_ptr<procedure> completed_wrapper =
            std::make_shared<procedure>(wrapper_proc_id,
                                        wrapper_args_tuple,
                                        wrapper_stmts,
                                        t, new_wrap_ct, "");
        m_wrapper = completed_wrapper;

        result_type rewritten =
            std::make_shared<procedure>(
                std::static_pointer_cast<name>(
                    get_node_ptr(n.id())),
                std::static_pointer_cast<tuple>(
                    get_node_ptr(n.args())),
                std::static_pointer_cast<suite>(
                    boost::apply_visitor(*this, n.stmts())),
                get_type_ptr(n.type()),
                new_ct,
                "");
        m_wrapping = false;
        return rewritten;
    } else {
        return this->copier::operator()(n);
    }
        
}
wrap::result_type wrap::operator()(const ret& n) {
    if (m_wrapping && detail::isinstance<name, node>(n.val())) {
        const name& val =
            boost::get<const name&>(n.val());
        if (detail::isinstance<ctype::sequence_t,
                               ctype::type_t>(val.ctype())) {
            std::shared_ptr<name> array_wrapped(
                new name(
                    detail::wrap_array_id(val.id()),
                    get_type_ptr(val.type()),
                    get_ctype_ptr(val.ctype())));
            return result_type(new ret(array_wrapped));
        }
    }
    return this->copier::operator()(n);
}
wrap::result_type wrap::operator()(const suite&n) {
    std::vector<std::shared_ptr<statement> > stmts;
    for(auto i = n.begin();
        i != n.end();
        i++) {
        stmts.push_back(
            std::static_pointer_cast<statement>(
                boost::apply_visitor(*this, *i)));
    }
    if (!m_wrapping && m_wrapper) {
        stmts.push_back(m_wrapper);
        m_wrapper = std::shared_ptr<procedure>();
    }
    return result_type(new suite(std::move(stmts)));
}

}