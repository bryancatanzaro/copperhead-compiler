#include "wrap.hpp"

#include "cuda_printer.hpp"

using std::string;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using std::move;
using backend::utility::make_vector;

namespace backend {

wrap::wrap(const string& entry_point) : m_entry_point(entry_point),
                                             m_wrapping(false),
                                             m_wrapper(),
                                             m_wrap_decl(){}


wrap::result_type wrap::operator()(const procedure &n) {
    if (n.id().id()  == m_entry_point) {
        m_wrapping = true;
                        
        vector<shared_ptr<expression> > wrapper_args;
        vector<shared_ptr<expression> > getter_args;
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
                    
                shared_ptr<type_t> t = i->p_type();
                    
                const ctype::sequence_t& arg_c_type =
                    boost::get<const ctype::sequence_t&>(i->ctype());
                shared_ptr<ctype::type_t> sub_ct =
                    arg_c_type.p_sub();
                shared_ptr<ctype::type_t> ct(
                    new ctype::cuarray_t(sub_ct));
                const name& arg_name =
                    boost::get<const name&>(*i);
                shared_ptr<name> wrapped_name(
                    new name(detail::wrap_array_id(arg_name.id()),
                             t, ct));
                wrapper_args.push_back(wrapped_name);
                    
                shared_ptr<ctype::tuple_t> tuple_sub_ct =
                    make_shared<ctype::tuple_t>(
                        make_vector<shared_ptr<ctype::type_t> >(
                            sub_ct));
                shared_ptr<templated_name> getter_name =
                    make_shared<templated_name>(
                        detail::get_remote_r(),
                        tuple_sub_ct);
                shared_ptr<tuple_t> wrapped_tuple_t =
                    make_shared<tuple_t>(
                        make_vector<shared_ptr<type_t> >(t));
                shared_ptr<ctype::tuple_t> wrapped_tuple_ct =
                    make_shared<ctype::tuple_t>(
                        make_vector<shared_ptr<ctype::type_t> >(ct));
                shared_ptr<tuple> wrapped_name_tuple =
                    make_shared<tuple>(
                        make_vector<shared_ptr<expression> >(wrapped_name),
                        wrapped_tuple_t, wrapped_tuple_ct);
                shared_ptr<apply> getter_apply =
                    make_shared<apply>(getter_name,
                                       wrapped_name_tuple);
                getter_args.push_back(getter_apply);
            } else {
                //Fallback
                shared_ptr<expression> passed =
                    static_pointer_cast<expression>(
                        boost::apply_visitor(*this, *i));
                wrapper_args.push_back(passed);
                getter_args.push_back(passed);                    
            }
        }

        //Derive the new output c type of the wrapper procedure
        const ctype::fn_t& previous_c_t =
            boost::get<const ctype::fn_t&>(n.ctype());
        std::shared_ptr<ctype::fn_t> p_previous_c_t =
            static_pointer_cast<ctype::fn_t>(n.p_ctype());
        const ctype::type_t& previous_c_res_t =
            previous_c_t.result();
            
        shared_ptr<ctype::type_t> new_wrap_ct;
        shared_ptr<ctype::type_t> new_ct;
        shared_ptr<ctype::type_t> new_res_ct;
        vector<shared_ptr<ctype::type_t> > wrap_arg_cts;
        for(auto i=wrapper_args.begin();
            i != wrapper_args.end();
            i++) {
            wrap_arg_cts.push_back(
                (*i)->p_ctype());
        }
        shared_ptr<ctype::tuple_t> new_wrap_args_ct =
            make_shared<ctype::tuple_t>(move(wrap_arg_cts));
        for(auto i = new_wrap_args_ct->p_begin();
            i != new_wrap_args_ct->p_end();
            i++) {
            
        }
        shared_ptr<ctype::tuple_t> new_args_ct =
            static_pointer_cast<ctype::tuple_t>(
                previous_c_t.p_args());
        
        if (detail::isinstance<ctype::sequence_t, ctype::type_t>(
                previous_c_res_t)) {
            const ctype::sequence_t& res_seq_t =
                boost::get<const ctype::sequence_t&>(previous_c_res_t);
            shared_ptr<ctype::type_t> sub_res_t =
                res_seq_t.p_sub();
                
            shared_ptr<ctype::type_t> new_wrap_res_ct(
                new ctype::cuarray_t(sub_res_t));
            new_wrap_ct = shared_ptr<ctype::fn_t>(
                new ctype::fn_t(new_wrap_args_ct, new_wrap_res_ct));

            new_res_ct =
                make_shared<ctype::polytype_t>(
                    make_vector<shared_ptr<ctype::type_t> >(
                        make_shared<ctype::polytype_t>(
                            make_vector<shared_ptr<ctype::type_t> >(
                                sub_res_t),
                            make_shared<ctype::monotype_t>("cuarray"))),
                    make_shared<ctype::monotype_t>(
                        "boost::shared_ptr"));
            new_ct = make_shared<ctype::fn_t>(
                new_args_ct,
                new_res_ct);
                
        } else {
            new_wrap_ct = make_shared<ctype::fn_t>(
                new_wrap_args_ct, previous_c_t.p_result());
            new_ct = p_previous_c_t;
            new_res_ct = previous_c_t.p_result();
        }

        shared_ptr<name> wrapper_proc_id =
            make_shared<name>(
                detail::wrap_proc_id(n.id().id()));
        auto t = n.p_type();
        auto res_t = static_pointer_cast<fn_t>(t)->p_result();
            
        shared_ptr<name> result_id =
            make_shared<name>(
                "result", res_t, new_res_ct);
        shared_ptr<bind> make_the_call =
            make_shared<bind>(
                result_id,
                make_shared<apply>(
                    static_pointer_cast<name>(get_node_ptr(n.id())),
                    make_shared<tuple>(
                        move(getter_args))));
        shared_ptr<ret> dynamize;

        if (detail::isinstance<ctype::sequence_t, ctype::type_t>(
                previous_c_res_t)) {
            dynamize =
                make_shared<ret>(
                    make_shared<apply>(
                        make_shared<name>("wrap_cuarray"),
                        make_shared<tuple>(
                            make_vector<shared_ptr<expression> >(
                                result_id))));
        } else {
            dynamize =
                make_shared<ret>(
                    result_id);
        }
        shared_ptr<suite> wrapper_stmts =
            make_shared<suite>(
                make_vector<shared_ptr<statement> >
                (make_the_call)
                (dynamize));
        shared_ptr<tuple> wrapper_args_tuple =
            make_shared<tuple>(
                move(wrapper_args));
        shared_ptr<procedure> completed_wrapper =
            make_shared<procedure>(wrapper_proc_id,
                                   wrapper_args_tuple,
                                   wrapper_stmts,
                                   t, new_wrap_ct, "");
        m_wrapper = completed_wrapper;
        m_wrap_decl =
            make_shared<procedure>(
                wrapper_proc_id,
                wrapper_args_tuple,
                make_shared<suite>(make_vector<shared_ptr<statement> >()),
                t,
                new_wrap_ct,
                "");
        
        result_type rewritten =
            make_shared<procedure>(
                static_pointer_cast<name>(
                    get_node_ptr(n.id())),
                static_pointer_cast<tuple>(
                    get_node_ptr(n.args())),
                static_pointer_cast<suite>(
                    boost::apply_visitor(*this, n.stmts())),
                n.p_type(),
                new_ct,
                "");
        m_wrapping = false;
        return rewritten;
    } else {
        return this->rewriter::operator()(n);
    }
        
}
wrap::result_type wrap::operator()(const ret& n) {
    if (m_wrapping && detail::isinstance<name, node>(n.val())) {
        const name& val =
            boost::get<const name&>(n.val());
        if (detail::isinstance<ctype::sequence_t,
                               ctype::type_t>(val.ctype())) {
            shared_ptr<name> array_wrapped =
                make_shared<name>(
                    detail::wrap_array_id(val.id()),
                    val.p_type(),
                    val.p_ctype());
            return result_type(new ret(array_wrapped));
        }
    }
    shared_ptr<ret> rewritten =
        static_pointer_cast<ret>(this->rewriter::operator()(n));
    
    return rewritten;
}
wrap::result_type wrap::operator()(const suite&n) {
    vector<shared_ptr<statement> > stmts;
    for(auto i = n.begin();
        i != n.end();
        i++) {
        stmts.push_back(
            static_pointer_cast<statement>(
                boost::apply_visitor(*this, *i)));
    }
    if (!m_wrapping && m_wrapper) {
        stmts.push_back(m_wrapper);
        m_wrapper = shared_ptr<procedure>();
    }
    return result_type(new suite(move(stmts)));
}

shared_ptr<procedure> wrap::p_wrap_decl() const {
    return m_wrap_decl;
}

}
