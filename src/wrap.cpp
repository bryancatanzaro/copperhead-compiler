#include "wrap.hpp"

#include "cpp_printer.hpp"
#include "type_printer.hpp"
using std::string;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using std::move;
using backend::utility::make_vector;

namespace backend {

wrap::wrap(const copperhead::system_variant& target,
           const string& entry_point)
    : m_target(target),
      m_entry_point(entry_point),
      m_wrapping(false) {}


wrap::result_type wrap::operator()(const procedure &n) {
    if (n.id().id()  == m_entry_point) {
        m_wrapping = true;
                        
        vector<shared_ptr<const expression> > wrapper_args;
        vector<shared_ptr<const expression> > getter_args;
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
                    
                shared_ptr<const type_t> t = i->type().ptr();
                    
                const ctype::sequence_t& arg_c_type =
                    boost::get<const ctype::sequence_t&>(i->ctype());
                shared_ptr<const ctype::type_t> sub_ct =
                    arg_c_type.sub().ptr();
                shared_ptr<const ctype::type_t> ct(
                    new ctype::cuarray_t(sub_ct));
                const name& arg_name =
                    boost::get<const name&>(*i);
                shared_ptr<const name> wrapped_name(
                    new name(detail::wrap_array_id(arg_name.id()),
                             t, ct));
                wrapper_args.push_back(wrapped_name);

                shared_ptr<const ctype::type_t> impl_seq_ct =
                    make_shared<const ctype::polytype_t>(
                        make_vector<shared_ptr<const ctype::type_t> >
                        (make_shared<const ctype::monotype_t>(copperhead::to_string(m_target)))
                        (sub_ct),
                        make_shared<const ctype::monotype_t>("sequence"));
        
                shared_ptr<const ctype::tuple_t> tuple_impl_seq_ct =
                    make_shared<const ctype::tuple_t>(
                        make_vector<shared_ptr<const ctype::type_t> >(impl_seq_ct));
                
                shared_ptr<const templated_name> getter_name =
                    make_shared<const templated_name>(
                        detail::make_sequence(),
                        tuple_impl_seq_ct);
                shared_ptr<const tuple_t> wrapped_tuple_t =
                    make_shared<const tuple_t>(
                        make_vector<shared_ptr<const type_t> >(t));
                shared_ptr<const ctype::tuple_t> wrapped_tuple_ct =
                    make_shared<const ctype::tuple_t>(
                        make_vector<shared_ptr<const ctype::type_t> >(ct));
                shared_ptr<const tuple> wrapped_name_tuple =
                    make_shared<const tuple>(
                        make_vector<shared_ptr<const expression> >(wrapped_name)
                        (make_shared<const apply>(
                            make_shared<const name>(copperhead::to_string(m_target)),
                            make_shared<const tuple>(make_vector<shared_ptr<const expression> >())))
                        (make_shared<const literal>("false")),
                        wrapped_tuple_t, wrapped_tuple_ct);
                shared_ptr<const apply> getter_apply =
                    make_shared<const apply>(getter_name,
                                             wrapped_name_tuple);
                getter_args.push_back(getter_apply);
            } else {
                //Fallback
                shared_ptr<const expression> passed =
                    static_pointer_cast<const expression>(
                        boost::apply_visitor(*this, *i));
                wrapper_args.push_back(passed);
                getter_args.push_back(passed);                    
            }
        }

        //Derive the new output c type of the wrapper procedure
        const ctype::fn_t& previous_c_t =
            boost::get<const ctype::fn_t&>(n.ctype());
        std::shared_ptr<const ctype::fn_t> p_previous_c_t =
            static_pointer_cast<const ctype::fn_t>(n.ctype().ptr());
        const ctype::type_t& previous_c_res_t =
            previous_c_t.result();
            
        shared_ptr<const ctype::type_t> new_wrap_ct;
        shared_ptr<const ctype::type_t> new_ct;
        shared_ptr<const ctype::type_t> new_res_ct;
        vector<shared_ptr<const ctype::type_t> > wrap_arg_cts;
        for(auto i=wrapper_args.begin();
            i != wrapper_args.end();
            i++) {
            wrap_arg_cts.push_back(
                (*i)->ctype().ptr());
        }
        shared_ptr<const ctype::tuple_t> new_wrap_args_ct =
            make_shared<const ctype::tuple_t>(move(wrap_arg_cts));
        shared_ptr<const ctype::tuple_t> new_args_ct =
            static_pointer_cast<const ctype::tuple_t>(
                previous_c_t.args().ptr());
        
        if (detail::isinstance<ctype::sequence_t, ctype::type_t>(
                previous_c_res_t)) {
            const ctype::sequence_t& res_seq_t =
                boost::get<const ctype::sequence_t&>(previous_c_res_t);
            shared_ptr<const ctype::type_t> sub_res_t =
                res_seq_t.sub().ptr();
                
            shared_ptr<const ctype::type_t> new_wrap_res_ct(
                new ctype::cuarray_t(sub_res_t));
            new_wrap_ct = shared_ptr<const ctype::fn_t>(
                new ctype::fn_t(new_wrap_args_ct, new_wrap_res_ct));

            new_res_ct = make_shared<const ctype::monotype_t>("sp_cuarray");
            new_ct = make_shared<const ctype::fn_t>(
                new_args_ct,
                new_res_ct);
                
        } else {
            new_wrap_ct = make_shared<const ctype::fn_t>(
                new_wrap_args_ct, previous_c_t.result().ptr());
            new_ct = p_previous_c_t;
            new_res_ct = previous_c_t.result().ptr();
        }

        shared_ptr<const name> wrapper_proc_id =
            make_shared<const name>(
                detail::wrap_proc_id(n.id().id()));

        const fn_t& n_fn_t = boost::get<const fn_t>(n.type());
        const type_t& res_t = n_fn_t.result();
        
        
        shared_ptr<const name> result_id =
            make_shared<const name>(
                "result", res_t.ptr(), new_res_ct);
        shared_ptr<const bind> make_the_call =
            make_shared<const bind>(
                result_id,
                make_shared<const apply>(
                    n.id().ptr(),
                    make_shared<const tuple>(
                        move(getter_args))));
        shared_ptr<const ret> dynamize = make_shared<const ret>(result_id);

        shared_ptr<const suite> wrapper_stmts =
            make_shared<const suite>(
                make_vector<shared_ptr<const statement> >
                (make_the_call)
                (dynamize));
        shared_ptr<const tuple> wrapper_args_tuple =
            make_shared<const tuple>(
                move(wrapper_args));
        shared_ptr<const procedure> completed_wrapper =
            make_shared<const procedure>(wrapper_proc_id,
                                         wrapper_args_tuple,
                                         wrapper_stmts,
                                         n.type().ptr(), new_wrap_ct, "");
        m_wrapper = completed_wrapper;
                    
        result_type rewritten =
            make_shared<const procedure>(
                static_pointer_cast<const name>(
                    n.id().ptr()),
                static_pointer_cast<const tuple>(
                    n.args().ptr()),
                static_pointer_cast<const suite>(
                    boost::apply_visitor(*this, n.stmts())),
                n.type().ptr(),
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
            shared_ptr<const name> array_wrapped =
                make_shared<const name>(
                    detail::wrap_array_id(val.id()),
                    val.type().ptr(),
                    val.ctype().ptr());
            return result_type(new ret(array_wrapped));
        }
    }
    shared_ptr<const ret> rewritten =
        static_pointer_cast<const ret>(this->rewriter::operator()(n));
    
    return rewritten;
}


}
