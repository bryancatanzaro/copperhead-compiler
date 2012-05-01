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

        vector<shared_ptr<const expression> > new_args;
        vector<shared_ptr<const statement> > new_stmts;

        for(auto i = n.args().begin();
            i != n.args().end();
            i++) {
            const expression& arg = *i;
            //If they're sequences, wrap them, construct extractors
            if (detail::isinstance<ctype::sequence_t, ctype::type_t>(
                    i->ctype())) {

                //-------------Derive Wrapper type--------------
                
                //Get ctype of argument
                const ctype::sequence_t& arg_ct =
                    boost::get<const ctype::sequence_t&>(i->ctype());
                
                //Since argument is a sequence, get its c subtype
                const ctype::type_t& arg_sub_ct = arg_ct.sub();

                //Construct a wrapper type with the subtype
                shared_ptr<const ctype::type_t> wrapped_arg_p_ct =
                    make_shared<const ctype::cuarray_t>(
                        arg_sub_ct.ptr());

                //-------------Build Wrapped argument--------------
                
                //Argument to procedure must be a name
                bool is_name = detail::isinstance<name>(arg);
                assert(is_name);

                const name& arg_name =
                    boost::get<const name&>(arg);
                
                shared_ptr<const name> p_wrapped_name(
                    new name(detail::wrap_array_id(arg_name.id()),
                             arg.type().ptr(), wrapped_arg_p_ct));
                new_args.push_back(p_wrapped_name);

                //-------------Build Extractor-------------------
                
                //Build type of implementation sequence
                shared_ptr<const ctype::type_t> p_impl_seq_ct =
                    make_shared<const ctype::polytype_t>(
                        make_vector<shared_ptr<const ctype::type_t> >
                        (make_shared<const ctype::monotype_t>(copperhead::to_string(m_target)))
                        (arg_sub_ct.ptr()),
                        make_shared<const ctype::monotype_t>("sequence"));

                //Stick it in a tuple for the templated_name
                shared_ptr<const ctype::tuple_t> p_tuple_impl_seq_ct =
                    make_shared<const ctype::tuple_t>(
                        make_vector<shared_ptr<const ctype::type_t> >(p_impl_seq_ct));
                //getter_name: make_sequence<sequence<tag, float> >
                shared_ptr<const templated_name> p_getter_name =
                    make_shared<const templated_name>(
                        detail::make_sequence(),
                        p_tuple_impl_seq_ct);
                
                //Build arguments for extractor
                shared_ptr<const tuple_t> p_wrapped_tuple_t =
                    make_shared<const tuple_t>(
                        make_vector<shared_ptr<const type_t> >(arg.type().ptr()));
                shared_ptr<const ctype::tuple_t> p_wrapped_tuple_ct =
                    make_shared<const ctype::tuple_t>(
                        make_vector<shared_ptr<const ctype::type_t> >(arg.ctype().ptr()));
                shared_ptr<const tuple> p_wrapped_name_tuple =
                    make_shared<const tuple>(
                        make_vector<shared_ptr<const expression> >(p_wrapped_name)
                        (make_shared<const apply>(
                            make_shared<const name>(copperhead::to_string(m_target)),
                            make_shared<const tuple>(make_vector<shared_ptr<const expression> >())))
                        (make_shared<const literal>("false")),
                        p_wrapped_tuple_t, p_wrapped_tuple_ct);
                shared_ptr<const apply> p_getter_apply =
                    make_shared<const apply>(p_getter_name,
                                             p_wrapped_name_tuple);
                
                //Bind extractor to arg id
                shared_ptr<const bind> p_extraction =
                    make_shared<const bind>(
                        arg.ptr(),
                        p_getter_apply);
                new_stmts.push_back(p_extraction);
                        
            } else {
                //Fallback
                shared_ptr<const expression> passed =
                    static_pointer_cast<const expression>(
                        boost::apply_visitor(*this, *i));
                new_args.push_back(passed);
            }
        }

        //Derive new output ctype
        vector<shared_ptr<const ctype::type_t> > new_arg_p_cts;
        for(auto i = new_args.begin();
            i != new_args.end();
            i++) {
            new_arg_p_cts.push_back((*i)->ctype().ptr());
        }
        const ctype::fn_t& previous_ct =
            boost::get<const ctype::fn_t&>(n.ctype());
        const ctype::type_t& previous_c_res_t =
            previous_ct.result();

        shared_ptr<const ctype::type_t> p_c_res_t;

        if (detail::isinstance<ctype::sequence_t>(
                previous_c_res_t)) {
            const ctype::sequence_t& res_seq_t =
                boost::get<const ctype::sequence_t&>(previous_c_res_t);
            shared_ptr<const ctype::type_t> sub_res_t =
                res_seq_t.sub().ptr();
                
            p_c_res_t = make_shared<const ctype::cuarray_t>(
                sub_res_t);
        } else {
            p_c_res_t = previous_c_res_t.ptr();
        }
        

        shared_ptr<const ctype::type_t> p_new_ct =
            make_shared<const ctype::fn_t>(
                make_shared<const ctype::tuple_t>(
                    move(new_arg_p_cts)),
                p_c_res_t);

        for(auto i = n.stmts().begin();
            i != n.stmts().end();
            i++) {
            new_stmts.push_back(
                static_pointer_cast<const statement>(
                    boost::apply_visitor(*this, *i)));
        }
        m_wrapping = false;
        
        return make_shared<const procedure>(
            n.id().ptr(),
            make_shared<const tuple>(
                move(new_args)),
            make_shared<const suite>(
                move(new_stmts)),
            n.type().ptr(),
            p_new_ct);
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
