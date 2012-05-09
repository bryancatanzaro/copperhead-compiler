#include "functorize.hpp"

using std::shared_ptr;
using std::make_shared;
using std::string;
using std::static_pointer_cast;
using std::vector;
using std::move;
using backend::utility::make_vector;

#include <iostream>
#include "type_printer.hpp"

namespace backend {

namespace detail {

type_corresponder::type_corresponder(const shared_ptr<const type_t>& input,
                                     type_map& corresponded)
    : m_working(input), m_corresponded(corresponded) {}

void type_corresponder::operator()(const monotype_t &n) {
    string id = n.name();
    m_corresponded.insert(std::make_pair(id, m_working));
}

void type_corresponder::operator()(const polytype_t &n) {
    //Polytypes are not allowed to be nested;
    assert(false);
}

void type_corresponder::operator()(const sequence_t &n) {
    //m_working must be a sequence_t or else the typechecking is wrong
    assert(detail::isinstance<sequence_t>(*m_working));
    m_working = static_pointer_cast<const sequence_t>(m_working)->sub().ptr();
    boost::apply_visitor(*this, n.sub());
}

void type_corresponder::operator()(const tuple_t &n) {
    assert(detail::isinstance<tuple_t>(*m_working));
    const tuple_t& working_tuple = boost::get<const tuple_t&>(*m_working);
    auto j = working_tuple.begin();
    for(auto i = n.begin();
        i != n.end();
        ++i, ++j) {
        m_working = j->ptr();
        boost::apply_visitor(*this, *i);
    }
    
}

void type_corresponder::operator()(const fn_t &n) {
    if (detail::isinstance<polytype_t>(*m_working)) {
        //If the fn type is a polytype, we can't harvest any correspondence
        return;
    }
    assert(detail::isinstance<fn_t>(*m_working));
    const fn_t& working_fn = boost::get<const fn_t&>(*m_working);
    m_working = working_fn.args().ptr();
    boost::apply_visitor(*this, n.args());
    m_working = working_fn.result().ptr();
    boost::apply_visitor(*this, n.result());
}

}

shared_ptr<const expression> functorize::instantiate_fn(const name& n,
                                                        const type_t &t) {
    string id = n.id();
    //Function id must exist in type registry
    assert(m_fns.find(id) != m_fns.end());
    const type_t& n_t = *m_fns[id];
    
    if (!detail::isinstance<polytype_t>(n_t)) {
        //The function is monomorphic. Instantiate a functor.
        return make_shared<const apply>(
            make_shared<const name>(detail::fnize_id(id)),
            make_shared<const tuple>(
                make_vector<shared_ptr<const expression> >()));
    }
    type_map tm;
    const polytype_t& n_pt = boost::get<const polytype_t&>(n_t);
    const monotype_t& n_mt = n_pt.monotype();
    detail::type_corresponder tc(t.ptr(), tm);
    boost::apply_visitor(tc, n_mt);

    vector<shared_ptr<const type_t> > instantiated_types;
    for(auto i = n_pt.begin();
        i != n_pt.end();
        i++) {
        string fn_t_name = i->name();
        //The name of this type should be in the type map
        assert(tm.find(fn_t_name)!=tm.end());
        instantiated_types.push_back(tm.find(fn_t_name)->second);
    }
    vector<shared_ptr<const ctype::type_t> > instantiated_ctypes;
    detail::cu_to_c ctc;
    for(auto i = instantiated_types.begin();
        i != instantiated_types.end();
        i++) {
        instantiated_ctypes.push_back(
            boost::apply_visitor(ctc, **i));
    }
    return make_shared<const apply>(
        make_shared<const templated_name>(
            detail::fnize_id(id),
            make_shared<const ctype::tuple_t>(move(instantiated_ctypes)),
            n.type().ptr(),
            n.ctype().ptr()),
        make_shared<const tuple>(
            make_vector<shared_ptr<const expression> >()));
}
    
functorize::functorize(const string& entry_point,
                       const registry& reg)
    : m_entry_point(entry_point), m_additionals(make_vector<result_type>()),
      m_reg(reg) {
    for(auto i = reg.fns().cbegin();
        i != reg.fns().cend();
        i++) {
        auto id = i->first;
        string fn_name = std::get<0>(id);
        std::shared_ptr<const type_t> fn_t = i->second.type().ptr();
        m_fns.insert(std::make_pair(fn_name, fn_t));
    }
}

functorize::result_type functorize::operator()(const apply &n) {
    //If the function being applied has no type, return the apply node
    //unchanged
    //This can happen due to synthesized apply nodes that shouldn't be
    //functorized.
    if (n.fn().type().ptr() == void_mt) {
        return n.ptr();
    }
    
        
    vector<shared_ptr<const expression> > n_arg_list;
    const tuple& n_args = n.args();
        
    shared_ptr<const fn_t> fn_type;
    if (detail::isinstance<fn_t>(n.fn().type())) {
        fn_type = static_pointer_cast<const fn_t>(
            n.fn().type().ptr());
    } else {
        //Must be a polytype_t(fn_t)
        assert(detail::isinstance<polytype_t>(n.fn().type()));
        const polytype_t& pt = boost::get<const polytype_t&>(n.fn().type());
        fn_type = static_pointer_cast<const fn_t>(
            pt.monotype().ptr());
    }
    const tuple_t& args_type = fn_type->args();
    auto arg_type = args_type.begin();
    for(auto n_arg = n_args.begin();
        n_arg != n_args.end();
        ++n_arg, ++arg_type) {
        if (detail::isinstance<closure>(*n_arg)) {
            const closure& n_closure = boost::get<const closure&>(*n_arg);
            //Can only close over function names
            assert(detail::isinstance<name>(n_closure.body()));
            const name& closed_fn = boost::get<const name&>(n_closure.body());
            auto found = m_fns.find(closed_fn.id());
            //Can only close over function names
            assert(found != m_fns.end());

            //Need to synthesize a new arg type that describes the
            //type of the closure, not the type of the argument

            //First, make sure the arg type is a function type
            assert(detail::isinstance<fn_t>(*arg_type));
            const fn_t& in_situ_fn_t =
                boost::get<const fn_t&>(*arg_type);
            const tuple_t& in_situ_fn_args_t =
                in_situ_fn_t.args();
            //Build the list of augmented arg types
            //That include the types from the original fn args
            //Plus the types of the closed over objects
            //Starting with the ones that were given
            vector<shared_ptr<const type_t> > augmented_args_t;
            for(auto i = in_situ_fn_args_t.begin();
                i != in_situ_fn_args_t.end();
                i++) {
                augmented_args_t.push_back(i->ptr());
            }
            //Translate arg types that were closed over, add
            //to fn arg types.
            for(auto i = n_closure.args().begin();
                i != n_closure.args().end();
                i++) {
                //XXX TRANSLATION DELETED
                augmented_args_t.push_back(
                    i->type().ptr());
            }
            shared_ptr<const fn_t> augmented_fn_t =
                make_shared<const fn_t>(
                    make_shared<const tuple_t>(
                        move(augmented_args_t)),
                    in_situ_fn_t.result().ptr());
            
            shared_ptr<const expression> instantiated_fn =
                instantiate_fn(closed_fn, *augmented_fn_t);
            n_arg_list.push_back(
                make_shared<const closure>(
                    n_closure.args().ptr(),
                    instantiated_fn,
                    n_closure.type().ptr(),
                    n_closure.ctype().ptr()));
        } else if (detail::isinstance<name>(*n_arg)) {
            const name& n_name = boost::get<const name&>(*n_arg);
            const string id = n_name.id();
            auto found = m_fns.find(id);
            if (found == m_fns.end()) {
                n_arg_list.push_back(
                    static_pointer_cast<const expression>(
                        boost::apply_visitor(*this, *n_arg)));
            } else {
                //We've found a function to instantiate
                n_arg_list.push_back(
                    instantiate_fn(n_name, *arg_type));
            }

        } else {
            //fallback
            n_arg_list.push_back(
                static_pointer_cast<const expression>(
                    boost::apply_visitor(*this, *n_arg)));
        }
    }
    auto n_fn = static_pointer_cast<const name>(this->rewriter::operator()(n.fn()));
    auto new_args = shared_ptr<const tuple>(new tuple(move(n_arg_list)));
    return shared_ptr<const apply>(new apply(n_fn, new_args));
}
    
functorize::result_type functorize::operator()(const suite &n) {
    vector<shared_ptr<const statement> > stmts;
    for(auto i = n.begin(); i != n.end(); i++) {
        auto p = static_pointer_cast<const statement>(
            boost::apply_visitor(*this, *i));
        stmts.push_back(p);
        while(m_additionals.size() > 0) {
            auto p = static_pointer_cast<const statement>(m_additionals.back());
            stmts.push_back(p);
            m_additionals.pop_back();
        }
    }
    return result_type(
        new suite(
            move(stmts)));
}

shared_ptr<const ctype::type_t> get_return_type(const procedure& n) {
    shared_ptr<const ctype::type_t> ret_type;
    if (detail::isinstance<ctype::polytype_t>(n.ctype())) {
        const ctype::monotype_t& n_mt =
            boost::get<const ctype::polytype_t&>(n.ctype()).monotype();
        assert(detail::isinstance<ctype::fn_t>(n_mt));
        const ctype::fn_t& n_t = boost::get<const ctype::fn_t&>(
            n_mt);
        ret_type = n_t.result().ptr();
    } else {
        assert(detail::isinstance<ctype::fn_t>(n.ctype()));
        const ctype::fn_t& n_t = boost::get<const ctype::fn_t&>(
            n.ctype());
        ret_type = n_t.result().ptr();
    }
    return ret_type;
}
        
functorize::result_type functorize::operator()(const procedure &n) {
    auto n_proc = static_pointer_cast<const procedure>(this->rewriter::operator()(n));
    if (n_proc->id().id() != m_entry_point) {
        //Wrap every non entry point function in a functor struct.
        
        //Add result_type declaration
        shared_ptr<const ctype::type_t> origin = get_return_type(n);
        shared_ptr<const ctype::type_t> rename(
            new ctype::monotype_t("result_type"));
        shared_ptr<const typedefn> res_defn(
            new typedefn(origin, rename));

            
        shared_ptr<const tuple> forward_args =
            static_pointer_cast<const tuple>(this->rewriter::operator()(n_proc->args()));
        shared_ptr<const name> forward_name =
            static_pointer_cast<const name>(this->rewriter::operator()(n_proc->id()));
        shared_ptr<const apply> op_call(new apply(forward_name, forward_args));
        shared_ptr<const ret> op_ret(new ret(op_call));
        vector<shared_ptr<const statement> > op_body_stmts =
            make_vector<shared_ptr<const statement> >(op_ret);
        shared_ptr<const suite> op_body(new suite(move(op_body_stmts)));
        auto op_args =
            static_pointer_cast<const tuple>(this->rewriter::operator()(n.args()));
        shared_ptr<const name> op_id(new name(string("operator()")));
        shared_ptr<const procedure> op(
            new procedure(
                op_id, op_args, op_body,
                n.type().ptr(),
                n.ctype().ptr()));
        shared_ptr<const suite> st_body =
            make_shared<const suite>(
                make_vector<shared_ptr<const statement> >(res_defn)(op));
        shared_ptr<const name> st_id =
            make_shared<const name>(detail::fnize_id(n_proc->id().id()));
        shared_ptr<const structure> st;
        if (detail::isinstance<ctype::polytype_t>(n.ctype())) {
            const ctype::polytype_t& pt =
                boost::get<const ctype::polytype_t&>(n.ctype());
            vector<shared_ptr<const ctype::type_t> > typevars;
            for(auto i = pt.begin();
                i != pt.end();
                i++) {
                typevars.push_back(i->ptr());
            }
            st = make_shared<const structure>(st_id, st_body, move(typevars));
        } else {
            st = make_shared<const structure>(st_id, st_body);
        }
        m_additionals.push_back(st);
        m_fns.insert(std::make_pair(
                         n_proc->id().id(),
                         n_proc->type().ptr()));
    }
    return n_proc;

}
    

}
