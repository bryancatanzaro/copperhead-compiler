#include "functorize.hpp"

using std::shared_ptr;
using std::make_shared;
using std::string;
using std::static_pointer_cast;
using std::vector;
using std::move;
using backend::utility::make_vector;

namespace backend {

namespace detail {

type_corresponder::type_corresponder(const shared_ptr<type_t>& input,
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
    m_working = static_pointer_cast<sequence_t>(m_working)->p_sub();
    boost::apply_visitor(*this, n.sub());
}

void type_corresponder::operator()(const tuple_t &n) {
    //m_working must be a tuple_t or else the typechecking is wrong
    assert(detail::isinstance<tuple_t>(*m_working));
    const tuple_t& working_tuple = boost::get<const tuple_t&>(*m_working);
    auto j = working_tuple.p_begin();
    for(auto i = n.begin();
        i != n.end();
        ++i, ++j) {
        m_working = *j;
        boost::apply_visitor(*this, *i);
    }
    
}

void type_corresponder::operator()(const fn_t &n) {
    if (detail::isinstance<polytype_t>(*m_working)) {
        //If the fn type is a polytype, we can't harvest any correspondence
        return;
    }
    if (!detail::isinstance<fn_t>(*m_working)) {
        std::cout << "This has a which of " << m_working->which() << std::endl;
        const monotype_t& mt =
            boost::get<const monotype_t&>(*m_working);
        std::cout << "This is a monotype of " << mt.name() << std::endl;
    }
    assert(detail::isinstance<fn_t>(*m_working));
    const fn_t& working_fn = boost::get<const fn_t&>(*m_working);
    m_working = working_fn.p_args();
    boost::apply_visitor(*this, n.args());
    m_working = working_fn.p_result();
    boost::apply_visitor(*this, n.result());
}

type_translator::type_translator(const type_translator::type_map& corresponded)
    : m_corresponded(corresponded) {}

type_translator::result_type
type_translator::operator()(const monotype_t& m) const {
    //Is monotype_t in type map?
    if(m_corresponded.find(m.name()) != m_corresponded.end()) {
        return m_corresponded.find(m.name())->second;
    } else {
        return result_type(new monotype_t(m.name()));
    }
}

type_translator::result_type
type_translator::operator()(const polytype_t& m) const {
    //Shouldn't be called
    assert(false);
    return void_mt;
}

type_translator::result_type
type_translator::operator()(const tuple_t& m) const {
    vector<shared_ptr<type_t> > subs;
    for(auto i = m.begin();
        i != m.end();
        i++) {
        subs.push_back(boost::apply_visitor(*this, *i));
    }
    return result_type(new tuple_t(move(subs)));
}

type_translator::result_type
type_translator::operator()(const sequence_t& m) const {
    return result_type(new sequence_t(boost::apply_visitor(*this, m.sub())));
}

type_translator::result_type
type_translator::operator()(const fn_t& m) const {
    shared_ptr<tuple_t> new_args = static_pointer_cast<tuple_t>(
        boost::apply_visitor(*this, m.args()));
    shared_ptr<type_t> new_result = boost::apply_visitor(*this, m.result());
    return result_type(new fn_t(new_args, new_result));
}

}



void functorize::make_type_map(const apply& n) {
    m_type_map.clear();
    const name& fn_name = n.fn();

    //If function name is not a polytype, the type map should be empty
    if (!detail::isinstance<polytype_t>(fn_name.type()))
        return;
    const polytype_t& fn_polytype = boost::get<const polytype_t&>(fn_name.type());
    //Polytype must contain a function type
    assert(detail::isinstance<fn_t>(fn_polytype.monotype()));
    const fn_t& fn_monotype = boost::get<const fn_t&>(fn_polytype.monotype());
    const tuple_t& fn_arg_t = fn_monotype.args();
    vector<shared_ptr<type_t> > arg_types;
    for(auto i = n.args().begin();
        i != n.args().end();
        i++) {
        arg_types.push_back(i->p_type());
    }
    shared_ptr<tuple_t> arg_t =
        make_shared<tuple_t>(move(arg_types));

    
    detail::type_corresponder tc(arg_t, m_type_map);
    boost::apply_visitor(tc, fn_arg_t);
        
}



shared_ptr<expression> functorize::instantiate_fn(const name& n,
                                                  shared_ptr<type_t> p_t) {
    string id = n.id();
    const type_t& n_t = n.type();
    if (!detail::isinstance<polytype_t>(n_t)) {
        //The function is monomorphic. Instantiate a functor.
        return make_shared<apply>(
            make_shared<name>(detail::fnize_id(id)),
            make_shared<tuple>(
                make_vector<shared_ptr<expression> >()));
    }
    //Use already populated type map to instantiate the
    //Polymorphic functor with the types it needs in situ

    //The function type is polymorphic
    //Find the monomorphic type
    const polytype_t& n_pt = boost::get<const polytype_t&>(n_t);
    const type_t& n_mt = n_pt.monotype();
        
    //First, create a type map relating the types of the
    //polymorphic function being instantiated to the
    //types in the apply which is instantiating this function.
    type_map fn_to_apl;
    detail::type_corresponder tc(p_t, fn_to_apl);
    boost::apply_visitor(tc, n_mt);

    vector<shared_ptr<type_t> > instantiated_types;
    for(auto i = n_pt.begin();
        i != n_pt.end();
        i++) {
        string fn_t_name = i->name();
        //The name of this type should be in the type map
        assert(fn_to_apl.find(fn_t_name)!=fn_to_apl.end());
        shared_ptr<type_t> apl_t = fn_to_apl.find(fn_t_name)->second;
        //Instantiate this type with the other type map
        instantiated_types.push_back(
            boost::apply_visitor(
                detail::type_translator(m_type_map),
                *apl_t));
    }
    vector<shared_ptr<ctype::type_t> > instantiated_ctypes;
    detail::cu_to_c ctc;
    for(auto i = instantiated_types.begin();
        i != instantiated_types.end();
        i++) {
        instantiated_ctypes.push_back(
            boost::apply_visitor(ctc, **i));
    }
    return make_shared<apply>(
        make_shared<templated_name>(
            detail::fnize_id(id),
            make_shared<ctype::tuple_t>(move(instantiated_ctypes)),
            n.p_type(),
            n.p_ctype()),
        make_shared<tuple>(
            make_vector<shared_ptr<expression> >()));

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
        m_fns.insert(fn_name);
    }
               

}

functorize::result_type functorize::operator()(const apply &n) {
    //If the function we're applying is polymorphic,
    //Figure out what types it's being instantiated with
    make_type_map(n);

        
    vector<shared_ptr<expression> > n_arg_list;
    const tuple& n_args = n.args();
        
    shared_ptr<fn_t> fn_type;
    if (detail::isinstance<fn_t>(n.fn().type())) {
        fn_type = static_pointer_cast<fn_t>(
            n.fn().p_type());
    } else {
        //Must be a polytype_t(fn_t)
        assert(detail::isinstance<polytype_t>(n.fn().type()));
        const polytype_t& pt = boost::get<const polytype_t&>(n.fn().type());
        fn_type = static_pointer_cast<fn_t>(
            pt.p_monotype());
    }
    const tuple_t& args_type = fn_type->args();
    auto arg_type = args_type.p_begin();
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
            assert(detail::isinstance<fn_t>(**arg_type));
            const fn_t& in_situ_fn_t =
                boost::get<const fn_t&>(**arg_type);
            const tuple_t& in_situ_fn_args_t =
                in_situ_fn_t.args();
            //Build the list of augmented arg types
            //That include the types from the original fn args
            //Plus the types of the closed over objects
            //Starting with the ones that were given
            vector<shared_ptr<type_t> > augmented_args_t(
                in_situ_fn_args_t.p_begin(),
                in_situ_fn_args_t.p_end());
            //Translate arg types that were closed over, add
            //to fn arg types.
            for(auto i = n_closure.args().begin();
                i != n_closure.args().end();
                i++) {
                augmented_args_t.push_back(
                    boost::apply_visitor(
                        detail::type_translator(
                            m_type_map),
                        i->type()));
            }
            shared_ptr<fn_t> augmented_fn_t =
                make_shared<fn_t>(
                    make_shared<tuple_t>(
                        move(augmented_args_t)),
                    in_situ_fn_t.p_result());
            
            shared_ptr<expression> instantiated_fn =
                instantiate_fn(closed_fn, augmented_fn_t);
            n_arg_list.push_back(
                make_shared<closure>(
                    n_closure.p_args(),
                    instantiated_fn,
                    n_closure.p_type(),
                    n_closure.p_ctype()));
        } else if (detail::isinstance<name>(*n_arg)) {
            const name& n_name = boost::get<const name&>(*n_arg);
            const string id = n_name.id();
            auto found = m_fns.find(id);
            if (found == m_fns.end()) {
                n_arg_list.push_back(
                    static_pointer_cast<expression>(
                        boost::apply_visitor(*this, *n_arg)));
            } else {
                //We've found a function to instantiate
                n_arg_list.push_back(
                    instantiate_fn(n_name, *arg_type));
            }

        } else {
            //fallback
            n_arg_list.push_back(
                static_pointer_cast<expression>(
                    boost::apply_visitor(*this, *n_arg)));
        }
    }
    auto n_fn = static_pointer_cast<name>(this->rewriter::operator()(n.fn()));
    auto new_args = shared_ptr<tuple>(new tuple(move(n_arg_list)));
    return shared_ptr<apply>(new apply(n_fn, new_args));
}
    
functorize::result_type functorize::operator()(const suite &n) {
    vector<shared_ptr<statement> > stmts;
    for(auto i = n.begin(); i != n.end(); i++) {
        auto p = static_pointer_cast<statement>(boost::apply_visitor(*this, *i));
        stmts.push_back(p);
        while(m_additionals.size() > 0) {
            auto p = static_pointer_cast<statement>(m_additionals.back());
            stmts.push_back(p);
            m_additionals.pop_back();
        }
    }
    return result_type(
        new suite(
            move(stmts)));
}

shared_ptr<ctype::type_t> get_return_type(const procedure& n) {
    shared_ptr<ctype::type_t> ret_type;
    if (detail::isinstance<ctype::polytype_t>(n.ctype())) {
        const ctype::monotype_t& n_mt =
            boost::get<const ctype::polytype_t&>(n.ctype()).monotype();
        assert(detail::isinstance<ctype::fn_t>(n_mt));
        const ctype::fn_t& n_t = boost::get<const ctype::fn_t&>(
            n_mt);
        ret_type = n_t.p_result();
    } else {
        assert(detail::isinstance<ctype::fn_t>(n.ctype()));
        const ctype::fn_t& n_t = boost::get<const ctype::fn_t&>(
            n.ctype());
        ret_type = n_t.p_result();
    }
    return ret_type;
}
        
functorize::result_type functorize::operator()(const procedure &n) {
    auto n_proc = static_pointer_cast<procedure>(this->rewriter::operator()(n));
    if (n_proc->id().id() != m_entry_point) {
        //Add result_type declaration
        shared_ptr<ctype::type_t> origin = get_return_type(n);
        shared_ptr<ctype::type_t> rename(
            new ctype::monotype_t("result_type"));
        shared_ptr<typedefn> res_defn(
            new typedefn(origin, rename));

            
        shared_ptr<tuple> forward_args = static_pointer_cast<tuple>(this->rewriter::operator()(n_proc->args()));
        shared_ptr<name> forward_name = static_pointer_cast<name>(this->rewriter::operator()(n_proc->id()));
        shared_ptr<apply> op_call(new apply(forward_name, forward_args));
        shared_ptr<ret> op_ret(new ret(op_call));
        vector<shared_ptr<statement> > op_body_stmts =
            make_vector<shared_ptr<statement> >(op_ret);
        shared_ptr<suite> op_body(new suite(move(op_body_stmts)));
        auto op_args = static_pointer_cast<tuple>(this->rewriter::operator()(n.args()));
        shared_ptr<name> op_id(new name(string("operator()")));
        shared_ptr<procedure> op(
            new procedure(
                op_id, op_args, op_body,
                n.p_type(),
                n.p_ctype()));
        shared_ptr<suite> st_body =
            make_shared<suite>(
                make_vector<shared_ptr<statement> >(res_defn)(op));
        shared_ptr<name> st_id =
            make_shared<name>(detail::fnize_id(n_proc->id().id()));
        shared_ptr<structure> st;
        if (detail::isinstance<ctype::polytype_t>(n.ctype())) {
            const ctype::polytype_t& pt =
                boost::get<const ctype::polytype_t&>(n.ctype());
            vector<shared_ptr<ctype::type_t> > typevars;
            for(auto i = pt.p_begin();
                i != pt.p_end();
                i++) {
                typevars.push_back(*i);
            }
            st = make_shared<structure>(st_id, st_body, move(typevars));
        } else {
            st = make_shared<structure>(st_id, st_body);
        }
        m_additionals.push_back(st);
        m_fns.insert(n_proc->id().id());
    }
    return n_proc;

}
    

}
