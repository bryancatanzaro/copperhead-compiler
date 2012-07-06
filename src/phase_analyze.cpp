#include "phase_analyze.hpp"

using std::string;
using std::pair;
using std::shared_ptr;
using std::make_shared;
using std::move;
using std::vector;
using std::static_pointer_cast;
using std::make_pair;
using backend::utility::make_vector;
using backend::utility::make_set;
using std::set;

namespace backend {

phase_analyze::phase_analyze(const string& entry_point,
                             const registry& reg)
    : m_entry_point(entry_point), m_in_entry(false) {
    for(auto i = reg.fns().cbegin();
        i != reg.fns().cend();
        i++) {
        auto id = i->first;
        string fn_name = std::get<0>(id);
        auto info = i->second;
        m_fns.insert(make_pair(fn_name, info.phase().ptr()));
    }
}

phase_analyze::result_type phase_analyze::operator()(const procedure& n) {
    if (n.id().id() == m_entry_point) {
        m_in_entry = true;
        m_completions.begin_scope();
        //All inputs to the entry point must be totally or invariantly formed
        for(auto i = n.args().begin();
            i != n.args().end();
            i++) {
            //Arguments to procedures must be names
            assert(detail::isinstance<name>(*i));
            const name& arg_name = detail::up_get<name>(*i);
            const std::string& arg_id = arg_name.id();

            //If the input is typed as a sequence, it's totally formed
            //Otherwise it's invariantly formed (scalars)
            if (detail::isinstance<sequence_t>(arg_name.type())) {
                m_completions.insert(make_pair(arg_id, completion::total));
            } else {
                m_completions.insert(make_pair(arg_id, completion::invariant));
            }
            
        }
        shared_ptr<const suite> stmts =
            static_pointer_cast<const suite>(
                boost::apply_visitor(*this, n.stmts()));
        result_type result =
            make_shared<const procedure>(
                n.id().ptr(),
                n.args().ptr(),
                stmts,
                n.type().ptr(),
                n.ctype().ptr(),
                n.place());
        m_in_entry = false;
        m_completions.end_scope();
        return result;
    } else {
        return n.ptr();
    }
}

bool phase_analyze::add_phase_boundary_tuple(const name& n) {
    assert(m_tuples.exists(n.id()));
    bool need_boundary = false;
    const vector<shared_ptr<const literal> >& sources =
        m_tuples.find(n.id())->second;
    for(auto i = sources.begin();
        i != sources.end();
        i++) {
        if (detail::isinstance<name>(**i)) {
            const name& i_name = boost::get<const name&>(**i);
            need_boundary |= add_phase_boundary(i_name);
        }
    }
    if (!need_boundary) {
        return false;
    }
    shared_ptr<const name> p_result =
        make_shared<const name>(
            detail::complete(n.id()),
            n.type().ptr());

    vector<shared_ptr<const expression> > expr_sources;
    for(auto i = sources.begin();
        i != sources.end();
        i++) {
        if (detail::isinstance<name>(**i)) {
            const name& i_name = boost::get<const name&>(**i);
            if (m_substitutions.exists(i_name.id())) {
                expr_sources.push_back(
                    m_substitutions.find(i_name.id())->second->ptr());
            } else {
                expr_sources.push_back(*i);
            }
        } else {
            expr_sources.push_back(*i);
        }
    }
    shared_ptr<const tuple> pb_args =
        make_shared<const tuple>(
            move(expr_sources));
    shared_ptr<const apply> pb_apply =
        make_shared<const apply>(
            make_shared<const name>(
                detail::snippet_make_tuple()),
            pb_args);

    shared_ptr<const bind> result =
        make_shared<const bind>(p_result, pb_apply);
    m_additionals.push_back(result);
        
    //Register completion
    m_completions.insert(
        make_pair(p_result->id(), completion::total));
        
    //Register substitution
    m_substitutions.insert(
        make_pair(n.id(), p_result));
    return true;
}

bool phase_analyze::add_phase_boundary(const name& n) {
    if (m_tuples.exists(n.id())) {
        return add_phase_boundary_tuple(n);
    }

    
    if (m_completions.exists(n.id())) {
        completion c = m_completions.find(n.id())->second;
        if ((c == completion::invariant) ||
            (c == completion::total))
            return false;
    }
    shared_ptr<const name> p_result =
        make_shared<const name>(
            detail::complete(n.id()),
            n.type().ptr());
    
    shared_ptr<const tuple> pb_args =
        make_shared<const tuple>(
            make_vector<shared_ptr<const expression> >(n.ptr()));
    shared_ptr<const name> pb_name =
        make_shared<const name>(detail::phase_boundary());
    shared_ptr<const apply> pb_apply =
        make_shared<const apply>(pb_name, pb_args);
    shared_ptr<const bind> result =
        make_shared<const bind>(p_result, pb_apply);
    m_additionals.push_back(result);
        
    //Register completion
    m_completions.insert(
        make_pair(p_result->id(), completion::total));
        
    //Register substitution
    m_substitutions.insert(
        make_pair(n.id(), p_result));
    return true;
}

phase_analyze::result_type phase_analyze::operator()(const apply& n) {
    if (!m_in_entry) {
        return n.ptr();
    }
    
    const name& fn_name = n.fn();
   
    //If function not declared, assume it can't trigger a phase boundary
    if (m_fns.find(fn_name.id()) == m_fns.end()) {
        return n.ptr();
    }
    
    shared_ptr<const phase_t> fn_phase = m_fns.find(fn_name.id())->second;
    phase_t::iterator j = fn_phase->begin();
    //The phase type for the function must match the args given to it
    assert(fn_phase->size() == n.args().arity());

    vector<shared_ptr<const expression> > new_args;
    
    for(auto i = n.args().begin();
        i != n.args().end();
        i++, j++) {
        shared_ptr<const expression> new_arg = i->ptr();
        //If we have something other than a name, assume it's invariant
        if (detail::isinstance<name>(*i)) {
            const name& id = detail::up_get<name>(*i);
            if (m_substitutions.exists(id.id())) {
                //Phase boundary already took place, use the complete version
                //HEURISTIC HAZARD:
                //This might not always be the right choice
                new_arg = m_substitutions.find(id.id())->second;
            } else {
                //If completion hasn't been recorded, assume it's invariant
                if (m_completions.exists(id.id())) {
                    completion arg_completion =
                        m_completions.find(id.id())->second;
                    //Do we need a phase boundary for this argument?
                    if (arg_completion < (*j)) {
                        add_phase_boundary(id);
                        new_arg = m_substitutions.find(id.id())->second;
                    }
                }
            } 
        }
        new_args.push_back(new_arg);
    }
    m_result_completion = fn_phase->result();
    return n.ptr();
}


phase_analyze::result_type phase_analyze::make_tuple_analyze(const bind& n) {
    assert(detail::isinstance<name>(n.lhs()));
    const name& lhs = boost::get<const name&>(n.lhs());
    assert(detail::isinstance<apply>(n.rhs()));
    const apply& rhs = boost::get<const apply&>(n.rhs());
    const name& fn_name = rhs.fn();
    assert(fn_name.id() == detail::snippet_make_tuple());
    vector<shared_ptr<const literal> > sources;
    for(auto i = rhs.args().begin(); i != rhs.args().end(); i++) {
        assert(detail::isinstance<literal>(*i));
        const literal& i_name = detail::up_get<const literal&>(*i);
        sources.push_back(i_name.ptr());
    }
    completion glb = completion::invariant;
    for(auto i = sources.begin(); i != sources.end(); i++) {
        if (m_completions.exists((*i)->id())) {
            completion other = m_completions.find((*i)->id())->second;
            if (other < glb) {
                glb = other;
            }
        }
    }
    m_completions.insert(make_pair(lhs.id(), glb));
    m_tuples.insert(make_pair(lhs.id(),
                              move(sources)));
    return n.ptr();
}

phase_analyze::result_type phase_analyze::form_suite(
    const shared_ptr<const statement>& stmt) {
    if (m_additionals.empty()) {
        return stmt;
    }
    vector<shared_ptr<const statement> > stmts(m_additionals.begin(),
                                               m_additionals.end());
    m_additionals.clear();
    stmts.push_back(stmt);
    return make_shared<const suite>(move(stmts));
}
    

phase_analyze::result_type phase_analyze::operator()(const bind& n) {
    if (!m_in_entry) {
        return n.ptr();
    }

    //XXX
    //If function is make_tuple, handle it separately
    //Reason: make_tuple's phase type is not representable
    //at present. When we redo phase inference, we'll want to
    //fix this.
    //The problem is that it needs to be a GLB type
    //And we don't have a way to represent GLB types in the phase
    //type system
    if (detail::isinstance<apply>(n.rhs())) {
        const apply& rhs_apply = boost::get<const apply&>(n.rhs());
        const name& fn_name = rhs_apply.fn();
        
        if (fn_name.id() == detail::snippet_make_tuple()) {
            return make_tuple_analyze(n);
        }
    }

    //If we're rebinding, ensure the source is as complete as the dest
    //This happens, for example, in the body of a while loop
    //The source can be incomplete, while the destination needs to be
    //Complete.  In that case, trigger a phase boundary
    if (detail::isinstance<name>(n.rhs())) {
        const name& source_name = boost::get<const name&>(n.rhs());
        shared_ptr<const name> rhs = source_name.ptr();
        assert(detail::isinstance<name>(n.lhs()));
        const name& dest_name = boost::get<const name&>(n.lhs());
        if (m_completions.exists(source_name.id()) &&
            m_completions.exists(dest_name.id())) {
            completion source_completion =
                m_completions.find(source_name.id())->second;
            completion dest_completion =
                m_completions.find(dest_name.id())->second;
            if (source_completion < dest_completion) {
                add_phase_boundary(source_name);
                rhs = m_substitutions.find(source_name.id())->second;
            }
        }
        
        if (rhs == n.rhs().ptr()) {
            return n.ptr();
        } else {
            return form_suite(make_shared<const bind>(
                                  n.lhs().ptr(),
                                  rhs));
        }
    }

    m_result_completion = completion::invariant;
    result_type rewritten = this->rewriter<phase_analyze>::operator()(n);
    //Update completion declarations
    if (detail::isinstance<name>(n.lhs())) {
        const name& lhs_name = detail::up_get<name>(n.lhs());
        m_completions.insert(make_pair(lhs_name.id(),
                                       m_result_completion));
    }
    return form_suite(static_pointer_cast<const statement>(rewritten));
}

phase_analyze::result_type phase_analyze::operator()(const ret& n) {
    if (!m_in_entry) {
        return n.ptr();
    }

    //Returns can only be names
    assert(detail::isinstance<name>(n.val()));
    shared_ptr<const expression> new_result =
        n.val().ptr();
    const name& return_name = detail::up_get<name>(n.val());
    if (m_substitutions.exists(return_name.id())) {
        //Phase boundary already happened, use complete version
        new_result = m_substitutions.find(return_name.id())->second;
    } else {
        if (m_completions.exists(return_name.id())) {
            completion result_completion =
                m_completions.find(return_name.id())->second;
            //Results must be totally complete!!
            if (result_completion < completion::total) {
                add_phase_boundary(return_name);
                new_result = m_substitutions.find(return_name.id())->second;
            }
        }
    }
    return form_suite(make_shared<const ret>(new_result));
}


}
