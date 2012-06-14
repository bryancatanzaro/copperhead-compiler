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

phase_analyze::result_type phase_analyze::operator()(const suite& n) {
    vector<shared_ptr<const statement> > stmts;
    for(auto i = n.begin(); i != n.end(); i++) {
        auto p = std::static_pointer_cast<const statement>(boost::apply_visitor(*this, *i));
        while(m_additionals.size() > 0) {
            auto p = std::static_pointer_cast<const statement>(m_additionals.back());
            stmts.push_back(p);
            m_additionals.pop_back();
        }
        stmts.push_back(p);
        
    }
    return make_shared<const suite>(move(stmts));
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
                m_sources.insert(
                    make_pair(
                        arg_id,
                        make_set<shared_ptr<const name> >(arg_name.ptr())));
            } else {
                m_completions.insert(make_pair(arg_id, completion::invariant));
                m_sources.insert(
                    make_pair(
                        arg_id,
                        make_set<shared_ptr<const name> >()));
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

void phase_analyze::add_phase_boundary(const name& n) {
    std::cout << "Adding phase boundary for " << n.id() << std::endl;
    if (!m_sources.exists(n.id())) {
        return;
    }
    
    const set<shared_ptr<const name> >& sources =
        m_sources.find(n.id())->second;
    std::cout << "  Found sources, there are " << sources.size() << std::endl;


    for(auto i = sources.begin();
        i != sources.end();
        i++) {
        const shared_ptr<const name> p_s = *i;
        const name& s = *p_s;
        std::cout << "    " << s.id() << std::endl;
        if (m_completions.exists(s.id())) {
            completion c = m_completions.find(s.id())->second;
            if ((c == completion::invariant) ||
                (c == completion::total))
                continue;
        }
            
        shared_ptr<const name> p_result =
            make_shared<const name>(
                detail::complete(s.id()),
                s.type().ptr());
        
        shared_ptr<const tuple> pb_args =
            make_shared<const tuple>(
                make_vector<shared_ptr<const expression> >(p_s));
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
            make_pair(s.id(), p_result));
    }

    m_completions.insert(
        make_pair(n.id(), completion::total));
        
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
                //XXX need to add check back.
                add_phase_boundary(id);
                // //If completion hasn't been recorded, assume it's invariant
                // if (m_completions.exists(id.id())) {
                //     completion arg_completion =
                //         m_completions.find(id.id())->second;
                //     //Do we need a phase boundary for this argument?
                //     if (arg_completion < (*j)) {
                //         add_phase_boundary(id);
                //         new_arg = m_substitutions.find(id.id())->second;
                //     }
                // }
            } 
        }
        new_args.push_back(new_arg);
    }
    m_result_completion = fn_phase->result();
    return n.ptr();
}

phase_analyze::result_type phase_analyze::operator()(const bind& n) {
    if (!m_in_entry) {
        return n.ptr();
    }
    m_result_completion = completion::invariant;
    result_type rewritten = this->rewriter::operator()(n);
    //Update completion declarations
    if (detail::isinstance<name>(n.lhs())) {
        const name& lhs_name = detail::up_get<name>(n.lhs());
        m_completions.insert(make_pair(lhs_name.id(),
                                       m_result_completion));
    }

    //Update sources
    if (detail::isinstance<name>(n.lhs())) {
        const name& lhs_name = detail::up_get<name>(n.lhs());
        
        if (detail::isinstance<sequence_t>(lhs_name.type())) {
            //If the lhs is a sequence, it sources itself

            m_sources.insert(
                make_pair(
                    lhs_name.id(),
                    make_set<shared_ptr<const name> >(lhs_name.ptr())));
        } else if (detail::isinstance<tuple_t>(lhs_name.type())) {
            //Sourcing a tuple.
            if (detail::isinstance<name>(n.rhs())) {
                //Copy all sources of my input
                const name& rhs_name = boost::get<const name&>(n.rhs());
                m_sources.insert(
                    make_pair(lhs_name.id(),
                              m_sources.find(
                                  rhs_name.id())->second));
            } else if (detail::isinstance<apply>(n.rhs())) {
                const apply& rhs_apply = boost::get<const apply&>(n.rhs());
                const string& fn_id = rhs_apply.fn().id();
                if (fn_id == detail::snippet_make_tuple()) {
                    std::cout << "Sourcing from make_tuple!" << std::endl;
                    set<shared_ptr<const name> > input_sources;
                    for(auto i = rhs_apply.args().begin();
                        i != rhs_apply.args().end();
                        i++) {
                        //Copy all the sources of my inputs
                        //This builds up the transitive closure
                        //of all sources this tuple draws from
                        if (detail::isinstance<name>(*i)) {
                            const name& arg = boost::get<const name&>(*i);
                            std::cout << "  " << arg.id() << std::endl;

                            const set<shared_ptr<const name> >& arg_sources =
                                m_sources.find(arg.id())->second;
                            input_sources.insert(
                                arg_sources.begin(),
                                arg_sources.end());
                        }
                    }
                    m_sources.insert(
                        make_pair(lhs_name.id(),
                                  std::move(input_sources)));
                } 
            }
        }

        //If we haven't found a sources set for this lhs, it sources
        //nothing
        //This is true for scalars
        //As well as tuples of literals, etc.
        if (!m_sources.exists(lhs_name.id())) {
            m_sources.insert(
                make_pair(lhs_name.id(),
                          make_set<shared_ptr<const name> >()));
        }
    }
        
    return rewritten;
}

phase_analyze::result_type phase_analyze::operator()(const ret& n) {
    if (!m_in_entry) {
        return n.ptr();
    }
;
    //Returns can only be names
    assert(detail::isinstance<name>(n.val()));
    shared_ptr<const expression> new_result =
        n.val().ptr();
    const name& return_name = detail::up_get<name>(n.val());
    if (m_substitutions.exists(return_name.id())) {
        //Phase boundary already happened, use complete version
        new_result = m_substitutions.find(return_name.id())->second;
    } else {
        add_phase_boundary(return_name);
        if (m_substitutions.exists(return_name.id())) {
            new_result = m_substitutions.find(return_name.id())->second;
        } else {
            new_result = return_name.ptr();
        }
        // if (m_completions.exists(return_name.id())) {
        //     completion result_completion =
        //         m_completions.find(return_name.id())->second;
        //     //Results must be totally complete!!
        //     if (result_completion < completion::total) {
        //         add_phase_boundary(return_name);
        //         new_result = m_substitutions.find(return_name.id())->second;
        //     }
        // }
    }
    return make_shared<const ret>(new_result);
}


}
