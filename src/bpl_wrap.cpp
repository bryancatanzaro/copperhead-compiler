#include "bpl_wrap.hpp"

using std::string;
using std::shared_ptr;
using std::make_shared;
using std::vector;
using std::move;
using std::static_pointer_cast;

namespace backend {


bpl_wrap::bpl_wrap(const string& entry_point,
                   const registry& reg)
    : m_entry_point(entry_point),
      m_outer(true) {
    for(auto i = reg.includes().cbegin();
        i != reg.includes().cend();
        i++) {
        m_includes.push_back(
            shared_ptr<include>(
                new include(
                    shared_ptr<literal>(
                        new literal(*i)))));
    }
    
}

bpl_wrap::result_type bpl_wrap::operator()(const procedure &n) {
    string wrapper_id = detail::wrap_proc_id(m_entry_point);
    //This is the wrapper procedure
    if (n.id().id() == wrapper_id) {
        //Create a blank declaration and store it for later use
        m_wrap_decl =
            make_shared<procedure>(
                static_pointer_cast<name>(get_node_ptr(n.id())),
                static_pointer_cast<tuple>(get_node_ptr(n.args())),
                make_shared<suite>(vector<shared_ptr<statement> >{}),
                get_type_ptr(n.type()),
                get_ctype_ptr(n.ctype()),
                n.place());
    }
    return this->copier::operator()(n);
}

bpl_wrap::result_type bpl_wrap::operator()(const suite& n) {
    if (!m_outer) {
        return this->copier::operator()(n);
    }
    m_outer = false;
    vector<shared_ptr<statement> > stmts;
        
    //Copy includes into statements
    stmts.insert(stmts.begin(),
                 m_includes.begin(),
                 m_includes.end());
    for(auto i = n.begin();
        i != n.end();
        i++) {
        stmts.push_back(
            std::static_pointer_cast<statement>(boost::apply_visitor(*this, *i)));
    }
    shared_ptr<name> entry_name(
        new name(detail::wrap_proc_id(m_entry_point)));
    shared_ptr<name> entry_generated_name(
        new name(detail::mark_generated_id(m_entry_point)));
    shared_ptr<literal> python_entry_name(
        new literal("\"" + entry_generated_name->id() + "\""));
    shared_ptr<tuple> def_args(
        new tuple(
            vector<shared_ptr<expression> >{
                python_entry_name, entry_name}));
    shared_ptr<name> def_fn(
        new name(
            detail::boost_python_def()));
    shared_ptr<call> def_call(
        new call(
            shared_ptr<apply>(
                new apply(def_fn, def_args))));
    shared_ptr<suite> def_suite(
        new suite(
            vector<shared_ptr<statement> >{def_call}));
    shared_ptr<name> bpl_module(
        new name(
            detail::boost_python_module()));
    shared_ptr<tuple> bpl_module_args(
        new tuple(
            vector<shared_ptr<expression> >{entry_generated_name}));
    shared_ptr<procedure> bpl_proc(
        new procedure(
            bpl_module,
            bpl_module_args,
            def_suite));


    shared_ptr<suite> device_code = make_shared<suite>(move(stmts));
    m_device = device_code;

    vector<shared_ptr<statement> > host_stmts;
    host_stmts.push_back(
        make_shared<include>(
            make_shared<literal>(
                "cudata/cudata.h")));
    //Make sure we found the wrapper procedure
    assert(m_wrap_decl);
    host_stmts.push_back(m_wrap_decl);
    
    host_stmts.push_back(bpl_proc);
    shared_ptr<suite> host_code = make_shared<suite>(move(host_stmts));
    m_host = host_code;

    return device_code;
        
        
}

}
