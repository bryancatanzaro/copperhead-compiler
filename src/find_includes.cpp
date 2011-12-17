#include "find_includes.hpp"

using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using std::vector;
using std::set;
using std::map;
using std::move;
using std::string;

namespace backend {

find_includes::find_includes(const registry& reg)
    : m_reg(reg), m_outer(true) {}

find_includes::result_type find_includes::operator()(const suite& n) {
    bool outer_suite = m_outer;
    m_outer = false;
    shared_ptr<suite> rewritten =
        static_pointer_cast<suite>(this->rewriter::operator()(n));
    if (!outer_suite) {
        return rewritten;
    }
    
    //We're the outer suite, add include statements
    vector<shared_ptr<statement> > augmented_statements;
    for(auto i = m_includes.begin();
        i != m_includes.end();
        i++) {
        augmented_statements.push_back(
            make_shared<include>(
                make_shared<literal>(
                    *i)));
    }
    
    augmented_statements.insert(augmented_statements.end(),
                                n.p_begin(), n.p_end());
    return make_shared<suite>(move(augmented_statements));
}

find_includes::result_type find_includes::operator()(const apply& n) {
    const string& fn_name = n.fn().id();
    const map<string, string>& include_map = m_reg.fn_includes();
    auto include_it = include_map.find(fn_name);
    if (include_it != include_map.end()) {
        m_includes.insert(include_it->second);
    }
    return this->rewriter::operator()(n);
}

}
