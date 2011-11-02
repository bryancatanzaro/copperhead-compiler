#include "copier.hpp"

namespace backend {

void copier::start_match() {
    m_matches.push(true);
}
bool copier::is_match() {
    bool result = m_matches.top();
    m_matches.pop();
    return result;
}

    
copier::result_type copier::operator()(const literal& n) {
    return get_node_ptr(n);
}
copier::result_type copier::operator()(const name &n) {
    return get_node_ptr(n);
}
copier::result_type copier::operator()(const tuple &n) {
    start_match();
    std::vector<std::shared_ptr<expression> > n_values;
    for(auto i = n.begin(); i != n.end(); i++) {
        auto n_i = std::static_pointer_cast<expression>(boost::apply_visitor(*this, *i));
        update_match(n_i, *i);
        n_values.push_back(n_i);
    }
    if (is_match())
        return get_node_ptr(n);
        
    std::shared_ptr<type_t> t = get_type_ptr(n.type());
    std::shared_ptr<ctype::type_t> ct = get_ctype_ptr(n.ctype());
        
    return result_type(new tuple(std::move(n_values), t, ct));
}
copier::result_type copier::operator()(const apply &n) {
    auto n_fn = std::static_pointer_cast<name>(boost::apply_visitor(*this, n.fn()));
    auto n_args = std::static_pointer_cast<tuple>((*this)(n.args()));
    start_match();
    update_match(n_fn, n.fn());
    update_match(n_args, n.args()); 
    if (is_match())
        return get_node_ptr(n);
    return result_type(new apply(n_fn, n_args));
}
copier::result_type copier::operator()(const lambda &n) {
    auto n_args = std::static_pointer_cast<tuple>((*this)(n.args()));
    auto n_body = std::static_pointer_cast<expression>(boost::apply_visitor(*this, n.body()));
    start_match();
    update_match(n_args, n.args());
    update_match(n_body, n.body());
    if (is_match())
        return get_node_ptr(n);
    std::shared_ptr<type_t> t = get_type_ptr(n.type());
    std::shared_ptr<ctype::type_t> ct = get_ctype_ptr(n.ctype());
    return result_type(new lambda(n_args, n_body, t, ct));
}
copier::result_type copier::operator()(const closure &n) {
    auto n_args = std::static_pointer_cast<tuple>((*this)(n.args()));
    auto n_body = std::static_pointer_cast<expression>(boost::apply_visitor(*this, n.body()));
    start_match();
    update_match(n_args, n.args());
    update_match(n_body, n.body());
    if (is_match())
        return get_node_ptr(n);
    std::shared_ptr<type_t> t = get_type_ptr(n.type());
    std::shared_ptr<ctype::type_t> ct = get_ctype_ptr(n.ctype());
    return result_type(new closure(n_args, n_body, t, ct));
}
copier::result_type copier::operator()(const conditional &n) {
    auto n_cond = std::static_pointer_cast<expression>(boost::apply_visitor(*this, n.cond()));
    auto n_then = std::static_pointer_cast<suite>(boost::apply_visitor(*this, n.then()));
    auto n_orelse = std::static_pointer_cast<suite>(boost::apply_visitor(*this, n.orelse()));
    start_match();
    update_match(n_cond, n.cond());
    update_match(n_then, n.then());
    update_match(n_orelse, n.orelse());
    if (is_match())
        return get_node_ptr(n);
    return result_type(new conditional(n_cond, n_then, n_orelse));
}
copier::result_type copier::operator()(const ret &n) {
    auto n_val = std::static_pointer_cast<expression>(boost::apply_visitor(*this, n.val()));
    start_match();
    update_match(n_val, n.val());
    if (is_match())
        return get_node_ptr(n);
    return result_type(new ret(n_val));
}
copier::result_type copier::operator()(const bind &n) {
    auto n_lhs = std::static_pointer_cast<expression>(boost::apply_visitor(*this, n.lhs()));
    auto n_rhs = std::static_pointer_cast<expression>(boost::apply_visitor(*this, n.rhs()));
    start_match();
    update_match(n_lhs, n.lhs());
    update_match(n_rhs, n.rhs());
    if (is_match())
        return get_node_ptr(n);
    return result_type(new bind(n_lhs, n_rhs));
}
copier::result_type copier::operator()(const call &n) {
    auto n_sub = std::static_pointer_cast<apply>(boost::apply_visitor(*this, n.sub()));
    start_match();
    update_match(n_sub, n.sub());
    if (is_match())
        return get_node_ptr(n);
    return result_type(new call(n_sub));
}
copier::result_type copier::operator()(const procedure &n) {
    auto n_id = std::static_pointer_cast<name>((*this)(n.id()));
    auto n_args = std::static_pointer_cast<tuple>((*this)(n.args()));
    auto n_stmts = std::static_pointer_cast<suite>((*this)(n.stmts()));
    start_match();
    update_match(n_id, n.id());
    update_match(n_args, n.args());
    update_match(n_stmts, n.stmts());
    if (is_match())
        return get_node_ptr(n);
    std::shared_ptr<type_t> t = get_type_ptr(n.type());
    std::shared_ptr<ctype::type_t> ct = get_ctype_ptr(n.ctype());

    return result_type(new procedure(n_id, n_args, n_stmts, t, ct));
}
copier::result_type copier::operator()(const suite &n) {
    start_match();
    std::vector<std::shared_ptr<statement> > n_stmts;
    ctype::ctype_printer cp(std::cout);
    for(auto i = n.begin(); i != n.end(); i++) {
        auto n_stmt = std::static_pointer_cast<statement>(boost::apply_visitor(*this, *i));
        update_match(n_stmt, *i);
        n_stmts.push_back(n_stmt);
    }
    if (is_match())
        return get_node_ptr(n);
    return result_type(new suite(std::move(n_stmts)));
}
copier::result_type copier::operator()(const structure &n) {
    auto n_id = std::static_pointer_cast<name>((*this)(n.id()));
    auto n_stmts = std::static_pointer_cast<suite>((*this)(n.stmts()));
    start_match();
    update_match(n_id, n.id());
    update_match(n_stmts, n.stmts());
    if (is_match())
        return get_node_ptr(n);
    return result_type(new structure(n_id, n_stmts));
}
copier::result_type copier::operator()(const templated_name &n) {
    return get_node_ptr(n);
}
copier::result_type copier::operator()(const include &n) {
    return get_node_ptr(n);
}
copier::result_type copier::operator()(const typedefn &n) {
    return get_node_ptr(n);
}

}