/*
 *   Copyright 2012      NVIDIA Corporation
 * 
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 * 
 *       http://www.apache.org/licenses/LICENSE-2.0
 * 
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 * 
 */

#pragma once

namespace backend {

template<typename Derived>
void rewriter<Derived>::start_match() {
    m_matches.push(true);
}

template<typename Derived>
bool rewriter<Derived>::is_match() {
    bool result = m_matches.top();
    m_matches.pop();
    return result;
}

    
template<typename Derived>
typename rewriter<Derived>::result_type rewriter<Derived>::operator()(const literal& n) {
    return n.ptr();
}

template<typename Derived>
typename rewriter<Derived>::result_type rewriter<Derived>::operator()(const name &n) {
    return n.ptr();
}

template<typename Derived>
typename rewriter<Derived>::result_type rewriter<Derived>::operator()(const tuple &n) {
    start_match();
    std::vector<std::shared_ptr<const expression> > n_values;
    for(auto i = n.begin(); i != n.end(); i++) {
        auto n_i =
            std::static_pointer_cast<const expression>(
                boost::apply_visitor(*static_cast<Derived*>(this), *i));
        update_match(n_i, *i);
        n_values.push_back(n_i);
    }
    if (is_match())
        return n.ptr();
        
    auto t = n.type().ptr();
    auto ct = n.ctype().ptr();
        
    return result_type(new tuple(move(n_values), t, ct));
}

template<typename Derived>
typename rewriter<Derived>::result_type rewriter<Derived>::operator()(const apply &n) {
    auto n_fn = std::static_pointer_cast<const name>(
        boost::apply_visitor(
            *static_cast<Derived*>(this), n.fn()));
    auto n_args = std::static_pointer_cast<const tuple>(
        (*static_cast<Derived*>(this))(n.args()));
    start_match();
    update_match(n_fn, n.fn());
    update_match(n_args, n.args()); 
    if (is_match())
        return n.ptr();
    
    return result_type(new apply(n_fn, n_args));
}

template<typename Derived>
typename rewriter<Derived>::result_type rewriter<Derived>::operator()(const lambda &n) {
    auto n_args = std::static_pointer_cast<const tuple>(
        (*static_cast<Derived*>(this))(n.args()));
    auto n_body = std::static_pointer_cast<const expression>(
        boost::apply_visitor(*static_cast<Derived*>(this), n.body()));
    start_match();
    update_match(n_args, n.args());
    update_match(n_body, n.body());
    if (is_match())
        return n.ptr();
    auto t = n.type().ptr();
    auto ct = n.ctype().ptr();
    return result_type(new lambda(n_args, n_body, t, ct));
}

template<typename Derived>
typename rewriter<Derived>::result_type rewriter<Derived>::operator()(const closure &n) {
    auto n_args = std::static_pointer_cast<const tuple>(
        (*static_cast<Derived*>(this))(n.args()));
    auto n_body = std::static_pointer_cast<const expression>(
        boost::apply_visitor(*static_cast<Derived*>(this), n.body()));
    start_match();
    update_match(n_args, n.args());
    update_match(n_body, n.body());
    if (is_match())
        return n.ptr();
    auto t = n.type().ptr();
    auto ct = n.ctype().ptr();
    return result_type(new closure(n_args, n_body, t, ct));
}

template<typename Derived>
typename rewriter<Derived>::result_type rewriter<Derived>::operator()(const subscript &n) {
    auto n_src = std::static_pointer_cast<const name>(
        (*static_cast<Derived*>(this))(n.src()));
    auto n_idx = std::static_pointer_cast<const expression>(
        boost::apply_visitor(*static_cast<Derived*>(this), n.idx()));
    start_match();
    update_match(n_src, n.src());
    update_match(n_idx, n.idx());
    if (is_match())
        n.ptr();
    auto t = n.type().ptr();
    auto ct = n.ctype().ptr();
    return result_type(new subscript(n_src, n_idx, t, ct));
}

template<typename Derived>
typename rewriter<Derived>::result_type rewriter<Derived>::operator()(const conditional &n) {
    auto n_cond = std::static_pointer_cast<const expression>(
        boost::apply_visitor(*static_cast<Derived*>(this), n.cond()));
    auto n_then = std::static_pointer_cast<const suite>(
        boost::apply_visitor(*static_cast<Derived*>(this), n.then()));
    auto n_orelse = std::static_pointer_cast<const suite>(
        boost::apply_visitor(*static_cast<Derived*>(this), n.orelse()));
    start_match();
    update_match(n_cond, n.cond());
    update_match(n_then, n.then());
    update_match(n_orelse, n.orelse());
    if (is_match())
        return n.ptr();
    return result_type(new conditional(n_cond, n_then, n_orelse));
}

template<typename Derived>
typename rewriter<Derived>::result_type rewriter<Derived>::operator()(const ret &n) {
    auto n_val = std::static_pointer_cast<const expression>(
        boost::apply_visitor(*static_cast<Derived*>(this), n.val()));
    start_match();
    update_match(n_val, n.val());
    if (is_match())
        return n.ptr();
    return result_type(new ret(n_val));
}

template<typename Derived>
typename rewriter<Derived>::result_type rewriter<Derived>::operator()(const bind &n) {
    auto n_lhs = std::static_pointer_cast<const expression>(
        boost::apply_visitor(*static_cast<Derived*>(this), n.lhs()));
    auto n_rhs = std::static_pointer_cast<const expression>(
        boost::apply_visitor(*static_cast<Derived*>(this), n.rhs()));
    start_match();
    update_match(n_lhs, n.lhs());
    update_match(n_rhs, n.rhs());
    if (is_match())
        return n.ptr();
    return result_type(new bind(n_lhs, n_rhs));
}

template<typename Derived>
typename rewriter<Derived>::result_type rewriter<Derived>::operator()(const call &n) {
    auto n_sub = std::static_pointer_cast<const apply>(
        boost::apply_visitor(*static_cast<Derived*>(this), n.sub()));
    start_match();
    update_match(n_sub, n.sub());
    if (is_match())
        return n.ptr();
    return result_type(new call(n_sub));
}

template<typename Derived>
typename rewriter<Derived>::result_type rewriter<Derived>::operator()(const procedure &n) {
    auto n_id = std::static_pointer_cast<const name>((*static_cast<Derived*>(this))(n.id()));
    auto n_args = std::static_pointer_cast<const tuple>((*static_cast<Derived*>(this))(n.args()));
    auto n_stmts = std::static_pointer_cast<const suite>((*static_cast<Derived*>(this))(n.stmts()));
    start_match();
    update_match(n_id, n.id());
    update_match(n_args, n.args());
    update_match(n_stmts, n.stmts());
    if (is_match())
        return n.ptr();
    auto t = n.type().ptr();
    auto ct = n.ctype().ptr();

    return result_type(new procedure(n_id, n_args, n_stmts, t, ct));
}

template<typename Derived>
typename rewriter<Derived>::result_type rewriter<Derived>::operator()(const suite &n) {
    start_match();
    std::vector<std::shared_ptr<const statement> > n_stmts;
    for(auto i = n.begin(); i != n.end(); i++) {
        auto n_stmt = std::static_pointer_cast<const statement>(boost::apply_visitor(*static_cast<Derived*>(this), *i));
        update_match(n_stmt, *i);
        n_stmts.push_back(n_stmt);
    }
    if (is_match())
        return n.ptr();
    return result_type(new suite(move(n_stmts)));
}

template<typename Derived>
typename rewriter<Derived>::result_type rewriter<Derived>::operator()(const structure &n) {
    auto n_id = std::static_pointer_cast<const name>((*static_cast<Derived*>(this))(n.id()));
    auto n_stmts = std::static_pointer_cast<const suite>((*static_cast<Derived*>(this))(n.stmts()));
    start_match();
    update_match(n_id, n.id());
    update_match(n_stmts, n.stmts());
    if (is_match())
        return n.ptr();
    //If structure has changed, assume struct template typevars are
    //remain the same.  Rewriters which want to change the template
    //typevars should reimplement this method directly.
    std::vector<std::shared_ptr<const ctype::type_t> > new_typevars;
    for(auto i = n.begin();
        i != n.end();
        i++) {
        new_typevars.push_back(i->ptr());
    }
    return result_type(new structure(n_id, n_stmts, move(new_typevars)));
}

template<typename Derived>
typename rewriter<Derived>::result_type rewriter<Derived>::operator()(const templated_name &n) {
    return n.ptr();
}

template<typename Derived>
typename rewriter<Derived>::result_type rewriter<Derived>::operator()(const include &n) {
    return n.ptr();
}

template<typename Derived>
typename rewriter<Derived>::result_type rewriter<Derived>::operator()(const typedefn &n) {
    return n.ptr();
}

template<typename Derived>
typename rewriter<Derived>::result_type rewriter<Derived>::operator()(const namespace_block &n) {
    return n.ptr();
}

}
