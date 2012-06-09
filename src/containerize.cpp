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
#include "containerize.hpp"
#include "utility/initializers.hpp"

using std::string;
using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using backend::utility::make_vector;
using std::vector;

namespace backend {

containerize::containerize(const string& entry_point) : m_entry_point(entry_point), m_in_entry(false) {}


containerize::result_type containerize::operator()(const name &n) {
    if (detail::isinstance<ctype::cuarray_t>(n.ctype())) {
        m_decl_containers.insert(n.id());
    }
}

containerize::result_type containerize::operator()(const suite &n) {
    vector<shared_ptr<const statement> > stmts;
    bool match = true;
    for(auto i = n.begin(); i != n.end(); i++) {
        containerize::result_type rewritten = boost::apply_visitor(*this, *i);
        match = match && (rewritten == i->ptr());
        if (detail::isinstance<statement>(*rewritten)) {
            stmts.push_back(static_pointer_cast<const statement>(rewritten));
        } else {
            assert(detail::isinstance<suite>(*rewritten));
            const suite& sub_suite = boost::get<const suite&>(*rewritten);
            for(auto j = sub_suite.begin(); j != sub_suite.end(); j++) {
                stmts.push_back(j->ptr());
            }
        }
    }
    if (match) {
        return n.ptr();
    } else {
        return result_type(
            new suite(
                move(stmts)));
    }
}

containerize::result_type containerize::operator()(const procedure &s) {
    m_in_entry = (s.id().id() == m_entry_point);
    m_decl_containers.begin_scope();
    containerize::result_type r = this->rewriter::operator()(s);
    m_decl_containers.end_scope();
    return r;
}

shared_ptr<const ctype::type_t> containerize::container_type(const ctype::type_t& t) {
    if (detail::isinstance<ctype::sequence_t>(t)) {
        const ctype::sequence_t seq = boost::get<const ctype::sequence_t&>(t);
        return make_shared<const ctype::cuarray_t>(
            seq.sub().ptr());
    } else if (!detail::isinstance<ctype::tuple_t>(t)) {
        return t.ptr();
    }
    bool match = true;
    const ctype::tuple_t& t_tuple = boost::get<const ctype::tuple_t&>(t);
    vector<shared_ptr<const ctype::type_t> > sub_types;
    for(auto i = t_tuple.begin(); i != t_tuple.end(); i++) {
        shared_ptr<const ctype::type_t> sub_container = container_type(*i);
        match = match && (sub_container == i->ptr());
        sub_types.push_back(container_type(*i));
    }
    if (match) {
        return t.ptr();
    }
    return make_shared<const ctype::tuple_t>(move(sub_types));
}

shared_ptr<const expression> containerize::container_args(const expression& t) {
    if (detail::isinstance<name>(t)) {
        if (container_type(t.ctype()) != t.ctype().ptr()) {
            const name& n = boost::get<const name&>(t);
            return make_shared<const name>(
                detail::wrap_array_id(n.id()),
                n.type().ptr(),
                n.ctype().ptr());
        }
    } else if (detail::isinstance<tuple>(t)) {
        bool match = true;
        vector<shared_ptr<const expression> > sub_exprs;
        const tuple& tup = boost::get<const tuple&>(t);
        for(auto i = tup.begin(); i != tup.end(); i++) {
            shared_ptr<const expression> sub = container_args(*i);
            sub_exprs.push_back(sub);
            match = match && (sub == i->ptr());
        }
        if (match) {
            return t.ptr();
        }
        return make_shared<const tuple>(
            move(sub_exprs),
            t.type().ptr(),
            t.ctype().ptr());
    }
    return t.ptr();
}

containerize::result_type containerize::operator()(const bind &n) {
    if (!m_in_entry) {
        return n.ptr();
    } else {
        const expression& rhs = n.rhs();
        if (!detail::isinstance<apply>(rhs)) {
            return n.ptr();
        }
        const apply& rhs_apply = boost::get<const apply&>(rhs);
        
        //Currently, we only process formation of tuples containing sequences
        //These will always be in the form of an apply of thrust::make_tuple
        const name& apply_fn = rhs_apply.fn();
        if (apply_fn.id() != detail::snippet_make_tuple()) {
            return n.ptr();
        }

        //Does this make_tuple need to be containerized?
        shared_ptr<const ctype::type_t> cont_type = container_type(n.lhs().ctype());
        //If the container is the same as the ctype, no
        if (cont_type == n.lhs().ctype().ptr()) {
            return n.ptr();
        }
        assert(detail::isinstance<name>(n.lhs()));
        
        const name& lhs = boost::get<const name&>(n.lhs());
        shared_ptr<const name> new_lhs =
            make_shared<const name>(
                detail::wrap_array_id(
                    lhs.id()),
                lhs.type().ptr(),
                cont_type);
        shared_ptr<const apply> new_rhs =
            make_shared<const apply>(
                apply_fn.ptr(),
                static_pointer_cast<const tuple>(container_args(rhs_apply.args())));

                return make_shared<const suite>(
                    make_vector<shared_ptr<const statement> >(n.ptr())
                    (make_shared<const bind>(
                        new_lhs,
                        new_rhs)));
    }
    return n.ptr();
}

}
