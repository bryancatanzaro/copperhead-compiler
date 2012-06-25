#include "allocate.hpp"
#include "utility/up_get.hpp"

using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::move;
using std::string;
using std::static_pointer_cast;
using backend::utility::make_vector;


namespace backend {


allocate::allocate(const copperhead::system_variant& target,
                   const string& entry_point) : m_target(target),
                                                m_entry_point(entry_point),
                                                m_in_entry(false),
                                                m_allocations()
        {}

allocate::result_type allocate::operator()(const procedure &n) {
    if (n.id().id()  == m_entry_point) {
        m_in_entry = true;
        vector<shared_ptr<const statement> > statements;
        for(auto i = n.stmts().begin();
            i != n.stmts().end();
            i++) {
            auto new_stmt = static_pointer_cast<const statement>(
                boost::apply_visitor(*this, *i));
            if (m_allocations.size() > 0) {
                for(auto j = m_allocations.begin();
                    j != m_allocations.end();
                    j++) {
                    statements.push_back(*j);
                }
                m_allocations.clear();
            }
            statements.push_back(new_stmt);
        }
        auto stmts = make_shared<const suite>(move(statements));
        auto args =
            static_pointer_cast<const tuple>(
                boost::apply_visitor(*this, n.args()));
        auto t = n.type().ptr();
        auto ct = n.ctype().ptr();
        auto id =
            static_pointer_cast<const name>(
                boost::apply_visitor(*this, n.id()));
        result_type allocated = make_shared<const procedure>(
            id, args, stmts, t, ct);

        m_in_entry = false;
        return allocated;
    } else {
        return this->rewriter::operator()(n);
    }
}

shared_ptr<const ctype::type_t> allocate::container_type(const ctype::type_t& t) {
    if (detail::isinstance<ctype::sequence_t>(t)) {
        const ctype::sequence_t& st = detail::up_get<const ctype::sequence_t&>(t);
        return make_shared<const ctype::cuarray_t>(st.sub().ptr());
    } else if (detail::isinstance<ctype::tuple_t>(t)) {
        const ctype::tuple_t& tt = boost::get<const ctype::tuple_t&>(t);
        vector<shared_ptr<const ctype::type_t> > subs;
        bool containerize = false;
        for(auto i = tt.begin(); i != tt.end(); i++) {
            shared_ptr<const ctype::type_t> container_type_i =
                container_type(*i);
            subs.push_back(container_type_i);
            containerize = containerize || (t.ptr() != container_type_i);
        }
        if (!containerize) {
            return t.ptr();
        }
        return make_shared<const ctype::tuple_t>(
            move(subs));
    } else {
        return t.ptr();
    }
}

allocate::result_type allocate::operator()(const bind &n) {
    if (m_in_entry &&
        detail::isinstance<apply>(n.rhs())) {

        shared_ptr<const ctype::type_t> containerized = container_type(n.lhs().ctype());
        if (containerized == n.lhs().ctype().ptr()) {
            return this->rewriter<allocate>::operator()(n);
        }
        
        bool lhs_is_name = detail::isinstance<name>(n.lhs());
        assert(lhs_is_name);

        const name& pre_lhs = boost::get<const name&>(n.lhs());

        //Construct cuarray for result
        //This cast is valid because we already tested the ctype in
        //the condition
        shared_ptr<const ctype::sequence_t> impl_seq_ct =
            static_pointer_cast<const ctype::sequence_t>(pre_lhs.ctype().ptr());
       
        shared_ptr<const ctype::tuple_t> tuple_impl_seq_ct =
            make_shared<const ctype::tuple_t>(
                make_vector<shared_ptr<const ctype::type_t> >(impl_seq_ct));

        shared_ptr<const type_t> result_t =
            pre_lhs.type().ptr();
        shared_ptr<const name> result_name = make_shared<const name>(
            detail::wrap_array_id(pre_lhs.id()),
            result_t,
            containerized);

        shared_ptr<const expression> new_rhs =
            static_pointer_cast<const expression>(
                boost::apply_visitor(*this, n.rhs()));
            
        shared_ptr<const bind> allocator = make_shared<const bind>(
            result_name, new_rhs);
        m_allocations.push_back(allocator);
            

        shared_ptr<const name> new_lhs = static_pointer_cast<const name>(
            boost::apply_visitor(*this, n.lhs()));
        shared_ptr<const templated_name> getter_name =
            make_shared<const templated_name>(
                detail::make_sequence(),
                tuple_impl_seq_ct);
        shared_ptr<const tuple> getter_args =
            make_shared<const tuple>(
                make_vector<shared_ptr<const expression> >(result_name)
                (make_shared<const apply>(
                    make_shared<const name>(copperhead::to_string(m_target)),
                    make_shared<const tuple>(make_vector<shared_ptr<const expression> >())))
                (make_shared<const literal>("true")));
        shared_ptr<const apply> getter_call =
            make_shared<const apply>(getter_name, getter_args);
        shared_ptr<const bind> retriever =
            make_shared<const bind>(new_lhs, getter_call);
        return retriever;
    } else {
        return this->rewriter<allocate>::operator()(n);
    }
}


}
