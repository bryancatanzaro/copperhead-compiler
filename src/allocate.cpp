#include "allocate.hpp"

using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::move;
using std::string;
using std::static_pointer_cast;

namespace backend {


allocate::allocate(const string& entry_point) : m_entry_point(entry_point),
                                                     m_in_entry(false),
                                                     m_allocations{}
        {}

allocate::result_type allocate::operator()(const procedure &n) {
    if (n.id().id()  == m_entry_point) {
        m_in_entry = true;
        vector<shared_ptr<statement> > statements;
        for(auto i = n.stmts().begin();
            i != n.stmts().end();
            i++) {
            auto new_stmt = static_pointer_cast<statement>(
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
        shared_ptr<suite> stmts = make_shared<suite>(move(statements));
        shared_ptr<tuple> args =
            static_pointer_cast<tuple>(
                boost::apply_visitor(*this, n.args()));
        auto t = n.p_type();
        auto ct = n.p_ctype();
        shared_ptr<name> id =
            static_pointer_cast<name>(
                boost::apply_visitor(*this, n.id()));
        result_type allocated = make_shared<procedure>(
            id, args, stmts, t, ct);

        m_in_entry = false;
        return allocated;
    } else {
        return this->rewriter::operator()(n);
    }
}

allocate::result_type allocate::operator()(const bind &n) {
    if (m_in_entry &&
        detail::isinstance<ctype::sequence_t>(
            n.lhs().ctype()) &&
        detail::isinstance<apply>(n.rhs())) {

        //We can only deal with names on the LHS of a bind
        //TUPLE FIX
        bool lhs_is_name = detail::isinstance<name>(n.lhs());
        assert(lhs_is_name);
        const name& pre_lhs = boost::get<const name&>(n.lhs());

        //Construct cuarray for result
        const ctype::sequence_t& pre_lhs_ct =
            boost::get<const ctype::sequence_t&>(pre_lhs.ctype());
        shared_ptr<ctype::type_t> sub_lhs_ct =
            pre_lhs_ct.p_sub();
        shared_ptr<ctype::tuple_t> tuple_sub_lhs_ct =
            make_shared<ctype::tuple_t>(
                vector<shared_ptr<ctype::type_t> >{sub_lhs_ct});
        shared_ptr<ctype::type_t> result_ct =
            make_shared<ctype::templated_t>(
                make_shared<ctype::monotype_t>(
                    "boost::shared_ptr"),
                vector<shared_ptr<ctype::type_t> >{
                    make_shared<ctype::templated_t>(
                        make_shared<ctype::monotype_t>(
                            "cuarray"),
                        vector<shared_ptr<ctype::type_t> >{
                            sub_lhs_ct})});
        shared_ptr<type_t> result_t =
            pre_lhs.p_type();
        shared_ptr<name> result_name = make_shared<name>(
            detail::wrap_array_id(pre_lhs.id()),
            result_t,
            result_ct);

        shared_ptr<expression> new_rhs =
            static_pointer_cast<expression>(
                boost::apply_visitor(*this, n.rhs()));
            
        shared_ptr<bind> allocator = make_shared<bind>(
            result_name, new_rhs);
        m_allocations.push_back(allocator);
            

        shared_ptr<name> new_lhs = static_pointer_cast<name>(
            boost::apply_visitor(*this, n.lhs()));
        shared_ptr<templated_name> getter_name =
            make_shared<templated_name>(
                detail::get_remote_w(),
                tuple_sub_lhs_ct);
        shared_ptr<tuple> getter_args =
            make_shared<tuple>(
                vector<shared_ptr<expression> >{result_name});
        shared_ptr<apply> getter_call =
            make_shared<apply>(getter_name, getter_args);
        shared_ptr<bind> retriever =
            make_shared<bind>(new_lhs, getter_call);
        return retriever;
    } else {
        return this->rewriter::operator()(n);
    }
}


}
