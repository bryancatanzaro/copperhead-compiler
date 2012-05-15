#include "tuple_break.hpp"

using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using std::string;
using std::move;
using backend::utility::make_vector;


namespace backend {

tuple_break::tuple_break()
    : m_supply("tuple") {}

tuple_break::result_type tuple_break::operator()(const bind& n) {
    bool lhs_tuple = detail::isinstance<tuple>(n.lhs());
    bool rhs_tuple = detail::isinstance<tuple>(n.rhs());
    if (lhs_tuple && rhs_tuple) {
        //tuple = tuple bind
        const tuple& lhs = boost::get<const tuple&>(n.lhs());
        const tuple& rhs = boost::get<const tuple&>(n.rhs());
        
        //This code assumes nested tuples have been flattened by
        //the expression flattener.
        //Because of this assumption, we expect the length of
        //both tuples in a tuple = tuple bind to be identical.
        //Assert to ensure this condition is not violated.
        assert(lhs.arity() == rhs.arity());
        vector<shared_ptr<const statement> > stmts;
        for(auto i = lhs.begin(), j = rhs.begin();
            i != lhs.end();
            i++, j++) {
            stmts.push_back(
                make_shared<bind>(
                    i->ptr(),
                    j->ptr()));
        }
        return make_shared<suite>(move(stmts));
    } else if (lhs_tuple && !rhs_tuple) {
        //Unpacking a tuple
        const tuple& lhs = boost::get<const tuple&>(n.lhs());
        vector<shared_ptr<const statement> > stmts;
        shared_ptr<const expression> p_rhs;
        if (detail::isinstance<apply>(n.rhs())) {
            //The RHS is a function call.  Call it, store the result,
            //then break it.  Here we call it and store the result.
            shared_ptr<const name> result = make_shared<const name>(
                m_supply.next(),
                lhs.type().ptr(),
                lhs.ctype().ptr());
            shared_ptr<const bind> call_stmt = make_shared<const bind>(result, 
                                                                       n.rhs().ptr());
            stmts.push_back(call_stmt);
            p_rhs = result;
        } else {
            //The RHS is not a function call, just use it directly
            p_rhs = n.rhs().ptr();
        }
        int number = 0;
        for(auto i = lhs.begin(); i != lhs.end(); i++, number++) {
            stmts.push_back(
                make_shared<const bind>(
                    i->ptr(),
                    make_shared<const apply>(
                        make_shared<const name>(
                            detail::snippet_get(number)),
                        make_shared<const tuple>(
                            make_vector<shared_ptr<const expression> >(p_rhs)))));
        }
        return make_shared<suite>(move(stmts));
    } else if (!lhs_tuple && rhs_tuple) {
        //Packing a tuple
        const tuple& rhs = boost::get<const tuple&>(n.rhs());
        
        vector<shared_ptr<const expression> > args;
        for(auto i = rhs.begin(); i != rhs.end(); i++) {
            args.push_back(i->ptr());
        }
        return make_shared<bind>(
            n.lhs().ptr(),
            make_shared<apply>(
                make_shared<name>(detail::snippet_make_tuple()),
                make_shared<tuple>(move(args))));
    } else {
        //No tuples in this bind, just return the original
        return n.ptr();
    }
}

void tuple_break::flatten_tuple(const tuple& t,
                                const std::vector<int> path,
                                flattened_tuple& ft) {
    int j = 0;
    for(auto i = t.begin();
        i != t.end();
        i++, j++) {
        if (detail::isinstance<tuple>(*i)) {
            std::vector<int> this_path = path;
            this_path.push_back(j);
            flatten_tuple(boost::get<const tuple&>(*i),
                          this_path,
                          ft);
        } else {
            assert(detail::isinstance<name>(*i));
            const name& n = boost::get<const name&>(*i);
            //Copy path
            std::vector<int> this_path = path;
            //Add path id
            this_path.push_back(j);
            //Record
            ft.push_back(
                std::make_tuple(std::move(this_path),
                                n.ptr()));
                    
        }
    }
}


tuple_break::result_type tuple_break::operator()(const procedure& n) {
    vector<shared_ptr<const expression> > args;
    vector<shared_ptr<const statement> > stmts;
    for(auto i = n.args().begin();
        i != n.args().end();
        i++) {
        if (detail::isinstance<name>(*i)) {
            args.push_back(
                i->ptr());
        } else {
            //We can only allow names and tuples as arguments of a function.
            assert(detail::isinstance<backend::tuple>(*i));
            const backend::tuple& tup =
                boost::get<const backend::tuple&>(*i);
            
            shared_ptr<const name> new_name =
                make_shared<const name>(
                    m_supply.next(),
                    tup.type().ptr(),
                    tup.ctype().ptr());
            args.push_back(new_name);
            flattened_tuple ft;
            flatten_tuple(tup, std::vector<int>(), ft);
            for(auto j = ft.cbegin();
                j != ft.cend();
                j++) {

                std::shared_ptr<const expression> gets = new_name;
                const std::vector<int>& path = std::get<0>(*j);
                for(auto k = path.cbegin();
                    k != path.cend();
                    k++) {
                    gets = make_shared<const apply>(
                        make_shared<const name>(
                            detail::snippet_get(*k)),
                        make_shared<const tuple>(
                            make_vector<shared_ptr<const expression> >(
                                gets)));
                }
                
                stmts.push_back(
                    make_shared<const bind>(
                        std::get<1>(*j),
                        gets));
            }
        }
    }
    if (stmts.size() == 0) {
        return this->rewriter::operator()(n);
    } else {
        shared_ptr<const suite> rewritten_stmts =
            static_pointer_cast<const suite>(
                this->rewriter::operator()(n.stmts()));
        for(auto i = rewritten_stmts->begin();
            i != rewritten_stmts->end();
            i++) {
            stmts.push_back(i->ptr());
        }
        return make_shared<const procedure>(
            n.id().ptr(),
            make_shared<backend::tuple>(std::move(args)),
            make_shared<backend::suite>(std::move(stmts)),
            n.type().ptr(),
            n.ctype().ptr(),
            n.place());
    }
    
}

tuple_break::result_type tuple_break::operator()(const suite& n) {
    vector<shared_ptr<const statement> > stmts;
    for(auto i = n.begin();
        i != n.end();
        i++) {
        auto r = boost::apply_visitor(*this, *i);
        //Turning one statement into a suite of statements requires
        //Individually adding each statement to the enclosing suite
        if (detail::isinstance<suite>(*r)) {
            //Add the suite of statements one by one
            const suite& s = boost::get<const suite&>(*r);
            for(auto j = s.begin(); j != s.end(); j++) {
                stmts.push_back(
                    j->ptr());
            }
        } else {
            //The rewritten statement was just one statement,
            //Add it directly.
            stmts.push_back(static_pointer_cast<const statement>(r));
        }
    }
    return make_shared<suite>(move(stmts));
}

}
