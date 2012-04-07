#include "typedefify.hpp"

using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;


namespace backend {

typedefify::typedefify() :
    m_typedef() {}

typedefify::result_type typedefify::operator()(const suite &n) {
    vector<shared_ptr<const statement> > stmts;
    for(auto i = n.begin();
        i != n.end();
        i++) {
        shared_ptr<const statement> s =
            static_pointer_cast<const statement>(
                boost::apply_visitor(*this, *i));
        if (m_typedef) {
            stmts.push_back(m_typedef);
            m_typedef = shared_ptr<const statement>();
        }
        stmts.push_back(s);
    }
    return result_type(
        new suite(std::move(stmts)));
}

typedefify::result_type typedefify::operator()(const bind &n) {

    //We can only deal with names in the LHS
    assert(detail::isinstance<name>(n.lhs()));

    const name& lhs = boost::get<const name&>(n.lhs());
        
    shared_ptr<const ctype::type_t> unique_type =
        std::make_shared<const ctype::monotype_t>(
            detail::typify(lhs.id()));
    shared_ptr<const expression> rhs =
        static_pointer_cast<const expression>(
            boost::apply_visitor(*this, n.rhs()));
    shared_ptr<const name> new_lhs =
        std::make_shared<const name>(lhs.id(),
                                     lhs.type().ptr(),
                                     unique_type);
    m_typedef =
        std::make_shared<const typedefn>(
            lhs.ctype().ptr(),
            unique_type);
    return result_type(
        new bind(new_lhs, rhs));
}
typedefify::result_type typedefify::operator()(const procedure &n) {
    const tuple& args = n.args();
    vector<shared_ptr<const statement> > stmts;
    for(auto i = args.begin();
        i != args.end();
        i++) {
        assert(detail::isinstance<name>(*i));
        const name& arg_name = boost::get<const name&>(*i);
        shared_ptr<const ctype::type_t> unique_type =
            std::make_shared<const ctype::monotype_t>(
                detail::typify(arg_name.id()));
        shared_ptr<const typedefn> arg_typedef =
            std::make_shared<const typedefn>(
                arg_name.ctype().ptr(),
                unique_type);
        stmts.push_back(arg_typedef);
    }
    for(auto i = n.stmts().begin();
        i != n.stmts().end();
        i++) {
        auto s = static_pointer_cast<const statement>(
            boost::apply_visitor(*this, *i));
        if (m_typedef) {
            stmts.push_back(m_typedef);
            m_typedef = shared_ptr<const statement>();
        }
        stmts.push_back(s);
          
    }
    shared_ptr<const name> n_name =
        static_pointer_cast<const name>(
            boost::apply_visitor(*this, n.id()));
    shared_ptr<const tuple> n_args =
        static_pointer_cast<const tuple>(
            boost::apply_visitor(*this, args));
    shared_ptr<const suite> n_stmts =
        std::make_shared<const suite>(std::move(stmts));
                
    return std::make_shared<const procedure>(
        n_name,
        n_args,
        n_stmts,
        n.type().ptr(),
        n.ctype().ptr());
}


}
