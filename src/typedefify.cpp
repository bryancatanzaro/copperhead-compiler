#include "typedefify.hpp"

namespace backend {

typedefify::typedefify() :
    m_typedef() {}

typedefify::result_type typedefify::operator()(const suite &n) {
    std::vector<std::shared_ptr<statement> > stmts;
    for(auto i = n.begin();
        i != n.end();
        i++) {
        std::shared_ptr<statement> s =
            std::static_pointer_cast<statement>(
                boost::apply_visitor(*this, *i));
        if (m_typedef) {
            stmts.push_back(m_typedef);
            m_typedef = std::shared_ptr<statement>();
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
        
    std::shared_ptr<ctype::type_t> unique_type =
        std::make_shared<ctype::monotype_t>(
            detail::typify(lhs.id()));
    std::shared_ptr<expression> rhs =
        std::static_pointer_cast<expression>(
            boost::apply_visitor(*this, n.rhs()));
    std::shared_ptr<name> new_lhs =
        std::make_shared<name>(lhs.id(),
                               get_type_ptr(lhs.type()),
                               unique_type);
    m_typedef =
        std::make_shared<typedefn>(
            get_ctype_ptr(lhs.ctype()),
            unique_type);
    return result_type(
        new bind(
            new_lhs, rhs));
}
typedefify::result_type typedefify::operator()(const procedure &n) {
    const tuple& args = n.args();
    std::vector<std::shared_ptr<statement> > stmts;
    for(auto i = args.begin();
        i != args.end();
        i++) {
        assert(detail::isinstance<name>(*i));
        const name& arg_name = boost::get<const name&>(*i);
        std::shared_ptr<ctype::type_t> unique_type =
            std::make_shared<ctype::monotype_t>(
                detail::typify(arg_name.id()));
        std::shared_ptr<typedefn> arg_typedef =
            std::make_shared<typedefn>(
                get_ctype_ptr(arg_name.ctype()),
                unique_type);
        stmts.push_back(arg_typedef);
    }
    for(auto i = n.stmts().begin();
        i != n.stmts().end();
        i++) {
        auto s = std::static_pointer_cast<statement>(
            boost::apply_visitor(*this, *i));
        if (m_typedef) {
            stmts.push_back(m_typedef);
            m_typedef = std::shared_ptr<statement>();
        }
        stmts.push_back(s);
          
    }
    std::shared_ptr<name> n_name =
        std::static_pointer_cast<name>(
            boost::apply_visitor(*this, n.id()));
    std::shared_ptr<tuple> n_args =
        std::static_pointer_cast<tuple>(
            boost::apply_visitor(*this, args));
    std::shared_ptr<suite> n_stmts =
        std::make_shared<suite>(std::move(stmts));
    std::shared_ptr<type_t> n_type =
        get_type_ptr(n.type());
    std::shared_ptr<ctype::type_t> n_ctype =
        get_ctype_ptr(n.ctype());
        
        
    return std::make_shared<procedure>(
        n_name,
        n_args,
        n_stmts,
        n_type,
        n_ctype);
}


}
