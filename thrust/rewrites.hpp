#pragma once
#include <map>
#include <string>
#include <sstream>

#include "../node.hpp"
#include "../type.hpp"
#include "../ctype.hpp"
#include "../copier.hpp"
#include "../utility/isinstance.hpp"
#include "../utility/markers.hpp"

#include "../type_printer.hpp"

namespace backend {

class thrust_rewriter
    : public copier {
private:
    static result_type map_rewrite(const bind& n) {
        //The rhs must be an apply
        assert(detail::isinstance<apply>(n.rhs()));
        const apply& rhs = boost::get<const apply&>(n.rhs());
        //The rhs must apply a "map"
        assert(rhs.fn().id() == std::string("map"));
        const tuple& ap_args = rhs.args();
        //Map must have arguments
        assert(ap_args.begin() != ap_args.end());
        auto init = ap_args.begin();
        std::shared_ptr<ctype::type_t> fn_t;

        if (detail::isinstance<literal>(*init)) {
                    
            const literal& fn_name = boost::get<const literal&>(
            *init);
            std::string fn_id = fn_name.id();
            size_t found = fn_id.find_last_not_of("()");
            assert(found != std::string::npos);
            fn_id.erase(found + 1);
            fn_t = std::make_shared<ctype::monotype_t>(fn_id);
        } else {
            //We must be dealing with a closure
            assert(detail::isinstance<closure>(*init));

            const closure& close = boost::get<const closure&>(
                *init);
            int arity = close.args().arity();
            //The closure must enclose something
            assert(arity > 0);
            std::stringstream ss;
            ss << "closure" << arity;
            std::string closure_t_name = ss.str();
            std::shared_ptr<ctype::monotype_t> closure_mt =
                std::make_shared<ctype::monotype_t>(closure_t_name);
            std::vector<std::shared_ptr<ctype::type_t> > cts;
            for(auto i = close.args().begin();
                i != close.args().end();
                i++) {
                cts.push_back(get_ctype_ptr(i->ctype()));
            }
            //Can only deal with names in the body of a closure
            assert(detail::isinstance<name>(close.body()));

            const name& body = boost::get<const name&>(close.body());
            std::string body_fn = detail::fnize_id(body.id());
            cts.push_back(
                std::make_shared<ctype::monotype_t>(
                    body_fn));
            fn_t = std::make_shared<ctype::templated_t>(
                closure_mt,
                std::move(cts));
        }
        std::vector<std::shared_ptr<ctype::type_t> > arg_types;
        for(auto i = init+1; i != ap_args.end(); i++) {
            //Assert we're looking at a name
            assert(detail::isinstance<name>(*i));
            arg_types.push_back(
                std::make_shared<ctype::monotype_t>(
                    detail::typify(boost::get<const name&>(*i).id())));
        }
        std::shared_ptr<ctype::templated_t> thrust_tupled =
            std::make_shared<ctype::templated_t>(
                std::make_shared<ctype::monotype_t>("thrust::tuple"),
                std::move(arg_types));
        std::shared_ptr<ctype::templated_t> transform_t =
            std::make_shared<ctype::templated_t>(
                std::make_shared<ctype::monotype_t>("transformed_sequence"),
                std::vector<std::shared_ptr<ctype::type_t> >{
                    fn_t, thrust_tupled});
        std::shared_ptr<apply> n_rhs =
            std::static_pointer_cast<apply>(get_node_ptr(n.rhs()));
        //Can only handle names on the LHS
        assert(detail::isinstance<name>(n.lhs()));
        const name& lhs = boost::get<const name&>(n.lhs());
        std::shared_ptr<name> n_lhs = std::make_shared<name>(lhs.id(),
                                 get_type_ptr(lhs.type()),
                                 transform_t);
        auto result = std::make_shared<bind>(n_lhs, n_rhs);
        return result;
        
    }
    static result_type indices_rewrite(const bind& n) {
        return get_node_ptr(n);
    }

    typedef result_type(*rewrite_fn)(const bind&);
    typedef std::map<std::string, rewrite_fn> fn_map;
    const fn_map m_lut; 
public:
    thrust_rewriter() :
        m_lut{
        {std::string("map"), &backend::thrust_rewriter::map_rewrite},
        {std::string("indices"), &backend::thrust_rewriter::indices_rewrite}
    } {}
    using copier::operator();
    result_type operator()(const bind& n) {
        const expression& rhs = n.rhs();
        if (!detail::isinstance<apply>(rhs)) {
            return get_node_ptr(n);
        }
        const apply& rhs_apply = boost::get<const apply&>(rhs);
        const name& fn_name = rhs_apply.fn();
        auto it_delegate = m_lut.find(fn_name.id());
        if (it_delegate != m_lut.end()) {
            return (it_delegate->second)(n);
        } else {
            return get_node_ptr(n);
        }
    }
};



}
