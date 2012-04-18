#include "thrust/rewrites.hpp"

using std::string;
using std::stringstream;
using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using std::vector;
using std::map;
using backend::utility::make_vector;
using backend::utility::make_map;

namespace backend {

thrust_rewriter::thrust_rewriter(const copperhead::system_variant& target)
    : m_target(target) {}


thrust_rewriter::result_type thrust_rewriter::map_rewrite(
    const bind& n) {
    //The rhs must be an apply
    assert(detail::isinstance<apply>(n.rhs()));
    const apply& rhs = boost::get<const apply&>(n.rhs());
    //The rhs must apply a "map"
    assert(rhs.fn().id().substr(0, 3) == string("map"));
    const tuple& ap_args = rhs.args();
    //Map must have arguments
    assert(ap_args.begin() != ap_args.end());
    auto init = ap_args.begin();
    shared_ptr<const ctype::type_t> fn_t;

    if (detail::isinstance<apply>(*init)) {
        //Function instantiation
        const apply& fn_inst = boost::get<const apply&>(
            *init);
        if (detail::isinstance<templated_name>(fn_inst.fn())) {
            const templated_name& tn =
                boost::get<const templated_name&>(fn_inst.fn());
            const ctype::tuple_t& tt =
                tn.template_types();
            vector<shared_ptr<const ctype::type_t> > ttc;
            for(auto i = tt.begin();
                i != tt.end();
                i++) {
                ttc.push_back(i->ptr());
            }
            shared_ptr<const ctype::monotype_t> base =
                make_shared<const ctype::monotype_t>(tn.id());
            fn_t = make_shared<const ctype::polytype_t>(
                std::move(ttc), base);
        } else {
            assert(detail::isinstance<name>(fn_inst.fn()));
            string fn_id = fn_inst.fn().id();
            fn_t = make_shared<const ctype::monotype_t>(fn_id);
        }
    } else {
        //We must be dealing with a closure
        assert(detail::isinstance<closure>(*init));

        const closure& close = boost::get<const closure&>(
            *init);
        int arity = close.args().arity();
        //The closure must enclose something
        assert(arity > 0);
        stringstream ss;
        ss << "closure" << arity;
        string closure_t_name = ss.str();
        shared_ptr<const ctype::monotype_t> closure_mt =
            make_shared<const ctype::monotype_t>(closure_t_name);
        vector<shared_ptr<const ctype::type_t> > cts;
        for(auto i = close.args().begin();
            i != close.args().end();
            i++) {
            cts.push_back(i->ctype().ptr());
        }
        //By this point, the body of the closure is an
        //instantiated functor (which must be an apply node)
        assert(detail::isinstance<apply>(close.body()));
        const apply& inst_fctor =
            boost::get<const apply&>(close.body());
        stringstream os;
        //It's either a plain name or a templated name
        if (detail::isinstance<templated_name>(inst_fctor.fn())) {
            const templated_name& tn =
                boost::get<const templated_name&>(inst_fctor.fn());
            os << tn.id() << "<";
            const ctype::tuple_t& template_types =
                tn.template_types();
            for(auto i = template_types.begin();
                i != template_types.end();
                i++) {
                //Template types contain ctype::monotype_t
                assert(detail::isinstance<ctype::monotype_t>(*i));
                const ctype::monotype_t& tmt =
                    boost::get<const ctype::monotype_t&>(*i);
                os << tmt.name();
                if (std::next(i) != template_types.end()) {
                    os << ", ";
                }
            }
            os << "> ";
        } else {
            assert(detail::isinstance<name>(inst_fctor.fn()));
            const name& fnn =
                boost::get<const name&>(inst_fctor.fn());
            os << fnn.id();
        }
        
        cts.push_back(
            make_shared<const ctype::monotype_t>(
                os.str()));
        fn_t = make_shared<const ctype::polytype_t>(
            std::move(cts),
            closure_mt);
    }
    vector<shared_ptr<const ctype::type_t> > arg_types;
    for(auto i = init+1; i != ap_args.end(); i++) {
        //Assert we're looking at a name
        assert(detail::isinstance<name>(*i));
        arg_types.push_back(
            make_shared<const ctype::monotype_t>(
                detail::typify(boost::get<const name&>(*i).id())));
    }
    shared_ptr<const ctype::polytype_t> thrust_tupled =
        make_shared<const ctype::polytype_t>(
            std::move(arg_types),
            make_shared<const ctype::monotype_t>("thrust::tuple"));
    shared_ptr<const ctype::polytype_t> transform_t =
        make_shared<const ctype::polytype_t>(
            make_vector<shared_ptr<const ctype::type_t> >
            (fn_t)(thrust_tupled),
            make_shared<const ctype::monotype_t>("transformed_sequence"));
    shared_ptr<const apply> n_rhs =
        static_pointer_cast<const apply>(n.rhs().ptr());
    //Can only handle names on the LHS
    assert(detail::isinstance<name>(n.lhs()));
    const name& lhs = boost::get<const name&>(n.lhs());
    shared_ptr<const name> n_lhs = make_shared<const name>(lhs.id(),
                                                           lhs.type().ptr(),
                                                           transform_t);
    auto result = make_shared<const bind>(n_lhs, n_rhs);
    return result;
        
}

thrust_rewriter::result_type thrust_rewriter::indices_rewrite(const bind& n) {
    //The rhs must be an apply
    assert(detail::isinstance<apply>(n.rhs()));
    const apply& rhs = boost::get<const apply&>(n.rhs());
    //The rhs must apply "indices"
    assert(rhs.fn().id() == string("indices"));
    const tuple& ap_args = rhs.args();
    //Indices must have arguments
    assert(ap_args.begin() != ap_args.end());
    const ctype::type_t& arg_t = ap_args.begin()->ctype();
    //Argument must have Seq[a] type
    assert(detail::isinstance<ctype::sequence_t>(arg_t));
        
    shared_ptr<const ctype::polytype_t> index_t =
        make_shared<const ctype::polytype_t>(
            make_vector<shared_ptr<const ctype::type_t> >
            (make_shared<const ctype::monotype_t>(copperhead::to_string(m_target))),
            make_shared<const ctype::monotype_t>("index_sequence"));
    shared_ptr<const apply> n_rhs =
        static_pointer_cast<const apply>(n.rhs().ptr());
    //Can only handle names on the LHS
    assert(detail::isinstance<name>(n.lhs()));
    const name& lhs = boost::get<const name&>(n.lhs());
    shared_ptr<const name> n_lhs =
        make_shared<const name>(lhs.id(),
                                lhs.type().ptr(),
                                index_t);
    auto result = make_shared<const bind>(n_lhs, n_rhs);
    return result;
}

thrust_rewriter::result_type thrust_rewriter::replicate_rewrite(const bind& n) {
    //The rhs must be an apply
    assert(detail::isinstance<apply>(n.rhs()));
    const apply& rhs = boost::get<const apply&>(n.rhs());
    //The rhs must apply "replicate"
    assert(rhs.fn().id() == string("replicate"));
    const tuple& ap_args = rhs.args();
    //replicate must have two arguments
    assert(ap_args.end() - ap_args.begin() == 2);

    //Need to add the target tag to this sequence.
    //Otherwise the machinery has nothing to pull a target from
    //To do this, we add an additional argument: the tag
    auto ap_arg_iterator = ap_args.begin();
    shared_ptr<const expression> tag_arg =
        make_shared<const apply>(
            make_shared<const name>(
                copperhead::to_string(m_target)),
            make_shared<const tuple>(
                make_vector<shared_ptr<const expression> >()));
    shared_ptr<const expression> arg1 = ap_arg_iterator->ptr();
    shared_ptr<const expression> arg2 = (ap_arg_iterator+1)->ptr();
    shared_ptr<const tuple> targeted_arguments =
        make_shared<const tuple>(
            make_vector<shared_ptr<const expression> >(tag_arg)(arg1)(arg2));
    
                    
    
    shared_ptr<const ctype::type_t> val_t =
        ap_args.begin()->ctype().ptr();
    
    
    shared_ptr<const ctype::polytype_t> constant_t =
        make_shared<const ctype::polytype_t>(
            make_vector<shared_ptr<const ctype::type_t> >
            (make_shared<const ctype::monotype_t>(copperhead::to_string(m_target)))
            (val_t),
            make_shared<const ctype::monotype_t>("constant_sequence"));

    shared_ptr<const apply> n_rhs =
        make_shared<const apply>(rhs.fn().ptr(),
                                 targeted_arguments);
            
    //Can only handle names on the LHS
    assert(detail::isinstance<name>(n.lhs()));
    const name& lhs = boost::get<const name&>(n.lhs());
    shared_ptr<const name> n_lhs =
        make_shared<const name>(lhs.id(),
                                lhs.type().ptr(),
                                constant_t);
    auto result = make_shared<const bind>(n_lhs, n_rhs);
    return result;
}

thrust_rewriter::result_type thrust_rewriter::shift_rewrite(const bind& n) {
    //The rhs must be an apply
    assert(detail::isinstance<apply>(n.rhs()));
    const apply& rhs = boost::get<const apply&>(n.rhs());
    //The rhs must apply "shift"
    assert(rhs.fn().id() == string("shift"));
    const tuple& ap_args = rhs.args();
    //replicate must have three arguments
    assert(ap_args.end() - ap_args.begin() == 3);

    shared_ptr<const ctype::type_t> val_t =
        ap_args.begin()->ctype().ptr();
    
    shared_ptr<const ctype::polytype_t> shifted_t =
        make_shared<const ctype::polytype_t>(
            make_vector<shared_ptr<const ctype::type_t> >(val_t),
            make_shared<const ctype::monotype_t>("shifted_sequence"));
    //Can only handle names on the LHS
    assert(detail::isinstance<name>(n.lhs()));
    const name& lhs = boost::get<const name&>(n.lhs());
    shared_ptr<const name> n_lhs =
        make_shared<const name>(lhs.id(),
                                lhs.type().ptr(),
                                shifted_t);
    auto result = make_shared<const bind>(n_lhs, rhs.ptr());
    return result;
}

thrust_rewriter::result_type thrust_rewriter::rotate_rewrite(const bind& n) {
    //The rhs must be an apply
    assert(detail::isinstance<apply>(n.rhs()));
    const apply& rhs = boost::get<const apply&>(n.rhs());
    //The rhs must apply "rotate"
    assert(rhs.fn().id() == string("rotate"));
    const tuple& ap_args = rhs.args();
    //replicate must have three arguments
    assert(ap_args.end() - ap_args.begin() == 2);

    shared_ptr<const ctype::type_t> val_t =
        ap_args.begin()->ctype().ptr();
    
    
    
    shared_ptr<const ctype::polytype_t> rotated_t =
        make_shared<const ctype::polytype_t>(
            make_vector<shared_ptr<const ctype::type_t> >(val_t),
            make_shared<const ctype::monotype_t>("rotated_sequence"));
    //Can only handle names on the LHS
    assert(detail::isinstance<name>(n.lhs()));
    const name& lhs = boost::get<const name&>(n.lhs());
    shared_ptr<const name> n_lhs =
        make_shared<const name>(lhs.id(),
                                lhs.type().ptr(),
                                rotated_t);
    auto result = make_shared<const bind>(n_lhs, rhs.ptr());
    return result;
}


thrust_rewriter::result_type thrust_rewriter::operator()(const bind& n) {
    const expression& rhs = n.rhs();
    if (!detail::isinstance<apply>(rhs)) {
        return n.ptr();
    }
    const apply& rhs_apply = boost::get<const apply&>(rhs);
    const name& fn_name = rhs_apply.fn();
    if (fn_name.id().substr(0, 3) == "map") {
        return map_rewrite(n);
    } else if (fn_name.id() == "indices") {
        return indices_rewrite(n);
    } else if (fn_name.id() == "replicate") {
        return replicate_rewrite(n);
    } else if (fn_name.id() == "rotate") {
        return rotate_rewrite(n);
    } else if (fn_name.id() == "shift") {
        return shift_rewrite(n);
    } else {
        return n.ptr();
    }

}




}
