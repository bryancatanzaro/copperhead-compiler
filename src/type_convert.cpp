#include "type_convert.hpp"
namespace backend {

namespace detail {

cu_to_c::result_type cu_to_c::operator()(const monotype_t& mt) {
    if (mt.name() == "Int32") {
        return ctype::int32_mt;
    } else if (mt.name() == "Int64") {
        return ctype::int64_mt;
    } else if (mt.name() == "Uint32") {
        return ctype::uint32_mt;
    } else if (mt.name() == "Uint64") {
        return ctype::uint64_mt;
    } else if (mt.name() == "Float32") {
        return ctype::float32_mt;
    } else if (mt.name() == "Float64") {
        return ctype::float64_mt;
    } else if (mt.name() == "Bool") {
        return ctype::bool_mt;
    } else if (mt.name() == "Void") {
        return ctype::void_mt;
    } else {
        return result_type(new ctype::monotype_t(mt.name()));
    }
}
cu_to_c::result_type cu_to_c::operator()(const sequence_t & st) {
    result_type sub = boost::apply_visitor(*this, st.sub());
    return result_type(new ctype::sequence_t(sub));
}
cu_to_c::result_type cu_to_c::operator()(const tuple_t& tt) {
    std::vector<result_type> subs;
    for(auto i = tt.begin(); i != tt.end(); i++) {
        subs.push_back(boost::apply_visitor(*this, *i));
    }
    return result_type(new ctype::tuple_t(std::move(subs)));
}
cu_to_c::result_type cu_to_c::operator()(const fn_t& ft) {
    std::shared_ptr<ctype::tuple_t> args =
        std::static_pointer_cast<ctype::tuple_t>(
            boost::apply_visitor(*this, ft.args()));
    std::shared_ptr<ctype::type_t> result =
        boost::apply_visitor(*this, ft.result());
    result_type fn_result(new ctype::fn_t(args, result));
    return fn_result;
}
//XXX Need polytypes! This code is probably not right.
cu_to_c::result_type cu_to_c::operator()(const polytype_t& p) {
    std::vector<result_type> subs;
    for(auto i = p.begin();
        i != p.end();
        i++) {
        subs.push_back(boost::apply_visitor(*this, *i));
    }
    result_type base = boost::apply_visitor(*this, p.monotype());
    return result_type(new ctype::templated_t(base, std::move(subs)));
}

}


type_convert::type_convert() : m_c() {}
type_convert::result_type type_convert::operator()(const procedure &p) {
    std::shared_ptr<name> id = std::static_pointer_cast<name>(this->operator()(p.id()));
    std::shared_ptr<tuple> args = std::static_pointer_cast<tuple>(this->operator()(p.args()));
    std::shared_ptr<suite> stmts = std::static_pointer_cast<suite>(this->operator()(p.stmts()));
    std::shared_ptr<type_t> t = p.p_type();
        
    //Yes, I really want to make a ctype from a type. That's the point!
    std::shared_ptr<ctype::type_t> ct = boost::apply_visitor(m_c, p.type());

    std::shared_ptr<node> result(new procedure(id, args, stmts, t, ct));
    return result;
}
type_convert::result_type type_convert::operator()(const name &p) {
    std::shared_ptr<type_t> t = p.p_type();
        
    //Yes, I really want to make a ctype from a type. That's the point!
    std::shared_ptr<ctype::type_t> ct = boost::apply_visitor(m_c, p.type());
    result_type result(new name(p.id(), t, ct));
    return result;
}


}
