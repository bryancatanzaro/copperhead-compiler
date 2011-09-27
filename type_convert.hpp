#pragma once
#include "copier.hpp"
#include "type.hpp"
#include "ctype.hpp"
#include "type_printer.hpp"

#include <iostream>
#include <cassert>

namespace backend {

namespace detail {
class cu_to_c
    : public boost::static_visitor<std::shared_ptr<ctype::type_t> >
{
public:
    result_type operator()(const monotype_t& mt) {
        assert(false);
        return result_type(new ctype::monotype_t("Empty"));
        
    }
    result_type operator()(const sequence_t & st) {
        result_type sub = boost::apply_visitor(*this, st.sub());
        return result_type(new ctype::sequence_t(sub));
    }
    result_type operator()(const tuple_t& tt) {
        std::vector<result_type> subs;
        for(auto i = tt.begin(); i != tt.end(); i++) {
            subs.push_back(boost::apply_visitor(*this, *i));
        }
        return result_type(new ctype::tuple_t(std::move(subs)));
    }
    result_type operator()(const fn_t& ft) {
        std::shared_ptr<ctype::tuple_t> args =
            std::static_pointer_cast<ctype::tuple_t>(
                boost::apply_visitor(*this, ft.args()));
        std::shared_ptr<ctype::type_t> result =
            boost::apply_visitor(*this, ft.result());
        result_type fn_result(new ctype::fn_t(args, result));
        return fn_result;
    }
    //XXX Need polytypes!
    result_type operator()(const polytype_t&) {
        return result_type(new ctype::void_mt());
    }
    result_type operator()(const int32_mt&) {
        return result_type(new ctype::int32_mt());
    }
    result_type operator()(const int64_mt&) {
        return result_type(new ctype::int64_mt());
    }
    result_type operator()(const uint32_mt&) {
        return result_type(new ctype::uint32_mt());
    }
    result_type operator()(const uint64_mt&) {
        return result_type(new ctype::uint64_mt());
    }
    result_type operator()(const float32_mt&) {
        return result_type(new ctype::float32_mt());
    }
    result_type operator()(const float64_mt&) {
        return result_type(new ctype::float64_mt());
    }
    result_type operator()(const bool_mt&) {
        return result_type(new ctype::bool_mt());
    }
    result_type operator()(const void_mt&) {
        return result_type(new ctype::void_mt());
    }
};
}

class type_convert
    : public copier
{
private:
    detail::cu_to_c m_c;
    type_copier m_tc;
public:
    type_convert() : m_c(), m_tc() {}
    using copier::operator();
    result_type operator()(const procedure &p) {
        std::shared_ptr<ctype::type_t> ct = boost::apply_visitor(m_c, p.type());
        std::shared_ptr<name> id = std::static_pointer_cast<name>(this->operator()(p.id()));
        std::shared_ptr<tuple> args = std::static_pointer_cast<tuple>(this->operator()(p.args()));
        std::shared_ptr<suite> stmts = std::static_pointer_cast<suite>(this->operator()(p.stmts()));
        std::shared_ptr<type_t> t = boost::apply_visitor(m_tc, p.type());
        std::shared_ptr<node> result(new procedure(id, args, stmts, t, ct));
        return result;
    }
};

}
