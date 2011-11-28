#pragma once
#include "rewriter.hpp"
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
    result_type operator()(const monotype_t& mt);
    
    result_type operator()(const sequence_t & st);
    
    result_type operator()(const tuple_t& tt);
    
    result_type operator()(const fn_t& ft);
    
    //XXX Need polytypes! This code is probably not right.
    result_type operator()(const polytype_t& p);
};
}

class type_convert
    : public rewriter
{
private:
    detail::cu_to_c m_c;
public:
    type_convert();

    using rewriter::operator();

    result_type operator()(const procedure &p);
    
    result_type operator()(const name &p);
};

}
