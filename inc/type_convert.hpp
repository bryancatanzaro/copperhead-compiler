#pragma once
#include "rewriter.hpp"
#include "type.hpp"
#include "ctype.hpp"
#include "type_printer.hpp"

#include <iostream>
#include <cassert>

namespace backend {

namespace detail {

//! Converts Copperhead types to C++ types
/*! Input ASTs typically have their C++ implementation types
  unspecified (defaulting to void). This visitor translates from
  Copperhead type to C++ type.
*/
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
/*! 
\addtogroup rewriters
@{
 */

//! A rewrite pass that converts Copperhead types to C++ types
/*! It does not change the structure of the AST, just
  creates a new AST where the C++ types are freshly derived
  from the Copperhead types embedded in the input AST.
*/
class type_convert
    : public rewriter
{
private:
    detail::cu_to_c m_c;
public:
    //! Constructor
    type_convert();

    using rewriter::operator();
    //! Rewrite rule for \p procedure nodes
    result_type operator()(const procedure &p);
    //! Rewrite rule for \p name nodes
    result_type operator()(const name &p);
    //! Rewrite rule for \p literal nodes
    result_type operator()(const literal &p);
};

/*!
  @}
*/
}
