#pragma once
#include <boost/variant.hpp>


namespace backend {
namespace detail {
/*!
  \addtogroup utilities
  @{
*/


//! Used internally.
template<typename T>
struct type_extractor
    : public boost::static_visitor<const T&>
{
    template<typename U>
    const T& operator()(const U& u) const {
        return (const T&)u;
    }
};

//! Similar to boost::get, but won't fail if you retrieve a base class.
/*! boost::get<T> allows you to extract a T object from a variant
    holding a T.  However, it fails if the variant holds U and you ask
    for T, even if U derives from T. This function succeeds in that case.

  \tparam T The type requested by the programmer.
  \param u The variant being examined.
  
  \return 
*/
template<typename T, typename U>
const T& up_get(const U& u) {
    return boost::apply_visitor(type_extractor<T>(), u);
}

/*!
  @}
*/


}
}
        
