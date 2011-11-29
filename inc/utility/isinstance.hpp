#pragma once
#include <boost/type_traits.hpp>
#include <boost/mpl/logical.hpp>


namespace backend {
namespace detail {
/*!
  \addtogroup utilities
  @{
*/

//! Used internally to examine types.
template<typename T>
class type_checker:
        public boost::static_visitor<>
{
public:
    explicit type_checker(bool& res) : m_res(res) {}
    template<typename U>
    typename boost::disable_if<
        boost::is_base_of<T, U> >::type
    operator()(const U& u) const {
        m_res = false;
    }
    template<typename U>
    typename boost::enable_if<
        boost::is_base_of<T, U> >::type
    operator()(const U& t) const {
        m_res = true;
    }
private:
    bool& m_res;
};
//! Checks if dynamic type is an instance of another type.
/*! This procedure examines a variant to discover if the dynamic type
  which the variant currently holds is an instance of another
  type. Will return true if the variant holds a derived type of the
  base type.
  \tparam T Base type.
  \param v Instance being checked.
  
  \return true if v is an instance of T, false otherwise.
*/
template<typename T, typename V>
bool isinstance(const V& v) {
    bool result = false;
    boost::apply_visitor(type_checker<T>(result), v);
    return result;
}

/*!
  @}
*/


}
}

