#pragma once
#include <boost/type_traits.hpp>
#include <boost/mpl/logical.hpp>

namespace backend {
namespace detail {

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

template<typename T, typename V>
bool isinstance(const V& v) {
    bool result = false;
    boost::apply_visitor(type_checker<T>(result), v);
    return result;
}

}
}
