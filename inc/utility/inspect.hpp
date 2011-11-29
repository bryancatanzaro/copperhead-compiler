#pragma once
#include <iostream>


namespace backend {
namespace detail {

struct inspect_impl : boost::static_visitor<> {
    template<typename S>
    void operator()(const S& n) {
        std::cout << typeid(n).name();
    }
};
/*!
  \addtogroup utilities
  @{
*/

//! Inspects a variant and prints out its dynamic type to std::cout
template<typename V>
void inspect(V& n) {
    inspect_impl g;
    boost::apply_visitor(g, n);
    return;
}
/*!
  @}
*/

}
}

