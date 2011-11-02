#pragma once
#include <boost/variant.hpp>

namespace backend {
namespace detail {

template<typename T>
struct type_extractor
    : public boost::static_visitor<const T&>
{
    template<typename U>
    const T& operator()(const U& u) const {
        return (const T&)u;
    }
};

template<typename T, typename U>
const T& up_get(const U& u) {
    return boost::apply_visitor(type_extractor<T>(), u);
}

}
}
        
