#pragma once

#include <boost/variant.hpp>
#include <functional>

namespace backend {

class monotype_t;
class polytype_t;


namespace detail {
typedef boost::variant<
    monotype_t &,
    polytype_t &
    > type_base;

struct make_type_base_visitor
    : boost::static_visitor<type_base>
{
    make_type_base_visitor(void *p)
        : ptr(p)
        {}
    template<typename Derived>
    type_base operator()(const Derived &) const {
        // use of std::ref disambiguates variant's copy constructor dispatch
        return type_base(std::ref(*reinterpret_cast<Derived*>(ptr)));
    }
    void *ptr;
};

type_base make_type_base(void *ptr, const type_base &other) {
    return boost::apply_visitor(make_type_base_visitor(ptr), other);
}

}

class type_t
    : public detail::type_base
{
public:
    typedef detail::type_base super_t;
    template<typename Derived>
    type_t(Derived &self)
        : super_t(std::ref(self)) //use of std::ref disambiguates variant's copy constructor dispatch
        {}

    type_t(const type_t &other)
        : super_t(detail::make_type_base(this, other))
        {}

};

}
