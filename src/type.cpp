#include "type.hpp"

namespace backend {
namespace detail {

make_type_base_visitor::make_type_base_visitor(void *p) : ptr(p) {}

type_base make_type_base(void *ptr, const type_base &other) {
    return boost::apply_visitor(make_type_base_visitor(ptr), other);
}

}

type_t::type_t(const type_t &other)
  : super_t(detail::make_type_base(this, other)) {}

}
