#pragma once

#include "ctype.hpp"

namespace backend {
namespace detail {

std::shared_ptr<const ctype::type_t> container_type(const ctype::type_t &t);

}
}
