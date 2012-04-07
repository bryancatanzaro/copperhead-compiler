#pragma once

#include "type.hpp"
#include "ctype.hpp"

namespace copperhead {

struct cu_and_c_types {
    std::shared_ptr<const backend::type_t> m_t;
    std::shared_ptr<const backend::ctype::type_t> m_ct;
};

}
