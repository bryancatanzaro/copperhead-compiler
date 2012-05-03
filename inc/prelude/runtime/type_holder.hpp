#pragma once

#include "type.hpp"
#include "ctype.hpp"
#include <stack>

namespace copperhead {

struct type_holder {
    std::shared_ptr<const backend::type_t> m_t;
    std::shared_ptr<const backend::ctype::type_t> m_ct;
    //XXX Simplify
    //This is used by various make_type_holder methods to iteratively
    //construct types
    //It is left empty when interatively constructing types is not necessary
    std::stack<std::vector<std::shared_ptr<const backend::type_t> > > m_i;
};

}
