#pragma once

namespace backend {
namespace detail {

std::string mark_generated_id(const std::string &in) {
    if (in[0] == '_') {
        return in;
    } else {
        return "_" + in;
    }
}

std::string wrap_array_id(const std::string &in) {
    return mark_generated_id(in + "_ary");
}
}
}
