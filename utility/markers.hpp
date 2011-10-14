#pragma once

namespace backend {
namespace detail {

std::string mark_generated_id(const std::string &in) {
    if (in[0] == '_') {
        return in.substr(1);
    } else {
        return in;
    }
}

std::string wrap_array_id(const std::string &in) {
    return mark_generated_id(in + "_ary");
}

std::string wrap_proc_id(const std::string &in) {
    return mark_generated_id(in + "_wrap");
}

std::string typify(const std::string &in) {
    return "T" + in;
}

}
}
