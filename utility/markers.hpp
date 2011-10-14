#pragma once

namespace backend {
namespace detail {

//XXX This isn't necessary any more, is it?
std::string mark_generated_id(const std::string &in) {
    if (in[0] == '_') {
        return in.substr(1);
    } else {
        return in;
    }
}

std::string fnize_id(const std::string &in) {
    if (in[0] == '_') {
        return mark_generated_id("fn" + in);
    } else {
        return mark_generated_id("fn_" + in);
    }
}

std::string wrap_array_id(const std::string &in) {
    return mark_generated_id("ary" + in);
}

std::string wrap_proc_id(const std::string &in) {
    return mark_generated_id("wrap" + in);
}

std::string typify(const std::string &in) {
    return "T" + in;
}

}
}
