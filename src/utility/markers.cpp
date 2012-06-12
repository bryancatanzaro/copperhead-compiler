#include "utility/markers.hpp"

namespace backend {
namespace detail {

std::string fnize_id(const std::string &in) {
    if (in[0] == '_') {
        return "fn" + in;
    } else {
        return "fn_" + in;
    }
}

std::string wrap_array_id(const std::string &in) {
    return "ary" + in;
}

std::string wrap_proc_id(const std::string &in) {
    return "wrap" + in;
}

std::string typify(const std::string &in) {
    return "T" + in;
}

std::string complete(const std::string &in) {
    return "comp" + in;
}

}
}
