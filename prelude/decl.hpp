#pragma once
#include <memory>
#include <vector>
#include <map>
#include <cstdlib>
#include "../import/library.hpp"

#define PRELUDE_PATH "PRELUDE_PATH"
#define PRELUDE_FILE "prelude.hpp"

#define GCC_VERSION (__GNUC__ * 10000                 \
                     + __GNUC_MINOR__ * 100           \
                     + __GNUC_PATCHLEVEL__)
      
#if GCC_VERSION < 40600
#define nullptr NULL
#endif

namespace backend {

namespace detail {

ident make_scalar(std::string &in) {
    return ident{in, iteration_structure::scalar};
}

std::vector<const char*> binary_scalar_operators = {
    "op_add",
    "op_sub",
    "op_mul",
    "op_div",
    "op_mod",
    "op_lshift",
    "op_rshift",
    "op_or",
    "op_xor",
    "op_and",
    "cmp_eq",
    "cmp_ne",
    "cmp_lt",
    "cmp_le",
    "cmp_gt",
    "cmp_ge"
};

std::vector<const char*> unary_scalar_operators = {
    "op_invert",
    "op_pos",
    "op_neg",
    "op_not"
};

void load_scalars(
    std::map<ident, fn_info>& fns,
    const std::vector<const char*>& names) {
    fn_info blank;
    for(auto i = names.begin();
        i != names.end();
        i++) {
        fns.insert(std::pair<ident, fn_info>{
                ident{std::string(*i), iteration_structure::scalar},
                    blank});
    }
}

const char* get_prelude_path() {
    char* path = getenv(PRELUDE_PATH);
    if (path != nullptr) {
        return path;
    } else {
        return "";
    }
}

}


std::shared_ptr<library> get_builtins() {
    std::map<ident, fn_info> fns;
    detail::load_scalars(fns, detail::unary_scalar_operators);
    detail::load_scalars(fns, detail::binary_scalar_operators);
    std::string path(detail::get_prelude_path());
    std::set<std::string> include_paths;
    if (path.length() > 0) {
        include_paths.insert(path);
    }
    std::string include(PRELUDE_FILE);
    std::shared_ptr<library> l(new library(fns,
                                           std::set<std::string>{include},
                                           include_paths));
    return l;
}


}
