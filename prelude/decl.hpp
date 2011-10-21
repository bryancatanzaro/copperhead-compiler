#pragma once
#include <memory>
#include <vector>
#include <map>
#include <cstdlib>
#include "../import/library.hpp"
#include "../import/paths.hpp"
#include "../type.hpp"
#include "../monotype.hpp"
#include "../polytype.hpp"


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

std::shared_ptr<monotype_t> t_a =
    std::make_shared<monotype_t>("a");

std::shared_ptr<type_t> bin_op_t =
    std::make_shared<polytype_t>(
        std::vector<std::shared_ptr<monotype_t> >{
            t_a},
        std::make_shared<fn_t>(
            std::make_shared<tuple_t>(
                std::vector<std::shared_ptr<type_t> >{
                    t_a, t_a}),
            t_a));

std::shared_ptr<type_t> bin_cmp_t =
    std::make_shared<polytype_t>(
        std::vector<std::shared_ptr<monotype_t> >{
            t_a},
        std::make_shared<fn_t>(
            std::make_shared<tuple_t>(
                std::vector<std::shared_ptr<type_t> >{
                    t_a, t_a}),
            std::make_shared<bool_mt>()));

std::shared_ptr<type_t> un_op_t =
     std::make_shared<polytype_t>(
        std::vector<std::shared_ptr<monotype_t> >{
            t_a},
        std::make_shared<fn_t>(
            std::make_shared<tuple_t>(
                std::vector<std::shared_ptr<type_t> >{
                    t_a}),
            t_a));

typedef std::tuple<const char*, fn_info> named_info;

std::vector<named_info> 
    binary_scalar_operators = {
    named_info{"op_add",    bin_op_t},
    named_info{"op_sub",    bin_op_t},
    named_info{"op_mul",    bin_op_t},
    named_info{"op_div",    bin_op_t},
    named_info{"op_mod",    bin_op_t},
    named_info{"op_lshift", bin_op_t},
    named_info{"op_rshift", bin_op_t},
    named_info{"op_or",     bin_op_t},
    named_info{"op_xor",    bin_op_t},
    named_info{"op_and",    bin_op_t},
    named_info{"cmp_eq",    bin_cmp_t},
    named_info{"cmp_ne",    bin_cmp_t},
    named_info{"cmp_lt",    bin_cmp_t},
    named_info{"cmp_le",    bin_cmp_t},
    named_info{"cmp_gt",    bin_cmp_t},
    named_info{"cmp_ge",    bin_cmp_t}
};

std::vector<named_info>
unary_scalar_operators = {
    named_info{"op_invert", un_op_t},
    named_info{"op_pos",    un_op_t},
    named_info{"op_neg",    un_op_t},
    named_info{"op_not",    un_op_t}
};

std::vector<named_info>
cpp_support_fns = {
    named_info{"wrap_cuarray", std::make_shared<void_mt>()}
};

void load_scalars(
    std::map<ident, fn_info>& fns,
    const std::vector<named_info>& names) {
    for(auto i = names.begin();
        i != names.end();
        i++) {
        fns.insert(std::pair<ident, fn_info>{
                ident{std::string(std::get<0>(*i)),
                        iteration_structure::scalar},
                    std::get<1>(*i)});
    }
}

}


std::shared_ptr<library> get_builtins() {
    std::map<ident, fn_info> fns;
    detail::load_scalars(fns, detail::unary_scalar_operators);
    detail::load_scalars(fns, detail::binary_scalar_operators);
    detail::load_scalars(fns, detail::cpp_support_fns);
    std::string path(detail::get_path(PRELUDE_PATH));
    std::set<std::string> include_paths;
    if (path.length() > 0) {
        include_paths.insert(path);
    }
    std::string include(PRELUDE_FILE);
    std::shared_ptr<library> l(new library(std::move(fns),
                                           std::set<std::string>{include},
                                           std::move(include_paths)));
    return l;
}


}
