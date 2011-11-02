#include "decl.hpp"


namespace backend {

namespace detail {

typedef std::tuple<const char*, fn_info> named_info;

namespace impl {

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
            bool_mt));

std::shared_ptr<type_t> un_op_t =
     std::make_shared<polytype_t>(
        std::vector<std::shared_ptr<monotype_t> >{
            t_a},
        std::make_shared<fn_t>(
            std::make_shared<tuple_t>(
                std::vector<std::shared_ptr<type_t> >{
                    t_a}),
            t_a));


std::shared_ptr<phase_t> bin_phase_t =
    std::make_shared<phase_t>(
        std::vector<completion>{
            completion::invariant,
                completion::invariant},
        completion::invariant);

std::shared_ptr<phase_t> un_phase_t =
    std::make_shared<phase_t>(
        std::vector<completion>{
            completion::invariant},
        completion::invariant);

fn_info bin_op_info(bin_op_t, bin_phase_t);
fn_info bin_cmp_info(bin_cmp_t, bin_phase_t);
fn_info un_op_info(un_op_t, un_phase_t);
fn_info nullary_info(void_mt,
                     std::make_shared<phase_t>(
                         std::vector<completion>{},
                         completion::invariant));

std::vector<named_info> 
    binary_scalar_operators = {
    named_info{"op_add",    bin_op_info},
    named_info{"op_sub",    bin_op_info},
    named_info{"op_mul",    bin_op_info},
    named_info{"op_div",    bin_op_info},
    named_info{"op_mod",    bin_op_info},
    named_info{"op_lshift", bin_op_info},
    named_info{"op_rshift", bin_op_info},
    named_info{"op_or",     bin_op_info},
    named_info{"op_xor",    bin_op_info},
    named_info{"op_and",    bin_op_info},
    named_info{"cmp_eq",    bin_cmp_info},
    named_info{"cmp_ne",    bin_cmp_info},
    named_info{"cmp_lt",    bin_cmp_info},
    named_info{"cmp_le",    bin_cmp_info},
    named_info{"cmp_gt",    bin_cmp_info},
    named_info{"cmp_ge",    bin_cmp_info}
};

std::vector<named_info>
unary_scalar_operators = {
    named_info{"op_invert", un_op_info},
    named_info{"op_pos",    un_op_info},
    named_info{"op_neg",    un_op_info},
    named_info{"op_not",    un_op_info}
};

std::vector<named_info>
cpp_support_fns = {
    named_info{"wrap_cuarray", nullary_info}
};

}

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
    detail::load_scalars(fns, detail::impl::unary_scalar_operators);
    detail::load_scalars(fns, detail::impl::binary_scalar_operators);
    detail::load_scalars(fns, detail::impl::cpp_support_fns);
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
