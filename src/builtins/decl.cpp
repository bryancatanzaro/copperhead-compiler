#include "builtins/decl.hpp"

using std::string;
using std::shared_ptr;
using std::make_shared;
using std::vector;
using std::map;
using std::set;
using std::make_pair;
using std::move;
using backend::utility::make_vector;
using backend::utility::make_set;
using backend::utility::make_map;

namespace backend {

namespace builtins {

typedef std::tuple<const char*, fn_info> named_info;

namespace detail {

shared_ptr<const monotype_t> t_a =
    make_shared<const monotype_t>("a");

shared_ptr<const type_t> bin_op_t =
    make_shared<const polytype_t>(
        make_vector<shared_ptr<const monotype_t> >(t_a),
        make_shared<const fn_t>(
            make_shared<const tuple_t>(
                make_vector<shared_ptr<const type_t> >(t_a)(t_a)),
            t_a));

shared_ptr<const type_t> bin_cmp_t =
    make_shared<const polytype_t>(
        make_vector<shared_ptr<const monotype_t> >(t_a),
        make_shared<const fn_t>(
            make_shared<const tuple_t>(
                make_vector<shared_ptr<const type_t> >(t_a)(t_a)),
            bool_mt));

shared_ptr<const type_t> un_op_t =
     make_shared<const polytype_t>(
         make_vector<shared_ptr<const monotype_t> >(
             t_a),
        make_shared<const fn_t>(
            make_shared<const tuple_t>(
                make_vector<shared_ptr<const type_t> >(t_a)),
            t_a));


shared_ptr<phase_t> bin_phase_t =
    make_shared<phase_t>(
        make_vector<completion>(completion::none)(completion::none),
        completion::invariant);

shared_ptr<phase_t> un_phase_t =
    make_shared<phase_t>(
        make_vector<completion>(
            completion::none),
        completion::invariant);

fn_info bin_op_info(bin_op_t, bin_phase_t);
fn_info bin_cmp_info(bin_cmp_t, bin_phase_t);
fn_info un_op_info(un_op_t, un_phase_t);
fn_info nullary_info(void_mt,
                     make_shared<phase_t>(
                         make_vector<completion>(),
                         completion::invariant));

vector<named_info> binary_scalar_operators =
    make_vector<named_info>
    (named_info("op_add",    bin_op_info))
    (named_info("op_sub",    bin_op_info))
    (named_info("op_mul",    bin_op_info))
    (named_info("op_div",    bin_op_info))
    (named_info("op_mod",    bin_op_info))
    (named_info("op_lshift", bin_op_info))
    (named_info("op_rshift", bin_op_info))
    (named_info("op_or",     bin_op_info))
    (named_info("op_xor",    bin_op_info))
    (named_info("op_and",    bin_op_info))
    (named_info("cmp_eq",    bin_cmp_info))
    (named_info("cmp_ne",    bin_cmp_info))
    (named_info("cmp_lt",    bin_cmp_info))
    (named_info("cmp_le",    bin_cmp_info))
    (named_info("cmp_gt",    bin_cmp_info))
    (named_info("cmp_ge",    bin_cmp_info));

vector<named_info> unary_scalar_operators =
    make_vector<named_info>
    (named_info("op_invert", un_op_info))
    (named_info("op_pos",    un_op_info))
    (named_info("op_neg",    un_op_info))
    (named_info("op_not",    un_op_info));

vector<named_info> cpp_support_fns =
    make_vector<named_info>
    (named_info("wrap_cuarray", nullary_info))
    (named_info("make_scalar", nullary_info))
    (named_info("unpack_scalar", nullary_info));

}

void load_scalars(
    std::map<ident, fn_info>& fns,
    const vector<named_info>& names) {
    for(auto i = names.begin();
        i != names.end();
        i++) {
        fns.insert(
            make_pair(
                make_pair(
                    string(std::get<0>(*i)),
                    iteration_structure::scalar),
                std::get<1>(*i)));
    }
}

}


shared_ptr<library> get_builtins() {
    map<ident, fn_info> fns;
    builtins::load_scalars(fns, builtins::detail::unary_scalar_operators);
    builtins::load_scalars(fns, builtins::detail::binary_scalar_operators);
    builtins::load_scalars(fns, builtins::detail::cpp_support_fns);
    string path(detail::get_path(PRELUDE_PATH));
    set<string> include_paths;
    if (path.length() > 0) {
        include_paths.insert(path);
    }
    string include(PRELUDE_FILE);
    shared_ptr<library> l(new library(move(fns),
                                      make_map<string, string>(),
                                      make_set<string>(include),
                                      move(include_paths)));
    return l;
}


}
