#pragma once
#include <memory>
#include <vector>
#include <map>
#include <cstdlib>
#include "../import/library.hpp"
#include "../import/paths.hpp"

#define PRELUDE_PATH "PRELUDE_PATH"
#define THRUST_PATH "THRUST_PATH"
#define THRUST_FILE "thrust.hpp"



namespace backend {

namespace detail {
std::vector<const char*> thrust_fn_names = {
    "adjacent_difference",
    "map",
    "scan",
    "rscan"/*,
    "exscan",
    "exrscan"*/
};

}


std::shared_ptr<library> get_thrust() {
    std::map<ident, fn_info> exported_fns;
    fn_info blank;
    for(auto i = detail::thrust_fn_names.begin();
        i != detail::thrust_fn_names.end();
        i++) {
        exported_fns.insert(std::pair<ident, fn_info>(
                       ident(std::string(*i), iteration_structure::parallel),
                       blank));

    }
    //XXX HACK.  NEED boost::filesystem path manipulation
    std::string library_path(std::string(detail::get_path(PRELUDE_PATH)) +
                             "/../thrust");
    std::string thrust_path(std::string(detail::get_path(THRUST_PATH)));
    std::set<std::string> include_paths;
    if (library_path.length() > 0) {
        include_paths.insert(library_path);
    }
    if (thrust_path.length() > 0) {
        include_paths.insert(thrust_path);
    }
    return std::shared_ptr<library>(
        new library(std::move(exported_fns),
                    std::set<std::string>{std::string(THRUST_FILE)},
                    std::move(include_paths)));
}

}
