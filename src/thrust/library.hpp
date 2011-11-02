#pragma once
#include <memory>
#include <vector>
#include <map>
#include <cstdlib>
#include "../import/library.hpp"
#include "../import/paths.hpp"
#include "../node.hpp"
#include "../type.hpp"
#include "../monotype.hpp"
#define PRELUDE_PATH "PRELUDE_PATH"
#define THRUST_PATH "THRUST_PATH"
#define THRUST_FILE "thrust.hpp"



namespace backend {

namespace detail {
extern std::vector<const char*> thrust_fn_names;
}

std::shared_ptr<library> get_thrust();

}
