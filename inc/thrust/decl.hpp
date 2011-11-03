#pragma once
#include <memory>
#include <vector>
#include <map>
#include <cstdlib>
#include <sstream>
#include "../import/library.hpp"
#include "../import/paths.hpp"
#include "../node.hpp"
#include "../type.hpp"
#include "../monotype.hpp"
#include "../polytype.hpp"
#define PRELUDE_PATH "PRELUDE_PATH"
#define THRUST_PATH "THRUST_PATH"
#define THRUST_FILE "thrust.h"



namespace backend {

std::shared_ptr<library> get_thrust();

}
