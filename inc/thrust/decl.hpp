/*
 *   Copyright 2012      NVIDIA Corporation
 * 
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 * 
 *       http://www.apache.org/licenses/LICENSE-2.0
 * 
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 * 
 */
#pragma once
#include <memory>
#include <vector>
#include <map>
#include <cstdlib>
#include <sstream>
#include "import/library.hpp"
#include "import/paths.hpp"
#include "node.hpp"
#include "type.hpp"
#include "monotype.hpp"
#include "polytype.hpp"
#include "utility/initializers.hpp"


#define PRELUDE_PATH "PRELUDE_PATH"
#define THRUST_PATH "THRUST_PATH"
#define THRUST_FILE "thrust.h"



namespace backend {

std::shared_ptr<library> get_thrust();

}
