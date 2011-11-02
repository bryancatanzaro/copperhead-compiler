#pragma once
#include <memory>
#include <vector>
#include <map>
#include <cstdlib>
#include "../type.hpp"
#include "../monotype.hpp"
#include "../polytype.hpp"
#include "../import/library.hpp"
#include "../import/paths.hpp"


#define PRELUDE_PATH "PRELUDE_PATH"
#define PRELUDE_FILE "prelude.h"

#define GCC_VERSION (__GNUC__ * 10000                 \
                     + __GNUC_MINOR__ * 100           \
                     + __GNUC_PATCHLEVEL__)
      
#if GCC_VERSION < 40600
#define nullptr NULL
#endif

namespace backend {

std::shared_ptr<library> get_builtins();

}
