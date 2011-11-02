#include "import/paths.hpp"

const char* backend::detail::get_path(const char* env_name) {
    char* path = getenv(env_name);
    if (path != nullptr) {
        return path;
    } else {
        return "";
    }
}
