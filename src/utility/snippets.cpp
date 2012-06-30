#include "utility/snippets.hpp"

namespace backend {
namespace detail {

const std::string anonymous_return() {
    return "result";
}

const std::string make_sequence() {
    return "make_sequence";
}

const std::string wrap() {
    return "wrap";
}

const std::string make_remote() {
    return "make_remote";
}

const std::string boost_python_module() {
    return "BOOST_PYTHON_MODULE";
}

const std::string boost_python_def() {
    return "boost::python::def";
}

const std::string phase_boundary() {
    return "phase_boundary";
}

const std::string snippet_get(int x) {
    std::ostringstream os;
    os << "thrust::get<" << x << ">";
    return os.str();
}

const std::string snippet_make_tuple() {
    return "thrust::make_tuple";
}

}
}
