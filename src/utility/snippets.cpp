#include "utility/snippets.hpp"

namespace backend {
namespace detail {



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

const std::string tag() {
    return "tag";
}

const std::string fake_tag_string(const copperhead::detail::fake_system_tag& t) {
    if (t == copperhead::detail::fake_omp_tag) {
        return "omp_tag";
    } else if (t == copperhead::detail::fake_cuda_tag) {
        return "cuda_tag";
    }
    throw std::invalid_argument("Unknown system tag");
}

}
}
