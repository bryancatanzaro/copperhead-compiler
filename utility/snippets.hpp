#pragma once

#include "../cppnode.hpp"

namespace backend {
namespace detail {



const std::string& get_remote_r() {
    return "get_remote_r";
}

const std::string& get_remote_w() {
    return "get_remote_w";
}

const std::string& wrap() {
    return "wrap";
}

const std::string& make_remote() {
    return "make_remote";
}

const std::string& boost_python_module() {
    return "BOOST_PYTHON_MODULE";
}

const std::string& boost_python_def() {
    return "boost::python::def";

}
}
