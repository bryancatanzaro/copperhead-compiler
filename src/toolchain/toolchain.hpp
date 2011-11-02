#pragma once
#include "../import/library.hpp"
#include "../utility/io.hpp"
#include <sstream>
#include <memory>

namespace backend {

struct toolchain {
protected:
    toolchain(std::shared_ptr<registry> &r)
        : m_registry(r) {}
    std::shared_ptr<registry> m_registry;
};

struct gcc_nvcc : public toolchain {
    gcc_nvcc(std::shared_ptr<registry> &r)
        : toolchain(r) {}
    std::string command_line() {
        std::stringstream sstrm;
        sstrm << "nvcc ";
        print_iterable(sstrm,
                       m_registry->include_dirs(),
                       std::string(" "),
                       std::string("-I"));
        print_iterable(sstrm,
                       m_registry->link_dirs(),
                       std::string(" "),
                       std::string("-L"));
        print_iterable(sstrm,
                       m_registry->links(),
                       std::string(" "),
                       std::string("-l"));
        
        
        return sstrm.str();

    }
};

}
