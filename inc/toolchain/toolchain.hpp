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
