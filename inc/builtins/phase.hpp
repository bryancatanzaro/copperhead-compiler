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
#include <sstream>
#include <vector>
#include <memory>

namespace backend {

enum struct iteration_structure {
    scalar,
    sequential,
    parallel,
    independent
};

enum struct completion {
    none,
    local,
    total,
    invariant
};


class phase_t
    : public std::enable_shared_from_this<phase_t> {
private:
    const std::vector<completion> m_args;
    const completion m_result;
public:
    phase_t(std::vector<completion>&& args, const completion& result);
    typedef std::vector<completion>::const_iterator iterator;
    iterator begin() const;
    iterator end() const;
    completion result() const;
    int size() const;
    std::shared_ptr<const phase_t> ptr() const;
};


        
}

//XXX Some compilers require this, others don't
bool operator<(backend::iteration_structure a, backend::iteration_structure b);

//To make iteration_structure enums print
std::ostream& operator<<(std::ostream& strm,
                         const backend::iteration_structure& is);

//To make completion enums print
std::ostream& operator<<(std::ostream& strm,
                         const backend::completion& cn);

//To make phase_t structs print
std::ostream& operator<<(std::ostream& strm,
                         const backend::phase_t& ct);
