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
#include "type.hpp"
#include "monotype.hpp"

namespace backend {

class polytype_t :
        public type_t {
private:
    std::vector<std::shared_ptr<monotype_t> > m_vars;
    std::shared_ptr<monotype_t> m_monotype;
public:
    polytype_t(std::vector<std::shared_ptr<monotype_t> >&& vars,
               std::shared_ptr<monotype_t> monotype);
    
    typedef decltype(boost::make_indirect_iterator(m_vars.cbegin())) const_iterator;
    
    const monotype_t& monotype() const;

    std::shared_ptr<monotype_t> p_monotype() const;

    const_iterator begin() const;

    const_iterator end() const;
};

}
