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
#include "dereference.hpp"
#include "utility/initializers.hpp"

using std::string;
using std::shared_ptr;
using std::make_shared;
using backend::utility::make_vector;

namespace backend {

dereference::dereference(const string& entry_point) : m_entry_point(entry_point), m_in_entry(false) {}

dereference::result_type dereference::operator()(const procedure &s) {
    m_in_entry = (s.id().id() == m_entry_point);
    return this->rewriter::operator()(s);
}

dereference::result_type dereference::operator()(const subscript &s) {
    if (!m_in_entry) {
        return s.ptr();
    } else {
        return make_shared<const apply>(
            make_shared<const name>("dereference"),
            make_shared<const tuple>(
                make_vector<shared_ptr<const expression> >
                (s.src().ptr())(s.idx().ptr())));
    }
}

}
