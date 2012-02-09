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
#include <map>
#include <set>
#include <vector>
#include <cassert>

namespace backend {

namespace detail {

template<typename key,
         typename value,
         typename store>
class environment_impl {
private:
    std::vector<store> m_stores;
public:
    typedef typename store::const_iterator citer;
    typedef typename store::value_type value_type;
    
    environment_impl() {
        begin_scope();
    }
    void begin_scope() {
        m_stores.push_back(store());
    }
    void end_scope() {
        m_stores.pop_back();
        assert(m_stores.size() > 0);
    }
    citer end() const {
        return m_stores[0].end();
    }
    citer find(const key& in) const {
        for(int i = m_stores.size()-1; i>= 0; i--) {
            const store& current_store = m_stores[i];
            citer found = current_store.find(in);
            if (found != current_store.end()) {
                return found;
            }
        }
        return end();
    }
    void insert(const value_type& in) {
        m_stores[m_stores.size()-1].insert(in);
    }
    bool exists(const key& in) const {
        return (find(in) != end());
    }
};

}

template<typename key, typename value=void>
class environment
    : public detail::environment_impl<key, value, std::map<key, value> > {};

template<typename key>
class environment<key, void>
    : public detail::environment_impl<key, void, std::set<key> >{};

}
