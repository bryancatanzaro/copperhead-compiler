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

#include <prelude/runtime/allocators.hpp>

namespace copperhead {

template<typename M>
class chunk {
  private:
    M m_s;
    void* m_d;
    size_t m_r;
  public:
    chunk(const M&s,
          size_t r) : m_s(s), m_d(NULL), m_r(r) {
    }
    ~chunk() {
        if (m_d != NULL) {
            m_s.deallocate(m_d);
            m_d = NULL;
        }
    }
    //movable
    chunk(chunk&& o)
        : m_s(o.m_s) {
        m_d = o.m_d;
        m_r = o.m_r;
        o.m_d = NULL;
    }
    chunk& operator=(chunk&& o) {
        m_s = o.m_s;
        m_d = o.m_d;
        m_r = o.m_r;
        o.m_d = NULL;
        return *this;
    }
    //not copyable
    chunk(const chunk&) = delete;
    chunk& operator=(const chunk&) = delete;
    void* ptr() {
        if (m_d == NULL) {
            //Lazy allocation - only allocate when pointer is requested
            m_d = m_s.allocate(m_r);
        } 
        return m_d;
    }
    size_t size() const {
        return m_r;
    }

};

}
