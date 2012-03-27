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

#include <cstddef>
#include <prelude/runtime/tags.h>

namespace copperhead {

class chunk {
private:
    system_variant m_s;
    void* m_d;
    size_t m_r;
public:
    chunk(const system_variant &s,
          size_t r);
    ~chunk();
private:
    //Not copyable
    chunk(const chunk& o);
    //Not assignable
    chunk& operator=(const chunk&);
public:
    void copy_from(chunk& o);
    void* ptr();
    size_t size() const;
};

}
