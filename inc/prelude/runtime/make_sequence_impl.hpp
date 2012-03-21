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
#include <vector>
#include <prelude/runtime/chunk.hpp>
#include <prelude/sequences/sequence.h>

namespace copperhead {

template<typename S>
struct make_seq_impl {};

template<typename T>
struct make_seq_impl<sequence<T, 0> > {
    static sequence<T, 0> fun(typename std::vector<std::shared_ptr<chunk> >::iterator d,
                              std::vector<size_t>::const_iterator l,
                              const size_t o=0) {
        return sequence<T, 0>(reinterpret_cast<T*>((*d)->ptr())+o, *l);
    }
};

template<typename T>
struct make_seq_impl<sequence<T, 1> > {
    static sequence<T, 1> fun(typename std::vector<std::shared_ptr<chunk> >::iterator d,
                              std::vector<size_t>::const_iterator l,
                              const size_t o=0) {
        sequence<size_t, 0> desc = make_seq_impl<sequence<size_t, 0> >::fun(d, l, o);
        sequence<T, 0> data = make_seq_impl<sequence<T, 0> >::fun(d+1, l+1);
        return sequence<T, 1>(desc, data);
    }
};

template<typename T, int D>
struct make_seq_impl<sequence<T, D > > {
    static sequence<T, D> fun(typename std::vector<std::shared_ptr<chunk> >::iterator d,
                              std::vector<size_t>::const_iterator l,
                              const size_t o=0) {
        sequence<size_t, 0> desc = make_seq_impl<sequence<size_t, 0> >::fun(d, l, o);
        sequence<T, D-1> sub = make_seq_impl<sequence<T, D-1> >::fun(d+1, l+1);
        return sequence<T, D>(desc, sub);
    }
};

template<typename S>
S make_sequence(sp_cuarray& in, detail::fake_system_tag t, bool write) {
    cuarray& r = *in;
    std::pair<std::vector<std::shared_ptr<chunk> >, bool>& s = r.m_d[t];
    //Do we need to copy?
    if (!s.second) {
        //Find a valid representation
        std::pair<std::vector<std::shared_ptr<chunk> >, bool> x;
        x.second = false;
        for(auto i = r.m_d.cbegin();
            (x.second == false) && (i != r.m_d.cend());
            i++) {
            x = i->second;
        }
        assert(x.second == true);
        //Copy from valid representation
        for(auto i = s.first.begin(), j = x.first.begin();
            i != s.first.end();
            i++, j++) {
            (*i)->copy_from(**j);
        }
    }
    //Do we need to invalidate?
    if (write) {
        for(auto i = r.m_d.begin();
            i != r.m_d.end();
            i++) {
            i->second.second = (i->first == t);
        }
    }
    return make_seq_impl<S>::fun(s.first.begin(), r.m_l.cbegin(), r.m_o);
}

}
