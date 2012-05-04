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
#include <prelude/sequences/zipped_sequence.h>
#include <cassert>
#include <iostream>

namespace copperhead {

namespace detail {

template<typename S>
struct make_seq_impl {};

template<typename Tag, typename T>
struct make_seq_impl<sequence<Tag, T, 0> > {
    static sequence<Tag, T, 0> fun(typename std::vector<boost::shared_ptr<chunk> >::iterator d,
                                   std::vector<size_t>::const_iterator l,
                                   const size_t o=0) {
        return sequence<Tag, T, 0>(reinterpret_cast<T*>((*d)->ptr())+o, *l);
    }
};

template<typename Tag, typename T>
struct make_seq_impl<sequence<Tag, T, 1> > {
    static sequence<Tag, T, 1> fun(typename std::vector<boost::shared_ptr<chunk> >::iterator d,
                                   std::vector<size_t>::const_iterator l,
                                   const size_t o=0) {
        sequence<Tag, size_t, 0> desc = make_seq_impl<sequence<Tag, size_t, 0> >::fun(d, l, o);
        sequence<Tag, T, 0> data = make_seq_impl<sequence<Tag, T, 0> >::fun(d+1, l+1);
        return sequence<Tag, T, 1>(desc, data);
    }
};

template<typename Tag, typename T, int D>
struct make_seq_impl<sequence<Tag, T, D > > {
    static sequence<Tag, T, D> fun(typename std::vector<boost::shared_ptr<chunk> >::iterator d,
                                   std::vector<size_t>::const_iterator l,
                                   const size_t o=0) {
        sequence<Tag, size_t, 0> desc = make_seq_impl<sequence<Tag, size_t, 0> >::fun(d, l, o);
        sequence<Tag, T, D-1> sub = make_seq_impl<sequence<Tag, T, D-1> >::fun(d+1, l+1);
        return sequence<Tag, T, D>(desc, sub);
    }
};

template<typename HT, typename TT>
struct make_seq_impl<thrust::detail::cons<HT, TT> > {
    static thrust::detail::cons<HT, TT> fun(typename std::vector<boost::shared_ptr<chunk> >::iterator d,
                                            std::vector<size_t>::const_iterator l,
                                            const size_t o=0) {
        return thrust::detail::cons<HT, TT>(
            make_seq_impl<HT>::fun(d, l, o),
            make_seq_impl<TT>::fun(d+1, l, o));
    }
};

template<typename S0,
         typename S1,
         typename S2,
         typename S3,
         typename S4,
         typename S5,
         typename S6,
         typename S7,
         typename S8,
         typename S9>
struct make_seq_impl<zipped_sequence<
                         thrust::tuple<S0, S1, S2, S3, S4, S5, S6, S7, S8, S9> > > {
    typedef thrust::tuple<S0, S1, S2, S3, S4, S5, S6, S7, S8, S9> sequences;
    static zipped_sequence<sequences> fun(typename std::vector<boost::shared_ptr<chunk> >::iterator d,
                                          std::vector<size_t>::const_iterator l,
                                          const size_t o=0) {
        std::cout << "Calling make_sequence constructor!" << std::endl;
        sequences s = make_seq_impl<
            thrust::detail::cons<
                typename sequences::head_type,
                typename sequences::tail_type> >::fun(d, l, o);
        return zipped_sequence<sequences>(s);
        
    }
};

template<>
struct make_seq_impl<thrust::null_type> {
    static thrust::null_type fun(typename std::vector<boost::shared_ptr<chunk> >::iterator d,
                                 std::vector<size_t>::const_iterator l,
                                 const size_t o=0) {
        return thrust::null_type();
    }
};



}

template<typename S>
S make_sequence(sp_cuarray& in, system_variant t, bool write) {
    cuarray& r = *in;
    system_variant canonical_tag = canonical_memory_tag(t);
    std::pair<std::vector<boost::shared_ptr<chunk> >, bool>& s = r.m_d[canonical_tag];
    //Do we need to copy?
    if (!s.second) {
        //Find a valid representation
        std::pair<std::vector<boost::shared_ptr<chunk> >, bool> x;
        x.second = false;
        for(typename data_map::iterator i = r.m_d.begin();
            (x.second == false) && (i != r.m_d.end());
            i++) {
            x = i->second;
        }
        assert(x.second == true);
        //Copy from valid representation
        for(std::vector<boost::shared_ptr<chunk> >::iterator i = s.first.begin(),
                j = x.first.begin();
            i != s.first.end();
            i++, j++) {
            (*i)->copy_from(**j);
        }
        s.second = true;
    }
    //Do we need to invalidate?
    if (write) {
        for(typename data_map::iterator i = r.m_d.begin();
            i != r.m_d.end();
            i++) {
            i->second.second = system_variant_equal(i->first, canonical_tag);
        }
    }
    return detail::make_seq_impl<S>::fun(s.first.begin(), r.m_l.begin(), r.m_o);
}

}
