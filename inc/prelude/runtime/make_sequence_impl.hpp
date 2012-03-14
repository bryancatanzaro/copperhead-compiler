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
#include <boost/iterator/indirect_iterator.hpp>

#include <prelude/runtime/allocators.hpp>
#include <prelude/runtime/chunk.hpp>
#include <prelude/sequences/sequence.h>

namespace copperhead {

template<typename S, typename M>
struct make_seq_impl {};

template<typename T, typename M>
struct make_seq_impl<sequence<T, 0>, M > {
    static sequence<T, 0> fun(typename std::vector<std::shared_ptr<chunk<M> > >::iterator d,
                              std::vector<size_t>::const_iterator l,
                              const size_t o=0) {
        return sequence<T, 0>(reinterpret_cast<T*>((*d)->ptr())+o, *l);
    }
};

template<typename T, typename M>
struct make_seq_impl<sequence<T, 1>, M > {
    static sequence<T, 1> fun(typename std::vector<std::shared_ptr<chunk<M> > >::iterator d,
                              std::vector<size_t>::const_iterator l,
                              const size_t o=0) {
        sequence<size_t, 0> desc = make_seq_impl<sequence<size_t, 0>, M >::fun(d, l, o);
        sequence<T, 0> data = make_seq_impl<sequence<T, 0>, M >::fun(d+1, l+1);
        return sequence<T, 1>(desc, data);
    }
};

template<typename T, int D, typename M>
struct make_seq_impl<sequence<T, D >, M > {
    static sequence<T, D> fun(typename std::vector<std::shared_ptr<chunk<M> > >::iterator d,
                              std::vector<size_t>::const_iterator l,
                              const size_t o=0) {
        sequence<size_t, 0> desc = make_seq_impl<sequence<size_t, 0>, M >::fun(d, l, o);
        sequence<T, D-1> sub = make_seq_impl<sequence<T, D-1>, M >::fun(d+1, l+1);
        return sequence<T, D>(desc, sub);
    }
};

#ifndef CUDA_SUPPORT
template<typename S>
S make_sequence(sp_cuarray& in, bool local, bool write) {
    cuarray& r = *in;
    return make_seq_impl<S, host_alloc>::fun(r.m_local.begin(), r.m_l.cbegin(), r.m_o);
}
#else
#include <cuda_runtime.h>

void retrieve(cuarray& r) {
    auto i = boost::make_indirect_iterator(r.m_local.begin());
    auto j = boost::make_indirect_iterator(r.m_remote.begin());
    auto e = boost::make_indirect_iterator(r.m_local.end());
    for(; i != e; ++i, ++j) {
        cudaMemcpy(i->ptr(), j->ptr(), i->size(), cudaMemcpyDeviceToHost);
    }
}

void exile(cuarray& r) {
    auto i = boost::make_indirect_iterator(r.m_local.begin());
    auto j = boost::make_indirect_iterator(r.m_remote.begin());
    auto e = boost::make_indirect_iterator(r.m_local.end());
    for(; i != e; ++i, ++j) {
        cudaMemcpy(j->ptr(), i->ptr(), i->size(), cudaMemcpyHostToDevice);
    }
}

template<typename S>
S make_sequence(sp_cuarray& in, bool local, bool write) {
    cuarray& r = *in;
    if (local) {
        if (!r.m_clean_local) {
            retrieve(r);
            r.m_clean_local = true;
        }
        if (write)
            r.m_clean_remote = false;
        return make_seq_impl<S, host_alloc>::fun(r.m_local.begin(), r.m_l.cbegin(), r.m_o);
    } else {
        if (!r.m_clean_remote) {
            exile(r);
            r.m_clean_remote = true;
        }
        if (write)
            r.m_clean_local = false;
        return make_seq_impl<S, cuda_alloc>::fun(r.m_remote.begin(), r.m_l.cbegin(), r.m_o);
    }
}

#endif

}
