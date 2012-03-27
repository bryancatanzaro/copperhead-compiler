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

#include <prelude/config.h>
#include <prelude/runtime/chunk.hpp>
#include <prelude/runtime/tags.h>
#include <prelude/runtime/tag_malloc_and_free.h>
#include <stdexcept>
#include <thrust/copy.h>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/not.hpp>

namespace copperhead {

namespace detail {

struct apply_malloc
    : public boost::static_visitor<void*> {
    const size_t m_ctr;
    apply_malloc(const size_t& ctr) : m_ctr(ctr) {}

    template<typename Tag>
    void* operator()(const Tag& t) const {
        return thrust::detail::tag_malloc(t, m_ctr);
    }
};

struct apply_free
    : public boost::static_visitor<> {
    void* m_p;
    apply_free(void* p) : m_p(p) {}

    template<typename Tag>
    void operator()(const Tag& t) const {
        thrust::detail::tag_free(t, m_p);
    }
};

struct apply_copy
    : public boost::static_visitor<> {
    void* m_d;
    void* m_s;
    size_t m_r;
    apply_copy(void* d, void* s, size_t r)
        : m_d(d), m_s(s), m_r(r) {}

    template<typename DTag, typename STag>
    typename boost::enable_if<
        boost::is_same<DTag, STag> >::type
    operator()(DTag, STag) const {
    }
    
    template<typename DTag, typename STag>
    typename boost::enable_if<
        boost::mpl::not_<
            boost::is_same<DTag, STag> > >::type
    operator()(DTag, STag) const {
        thrust::pointer<char, STag> s_start((char*)m_s);
        thrust::pointer<char, STag> s_end = s_start + m_r;
        thrust::pointer<char, DTag> d_start((char*)m_d);
        thrust::copy(s_start, s_end, d_start);
    }
};
            
}

chunk::chunk(const system_variant &s,
             size_t r) : m_s(s), m_d(NULL), m_r(r) {}

chunk::~chunk() {
    if (m_d != NULL) {
        boost::apply_visitor(
            detail::apply_free(m_d),
            m_s);
    }
}

void chunk::copy_from(chunk& o) {
    if (boost::apply_visitor(detail::compare_tags(), m_s, o.m_s)) {
        //XXX Investigate why this doesn't work
        throw std::invalid_argument("Internal error: can't copy a chunk into the same memory space");
    }
    if (m_r != o.m_r) {
        throw std::invalid_argument("Internal error: can't copy chunks of different size");
    }
    boost::apply_visitor(detail::apply_copy(ptr(),
                                            o.ptr(),
                                            m_r),
                         m_s,
                         o.m_s);
}


void* chunk::ptr() {
    if (m_d == NULL) {
        //Lazy allocation - only allocate when pointer is requested
        m_d = boost::apply_visitor(
            detail::apply_malloc(m_r),
            m_s);
    } 
    return m_d;
}

size_t chunk::size() const {
    return m_r;
}


}
