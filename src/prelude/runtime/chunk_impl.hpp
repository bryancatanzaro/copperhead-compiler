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

#include <prelude/runtime/chunk.hpp>
#include <prelude/config.h>
#include <prelude/runtime/tags.h>
#include <prelude/runtime/tag_malloc_and_free.h>
#include <stdexcept>

namespace copperhead {

namespace detail {

struct apply_malloc
    : public boost::static_visitor<void*> {
    const size_t m_ctr;
    apply_malloc(const size_t& ctr) : m_ctr(ctr) {}

    template<typename Tag>
    void* operator()(const Tag& t) const {
        return thrust::detail::tag_malloc(Tag(), m_ctr);
    }
};

struct apply_free
    : public boost::static_visitor<> {
    void* m_p;
    apply_free(void* p) : m_p(p) {}

    template<typename Tag>
    void operator()(const Tag& t) const {
        thrust::detail::tag_free(Tag(), m_p);
    }
};

system_variant fake_to_real(const detail::fake_system_tag& t) {
    if (t == detail::fake_omp_tag) {
        return omp_tag();
    }
#ifdef CUDA_SUPPORT
    else if (t == detail::fake_cuda_tag) {
        return cuda_tag();
    }
#endif
    else {
        throw std::invalid_argument("Internal error due to non typesafe enum.");
    }
}
            

}

chunk::chunk(const detail::fake_system_tag &s,
             size_t r) : m_s(s), m_d(NULL), m_r(r) {}

chunk::~chunk() {
    if (m_d != NULL) {
        system_variant s = fake_to_real(m_s);
        boost::apply_visitor(
            detail::apply_free(m_d),
            s);
    }
}

void* chunk::ptr() {
    if (m_d == NULL) {
        //Lazy allocation - only allocate when pointer is requested
        system_variant s = fake_to_real(m_s);
        m_d = boost::apply_visitor(
            detail::apply_malloc(m_r),
            s);
    } 
    return m_d;
}

size_t chunk::size() const {
    return m_r;
}


}
