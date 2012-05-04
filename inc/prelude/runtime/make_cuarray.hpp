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

#include <prelude/runtime/cuarray.hpp>
#include <prelude/runtime/make_type_holder.hpp>
#include <prelude/runtime/tags.h>

namespace copperhead {

namespace detail {

template<typename T>
struct make_cuarray_impl {
    static void fun(sp_cuarray r, size_t s) {
        add_type(r->m_t.get(), T());
        r->add_chunk(boost::shared_ptr<chunk>(new chunk(cpp_tag(), s * sizeof(T))), true);
#ifdef CUDA_SUPPORT
        r->add_chunk(boost::shared_ptr<chunk>(new chunk(cuda_tag(), s * sizeof(T))), true);
#endif
    
    }
};

template<typename HT, typename TT>
struct make_cuarray_impl<thrust::detail::cons<HT, TT> > {
    static void fun(sp_cuarray r, size_t s) {
        make_cuarray_impl<HT>::fun(r, s);
        make_cuarray_impl<TT>::fun(r, s);
    }
};

template<typename T0,
         typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename T6,
         typename T7,
         typename T8,
         typename T9>
struct make_cuarray_impl<
    thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> > {
    typedef thrust::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> result_type;
    static void fun(
        sp_cuarray r, size_t s) {
        detail::begin(r->m_t.get());
        make_cuarray_impl<
            thrust::detail::cons<
                typename result_type::head_type,
                typename result_type::tail_type> >::fun(r, s);
        detail::end_tuple(r->m_t.get());
    }
};

template<>
struct make_cuarray_impl<thrust::null_type> {
    static void fun(sp_cuarray, size_t) {
    }
};

}

template<typename T>
sp_cuarray make_cuarray(size_t s) {
    type_holder* th = detail::make_type_holder();
    detail::begin(th);
    sp_cuarray r(new cuarray(th));
    r->push_back_length(s);    
    detail::make_cuarray_impl<T>::fun(r, s);
    detail::end_sequence(th);
    detail::finalize_type(th);
    return r;
}


}
