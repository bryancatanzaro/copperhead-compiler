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
#include "tuple_cat.h"
#include "apply_from_tuple.h"

namespace copperhead {

template<typename F,
         typename T>
struct closure {
    typedef typename F::result_type result_type;
    F m_f;
    T m_t;

    __host__ __device__ closure(const F& f, const T& t)
        : m_f(f),
          m_t(t) {}

    __host__ __device__ result_type operator()() {
        return detail::apply_from_tuple(m_f, m_t);
    }
    
    template<typename T0>
    __host__ __device__ result_type operator()(const T0& t0) {
        return detail::apply_from_tuple(m_f,
                                        thrust::tuple_cat(
                                            thrust::make_tuple(t0),
                                            m_t));
    }
    template<typename T0, typename T1>
    __host__ __device__ result_type operator()(const T0& t0,
                                               const T1& t1) {
        return detail::apply_from_tuple(m_f,
                                        thrust::tuple_cat(
                                            thrust::make_tuple(t0, t1),
                                            m_t));
    }

    template<typename T0, typename T1, typename T2>
    __host__ __device__ result_type operator()(const T0& t0,
                                               const T1& t1,
                                               const T2& t2) {
        return detail::apply_from_tuple(m_f,
                                        thrust::tuple_cat(
                                            thrust::make_tuple(t0, t1, t2),
                                            m_t));
    }
    
    template<typename T0, typename T1, typename T2, typename T3>
    __host__ __device__ result_type operator()(const T0& t0,
                                               const T1& t1,
                                               const T2& t2,
                                               const T3& t3) {
        return detail::apply_from_tuple(m_f,
                                        thrust::tuple_cat(
                                            thrust::make_tuple(t0, t1, t2, t3),
                                            m_t));
    }

    template<typename T0, typename T1, typename T2, typename T3, typename T4>
    __host__ __device__ result_type operator()(const T0& t0,
                                               const T1& t1,
                                               const T2& t2,
                                               const T3& t3,
                                               const T4& t4) {
        return detail::apply_from_tuple(m_f,
                                        thrust::tuple_cat(
                                            thrust::make_tuple(t0, t1, t2, t3, t4),
                                            m_t));
    } 
    
    template<typename T0, typename T1, typename T2, typename T3, typename T4,
             typename T5>
    __host__ __device__ result_type operator()(const T0& t0,
                                               const T1& t1,
                                               const T2& t2,
                                               const T3& t3,
                                               const T4& t4,
                                               const T5& t5) {
        return detail::apply_from_tuple(m_f,
                                        thrust::tuple_cat(
                                            thrust::make_tuple(
                                                t0, t1, t2, t3, t4,
                                                t5),
                                            m_t));
    }
    
    template<typename T0, typename T1, typename T2, typename T3, typename T4,
             typename T5, typename T6>
    __host__ __device__ result_type operator()(const T0& t0,
                                               const T1& t1,
                                               const T2& t2,
                                               const T3& t3,
                                               const T4& t4,
                                               const T5& t5,
                                               const T6& t6) {
        return detail::apply_from_tuple(m_f,
                                        thrust::tuple_cat(
                                            thrust::make_tuple(
                                                t0, t1, t2, t3, t4,
                                                t5, t6),
                                            m_t));
    }
    
    template<typename T0, typename T1, typename T2, typename T3, typename T4,
             typename T5, typename T6, typename T7>
    __host__ __device__ result_type operator()(const T0& t0,
                                               const T1& t1,
                                               const T2& t2,
                                               const T3& t3,
                                               const T4& t4,
                                               const T5& t5,
                                               const T6& t6,
                                               const T7& t7) {
        return detail::apply_from_tuple(m_f,
                                        thrust::tuple_cat(
                                            thrust::make_tuple(
                                                t0, t1, t2, t3, t4,
                                                t5, t6, t7),
                                            m_t));
    }

    template<typename T0, typename T1, typename T2, typename T3, typename T4,
             typename T5, typename T6, typename T7, typename T8>
    __host__ __device__ result_type operator()(const T0& t0,
                                               const T1& t1,
                                               const T2& t2,
                                               const T3& t3,
                                               const T4& t4,
                                               const T5& t5,
                                               const T6& t6,
                                               const T7& t7,
                                               const T8& t8) {
        return detail::apply_from_tuple(m_f,
                                        thrust::tuple_cat(
                                            thrust::make_tuple(
                                                t0, t1, t2, t3, t4,
                                                t5, t6, t7, t8),
                                            m_t));
    }
    
    template<typename T0, typename T1, typename T2, typename T3, typename T4,
             typename T5, typename T6, typename T7, typename T8, typename T9>
    __host__ __device__ result_type operator()(const T0& t0,
                                               const T1& t1,
                                               const T2& t2,
                                               const T3& t3,
                                               const T4& t4,
                                               const T5& t5,
                                               const T6& t6,
                                               const T7& t7,
                                               const T8& t8,
                                               const T9& t9) {
        return detail::apply_from_tuple(m_f,
                                        thrust::tuple_cat(
                                            thrust::make_tuple(
                                                t0, t1, t2, t3, t4,
                                                t5, t6, t7, t8, t9),
                                            m_t));
    }

};

}
