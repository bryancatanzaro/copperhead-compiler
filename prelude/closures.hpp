#pragma once

//If NVCC ever supports variadic templates...

template<typename K0,
         typename F,
         typename R>
struct closure1 {
    typedef R result_type;
    K0 m_k0;
    F m_f;
    __host__ __device__ closure1(const K0& k0,
                                 const F& f)
        : m_k0(k0),
          m_f(f) {}

    template<typename T0>
    __host__ __device__ R operator()(const T0& t0) {
        return m_f(t0,
                   m_k0);
    }
    template<typename T0,
             typename T1>
    __host__ __device__ R operator()(const T0& t0,
                                     const T1& t1) {
        return m_f(t0,
                   t1,
                   m_k0);
    }

    template<typename T0,
             typename T1,
             typename T2>
    __host__ __device__ R operator()(const T0& t0,
                                     const T1& t1,
                                     const T2& t2) {
        return m_f(t0,
                   t1,
                   t2,
                   m_k0);
    }
    
    template<typename T0,
             typename T1,
             typename T2,
             typename T3>
    __host__ __device__ R operator()(const T0& t0,
                                     const T1& t1,
                                     const T2& t2,
                                     const T3& t3) {
        return m_f(t0,
                   t1,
                   t2,
                   t3,
                   m_k0);
    }

    template<typename T0,
             typename T1,
             typename T2,
             typename T3,
             typename T4>
    __host__ __device__ R operator()(const T0& t0,
                                     const T1& t1,
                                     const T2& t2,
                                     const T3& t3,
                                     const T4& t4) {
        return m_f(t0,
                   t1,
                   t2,
                   t3,
                   t4,
                   m_k0);
    }
    
    template<typename T0,
             typename T1,
             typename T2,
             typename T3,
             typename T4,
             typename T5>
    __host__ __device__ R operator()(const T0& t0,
                                     const T1& t1,
                                     const T2& t2,
                                     const T3& t3,
                                     const T4& t4,
                                     const T5& t5) {
        return m_f(t0,
                   t1,
                   t2,
                   t3,
                   t4,
                   t5,
                   m_k0);
    }
    
    template<typename T0,
             typename T1,
             typename T2,
             typename T3,
             typename T4,
             typename T5,
             typename T6>
    __host__ __device__ R operator()(const T0& t0,
                                     const T1& t1,
                                     const T2& t2,
                                     const T3& t3,
                                     const T4& t4,
                                     const T5& t5,
                                     const T6& t6) {
        return m_f(t0,
                   t1,
                   t2,
                   t3,
                   t4,
                   t5,
                   t6,
                   m_k0);
    }
    
    template<typename T0,
             typename T1,
             typename T2,
             typename T3,
             typename T4,
             typename T5,
             typename T6,
             typename T7>
    __host__ __device__ R operator()(const T0& t0,
                                     const T1& t1,
                                     const T2& t2,
                                     const T3& t3,
                                     const T4& t4,
                                     const T5& t5,
                                     const T6& t6,
                                     const T7& t7) {
        return m_f(t0,
                   t1,
                   t2,
                   t3,
                   t4,
                   t5,
                   t6,
                   t7,
                   m_k0);
    }

    template<typename T0,
             typename T1,
             typename T2,
             typename T3,
             typename T4,
             typename T5,
             typename T6,
             typename T7,
             typename T8>
    __host__ __device__ R operator()(const T0& t0,
                                     const T1& t1,
                                     const T2& t2,
                                     const T3& t3,
                                     const T4& t4,
                                     const T5& t5,
                                     const T6& t6,
                                     const T7& t7,
                                     const T8& t8) {
        return m_f(t0,
                   t1,
                   t2,
                   t3,
                   t4,
                   t5,
                   t6,
                   t7,
                   t8,
                   m_k0);
    }
    
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
    __host__ __device__ R operator()(const T0& t0,
                                     const T1& t1,
                                     const T2& t2,
                                     const T3& t3,
                                     const T4& t4,
                                     const T5& t5,
                                     const T6& t6,
                                     const T7& t7,
                                     const T8& t8,
                                     const T9& t9) {
        return m_f(t0,
                   t1,
                   t2,
                   t3,
                   t4,
                   t5,
                   t6,
                   t7,
                   t8,
                   t9,
                   m_k0);
    }

};


template<typename K0,
         typename K1,
         typename F,
         typename R>
struct closure2 {
    typedef R result_type;
    K0 m_k0;
    K1 m_k1;
    F m_f;
    __host__ __device__ closure1(const K0& k0,
                                 const K1& k1,
                                 const F& f)
        : m_k0(k0),
          m_f(f) {}

    template<typename T0>
    __host__ __device__ R operator()(const T0& t0) {
        return m_f(t0,
                   m_k0,
                   m_k1);
    }
    template<typename T0,
             typename T1>
    __host__ __device__ R operator()(const T0& t0,
                                     const T1& t1) {
        return m_f(t0,
                   t1,
                   m_k0,
                   m_k1);
    }

    template<typename T0,
             typename T1,
             typename T2>
    __host__ __device__ R operator()(const T0& t0,
                                     const T1& t1,
                                     const T2& t2) {
        return m_f(t0,
                   t1,
                   t2,
                   m_k0,
                   m_k1);
    }
    
    template<typename T0,
             typename T1,
             typename T2,
             typename T3>
    __host__ __device__ R operator()(const T0& t0,
                                     const T1& t1,
                                     const T2& t2,
                                     const T3& t3) {
        return m_f(t0,
                   t1,
                   t2,
                   t3,
                   m_k0,
                   m_k1);
    }

    template<typename T0,
             typename T1,
             typename T2,
             typename T3,
             typename T4>
    __host__ __device__ R operator()(const T0& t0,
                                     const T1& t1,
                                     const T2& t2,
                                     const T3& t3,
                                     const T4& t4) {
        return m_f(t0,
                   t1,
                   t2,
                   t3,
                   t4,
                   m_k0,
                   m_k1);
    }
    
    template<typename T0,
             typename T1,
             typename T2,
             typename T3,
             typename T4,
             typename T5>
    __host__ __device__ R operator()(const T0& t0,
                                     const T1& t1,
                                     const T2& t2,
                                     const T3& t3,
                                     const T4& t4,
                                     const T5& t5) {
        return m_f(t0,
                   t1,
                   t2,
                   t3,
                   t4,
                   t5,
                   m_k0,
                   m_k1);
    }
    
    template<typename T0,
             typename T1,
             typename T2,
             typename T3,
             typename T4,
             typename T5,
             typename T6>
    __host__ __device__ R operator()(const T0& t0,
                                     const T1& t1,
                                     const T2& t2,
                                     const T3& t3,
                                     const T4& t4,
                                     const T5& t5,
                                     const T6& t6) {
        return m_f(t0,
                   t1,
                   t2,
                   t3,
                   t4,
                   t5,
                   t6,
                   m_k0,
                   m_k1);
    }
    
    template<typename T0,
             typename T1,
             typename T2,
             typename T3,
             typename T4,
             typename T5,
             typename T6,
             typename T7>
    __host__ __device__ R operator()(const T0& t0,
                                     const T1& t1,
                                     const T2& t2,
                                     const T3& t3,
                                     const T4& t4,
                                     const T5& t5,
                                     const T6& t6,
                                     const T7& t7) {
        return m_f(t0,
                   t1,
                   t2,
                   t3,
                   t4,
                   t5,
                   t6,
                   t7,
                   m_k0,
                   m_k1);
    }

    template<typename T0,
             typename T1,
             typename T2,
             typename T3,
             typename T4,
             typename T5,
             typename T6,
             typename T7,
             typename T8>
    __host__ __device__ R operator()(const T0& t0,
                                     const T1& t1,
                                     const T2& t2,
                                     const T3& t3,
                                     const T4& t4,
                                     const T5& t5,
                                     const T6& t6,
                                     const T7& t7,
                                     const T8& t8) {
        return m_f(t0,
                   t1,
                   t2,
                   t3,
                   t4,
                   t5,
                   t6,
                   t7,
                   t8,
                   m_k0,
                   m_k1);
    }
    
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
    __host__ __device__ R operator()(const T0& t0,
                                     const T1& t1,
                                     const T2& t2,
                                     const T3& t3,
                                     const T4& t4,
                                     const T5& t5,
                                     const T6& t6,
                                     const T7& t7,
                                     const T8& t8,
                                     const T9& t9) {
        return m_f(t0,
                   t1,
                   t2,
                   t3,
                   t4,
                   t5,
                   t6,
                   t7,
                   t8,
                   t9,
                   m_k0,
                   m_k1);
    }

};
