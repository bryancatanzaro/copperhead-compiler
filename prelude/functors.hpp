#pragma once
#include "operators.hpp"

template<typename a>
struct fn_op_add {
    typedef a result_type;
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_add(l, r);
    }
};

template<typename a>
struct fn_op_sub {
    typedef a result_type;
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_sub(l, r);
    }
};

template<typename a>
struct fn_op_mul {
    typedef a result_type;
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_mul(l, r);
    }
};

template<typename a>
struct fn_op_div {
    typedef a result_type;
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_div(l, r);
    }
};

template<typename a>
struct fn_op_mod {
    typedef a result_type;
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_mod(l, r);
    }
};

template<typename a>
struct fn_op_lshift {
    typedef a result_type;
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_lshift(l, r);
    }
};

template<typename a>
struct fn_op_rshift {
    typedef a result_type;
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_rshift(l, r);
    }
};

template<typename a>
struct fn_op_or {
    typedef a result_type;
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_or(l, r);
    }
};

template<typename a>
struct fn_op_xor {
    typedef a result_type;
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_xor(l, r);
    }
};

template<typename a>
struct fn_op_and {
    typedef a result_type;
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_and(l, r);
    }
};

template<typename a>
struct fn_cmp_eq {
    typedef bool result_type;
    __host__ __device__ bool operator()(const a &l, const a &r) {
        return cmp_eq(l, r);
    }
};

template<typename a>
struct fn_cmp_ne {
    typedef bool result_type;
    __host__ __device__ bool operator()(const a &l, a r) {
        return cmp_ne(l, r);
    }
};

template<typename a>
struct fn_cmp_lt {
    typedef bool result_type;
    __host__ __device__ bool operator()(const a &l, const a &r) {
        return cmp_lt(l, r);
    }
};

template<typename a>
struct fn_cmp_le {
    typedef bool result_type;
    __host__ __device__ bool operator()(a l, const a &r) {
        return cmp_le(l, r);
    }
};

template<typename a>
struct fn_cmp_gt {
    typedef bool result_type;
    __host__ __device__ bool operator()(const a &l, a r) {
        return cmp_gt(l, r);
    }
};

template<typename a>
struct fn_cmp_ge {
    typedef bool result_type;
    __host__ __device__ bool operator()(const a &l, const a &r) {
        return cmp_ge(l, r);
    }
};

template<typename a>
struct fn_op_invert {
    typedef a result_type;
    __host__ __device__ a operator()(const a &i) {
        return op_invert(i);
    }
};

template<typename a>
struct fn_op_pos {
    typedef a result_type;
    __host__ __device__ a operator()(const a &i) {
        return op_pos(i);
    }
};

template<typename a>
struct fn_op_neg {
    typedef a result_type;
    __host__ __device__ a operator()(const a &i) {
        return op_neg(i);
    }
};

template<typename a>
struct fn_op_not {
    typedef a result_type;
    __host__ __device__ a operator()(const a &i) {
        return op_not(i);
    }
};
