#pragma once
#include "operators.hpp"

struct fn_op_add {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_add(l, r);
    }
};

struct fn_op_sub {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_sub(l, r);
    }
};

struct fn_op_mul {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_mul(l, r);
    }
};

struct fn_op_div {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_div(l, r);
    }
};

struct fn_op_mod {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_mod(l, r);
    }
};

struct fn_op_lshift {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_lshift(l, r);
    }
};

struct fn_op_rshift {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_rshift(l, r);
    }
};

struct fn_op_or {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_or(l, r);
    }
};

struct fn_op_xor {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_xor(l, r);
    }
};

struct fn_op_and {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_and(l, r);
    }
};

struct fn_cmp_eq {
    template<typename a>
    __host__ __device__ bool operator()(const a &l, const a &r) {
        return cmp_eq(l, r);
    }
};

struct fn_cmp_ne {
    template<typename a>
    __host__ __device__ bool operator()(const a &l, a r) {
        return cmp_ne(l, r);
    }
};

struct fn_cmp_lt {
    template<typename a>
    __host__ __device__ bool operator()(const a &l, const a &r) {
        return cmp_lt(l, r);
    }
};

struct fn_cmp_le {
    template<typename a>
    __host__ __device__ bool operator()(a l, const a &r) {
        return cmp_le(l, r);
    }
};

struct fn_cmp_gt {
    template<typename a>
    __host__ __device__ bool operator()(const a &l, a r) {
        return cmp_gt(l, r);
    }
};

struct fn_cmp_ge {
    template<typename a>
    __host__ __device__ bool operator()(const a &l, const a &r) {
        return cmp_ge(l, r);
    }
};

struct fn_op_invert {
    template<typename a>
    __host__ __device__ a operator()(const a &i) {
        return op_invert(i);
    }
};

struct fn_op_pos {
    template<typename a>
    __host__ __device__ a operator()(const a &i) {
        return op_pos(i);
    }
};

struct fn_op_neg {
    template<typename a>
    __host__ __device__ a operator()(const a &i) {
        return op_neg(i);
    }
};

struct fn_op_not {
    template<typename a>
    __host__ __device__ a operator()(const a &i) {
        return op_not(i);
    }
};
