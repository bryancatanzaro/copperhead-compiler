#pragma once
#include "operators.hpp"

struct op_add_fn {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_add(l, r);
    }
};

struct op_sub_fn {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_sub(l, r);
    }
};

struct op_mul_fn {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_mul(l, r);
    }
};

struct op_div_fn {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_div(l, r);
    }
};

struct op_mod_fn {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_mod(l, r);
    }
};

struct op_lshift_fn {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_lshift(l, r);
    }
};

struct op_rshift_fn {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_rshift(l, r);
    }
};

struct op_or_fn {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_or(l, r);
    }
};

struct op_xor_fn {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_xor(l, r);
    }
};

struct op_and_fn {
    template<typename a>
    __host__ __device__ a operator()(const a &l, const a &r) {
        return op_and(l, r);
    }
};

struct cmp_eq_fn {
    template<typename a>
    __host__ __device__ bool operator()(const a &l, const a &r) {
        return cmp_eq(l, r);
    }
};

struct cmp_ne_fn {
    template<typename a>
    __host__ __device__ bool operator()(const a &l, a r) {
        return cmp_ne(l, r);
    }
};

struct cmp_lt_fn {
    template<typename a>
    __host__ __device__ bool operator()(const a &l, const a &r) {
        return cmp_lt(l, r);
    }
};

struct cmp_le_fn {
    template<typename a>
    __host__ __device__ bool operator()(a l, const a &r) {
        return cmp_le(l, r);
    }
};

struct cmp_gt_fn {
    template<typename a>
    __host__ __device__ bool operator()(const a &l, a r) {
        return cmp_gt(l, r);
    }
};

struct cmp_ge_fn {
    template<typename a>
    __host__ __device__ bool operator()(const a &l, const a &r) {
        return cmp_ge(l, r);
    }
};

struct op_invert_fn {
    template<typename a>
    __host__ __device__ a operator()(const a &i) {
        return op_invert(i);
    }
};

struct op_pos_fn {
    template<typename a>
    __host__ __device__ a operator()(const a &i) {
        return op_pos(i);
    }
};

struct op_neg_fn {
    template<typename a>
    __host__ __device__ a operator()(const a &i) {
        return op_neg(i);
    }
};

struct op_not_fn {
    template<typename a>
    __host__ __device__ a operator()(const a &i) {
        return op_not(i);
    }
};
