#pragma once
template<typename a>
__host__ __device__ a op_add(const a &l, const a &r) {
    return l + r;
}

template<typename a>
__host__ __device__ a op_sub(const a &l, const a &r) {
    return l - r;
}

template<typename a>
__host__ __device__ a op_mul(const a &l, const a &r) {
    return l * r;
}

template<typename a>
__host__ __device__ a op_div(const a &l, const a &r) {
    return l / r;
}

template<typename a>
__host__ __device__ a op_mod(const a &l, const a &r) {
    return l % r;
}


template<typename a>
__host__ __device__ a op_lshift(const a &l, const a &r) {
    return l << r;
}

template<typename a>
__host__ __device__ a op_rshift(const a &l, const a &r) {
    return l >> r;
}

template<typename a>
__host__ __device__ a op_or(const a &l, const a &r) {
    return l | r;
}

template<typename a>
__host__ __device__ a op_xor(const a &l, const a &r) {
    return l ^ r;
}

template<typename a>
__host__ __device__ a op_and(const a &l, const a &r) {
    return l & r;
}

template<typename a>
__host__ __device__ bool cmp_eq(const a &l, const a &r) {
    return l == r;
}

template<typename a>
__host__ __device__ bool cmp_ne(const a &l, const a &r) {
    return l != r;
}

template<typename a>
__host__ __device__ bool cmp_lt(const a &l, const a &r) {
    return l < r;
}

template<typename a>
__host__ __device__ bool cmp_le(const a &l, const a &r) {
    return l <= r;
}

template<typename a>
__host__ __device__ bool cmp_gt(const a &l, const a &r) {
    return l > r;
}

template<typename a>
__host__ __device__ bool cmp_ge(const a &l, const a &r) {
    return l >= r;
}

template<typename a>
__host__ __device__ a op_invert(const a &i) {
    return ~i;
}

template<typename a>
__host__ __device__ a op_pos(const a &i) {
    return +i;
}

template<typename a>
__host__ __device__ a op_neg(const a &i) {
    return -i;
}

template<typename a>
__host__ __device__ a op_not(const a &i) {
    return !i;
}
