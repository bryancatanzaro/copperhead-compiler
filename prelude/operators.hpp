#pragma once
template<typename a>
__host__ __device__ a op_add(a l, a r) {
    return l + r;
}

template<typename a>
__host__ __device__ a op_sub(a l, a r) {
    return l - r;
}

template<typename a>
__host__ __device__ a op_mul(a l, a r) {
    return l * r;
}

template<typename a>
__host__ __device__ a op_div(a l, a r) {
    return l / r;
}

template<typename a>
__host__ __device__ a op_mod(a l, a r) {
    return l % r;
}


template<typename a>
__host__ __device__ a op_lshift(a l, a r) {
    return l << r;
}

template<typename a>
__host__ __device__ a op_rshift(a l, a r) {
    return l >> r;
}

template<typename a>
__host__ __device__ a op_or(a l, a r) {
    return l | r;
}

template<typename a>
__host__ __device__ a op_xor(a l, a r) {
    return l ^ r;
}

template<typename a>
__host__ __device__ a op_and(a l, a r) {
    return l & r;
}

template<typename a>
__host__ __device__ bool cmp_eq(a l, a r) {
    return l == r;
}

template<typename a>
__host__ __device__ bool cmp_ne(a l, a r) {
    return l != r;
}

template<typename a>
__host__ __device__ bool cmp_lt(a l, a r) {
    return l < r;
}

template<typename a>
__host__ __device__ bool cmp_le(a l, a r) {
    return l <= r;
}

template<typename a>
__host__ __device__ bool cmp_gt(a l, a r) {
    return l > r;
}

template<typename a>
__host__ __device__ bool cmp_ge(a l, a r) {
    return l >= r;
}

template<typename a>
__host__ __device__ a op_invert(a i) {
    return ~i;
}

template<typename a>
__host__ __device__ a op_pos(a i) {
    return +i;
}

template<typename a>
__host__ __device__ a op_neg(a i) {
    return -i;
}

template<typename a>
__host__ __device__ a op_not(a i) {
    return !i;
}
