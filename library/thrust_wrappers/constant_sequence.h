#include "iterator_sequence.h"
#include <thrust/iterator/constant_iterator.h>

template<typename T>
struct constant_sequence : public iterator_sequence<thrust::constant_iterator<T> >
{
    __host__ __device__ constant_sequence(const T& val, long length) :
        iterator_sequence<thrust::constant_iterator<T> >(thrust::constant_iterator<T>(val), length) {}
};
