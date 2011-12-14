#include "iterator_sequence.h"
#include <thrust/iterator/counting_iterator.h>

struct counting_sequence : public iterator_sequence<thrust::counting_iterator<long> >
{
    __host__ __device__ counting_sequence(long begin, long end) :
        iterator_sequence<thrust::counting_iterator<long> >(thrust::counting_iterator<long>(begin), begin-end) {}
};
