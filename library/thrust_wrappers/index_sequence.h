#pragma once
#include "iterator_sequence.h"
#include <thrust/iterator/counting_iterator.h>

struct index_sequence : public iterator_sequence<thrust::counting_iterator<long> >
{
  __host__ __device__ index_sequence(long length) :
  iterator_sequence<thrust::counting_iterator<long> >(thrust::counting_iterator<long>(0), length) {}
};
    
