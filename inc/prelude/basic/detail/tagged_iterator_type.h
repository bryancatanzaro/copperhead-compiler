#include <thrust/iterator/detail/tagged_iterator.h>

#pragma once
namespace copperhead {
namespace detail {

template<typename I, typename Tag>
struct tagged_iterator_type {
    typedef typename thrust::detail::tagged_iterator<I, Tag> type;
};
}
}
