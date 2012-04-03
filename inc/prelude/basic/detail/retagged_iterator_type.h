#include <thrust/iterator/detail/tagged_iterator.h>
#include <thrust/detail/pointer.h>

#pragma once
namespace copperhead {
namespace detail {

template<typename I, typename Tag>
struct retagged_iterator_type {
    typedef typename thrust::detail::tagged_iterator<I, Tag> type;
};

//Avoid nested tagged_iterators
template<typename OI, typename OT, typename Tag>
struct retagged_iterator_type<thrust::detail::tagged_iterator<OI, OT>, Tag> {
    typedef typename thrust::detail::tagged_iterator<OI, Tag> type;
};

//Specialize for pointers
template<typename T, typename OT, typename Tag>
struct retagged_iterator_type<thrust::pointer<T, OT>, Tag> {
    typedef typename thrust::pointer<T, Tag> type;
};

//Specialize for raw pointers
template<typename T, typename Tag>
struct retagged_iterator_type<T*, Tag> {
    typedef typename thrust::pointer<T, Tag> type;
};

}
}
