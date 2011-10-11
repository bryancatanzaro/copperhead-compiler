#pragma once
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <thrust/detail/tuple_meta_transform.h>

namespace detail {

template<typename Seq>
struct extract_value {
    typedef typename Seq::value_type type;
};

template<typename Seq>
struct extract_reference {
    typedef typename Seq::value_type& type;
};
template<typename Seq>
struct extract_iterator {
    typedef typename Seq::iterator_type type;
};
struct index_sequence {
    int m_index;
    index_sequence(int index) : m_index(index) {}
    //XXX Fix references to zipped sequences!
    template<typename Seq>
    typename Seq::value_type operator()(Seq& in) {
        return in[m_index];
    }
};
struct extract_begin {
    template<typename Seq>
    typename Seq::iterator_type operator()(const Seq& in) {
        return in.begin();
    }
};
struct extract_end {
    template<typename Seq>
    typename Seq::iterator_type operator()(const Seq& in) {
        return in.end();
    }
};
}

template<typename S>
struct zipped_sequence {
    S m_seqs;
    typedef typename thrust::detail::tuple_meta_transform<
        S, detail::extract_value>::type value_type;
    //XXX Fix references to zipped sequences!
    typedef typename thrust::detail::tuple_meta_transform<
        S, detail::extract_value>::type reference_type;
    typedef typename thrust::zip_iterator<
        typename thrust::detail::tuple_meta_transform<
            S, detail::extract_iterator>::type > iterator_type;
    zipped_sequence(S seqs) : m_seqs(seqs) {}
    reference_type operator[](int index) {
        //XXX Fix references to zipped sequences!
        return thrust::detail::tuple_host_transform
            <detail::extract_value, S, detail::index_sequence>(
                m_seqs,
                detail::index_sequence(index));
    }
    iterator_type begin() const {
        return thrust::detail::tuple_host_transform
            <detail::extract_iterator, S, detail::extract_begin>(
                m_seqs,
                detail::extract_begin());
    }
    iterator_type end() const {
        return thrust::detail::tuple_host_transform
            <detail::extract_iterator, S, detail::extract_end>(
                m_seqs,
                detail::extract_end());
    }
};
