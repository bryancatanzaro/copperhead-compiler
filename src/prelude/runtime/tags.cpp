#include <prelude/runtime/tags.h>
#include <boost/variant.hpp>

#ifdef CUDA_SUPPORT
#include <thrust/system/cuda/memory.h>
#endif

namespace copperhead {

namespace detail {

struct canonicalize_memory_tag
    : boost::static_visitor<copperhead::system_variant> {
    template<typename T>
    copperhead::system_variant operator()(const T&) const {
        typedef typename canonical_memory_tag<T>::tag canonical_T;
        return canonical_T();
    }
};

//Assign a unique identifier to all canonical memory tags
//This is only used for comparison.
template<typename T>
struct tag_number{};

template<>
struct tag_number<cpp_tag> {
    static const int value = 0;
};

#ifdef CUDA_SUPPORT
template<>
struct tag_number<cuda_tag> {
    static const int value = 1;
};
#endif


struct system_variant_less_impl
    : boost::static_visitor<bool> {

    template<typename T, typename U>
    bool operator()(const T&, const U&) const {
        typedef typename canonical_memory_tag<T>::tag canonical_T;
        typedef typename canonical_memory_tag<U>::tag canonical_U;
        return tag_number<canonical_T>::value < tag_number<canonical_U>::value;
    }
};

struct system_variant_equal_impl
    : boost::static_visitor<bool> {

    template<typename T, typename U>
    bool operator()(const T&, const U&) const {
        typedef typename canonical_memory_tag<T>::tag canonical_T;
        typedef typename canonical_memory_tag<U>::tag canonical_U;
        return tag_number<canonical_T>::value == tag_number<canonical_U>::value;
    }
};

}


bool system_variant_less::operator()(const system_variant& x,
                                     const system_variant& y) const {
    return boost::apply_visitor(detail::system_variant_less_impl(), x, y);
}

std::string detail::system_variant_to_string::operator()(const cpp_tag&) const {
    return "cpp_tag";
}

#ifdef OMP_SUPPORT
std::string detail::system_variant_to_string::operator()(const omp_tag&) const {
    return "omp_tag";
}
#endif

#ifdef TBB_SUPPORT
std::string detail::system_variant_to_string::operator()(const tbb_tag&) const {
    return "tbb_tag";
}
#endif
        
#ifdef CUDA_SUPPORT
std::string detail::system_variant_to_string::operator()(const cuda_tag&) const {
    return "cuda_tag";
}
#endif

std::string to_string(const system_variant& x) {
    return boost::apply_visitor(detail::system_variant_to_string(), x);
}

bool system_variant_equal(const system_variant& x,
                          const system_variant& y) {
    return boost::apply_visitor(detail::system_variant_equal_impl(), x, y);
}

system_variant canonical_memory_tag(const system_variant& x) {
    return boost::apply_visitor(detail::canonicalize_memory_tag(), x);
}

}
