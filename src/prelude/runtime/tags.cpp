#include <prelude/runtime/tags.h>
#include <boost/type_traits.hpp>
#include <boost/variant.hpp>

namespace copperhead {

namespace detail {

template<typename T>
struct tag_number{};

template<>
struct tag_number<cpp_tag> {
    static const int value = 0;
};

template<>
struct tag_number<omp_tag> {
    static const int value = 0;
};

#ifdef CUDA_SUPPORT
template<>
struct tag_number<cuda_tag> {
    static const int value = 0;
};
#endif


struct system_variant_less_impl
    : boost::static_visitor<bool> {

    template<typename T, typename U>
    typename boost::enable_if<
        boost::is_convertible<T, U>,
        bool>::type
    operator()(const T&, const U&) const {
        return false;
    }

    template<typename T, typename U>
    typename boost::disable_if<
        boost::is_convertible<T, U>,
        bool>::type
    operator()(const T&, const U&) const {
        return tag_number<T>::value < tag_number<U>::value;
    }  
};

struct system_variant_equal_impl
    : boost::static_visitor<bool> {

    template<typename T, typename U>
    typename boost::enable_if<
        boost::is_convertible<T, U>,
        bool>::type
    operator()(const T&, const U&) const {
        return true;
    }

    template<typename T, typename U>
    typename boost::disable_if<
        boost::is_convertible<T, U>,
        bool>::type
    operator()(const T& t, const U& u) const {
        return false;
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

std::string detail::system_variant_to_string::operator()(const omp_tag&) const {
    return "omp_tag";
}

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

}
