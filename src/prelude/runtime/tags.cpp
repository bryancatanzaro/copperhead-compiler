#include <prelude/runtime/tags.h>

using namespace copperhead;
bool system_variant_less::operator()(const system_variant& x,
                                     const system_variant& y) const {
    return x.which() < y.which();
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
    return x.which() == y.which();
}
