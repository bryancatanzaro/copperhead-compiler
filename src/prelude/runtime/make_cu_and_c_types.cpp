#include <prelude/runtime/make_cu_and_c_types.hpp>
#include <prelude/runtime/cu_and_c_types.hpp>
#include <monotype.hpp>

namespace copperhead {

cu_and_c_types* make_type_holder_helper(std::shared_ptr<const backend::type_t> t) {
    cu_and_c_types* holder = new cu_and_c_types();
    holder->m_t = std::make_shared<const backend::sequence_t>(t);
    return holder;
}

cu_and_c_types* make_type_holder(int) {
    return make_type_holder_helper(backend::int32_mt);
}

cu_and_c_types* make_type_holder(long) {
    return make_type_holder_helper(backend::int64_mt);
}

cu_and_c_types* make_type_holder(float) {
    return make_type_holder_helper(backend::float32_mt);
}

cu_and_c_types* make_type_holder(double) {
    return make_type_holder_helper(backend::float64_mt);
}

cu_and_c_types* make_type_holder(bool) {
    return make_type_holder_helper(backend::bool_mt);
}

}

