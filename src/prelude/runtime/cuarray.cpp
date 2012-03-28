#include <prelude/runtime/cuarray.hpp>
#include <prelude/runtime/cu_and_c_types.hpp>

copperhead::cuarray::cuarray(copperhead::data_map&& d,
                    std::vector<size_t>&& l,
                    cu_and_c_types* t,
                    size_t o)
    : m_d(d), m_l(l), m_t(t), m_o(o) {}

size_t copperhead::cuarray::size() const {
    size_t s = m_l[0];
    if (m_l.size() > 1) {
        s--;
    }
    return s;
}
