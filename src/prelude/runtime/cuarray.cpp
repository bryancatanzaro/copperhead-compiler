#include <prelude/runtime/cuarray.hpp>
#include <prelude/runtime/cu_and_c_types.hpp>

namespace copperhead {

cuarray::cuarray(cu_and_c_types* t,
                 size_t o)
    : m_t(t), m_o(o) {}

cuarray::~cuarray() {
    //This is done just to move the destructor to somewhere nvcc can't see
    //because boost::scoped_ptr requires a complete type upon destruction
    //And nvcc can't see the complete type of the cu_and_c_types object
}

void cuarray::push_back_length(size_t l) {
    m_l.push_back(l);
}

void cuarray::add_chunk(boost::shared_ptr<chunk> c,
                        const bool& v) {
    system_variant t = c->tag();
    if (m_d.find(t) == m_d.end()) {
        std::vector<boost::shared_ptr<chunk> > new_vector;
        new_vector.push_back(c);
        std::pair<std::vector<boost::shared_ptr<chunk> >, bool> new_pair =
            std::make_pair(std::move(new_vector), v);
        m_d[t] = std::move(new_pair);
    } else {
        std::pair<std::vector<boost::shared_ptr<chunk> >, bool>& e = m_d[t];
        e.second = v;
        e.first.push_back(c);
    }
}

size_t cuarray::size() const {
    size_t s = m_l[0];
    if (m_l.size() > 1) {
        s--;
    }
    return s;
}

std::vector<boost::shared_ptr<chunk> >& cuarray::get_chunks(const system_variant& t) {
    return m_d[t].first;
}

bool cuarray::clean(const system_variant& t) {
    return m_d[t].second;
}

}
