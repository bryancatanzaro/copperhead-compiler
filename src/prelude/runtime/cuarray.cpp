#include <prelude/runtime/cuarray.hpp>
#include <prelude/runtime/type_holder.hpp>

namespace copperhead {

cuarray::cuarray(type_holder* t,
                 size_t o)
    : m_t(t), m_o(o) {}

cuarray::~cuarray() {
    //This is done just to move the destructor to somewhere nvcc can't see
    //because boost::scoped_ptr requires a complete type upon destruction
    //And nvcc can't see the complete type of the type_holder object
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

std::vector<boost::shared_ptr<chunk> >& cuarray::get_chunks(const system_variant& t, bool write) {
    system_variant canonical_tag = canonical_memory_tag(t);
    std::pair<std::vector<boost::shared_ptr<chunk> >, bool>& s = m_d[canonical_tag];
    //Do we need to copy?
    if (!s.second) {
        //Find a valid representation
        std::pair<std::vector<boost::shared_ptr<chunk> >, bool> x;
        x.second = false;
        for(typename data_map::iterator i = m_d.begin();
            (x.second == false) && (i != m_d.end());
            i++) {
            x = i->second;
        }
        assert(x.second == true);
        //Copy from valid representation
        for(std::vector<boost::shared_ptr<chunk> >::iterator i = s.first.begin(),
                j = x.first.begin();
            i != s.first.end();
            i++, j++) {
            (*i)->copy_from(**j);
        }
        s.second = true;
    }
    //Do we need to invalidate?
    if (write) {
        for(typename data_map::iterator i = m_d.begin();
            i != m_d.end();
            i++) {
            i->second.second = system_variant_equal(i->first, canonical_tag);
        }
    }
    return s.first;
}

bool cuarray::clean(const system_variant& t) {
    return m_d[t].second;
}

}
