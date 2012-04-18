#include "polytype.hpp"

using std::vector;
using std::shared_ptr;
using std::static_pointer_cast;

namespace backend {

polytype_t::polytype_t(vector<shared_ptr<const monotype_t> >&& vars,
                       shared_ptr<const monotype_t> monotype)
    : type_t(*this), m_vars(std::move(vars)), m_monotype(monotype) {}

    
const monotype_t& polytype_t::monotype() const {
    return *m_monotype;
}

polytype_t::const_iterator polytype_t::begin() const {
    return boost::make_indirect_iterator(m_vars.cbegin());
}

polytype_t::const_iterator polytype_t::end() const {
    return boost::make_indirect_iterator(m_vars.cend());
}
    
shared_ptr<const polytype_t> polytype_t::ptr() const {
    return static_pointer_cast<const polytype_t>(this->shared_from_this());
}

}
