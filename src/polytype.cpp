#include "polytype.hpp"

namespace backend {

polytype_t::polytype_t(std::vector<std::shared_ptr<monotype_t> >&& vars,
                       std::shared_ptr<monotype_t> monotype)
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
    

}
