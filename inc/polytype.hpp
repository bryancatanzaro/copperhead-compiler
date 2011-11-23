#pragma once
#include "type.hpp"
#include "monotype.hpp"

namespace backend {

class polytype_t :
        public type_t {
private:
    std::vector<std::shared_ptr<monotype_t> > m_vars;
    std::shared_ptr<monotype_t> m_monotype;
public:
    polytype_t(std::vector<std::shared_ptr<monotype_t> >&& vars,
               std::shared_ptr<monotype_t> monotype);
    
    typedef decltype(boost::make_indirect_iterator(m_vars.cbegin())) const_iterator;
    
    const monotype_t& monotype() const;

    std::shared_ptr<monotype_t> p_monotype() const;

    const_iterator begin() const;

    const_iterator end() const;
};

}
