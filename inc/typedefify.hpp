#pragma once
#include "node.hpp"
#include "type.hpp"
#include "ctype.hpp"
#include "utility/isinstance.hpp"
#include "utility/markers.hpp"
#include "utility/snippets.hpp"
#include "copier.hpp"

namespace backend {


class typedefify
    : public copier
{
private:
    std::shared_ptr<statement> m_typedef;
public:
    typedefify();
    
    using copier::operator();
    
    result_type operator()(const suite &n);
    
    result_type operator()(const bind &n);
    
    result_type operator()(const procedure &n);
        
};

}
