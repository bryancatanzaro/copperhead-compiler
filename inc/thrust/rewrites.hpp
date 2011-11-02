#pragma once
#include <map>
#include <string>
#include <sstream>

#include "../node.hpp"
#include "../type.hpp"
#include "../ctype.hpp"
#include "../copier.hpp"
#include "../utility/isinstance.hpp"
#include "../utility/markers.hpp"

#include "../type_printer.hpp"

namespace backend {

class thrust_rewriter
    : public copier {
private:
    static result_type map_rewrite(const bind& n);
    
    static result_type indices_rewrite(const bind& n);

    typedef result_type(*rewrite_fn)(const bind&);
    
    typedef std::map<std::string, rewrite_fn> fn_map;
    
    const fn_map m_lut; 
public:
    thrust_rewriter();
    
    using copier::operator();
    
    result_type operator()(const bind& n);
    
};



}
