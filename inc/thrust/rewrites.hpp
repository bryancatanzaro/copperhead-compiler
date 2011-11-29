#pragma once
#include <map>
#include <string>
#include <sstream>

#include "../node.hpp"
#include "../type.hpp"
#include "../ctype.hpp"
#include "../rewriter.hpp"
#include "../utility/isinstance.hpp"
#include "../utility/markers.hpp"

#include "../type_printer.hpp"

namespace backend {

/*! 
  \addtogroup rewriters
  @{
 */

//! Rewriter for Thrust calls
/*! This rewriter performs all rewrites specific to the Thrust library.
  For example, it makes mapn calls produce a transformed_sequence<>
  C++ implementation type, or indices produce a counting_sequence
  C++ implementation type.
*/
class thrust_rewriter
    : public rewriter {
private:
    static result_type map_rewrite(const bind& n);
    
    static result_type indices_rewrite(const bind& n);

    typedef result_type(*rewrite_fn)(const bind&);
    
    typedef std::map<std::string, rewrite_fn> fn_map;
    
    const fn_map m_lut; 
public:
    //! Constructor
    thrust_rewriter();
    
    using rewriter::operator();
    
    result_type operator()(const bind& n);
    
};

/*! 
  @}
 */


}
