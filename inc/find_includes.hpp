#pragma once
#include "node.hpp"
#include "type.hpp"
#include "ctype.hpp"
#include "cppnode.hpp"
#include "rewriter.hpp"
#include "import/library.hpp"
#include <set>
#include <map>

namespace backend {

/*!
  \addtogroup rewriters
  @{
*/

//! A rewrite pass which finds include files needed by program
/*! Because the compiler assembles code fragments from various places,
    we need to examine the generated code and determine what include
    statements to add to make the program complete.
  
*/
class find_includes
    : public rewriter
{
private:
    const registry& m_reg;
    std::set<std::string> m_includes;
    bool m_outer;
public:
    //! Constructor
    /*! \param reg The registry maintained by the compiler
     */
    find_includes(const registry& reg);
    using rewriter::operator();
    result_type operator()(const suite& n);
    result_type operator()(const apply& n);
};

}
