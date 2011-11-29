#pragma once
#include "node.hpp"
#include "type.hpp"
#include "ctype.hpp"
#include "utility/isinstance.hpp"
#include "utility/markers.hpp"
#include "utility/snippets.hpp"
#include "py_printer.hpp"
#include "rewriter.hpp"

namespace backend {

/*!
  \addtogroup rewriters
  @{
*/

//! A rewrite pass which constructs the entry point wrapper
/*! The entry point is a little special. It needs to operate on
  containers that are held by the broader context of the program,
  whereas the rest of the program operates solely on views.  This pass
  adds a wrapper which operates on containers, derives views, and then
  calls the body of the entry point.
  
*/
class wrap
    : public rewriter
{
private:
    const std::string& m_entry_point;
    bool m_wrapping;
    std::shared_ptr<procedure> m_wrapper;
    std::shared_ptr<procedure> m_wrap_decl;
public:
    //! Constructor
/*! 
  
  \param entry_point Name of the entry point procedure
*/
    wrap(const std::string& entry_point);
    
    using rewriter::operator();
    //! Rewrite rule for \p procedure nodes
    result_type operator()(const procedure &n);
    //! Rewrite rule for \p ret nodes
    result_type operator()(const ret& n);
    //! Rewrite rule for \p suite nodes
    result_type operator()(const suite&n);
    //! Gets the wrapper produced by this rewrite
/*! The wrapper needs to be introspected by other parts of the
  compiler and runtime. This provides access to the generated
  wrapper. It will produce a null pointer if called before the
  wrap pass is executed correctly.
*/
    std::shared_ptr<procedure> p_wrap_decl() const;
};

/*!
  @}
*/

}
