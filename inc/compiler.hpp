/*! \file compiler.hpp
 *  \brief The declaration of the \p compiler class.
 */


#pragma once
#include <string>
#include "node.hpp"
#include "functorize.hpp"
#include "type_convert.hpp"
#include "allocate.hpp"
#include "wrap.hpp"
#include <iostream>
#include "thrust/decl.hpp"
#include "prelude/decl.hpp"
#include "cuda_printer.hpp"
#include "typedefify.hpp"
#include "phase_analyze.hpp"

//XXX We need an interface for libraries to insert compiler passes
//In lieu of such an interface, this is hard coded.
//And must be changed!
#include "thrust/rewrites.hpp"

namespace backend {
//! The compiler used with the Python Copperhead project.
/*! Assembles multiple \ref rewriters to form a compiler for
  the Python Copperhead project.

  This compiler accepts input only in a normalized form:

  - Input is a \ref backend::suite "suite" of \ref backend::procedure
  "procedure" nodes.

  - One of the procedures in the input suite is designated as the
   entry point.

   - The input suite includes the entire program: the
   transitive closure of all procedures called from the entry point,
   excepting procedures supplied by the compiler itself, which are
   tracked by a \ref backend::registry "registry" maintained by the
   compiler.

   - All expressions in the input suite are flattened.  For
   example, \verbatim a = b + c * d \endverbatim must have been
   flattened to \verbatim e0 = c * d
a = b + e0 \endverbatim

   - All closures are made explicit with \ref backend::closure
   "closure" objects.

   - All \ref backend::lambda "lambda" functions are lifted to
   procedures.

   - All nested procedures have been flattened.

   - The \p suite has been typechecked, and AST nodes in the \p suite
   are populated with type information.
 */
class compiler {
private:
    /*! The name of the entry point function.*/
    std::string m_entry_point;
    /*! The registry used by the compiler.*/
    std::shared_ptr<registry> m_registry;
    /*! A helper function to apply a compiler pass.*/
    template<typename P>
    std::shared_ptr<suite> apply(P& pass, const suite &n) {
        return std::static_pointer_cast<suite>(pass(n));
    }
    /*! A helper function to apply a compiler pass.*/
    template<typename P>
    std::shared_ptr<suite> apply(P& pass, const std::shared_ptr<suite> n) {
        return apply(pass, *n);
    }
    /*! After compilation, this holds a \p procedure describing the
     *  wrapped entry point.*/
    std::shared_ptr<procedure> m_wrap_decl;
public:
    //! Constructor.
    /*!\param entry_point The name of the entry point function.
       When the compiler rewrites a \p suite node, it assumes
       there will be a \p procedure node contained in the \p suite
       named \p entry_point, which will serve as the externally
       visible entry point to the entire program being compiled.
     */
    compiler(const std::string& entry_point);
    /*!Compiles a \ref backend::suite suite node.
       \param n The suite node containing the entire program to be
       compiled.
    */
    std::shared_ptr<suite> operator()(const suite &n);
    //! Gets the name of the entry point function
    const std::string& entry_point() const;
    //! Gets the \ref backend::registry "registry" used by the compiler
    const registry& reg() const;
    //! Gets a pointer to the \ref backend::registry "registry" used by the compiler
    std::shared_ptr<registry> p_reg() const;

    //! Gets a \p shared_ptr to the declaration of the wrapper procedure    
    /*! The compiler generates a wrapper procedure, which encapsulates
       input and output type information for the wrapped entry point
       procedure and therefore facilitates calling the wrapped procedure.
     */
    std::shared_ptr<procedure> p_wrap_decl() const;
};

}
