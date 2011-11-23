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
/*! \p compiler contains state and methods for compiling programs.
 */
class compiler {
private:
    /*! The name of the entry point function.*/
    std::string m_entry_point;
    /*! The registry used by the compiler.*/
    registry m_registry;
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
    /*!\param entry_point The name of the entry point function.
       When the compiler rewrites a \p suite node, it assumes
       there will be a \p procedure node contained in the \p suite
       named \p entry_point, which will serve as the externally
       visible entry point to the entire program being compiled.
     */
    compiler(const std::string& entry_point);
    /*!Rewrite for \p suite nodes .
       \param n The suite node containing the entire program to be
       compiled.
    */
    std::shared_ptr<suite> operator()(const suite &n);
    /*! Retrieve the name of the entry point function.
     */
    const std::string& entry_point() const;
    /*! Retrieve the registry used by this compiler.
     */
    const registry& reg() const;
    /*! Retrieve a pointer to the declaration of the
       wrapper procedure generated by the compiler.
       This encapsulates input and output
       type information for the wrapped entry point procedure,
       which facilitates calling the wrapped procedure.
     */
    std::shared_ptr<procedure> p_wrap_decl() const;
};

}
