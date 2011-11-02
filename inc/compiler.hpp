/*! \file compiler.hpp
 *  \brief The compiler itself.
 */


#pragma once
#include <string>
#include "node.hpp"
#include "functorize.hpp"
#include "type_convert.hpp"
#include "allocate.hpp"
#include "wrap.hpp"
#include "bpl_wrap.hpp"
#include <iostream>
#include "thrust/decl.hpp"
#include "prelude/decl.hpp"
#include "cuda_printer.hpp"
#include "typedefify.hpp"

//XXX We need an interface for libraries to insert compiler passes
//In lieu of such an interface, this is hard coded.
//And must be changed!
#include "thrust/rewrites.hpp"

namespace backend {
/*! \p compiler contains state and methods for compiling programs.
 */
class compiler {
private:
    std::string m_entry_point;
    registry m_registry;
    template<typename P>
    std::shared_ptr<suite> apply(P& pass, const suite &n) {
        return std::static_pointer_cast<suite>(pass(n));
    }
    template<typename P>
    std::shared_ptr<suite> apply(P& pass, const std::shared_ptr<suite> n) {
        return apply(pass, *n);
    }
public:
    /*! \param entry_point The name of the outermost function being compiled
     */
    compiler(const std::string& entry_point);
    std::shared_ptr<suite> operator()(const suite &n);
    const std::string& entry_point() const;
    const registry& reg() const;
};

}
