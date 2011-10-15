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
#include "thrust/library.hpp"
#include "prelude/decl.hpp"
#include "cuda_printer.hpp"
#include "typedefify.hpp"

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
    compiler(const std::string& entry_point)
        : m_entry_point(entry_point) {
        std::shared_ptr<library> thrust = get_thrust();
        m_registry.add_library(thrust);
        std::shared_ptr<library> prelude = get_builtins();
        m_registry.add_library(prelude);

    }
    std::shared_ptr<suite> operator()(const suite &n) {
        cuda_printer cp(m_entry_point, m_registry, std::cout);
        
        type_convert type_converter;
        auto type_converted = apply(type_converter, n);

        functorize functorizer(m_entry_point, m_registry);
        auto functorized = apply(functorizer, type_converted);

        allocate allocator(m_entry_point);
        auto allocated = apply(allocator, functorized);
        
        typedefify typedefifier;
        auto typedefified = apply(typedefifier, allocated);
        
        wrap wrapper(m_entry_point);
        auto wrapped = apply(wrapper, typedefified);
        
        bpl_wrap bpl_wrapper(m_entry_point, m_registry);
        auto bpl_wrapped = apply(bpl_wrapper, wrapped);
        return bpl_wrapped;
    }
    const std::string& entry_point() const {
        return m_entry_point;
    }
    const registry& reg() const {
        return m_registry;
    }
};

}
