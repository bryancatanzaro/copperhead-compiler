/*! \file compiler.cpp
 *  \brief The compiler implementation.
 */
#include "compiler.hpp"

namespace backend {
compiler::compiler(const std::string& entry_point)
    : m_entry_point(entry_point) {
    std::shared_ptr<library> thrust = get_thrust();
    m_registry.add_library(thrust);
    std::shared_ptr<library> prelude = get_builtins();
    m_registry.add_library(prelude);

}
std::shared_ptr<suite> compiler::operator()(const suite &n) {
    cuda_printer cp(m_entry_point, m_registry, std::cout);

    phase_analyze phase_analyzer(m_entry_point, m_registry);
    auto phase_analyzed = apply(phase_analyzer, n);
#ifdef TRACE
    std::cout << "Phase analyzed" << std::endl;
    boost::apply_visitor(cp, *phase_analyzed);
#endif
    
    type_convert type_converter;
    auto type_converted = apply(type_converter, phase_analyzed);
#ifdef TRACE
    std::cout << "Type converted" << std::endl;
    boost::apply_visitor(cp, *type_converted);
#endif
    functorize functorizer(m_entry_point, m_registry);
    auto functorized = apply(functorizer, type_converted);
#ifdef TRACE
    std::cout << "Functorized" << std::endl;
    boost::apply_visitor(cp, *functorized);
#endif
    thrust_rewriter thrustizer;
    auto thrust_rewritten = apply(thrustizer, functorized);
#ifdef TRACE
    std::cout << "Thrust rewritten" << std::endl;
    boost::apply_visitor(cp, *thrust_rewritten);
#endif
    allocate allocator(m_entry_point);
    auto allocated = apply(allocator, thrust_rewritten);
#ifdef TRACE
    std::cout << "Allocated" << std::endl;
    boost::apply_visitor(cp, *allocated);
#endif
    typedefify typedefifier;
    auto typedefified = apply(typedefifier, allocated);
#ifdef TRACE
    std::cout << "Typedefified" << std::endl;
    boost::apply_visitor(cp, *typedefified);
#endif        
    wrap wrapper(m_entry_point);
    auto wrapped = apply(wrapper, typedefified);
    m_wrap_decl = wrapper.p_wrap_decl();
#ifdef TRACE
    std::cout << "Wrapped" << std::endl;
    boost::apply_visitor(cp, *wrapped);
#endif
    return wrapped;
}
const std::string& compiler::entry_point() const {
    return m_entry_point;
}
const registry& compiler::reg() const {
    return m_registry;
}
std::shared_ptr<procedure> compiler::p_wrap_decl() const {
    return m_wrap_decl;
}

}
