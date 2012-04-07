/*! \file compiler.cpp
 *  \brief The compiler implementation.
 */
#include "compiler.hpp"

namespace backend {
compiler::compiler(const std::string& entry_point,
                   const copperhead::system_variant& backend_tag)
    : m_entry_point(entry_point), m_backend_tag(backend_tag), m_registry(){
    std::shared_ptr<library> thrust = get_thrust();
    m_registry.add_library(thrust);
    std::shared_ptr<library> prelude = get_builtins();
    m_registry.add_library(prelude);

}
std::shared_ptr<const suite> compiler::operator()(const suite &n) {
    cpp_printer cp(m_backend_tag, m_entry_point, m_registry, std::cout);

    phase_analyze phase_analyzer(m_entry_point, m_registry);
    auto phase_analyzed = apply(phase_analyzer, n);

    
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
    thrust_rewriter thrustizer(m_backend_tag);
    auto thrust_rewritten = apply(thrustizer, functorized);
#ifdef TRACE
    std::cout << "Thrust rewritten" << std::endl;
    boost::apply_visitor(cp, *thrust_rewritten);
#endif
    allocate allocator(m_backend_tag, m_entry_point);
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
    wrap wrapper(m_backend_tag, m_entry_point);
    auto wrapped = apply(wrapper, typedefified);
#ifdef TRACE
    std::cout << "Wrapped" << std::endl;
    boost::apply_visitor(cp, *wrapped);
#endif
    find_includes include_finder(m_registry);
    auto included = apply(include_finder, wrapped);
#ifdef TRACE
    std::cout << "Included" << std::endl;
    boost::apply_visitor(cp, *included);
#endif
    return included;
}
const std::string& compiler::entry_point() const {
    return m_entry_point;
}
const registry& compiler::reg() const {
    return m_registry;
}

const copperhead::system_variant& compiler::target() const {
    return m_backend_tag;
}

}
