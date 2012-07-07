/*! \file compiler.cpp
 *  \brief The compiler implementation.
 */
#include "compiler.hpp"
#include <typeinfo>

#ifndef TRACE
#define TRACE false
#endif

namespace backend {
compiler::compiler(const std::string& entry_point,
                   const copperhead::system_variant& backend_tag)
    : m_entry_point(entry_point), m_backend_tag(backend_tag), m_registry(){
    std::shared_ptr<library> thrust = get_thrust();
    m_registry.add_library(thrust);
    std::shared_ptr<library> prelude = get_builtins();
    m_registry.add_library(prelude);

}

namespace detail {

template<int N, bool D=false>
struct pipeline_helper {
    typedef std::shared_ptr<const suite> result_type;
    template<class Tuple>
    static inline result_type impl(
        Tuple& t,
        const result_type& i,
        cpp_printer& cp) {
        result_type rewritten =
            std::static_pointer_cast<const suite>(
                boost::apply_visitor(
                    std::get<std::tuple_size<Tuple>::value-N>(t),
                    *i));
        if (D) {
            std::cout << "After " <<
                typeid(std::get<std::tuple_size<Tuple>::value-N>(t)).name() << std::endl;
            boost::apply_visitor(cp, *rewritten);
        }
        return pipeline_helper<N-1, D>::impl(t, rewritten, cp);
    }        
};

template<bool D>
struct pipeline_helper<0, D> {
    typedef std::shared_ptr<const suite> result_type;
    template<typename Tuple>
    static inline result_type impl(
        const Tuple& t,
        const result_type& i,
        const cpp_printer&) {
        return i;
    }        
};

}

template<class Tuple>
static inline std::shared_ptr<const suite> apply(Tuple& t,
                                   const suite& n,
                                   cpp_printer& cp) {
    return detail::
        pipeline_helper<std::tuple_size<Tuple>::value,
                        TRACE>::
        impl(t, n.ptr(), cp);
}


std::shared_ptr<const suite> compiler::operator()(const suite &n) {
    //Defines the compiler pipeline
    //Passes will be processed sequentially, with outputs chained to inputs
    auto passes = std::make_tuple(
        tuple_break(),
        iterizer(),
        phase_analyze(m_entry_point, m_registry),
        type_convert(),
        functorize(m_entry_point, m_registry),
        thrust_rewriter(m_backend_tag),
        dereference(m_entry_point),
        allocate(m_backend_tag, m_entry_point),
        wrap(m_backend_tag, m_entry_point),
        containerize(m_entry_point),
        typedefify(),
        find_includes(m_registry),
        prune());

    cpp_printer cp(m_backend_tag, m_entry_point, m_registry, std::cout);
    auto result = apply(passes, n, cp);
    return result;
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
