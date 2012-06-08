/*! \file compiler.cpp
 *  \brief The compiler implementation.
 */
#include "compiler.hpp"
#include <typeinfo>

#define TRACE false //true

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
    template<class... Args>
    static result_type impl(
        std::tuple<Args...>& t,
        const result_type& i,
        cpp_printer& cp) {
        result_type rewritten =
            std::static_pointer_cast<const suite>(
                boost::apply_visitor(
                    std::get<sizeof...(Args)-N>(t),
                    *i));
        if (D) {
            std::cout << "After " << typeid(std::get<sizeof...(Args)-N>(t)).name() << std::endl;
            boost::apply_visitor(cp, *rewritten);
        }
        return pipeline_helper<N-1, D>::impl(t, rewritten, cp);
    }        
};

template<bool D>
struct pipeline_helper<0, D> {
    typedef std::shared_ptr<const suite> result_type;
    template<class... Args>
    static result_type impl(
        std::tuple<Args...> const& t,
        const result_type& i,
        cpp_printer&) {
        return i;
    }        
};

}

template<class... Args>
std::shared_ptr<const suite> apply(std::tuple<Args...>& t,
                                   const suite& n,
                                   cpp_printer& cp) {
    return detail::pipeline_helper<sizeof...(Args), TRACE>::impl(t, n.ptr(), cp);
}


std::shared_ptr<const suite> compiler::operator()(const suite &n) {
    auto passes = std::make_tuple(
        tuple_break(),
        phase_analyze(m_entry_point, m_registry),
        type_convert(),
        functorize(m_entry_point, m_registry),
        thrust_rewriter(m_backend_tag),
        dereference(m_entry_point),
        allocate(m_backend_tag, m_entry_point),
        wrap(m_backend_tag, m_entry_point),
        containerize(m_entry_point),
        typedefify(),
        find_includes(m_registry));

    cpp_printer cp(m_backend_tag, m_entry_point, m_registry, std::cout);
    return apply(passes, n, cp);
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
