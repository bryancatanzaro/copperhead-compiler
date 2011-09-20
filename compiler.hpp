/*! \file compiler.hpp
 *  \brief The compiler itself.
 */


#pragma once
#include <string>
#include "node.hpp"
#include "functorize.hpp"

namespace backend {
/*! \p compiler contains state and methods for compiling programs.
 */
class compiler {
private:
    std::string m_entry_point;
    template<typename P>
    std::shared_ptr<suite> apply(P& pass, const suite &n) {
        return std::static_pointer_cast<suite>(pass(n));
    }
public:
    /*! \param entry_point The name of the outermost function being compiled
     */
    compiler(const std::string& entry_point)
        : m_entry_point(entry_point) {}
    std::shared_ptr<suite> operator()(const suite &n) {
        functorize functorizer(m_entry_point);
        auto functorized = apply(functorizer, n);
        return functorized;
    }
    const std::string& entry_point() const {
        return m_entry_point;
    }
};

}
