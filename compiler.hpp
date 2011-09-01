/*! \file compiler.hpp
 *  \brief The compiler itself.
 */


#pragma once
#include <string>

/*! \p compiler contains state and methods for compiling programs.
 */
class compiler {
private:
    std::string m_entry_point;
public:
    /*! \param entry_point The name of the outermost function being compiled
     */
    compiler(const std::string& entry_point)
        : m_entry_point(entry_point) {}
};
