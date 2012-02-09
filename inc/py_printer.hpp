/*
 *   Copyright 2012      NVIDIA Corporation
 * 
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 * 
 *       http://www.apache.org/licenses/LICENSE-2.0
 * 
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 * 
 */
#pragma once
#include <cassert>
#include <sstream>

#include "node.hpp"
#include "expression.hpp"
#include "statement.hpp"
#include "cppnode.hpp"

namespace backend
{
namespace detail {
template<typename Visitor,
         typename Iterable>
inline void list(Visitor& v,
                 const Iterable &l) {
    for(auto i = l.begin();
        i != l.end();
        i++) {
        boost::apply_visitor(v, *i);
        if (std::next(i) != l.end())
            v.sep();
    }
}
}

class py_printer
    : public no_op_visitor<>
{
public:
    py_printer(std::ostream &os);
    
    void operator()(const literal &n);

    void operator()(const tuple &n);

    void operator()(const apply &n);
    
    void operator()(const lambda &n);
    
    void operator()(const closure &n);
    
    void operator()(const conditional &n);
    
    void operator()(const ret &n);
    
    void operator()(const bind &n);
    
    void operator()(const call & n);
    
    void operator()(const procedure &n);
    
    void operator()(const suite &n);

    void operator()(const structure &n);
    
    void operator()(const include &n);

    void operator()(const typedefn &n);

    void operator()(const namespace_block &n);
    
    void operator()(const std::string &s);
    
    template<typename T>
        void operator()(const std::vector<T> &v) const {
        detail::list(this, v);
    }

    void sep(void) const;
    
    void open(void) const;
    
    void close(void) const;
    
    protected:

    std::ostream &m_os;
    int indent_level;
    std::string indent_atom;
    std::string current_indent;
    void indent(int amount=1);
    
    void dedent();
    
    void indentation();
};


}
