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
#include <sstream>
#include "node.hpp"
#include "expression.hpp"
#include "statement.hpp"
#include "cppnode.hpp"

namespace backend
{

class repr_printer
    : public no_op_visitor<>
{
public:
    repr_printer(std::ostream &os);

    using backend::no_op_visitor<>::operator();

    void operator()(const name &n);
    
    void operator()(const literal &n);
    
    void operator()(const tuple &n);

    void operator()(const bind &n);

    void operator()(const apply &n);
    
    void operator()(const lambda &n);
    
    void operator()(const closure &n);
    
    void operator()(const ret &n);
    
    void operator()(const procedure &n);
    
    void operator()(const suite &n);
    
    void operator()(const include &n);
    
    void operator()(const typedefn &n);
    
    void operator()(const conditional &n);
    
    void operator()(const std::string &s);
    
    template<typename T>
        void operator()(const std::vector<T> &v) {
        list(v);
    }
private:
    void sep(void) const;
    
    void open(void) const;
    
    void close(void) const;
    
    template<typename Value>
    void print_name(const std::string &n,
                    const Value &val) {
        m_os << n;
        open();
        (*this)(val);
        close();
    }
    template<typename Iterable>
    void list(const Iterable &l,
              const std::string sep = ", ") {
        for(auto i = l.begin();
            i != l.end();
            i++) {
            boost::apply_visitor(*this, *i);
            if (std::next(i) != l.end())
                m_os << sep;
        }
    }
    
    std::ostream &m_os;
};

}

