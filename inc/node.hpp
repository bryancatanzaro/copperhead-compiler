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

#include <boost/variant.hpp>
#include <functional>
#include <vector>
#include <memory>
#include <boost/iterator/indirect_iterator.hpp>
#include <iostream>
#include "utility/inspect.hpp"


namespace backend
{

class literal;
class name;
class apply;
class lambda;
class closure;
class conditional;
class tuple;
class subscript;
class ret;
class bind;
class call;
class procedure;
class suite;
class statement;
class structure;
class templated_name;
class include;
class typedefn;
class namespace_block;

class node;
class statement;
class expression;
class cppnode;


namespace detail
{

typedef boost::variant<
    literal &,
    name &,
    apply &,
    lambda &,
    closure &,
    conditional &,
    tuple &,
    subscript &,
    ret &,
    bind &,
    call &,
    procedure &,
    suite &,
    structure &,
    templated_name &,
    include &,
    typedefn &,
    namespace_block &
    > node_base;

struct make_node_base_visitor
    : boost::static_visitor<node_base>
{
    make_node_base_visitor(void *p);

    template<typename Derived>
    node_base operator()(const Derived &) const {
        // use of std::ref disambiguates variant's copy constructor dispatch
        return node_base(std::ref(*reinterpret_cast<Derived*>(ptr)));
    }

    void* ptr;
};

node_base make_node_base(void *ptr, const node_base &other);

} // end detail

//! \addtogroup nodes
/*! @{
 */
//! The base AST node class
/*! 
  All AST nodes derive from this class. Not intended to be
  instantiated directly.
 */

class node
    : public detail::node_base,
      public std::enable_shared_from_this<node>
{
protected:
    typedef detail::node_base super_t;

    template<typename Derived>
    node(Derived &self)
        : super_t(std::ref(self)) // use of std::ref disambiguates variant's copy constructor dispatch
        {}
public:
    //copy constructor requires special handling
    node(const node &other);
    /*! When we need to get at the pointer holding a node */
    std::shared_ptr<const node> ptr() const;
};

//! Prints AST \p node objects
/*! 
  
  \param strm Output stream to print the node to.
  \param n The node to be printed.
  
  \return Modified stream that has been printed to.
*/
std::ostream& operator<<(std::ostream& strm, const node& n);

//! @}

template<typename ResultType = void>
struct no_op_visitor
    : boost::static_visitor<ResultType>
{
    inline ResultType operator()(const node &) const {
        return ResultType();
    }
};

}


    
