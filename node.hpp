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

class number;
class name;
class apply;
class lambda;
class closure;
class conditional;
class tuple;
class ret;
class bind;
class call;
class procedure;
class suite;
class statement;
class structure;
class templated_name;
class include;

class node;
class statement;
class expression;
class literal;
class cppnode;


namespace detail
{

typedef boost::variant<
    number &,
    name &,
    apply &,
    lambda &,
    closure &,
    conditional &,
    tuple &,
    ret &,
    bind &,
    call &,
    procedure &,
    suite &,
    structure &,
    templated_name &,
    include &
    > node_base;

struct make_node_base_visitor
    : boost::static_visitor<node_base>
{
    make_node_base_visitor(void *p)
        : ptr(p)
        {}

    template<typename Derived>
    node_base operator()(const Derived &) const {
        // use of std::ref disambiguates variant's copy constructor dispatch
        return node_base(std::ref(*reinterpret_cast<Derived*>(ptr)));
    }

    void* ptr;
};

node_base make_node_base(void *ptr, const node_base &other) {
    return boost::apply_visitor(make_node_base_visitor(ptr), other);
}

} // end detail

class node
    : public detail::node_base
{
protected:
    typedef detail::node_base super_t;

#ifdef DEBUG
    static int counter;
    int id;
#endif
    template<typename Derived>
    node(Derived &self)
        : super_t(std::ref(self)) // use of std::ref disambiguates variant's copy constructor dispatch
        {
#ifdef DEBUG
            id = ++counter;
            std::cout << "Making node[" << id << "] from ";
            detail::inspect(self);
            std::cout << std::endl;
#endif
        }

    //copy constructor requires special handling
    node(const node &other)
        : super_t(detail::make_node_base(this, other))
        {
#ifdef DEBUG
            id = ++counter;
            std::cout << "Copying node[" << id << "] from ";
            detail::inspect(other);
            std::cout << std::endl;
#endif
        }
#ifdef DEBUG
    ~node() {
        std::cout << "Destroying node[" << id << "]" << std::endl;
    }
#endif
};

#ifdef DEBUG
int node::counter = 0;
#endif

template<typename ResultType = void>
struct no_op_visitor
    : boost::static_visitor<ResultType>
{
    inline ResultType operator()(const node &) const {
        return ResultType();
    }
};

template<>
struct no_op_visitor<void>
    : boost::static_visitor<>
{
    inline void operator()(const node &) const {}
};

}


    
