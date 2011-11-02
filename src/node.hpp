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
    ret &,
    bind &,
    call &,
    procedure &,
    suite &,
    structure &,
    templated_name &,
    include &,
    typedefn &
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

class node
    : public detail::node_base,
      public std::enable_shared_from_this<node>
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
    node(const node &other);
#ifdef DEBUG
    ~node();
#endif
};

template<typename ResultType = void>
struct no_op_visitor
    : boost::static_visitor<ResultType>
{
    inline ResultType operator()(const node &) const {
        return ResultType();
    }
};

}


    
