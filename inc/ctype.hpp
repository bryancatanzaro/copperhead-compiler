#pragma once
#include <vector>
#include <boost/variant.hpp>
#include <boost/iterator/indirect_iterator.hpp>
#include <memory>
#include <iostream>

namespace backend {

namespace ctype {

class monotype_t;
class polytype_t;
class sequence_t;
class tuple_t;
class fn_t;
class cuarray_t;
class templated_t;

namespace detail {
typedef boost::variant<
    monotype_t &,
    polytype_t &,
    sequence_t &,
    tuple_t &,
    fn_t &,
    cuarray_t &,
    templated_t &
    > type_base;

struct make_type_base_visitor
    : boost::static_visitor<type_base>
{
    make_type_base_visitor(void *p);
    
    template<typename Derived>
    type_base operator()(const Derived &) const {
        // use of std::ref disambiguates variant's copy constructor dispatch
        return type_base(std::ref(*reinterpret_cast<Derived*>(ptr)));
    }
    void *ptr;
};

type_base make_type_base(void *ptr, const type_base &other);

}

/*! 
  \addtogroup ctypes
  @{
 */

//! Parent type for all C++ implementation types
/*! Not intended to be instantiated directly 
  
  \param self 
  
  \return 
*/
class type_t
    : public detail::type_base
{
protected:
    typedef detail::type_base super_t;
    //! Derived constructor
/*! To be called by derived object during construction.
  
  \param self Reference to derived object being constructed.
*/
    template<typename Derived>
    type_t(Derived &self)
        : super_t(std::ref(self)) //use of std::ref disambiguates variant's copy constructor dispatch
        {}
public:
    //! Copy constructor
    type_t(const type_t &other);

};

//! Monomorphic type
/*! Can be used standalone or as a parent class */
class monotype_t :
    public type_t
{
protected:
    const std::string m_name;
   
public:
    //! Constructor
/*! 
  \param name Name of type.
*/
    monotype_t(const std::string &name);
    //! Derived constructor
/*! To be called during construction of derived object
  
  \param self Reference to derived object under construction.
  \param name Name of type.
  
*/
    template<typename Derived>
    monotype_t(Derived &self,
               const std::string &name)
        : type_t(self),
          m_name(name)
        {}
    //! Gets name of type.
    const std::string& name(void) const;

};

extern std::shared_ptr<monotype_t> int32_mt;
extern std::shared_ptr<monotype_t> int64_mt;
extern std::shared_ptr<monotype_t> uint32_mt;
extern std::shared_ptr<monotype_t> uint64_mt;
extern std::shared_ptr<monotype_t> float32_mt;
extern std::shared_ptr<monotype_t> float64_mt;
extern std::shared_ptr<monotype_t> bool_mt;
extern std::shared_ptr<monotype_t> void_mt;

//! Sequence type.
/* Can be used directly or as a parent class */
class sequence_t :
        public monotype_t
{
protected:
    std::shared_ptr<type_t> m_sub;
public:
    //! Basic constructor
/*!   
  \param sub Type of element of the Sequence
*/
    sequence_t(const std::shared_ptr<type_t> &sub);
    //! Derived constructor
/*! To be called during the construction of a derived object
  
  \param self Reference to the derived object under construction
  \param name Name of the Sequence-like type.
  \param sub Type of the element of the Sequence-like type.
  
*/
    template<typename Derived>
    sequence_t(Derived &self,
                      const std::string& name,
                      const std::shared_ptr<type_t> &sub) :
        monotype_t(self, name), m_sub(sub) {}
    //! Gets the type of the element of the Sequence
    const type_t& sub() const;
    //! Gets a pointer to the type of the element of the Sequence
    std::shared_ptr<type_t> p_sub() const;
};

//! Tuple type.
class tuple_t :
        public monotype_t
{
private:
    std::vector<std::shared_ptr<type_t> > m_sub;
public:
    //! Constructor
/*! 
  \param sub A vector of types contained in this tuple.
*/
    tuple_t(std::vector<std::shared_ptr<type_t> > && sub);
    //! An iterator type over the tuple subtypes
    typedef decltype(boost::make_indirect_iterator(m_sub.cbegin())) const_iterator;
    //! Gets an iterator to the first type held by this tuple
    const_iterator begin() const;
    //! Gets an iterator to the last type held by this tuple
    const_iterator end() const;
    //! An iterator type over the pointers holding the tuple subtypes
    typedef decltype(m_sub.cbegin()) const_ptr_iterator;
    //! Gets an iterator to the pointer of the first type held by this tuple
    const_ptr_iterator p_begin() const;
    //! Gets an iterator to the pointer of the last type held by this tuple
    const_ptr_iterator p_end() const;
};

//! Function type
class fn_t :
        public monotype_t
{
private:
    std::shared_ptr<tuple_t> m_args;
    std::shared_ptr<type_t> m_result;
public:
    //! Constructor
/*! 
  \param args Tuple of argument types.
  \param result Result type.
*/
    fn_t(const std::shared_ptr<tuple_t> args,
                const std::shared_ptr<type_t> result);
    //! Gets the tuple of argument types.
    const tuple_t& args() const;
    //! Gets the result type.
    const type_t& result() const;
    //! Gets a pointer to the tuple of argument types.
    std::shared_ptr<tuple_t> p_args() const;
    //! Gets a pointer to the result type.
    std::shared_ptr<type_t> p_result() const;
};

//! Polymorphic type
/*! This is currently a stub. */
class polytype_t :
        public type_t {
    polytype_t();
};

//! Cuarray type
/*! This is a sequence type, but is actually a container in C++
   and has to print differently. */
class cuarray_t :
        public sequence_t {
public:
    cuarray_t(const std::shared_ptr<type_t> sub);
};

//! Templated type
/*! This represents a type which is templated by other types.
  Perhaps this should become the implementation for
  \ref backend::ctype::polytype_t "polytype_t"
 */
class templated_t
    : public type_t {
private:
    std::shared_ptr<type_t> m_base;
    std::vector<std::shared_ptr<type_t> > m_sub;
public:
    templated_t(std::shared_ptr<type_t> base, std::vector<std::shared_ptr<type_t> > && sub);

    const type_t& base() const;
    std::shared_ptr<type_t> p_base() const;
    typedef decltype(boost::make_indirect_iterator(m_sub.cbegin())) const_iterator;
    const_iterator begin() const;
    const_iterator end() const;
    typedef decltype(m_sub.cbegin()) const_ptr_iterator;
    const_ptr_iterator p_begin() const;
    const_ptr_iterator p_end() const;
};

/*
  @}
*/

}
}
