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

class type_t
    : public detail::type_base,
      public std::enable_shared_from_this<type_t>
{
public:
    typedef detail::type_base super_t;
  
    template<typename Derived>
    type_t(Derived &self)
        : super_t(std::ref(self)) //use of std::ref disambiguates variant's copy constructor dispatch
        {}

    type_t(const type_t &other);

};

class monotype_t :
    public type_t
{
protected:
    const std::string m_name;
   
public:
    monotype_t(const std::string &name);
    
    template<typename Derived>
    monotype_t(Derived &self,
               const std::string &name)
        : type_t(self),
          m_name(name)
        {}
    
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

class sequence_t :
        public monotype_t
{
protected:
    std::shared_ptr<type_t> m_sub;
public:
    sequence_t(const std::shared_ptr<type_t> &sub);
    template<typename Derived>
    sequence_t(Derived &self,
                      const std::string& name,
                      const std::shared_ptr<type_t> &sub) :
        monotype_t(self, name), m_sub(sub) {}
    const type_t& sub() const;
    std::shared_ptr<type_t> p_sub() const;
};

class tuple_t :
        public monotype_t
{
private:
    std::vector<std::shared_ptr<type_t> > m_sub;
public:
    tuple_t(std::vector<std::shared_ptr<type_t> > && sub);
    typedef decltype(boost::make_indirect_iterator(m_sub.cbegin())) const_iterator;
    const_iterator begin() const;
    const_iterator end() const;
    typedef decltype(m_sub.cbegin()) const_ptr_iterator;
    const_ptr_iterator p_begin() const;
    const_ptr_iterator p_end() const;
};

class fn_t :
        public monotype_t
{
private:
    std::shared_ptr<tuple_t> m_args;
    std::shared_ptr<type_t> m_result;
public:
    fn_t(const std::shared_ptr<tuple_t> args,
                const std::shared_ptr<type_t> result);
    const tuple_t& args() const;
    const type_t& result() const;
    std::shared_ptr<tuple_t> p_args() const;
    std::shared_ptr<type_t> p_result() const;
};

class polytype_t :
        public type_t {
    polytype_t();
};

class cuarray_t :
        public sequence_t {
public:
    cuarray_t(const std::shared_ptr<type_t> sub);
};

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

}
}
