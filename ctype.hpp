#pragma once
#include <vector>
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
class int32_mt;
class int64_mt;
class uint32_mt;
class uint64_mt;
class float32_mt;
class float64_mt;
class bool_mt;
class void_mt;

namespace detail {
typedef boost::variant<
    monotype_t &,
    polytype_t &,
    sequence_t &,
    tuple_t &,
    fn_t &,
    int32_mt &,
    int64_mt &,
    uint32_mt &,
    uint64_mt &,
    float32_mt &,
    float64_mt &,
    bool_mt &,
    void_mt &
    > type_base;

struct make_type_base_visitor
    : boost::static_visitor<type_base>
{
    make_type_base_visitor(void *p)
        : ptr(p)
        {}
    template<typename Derived>
    type_base operator()(const Derived &) const {
        // use of std::ref disambiguates variant's copy constructor dispatch
        return type_base(std::ref(*reinterpret_cast<Derived*>(ptr)));
    }
    void *ptr;
};

type_base make_type_base(void *ptr, const type_base &other) {
    return boost::apply_visitor(make_type_base_visitor(ptr), other);
}

}

class type_t
    : public detail::type_base
{
public:
    typedef detail::type_base super_t;
  
    template<typename Derived>
    type_t(Derived &self)
        : super_t(std::ref(self)) //use of std::ref disambiguates variant's copy constructor dispatch
        {}

    type_t(const type_t &other)
        : super_t(detail::make_type_base(this, other))
        {}
    ~type_t() {}

};

class monotype_t :
    public type_t
{
protected:
    const std::string m_name;
   
public:
    monotype_t(const std::string &name)
        : type_t(*this),
          m_name(name)
        {}
    template<typename Derived>
    monotype_t(Derived &self,
               const std::string &name)
        : type_t(self),
          m_name(name)
        {}
    const std::string& name(void) const {
        return m_name;
    }

};


struct int32_mt :
        public monotype_t
{
    int32_mt() : monotype_t(*this, "int") {}
};

struct int64_mt :
        public monotype_t
{
    int64_mt() : monotype_t(*this, "long") {}
};

struct uint32_mt :
        public monotype_t
{
    uint32_mt() : monotype_t(*this, "unsigned int") {}
};

struct uint64_mt :
        public monotype_t
{
    uint64_mt() : monotype_t(*this, "unsigned long") {}
};

struct float32_mt :
        public monotype_t
{
    float32_mt() : monotype_t(*this, "float") {}
};

struct float64_mt :
        public monotype_t
{
    float64_mt() : monotype_t(*this, "double") {}
};

struct bool_mt :
        public monotype_t
{
    bool_mt() : monotype_t(*this, "bool") {}
};

struct void_mt :
        public monotype_t
{
    void_mt() : monotype_t(*this, "void") {}
};

class sequence_t :
        public monotype_t
{
private:
    std::shared_ptr<type_t> m_sub;
public:
    inline sequence_t(const std::shared_ptr<type_t> &sub)
        : monotype_t(*this, "sequence"), m_sub(sub) {}
    const type_t& sub() const {
        return *m_sub;
    }
};

class tuple_t :
        public monotype_t
{
private:
    std::vector<std::shared_ptr<type_t> > m_sub;
public:
    inline tuple_t(std::vector<std::shared_ptr<type_t> > && sub)
        : monotype_t(*this, "Tuple"), m_sub(std::move(sub))
        {}
        typedef decltype(boost::make_indirect_iterator(m_sub.cbegin())) const_iterator;
    const_iterator begin() const {
        return boost::make_indirect_iterator(m_sub.cbegin());
    }

    const_iterator end() const {
        return boost::make_indirect_iterator(m_sub.cend());
    }
};

class fn_t :
        public monotype_t
{
private:
    std::shared_ptr<tuple_t> m_args;
    std::shared_ptr<type_t> m_result;
public:
    inline fn_t(const std::shared_ptr<tuple_t> args,
                const std::shared_ptr<type_t> result)
        : monotype_t(*this, "Fn"), m_args(args), m_result(result)
        {}
    inline const tuple_t& args() const {
        return *m_args;
    }
    inline const type_t& result() const {
        return *m_result;
    }
};

class polytype_t :
        public type_t {
    polytype_t() : type_t(*this) {}

};
}
}
