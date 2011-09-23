#pragma once
#include <vector>
#include <boost/iterator/indirect_iterator.hpp>


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

};

class monotype_t :
    public type_t
{
private:
    const std::string m_name;
    std::vector<std::shared_ptr<type_t> > m_params;
   
public:
    monotype_t(const std::string &name)
        : type_t(*this),
          m_name(name)
        {}
    monotype_t(const std::string &name,
               std::vector<std::shared_ptr<type_t > > &&params)
        : type_t(*this),
          m_name(name),
          m_params(std::move(params))
        {}
    const std::string& name(void) const {
        return m_name;
    }
    typedef decltype(boost::make_indirect_iterator(m_params.cbegin())) const_iterator;
    const_iterator begin() const {
        return boost::make_indirect_iterator(m_params.cbegin());
    }

    const_iterator end() const {
        return boost::make_indirect_iterator(m_params.cend());
    }

};

template<const char *s>
struct concrete_t :
        public monotype_t
{
    concrete_t() : monotype_t(s) {}
};


//XXX need ifdefs for Windows and the various 64 bit C data models
//These assume Linux/OS X style models
char int32_s[] = "int";
char int64_s[] = "long";
char uint32_s[] = "unsigned int";
char uint64_s[] = "unsigned long";
char float32_s[] = "float";
char float64_s[] = "double";
char bool_s[] = "bool";
char void_s[] = "void";



class int32_mt : public concrete_t<int32_s>{};
class int64_mt : public concrete_t<int64_s>{};
class uint32_mt : public concrete_t<uint32_s>{};
class uint64_mt : public concrete_t<uint64_s>{};
class float32_mt : public concrete_t<float32_s>{};
class float64_mt : public concrete_t<float64_s>{};
class bool_mt : public concrete_t<bool_s>{};
class void_mt : public concrete_t<void_s>{};

class sequence_t :
        public monotype_t
{
public:
    inline sequence_t(const std::shared_ptr<type_t> &sub)
        : monotype_t("Seq", std::vector<std::shared_ptr<type_t> >{sub})
        {}
};

class tuple_t :
        public monotype_t
{
public:
    inline tuple_t(std::vector<std::shared_ptr<type_t> > && sub)
        : monotype_t("Tuple", std::move(sub))
        {}
};

class fn_t :
        public monotype_t
{
public:
    inline fn_t(const std::shared_ptr<tuple_t> args,
                const std::shared_ptr<type_t> result)
        : monotype_t("Fn", std::vector<std::shared_ptr<type_t> >{args, result})
        {}
};

}
}
