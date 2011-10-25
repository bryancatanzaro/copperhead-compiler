#pragma once
#include <vector>
#include <boost/iterator/indirect_iterator.hpp>
#include "type.hpp"
#include <iostream>


namespace backend {

class monotype_t :
    public type_t
{
protected:
    const std::string m_name;
    std::vector<std::shared_ptr<type_t> > m_params;
   
public:
    monotype_t(const std::string &name)
        : type_t(*this),
          m_name(name)
        {}

    template<typename Derived>
    monotype_t(Derived& self,
               const std::string &name) :
        type_t(self), m_name(name) {}

    template<typename Derived>
    monotype_t(Derived& self,
               const std::string &name,
               std::vector<std::shared_ptr<type_t> > &&params)
        : type_t(self), m_name(name), m_params(std::move(params)) {}
    
    const std::string& name(void) const {
        return m_name;
    }

    typedef decltype(boost::make_indirect_iterator(m_params.cbegin())) const_iterator;
    const_iterator begin() const {
        return m_params.begin();
    }
    const_iterator end() const {
        return m_params.end();
    }
    int size() const {
        return m_params.size();
    }

};



struct int32_mt :
        public monotype_t
{
    int32_mt() : monotype_t(*this, "Int32") {}
};

struct int64_mt :
        public monotype_t
{
    int64_mt() : monotype_t(*this, "Int64") {}
};

struct uint32_mt :
        public monotype_t
{
    uint32_mt() : monotype_t(*this, "Uint32") {}
};

struct uint64_mt :
        public monotype_t
{
    uint64_mt() : monotype_t(*this, "Uint64") {}
};

struct float32_mt :
        public monotype_t
{
    float32_mt() : monotype_t(*this, "Float32") {}
};

struct float64_mt :
        public monotype_t
{
    float64_mt() : monotype_t(*this, "Float64") {}
};

struct bool_mt :
        public monotype_t
{
    bool_mt() : monotype_t(*this, "Bool") {}
};

struct void_mt :
        public monotype_t
{
    void_mt() : monotype_t(*this, "Void") {}
};

class sequence_t :
        public monotype_t
{
public:
    inline sequence_t(const std::shared_ptr<type_t> &sub)
        : monotype_t(*this,
                     "Seq",
                     std::vector<std::shared_ptr<type_t> >{sub})
        {}
    const type_t& sub() const {
        return *m_params[0];
    }
};

class tuple_t :
        public monotype_t
{
public:
    inline tuple_t(std::vector<std::shared_ptr<type_t> > && sub)
        : monotype_t(*this, "Tuple", std::move(sub))
        {}
    template<typename Derived>
    inline tuple_t(Derived& self,
                   const std::string& name,
                   std::vector<std::shared_ptr<type_t> > && sub)
        : monotype_t(self, name, std::move(sub))
        {}
};

class fn_t :
        public monotype_t
{
public:
    inline fn_t(const std::shared_ptr<tuple_t> args,
                const std::shared_ptr<type_t> result)
        : monotype_t(*this,
                     "Fn",
                     std::vector<std::shared_ptr<type_t> >{args, result})
        {}
    inline const tuple_t& args() const {
        return *std::static_pointer_cast<tuple_t>(m_params[0]);
    }
    inline const type_t& result() const {
        return *m_params[1];
    }
};

class var_t
    : public monotype_t
{
public:
    inline var_t(const std::shared_ptr<monotype_t> sub)
        : monotype_t(*this,
                     "Var",
                     std::vector<std::shared_ptr<type_t> >{sub}) {}
    inline const monotype_t& sub() const {
        return *std::static_pointer_cast<monotype_t>(m_params[0]);
    }
};

class vartuple_t
    : public tuple_t
{
public:
    inline vartuple_t(const std::shared_ptr<monotype_t> sub)
        : tuple_t(*this,
                  "Vartuple",
                  std::vector<std::shared_ptr<type_t> >{sub}) {}
    inline const monotype_t& sub() const {
        return *std::static_pointer_cast<monotype_t>(m_params[0]);
    }
};
        


}
