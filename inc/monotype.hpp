#pragma once
#include <vector>
#include <boost/iterator/indirect_iterator.hpp>
#include "type.hpp"
#include <iostream>
#include <string>

namespace backend {

class monotype_t :
    public type_t
{
protected:
    const std::string m_name;
    std::vector<std::shared_ptr<type_t> > m_params;
   
public:
    monotype_t(const std::string &name);

    template<typename Derived>
    monotype_t(Derived& self,
               const std::string &name) :
        type_t(self), m_name(name) {}

    template<typename Derived>
    monotype_t(Derived& self,
               const std::string &name,
               std::vector<std::shared_ptr<type_t> > &&params)
        : type_t(self), m_name(name), m_params(std::move(params)) {}

    //XXX Should this be id() to be consistent?
    const std::string& name(void) const;

    typedef decltype(boost::make_indirect_iterator(m_params.cbegin())) const_iterator;
    const_iterator begin() const;
    const_iterator end() const;
    int size() const;

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
public:
    sequence_t(const std::shared_ptr<type_t> &sub);
    const type_t& sub() const;
    const std::shared_ptr<type_t> p_sub() const;
};

class tuple_t :
        public monotype_t
{
public:
    tuple_t(std::vector<std::shared_ptr<type_t> > && sub);
    template<typename Derived>
    inline tuple_t(Derived& self,
                   const std::string& name,
                   std::vector<std::shared_ptr<type_t> > && sub)
        : monotype_t(self, name, std::move(sub))
        {}
    typedef decltype(m_params.cbegin()) const_ptr_iterator;
    const_ptr_iterator p_begin() const;
    const_ptr_iterator p_end() const;
};

class fn_t :
        public monotype_t
{
public:
    fn_t(const std::shared_ptr<tuple_t> args,
                const std::shared_ptr<type_t> result);
    const tuple_t& args() const;
    const type_t& result() const;
};



}
