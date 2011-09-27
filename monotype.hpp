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
    
    const std::string& name(void) const {
        return m_name;
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
private:
    std::shared_ptr<type_t> m_sub;
public:
    inline sequence_t(const std::shared_ptr<type_t> &sub)
        : monotype_t(*this, "Seq"), m_sub(sub)
        {}
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

    int size() const {
        return m_sub.size();
    }
    
};

class fn_t :
        public monotype_t
{
private:
    const std::shared_ptr<tuple_t> m_args;
    const std::shared_ptr<type_t> m_result;
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

}
