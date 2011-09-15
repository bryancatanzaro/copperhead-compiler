#pragma once
#include "type.hpp"

namespace backend {

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

char int32_s[]   =   "Int32";
char int64_s[]   =   "Int64";
char uint32_s[]  =  "Uint32";
char uint64_s[]  =  "Uint64";
char float32_s[] = "Float32";
char float64_s[] = "Float64";
char bool_s[]    =    "Bool";
char void_s[]    =    "Void";

typedef concrete_t<int32_s>     int32_mt;
typedef concrete_t<int64_s>     int64_mt;
typedef concrete_t<uint32_s>   uint32_mt;
typedef concrete_t<uint64_s>   uint64_mt;
typedef concrete_t<float32_s> float32_mt;
typedef concrete_t<float64_s> float64_mt;
typedef concrete_t<bool_s>       bool_mt;
typedef concrete_t<void_s>       void_mt;

class sequence_t :
        public monotype_t
{
public:
    inline sequence_t(const std::shared_ptr<type_t> &sub)
        : monotype_t("Seq", std::vector<std::shared_ptr<type_t>>{sub})
        {}
};



}
