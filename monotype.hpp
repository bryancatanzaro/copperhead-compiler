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

std::shared_ptr<monotype_t> int32_mt(new monotype_t("Int32"));
std::shared_ptr<monotype_t> int64_mt(new monotype_t("Int64"));
std::shared_ptr<monotype_t> uint32_mt(new monotype_t("Uint32"));
std::shared_ptr<monotype_t> uint64_mt(new monotype_t("Uint64"));
std::shared_ptr<monotype_t> float32_mt(new monotype_t("Float32"));
std::shared_ptr<monotype_t> float64_mt(new monotype_t("Float64"));
std::shared_ptr<monotype_t> bool_mt(new monotype_t("Bool"));
std::shared_ptr<monotype_t> void_mt(new monotype_t("Void"));

class sequence_t :
        public monotype_t
{
public:
    inline sequence_t(const std::shared_ptr<type_t> &sub)
        : monotype_t("Seq", std::vector<std::shared_ptr<type_t>>{sub})
        {}
};

}
