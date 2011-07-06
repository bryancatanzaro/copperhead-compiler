#pragma once
#include "type.hpp"

namespace backend {

class monotype_t :
    public type_t
{
public:
    monotype_t(const std::string &name)
        : type_t(*this),
          m_name(name)
        {}
private:
    std::vector<std::shared_ptr<type_t> > m_params;
    const std::string m_name;
};

monotype_t int32_mt("int32");
monotype_t int64_mt("int64");
monotype_t uint32_mt("uint32");
monotype_t uint64_mt("uint64");
monotype_t float32_mt("float32");
monotype_t float64_mt("float64");
monotype_t bool_mt("bool");
monotype_t void_mt("void");

}
