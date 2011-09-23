#pragma once
#include <vector>
#include <boost/iterator/indirect_iterator.hpp>
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
        : monotype_t("Seq", std::vector<std::shared_ptr<type_t>>{sub})
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
        : monotype_t("Fn", std::vector<std::shared_ptr<type_t>>{args, result})
        {}
};

}
