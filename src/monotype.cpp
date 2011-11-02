#include "monotype.hpp"

namespace backend {

monotype_t::monotype_t(const std::string &name)
        : type_t(*this),
          m_name(name)
        {}

//XXX Should this be id() to be consistent?
const std::string& monotype_t::name(void) const {
    return m_name;
}

monotype_t::const_iterator monotype_t::begin() const {
    return m_params.begin();
}
monotype_t::const_iterator monotype_t::end() const {
    return m_params.end();
}
int monotype_t::size() const {
    return m_params.size();
}

std::shared_ptr<monotype_t> int32_mt = std::make_shared<monotype_t>("Int32");
std::shared_ptr<monotype_t> int64_mt = std::make_shared<monotype_t>("Int64");
std::shared_ptr<monotype_t> uint32_mt = std::make_shared<monotype_t>("Uint32");
std::shared_ptr<monotype_t> uint64_mt = std::make_shared<monotype_t>("Uint64");
std::shared_ptr<monotype_t> float32_mt = std::make_shared<monotype_t>("Float32");
std::shared_ptr<monotype_t> float64_mt = std::make_shared<monotype_t>("Float64");
std::shared_ptr<monotype_t> bool_mt = std::make_shared<monotype_t>("Bool");
std::shared_ptr<monotype_t> void_mt = std::make_shared<monotype_t>("Void");


sequence_t::sequence_t(const std::shared_ptr<type_t> &sub)
    : monotype_t(*this,
                 "Seq",
                 std::vector<std::shared_ptr<type_t> >{sub}) {}
const type_t& sequence_t::sub() const {
    return *m_params[0];
}

tuple_t::tuple_t(std::vector<std::shared_ptr<type_t> > && sub)
    : monotype_t(*this, "Tuple", std::move(sub)) {}

fn_t::fn_t(const std::shared_ptr<tuple_t> args,
           const std::shared_ptr<type_t> result)
    : monotype_t(*this,
                 "Fn",
                 std::vector<std::shared_ptr<type_t> >{args, result}) {}

const tuple_t& fn_t::args() const {
    return *std::static_pointer_cast<tuple_t>(m_params[0]);
}
const type_t& fn_t::result() const {
        return *m_params[1];
}

}
