#include "monotype.hpp"

using std::string;
using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using std::vector;
using backend::utility::make_vector;


namespace backend {

monotype_t::monotype_t(const string &name)
        : type_t(*this),
          m_name(name)
        {}

//XXX Should this be id() to be consistent?
const string& monotype_t::name(void) const {
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
//! The instantiated type object representing the Int32 type
shared_ptr<monotype_t> int32_mt = make_shared<monotype_t>("Int32");
//! The instantiated type object representing the Int64 type
shared_ptr<monotype_t> int64_mt = make_shared<monotype_t>("Int64");
//! The instantiated type object representing the Uint32 type
shared_ptr<monotype_t> uint32_mt = make_shared<monotype_t>("Uint32");
//! The instantiated type object representing the Uint64 type
shared_ptr<monotype_t> uint64_mt = make_shared<monotype_t>("Uint64");
//! The instantiated type object representing the Float32 type
shared_ptr<monotype_t> float32_mt = make_shared<monotype_t>("Float32");
//! The instantiated type object representing the Float64 type
shared_ptr<monotype_t> float64_mt = make_shared<monotype_t>("Float64");
//! The instantiated type object representing the Bool type
shared_ptr<monotype_t> bool_mt = make_shared<monotype_t>("Bool");
//! The instantiated type object representing the Void type
shared_ptr<monotype_t> void_mt = make_shared<monotype_t>("Void");


sequence_t::sequence_t(const shared_ptr<type_t> &sub)
    : monotype_t(*this,
                 "Seq",
                 make_vector<shared_ptr<type_t> >(sub)) {}
const type_t& sequence_t::sub() const {
    return *m_params[0];
}
shared_ptr<type_t> sequence_t::p_sub() const {
    return m_params[0];
}
tuple_t::tuple_t(vector<shared_ptr<type_t> > && sub)
    : monotype_t(*this, "Tuple", std::move(sub)) {}

tuple_t::const_ptr_iterator tuple_t::p_begin() const {
    return m_params.cbegin();
}
tuple_t::const_ptr_iterator tuple_t::p_end() const {
    return m_params.cend();
}

fn_t::fn_t(const shared_ptr<tuple_t> args,
           const shared_ptr<type_t> result)
    : monotype_t(*this,
                 "Fn",
                 make_vector<shared_ptr<type_t> >(args)(result)) {}

const tuple_t& fn_t::args() const {
    return *static_pointer_cast<tuple_t>(m_params[0]);
}
const type_t& fn_t::result() const {
        return *m_params[1];
}

shared_ptr<tuple_t> fn_t::p_args() const {
    return static_pointer_cast<tuple_t>(m_params[0]);
}

shared_ptr<type_t> fn_t::p_result() const {
    return m_params[1];
}

}
