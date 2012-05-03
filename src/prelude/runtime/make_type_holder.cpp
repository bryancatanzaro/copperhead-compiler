#include <prelude/runtime/make_type_holder.hpp>
#include <prelude/runtime/type_holder.hpp>
#include <monotype.hpp>

using std::shared_ptr;
using std::make_shared;
using std::vector;

namespace copperhead {

namespace detail {

type_holder* make_type_holder() {
    type_holder* holder = new type_holder();
    holder->m_i.push(vector<shared_ptr<const backend::type_t> >());
    return holder;
}


void add_type(type_holder* t, int) {
    t->m_i.top().push_back(backend::int32_mt);
}

void add_type(type_holder* t, long) {
    t->m_i.top().push_back(backend::int64_mt);
}

void add_type(type_holder* t, float) {
    t->m_i.top().push_back(backend::float32_mt);
}

void add_type(type_holder* t, double) {
    t->m_i.top().push_back(backend::float64_mt);
}

void add_type(type_holder* t, bool) {
    t->m_i.top().push_back(backend::bool_mt);
}

void begin(type_holder* t) {
    t->m_i.push(vector<shared_ptr<const backend::type_t> >());
}

void end_sequence(type_holder* t) {
    shared_ptr<const backend::sequence_t> s =
        make_shared<const backend::sequence_t>(
            t->m_i.top()[0]);
    t->m_i.pop();
    t->m_i.top().push_back(s);
}

void end_tuple(type_holder* t) {
    shared_ptr<const backend::tuple_t> s =
        make_shared<const backend::tuple_t>(
            std::move(t->m_i.top()));
    t->m_i.pop();
    t->m_i.top().push_back(s);
}

void finalize_type(type_holder* t) {
    t->m_t = t->m_i.top()[0];
    t->m_i.pop();

}

}

}
