#pragma once

namespace backend {
namespace detail {

template<typename T>
class type_checker:
        public boost::static_visitor<>
{
public:
    explicit type_checker(bool& res) : m_res(res) {}
    template<typename U>
    void operator()(const U& u) const {
        m_res = false;
    }

    void operator()(const T& t) const {
        m_res = true;
    }
private:
    bool& m_res;
};

template<typename T, typename V>
bool isinstance(const V& v) {
    bool result = false;
    boost::apply_visitor(type_checker<T>(result), v);
    return result;
}

}
}
