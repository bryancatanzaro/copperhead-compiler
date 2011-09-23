#pragma once

#include "monotype.hpp"

namespace backend
{

class repr_type_printer
    : public boost::static_visitor<>
{
public:
    inline repr_type_printer(std::ostream &os)
        : m_os(os)
        {}
    inline void operator()(const monotype_t &mt) {
        m_os << mt.name();
        if (mt.begin() != mt.end()) {
            open();
            for(auto i = mt.begin();
                i != mt.end();
                i++) {
                boost::apply_visitor(*this, *i);
                if (std::next(i) != mt.end()) 
                    sep();

            }
            close();
        }
    }
    template<const char* S>
    inline void operator()(const concrete_t<S> &c) {
        m_os << s;
    }
    inline void operator()(const polytype_t &pt) {
    }
private:
    std::ostream &m_os;
    inline void sep() const {
        m_os << ", ";
    }
    inline void open() const {
        m_os << "(";
    }
    inline void close() const {
        m_os << ")";
    }
};

}
