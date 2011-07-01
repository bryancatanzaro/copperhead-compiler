#pragma once

#include "../backend.hpp"

namespace backend {

class expression
    : public node
{
protected:
    template<typename Derived>
    expression(Derived &self)
        : node(self)
        {}

};

class statement
    : public node
{
protected:
    template<typename Derived>
    statement(Derived &self)
        : node(self)
        {}

};

class literal
    : public expression
{
protected:
    template<typename Derived>
    literal(Derived &self)
        : expression(self)
        {}

};

class number
    : public literal
{
public:
    number(const std::string &val)
        : literal(*this),
          m_val(val)
        {}
private:
    std::string m_val;
};

class name
    : public literal
{
public:
    name(const std::string &val)
        : literal(*this),
          m_val(val)
        {}
private:
    std::string m_val;
};

// class tuple
//     : public expression
// {
// public:
//     tuple(std::vector<std::shared_ptr<expression> > &&values)
//         : expression(*this),
//           m_values(std::move(values))
//         {}
// private:
//     std::vector<std::shared_ptr<expression> > m_values;
// };

}
