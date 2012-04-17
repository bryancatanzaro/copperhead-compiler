#pragma once

#include <string>

namespace backend {
namespace detail {

class name_supply {
private:
    std::string m_prefix;
    int m_state;
public:
    name_supply(const std::string&);
    std::string next();
};

}
}
