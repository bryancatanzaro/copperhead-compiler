#include "utility/name_supply.hpp"
#include <sstream>

using std::string;
using std::ostringstream;

namespace backend {
namespace detail {

name_supply::name_supply(const string& prefix) : m_prefix(prefix), m_state(0) {}
    
string name_supply::next() {
    ostringstream os;
    os << m_prefix << m_state;
    m_state++;
    return os.str();        
}

}
}
