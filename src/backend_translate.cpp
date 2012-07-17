#include "backend_translate.hpp"

using std::string;
using std::make_shared;
using std::make_pair;

namespace backend {

backend_translate::backend_translate() {
    m_table.insert(make_pair(string("True"), string("true")));
    m_table.insert(make_pair(string("False"), string("false")));
}

backend_translate::result_type backend_translate::operator()(const name& n) {
    auto it = m_table.find(n.id());
    if (it != m_table.end()) {
        return make_shared<const name>(it->second,
                                       n.type().ptr(),
                                       n.ctype().ptr());
    }
    return n.ptr();
}

}
