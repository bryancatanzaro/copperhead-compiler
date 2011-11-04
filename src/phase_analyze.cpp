#include "phase_analysis.hpp"

using std::string;
using std::pair;
using std::shared_ptr;

namespace backend {

phase_analyzer::phase_analyzer(const registry& reg) {
    for(auto i = reg.fns().cbegin();
        i != reg.fns().cend();
        i++) {
        auto id = i->first;
        string fn_name = std::get<0>(id);
        auto info = i->second;
        m_fns.insert(pair<string, shared_ptr<phase_t> >{
                fn_name, info.p_phase()});
    }
}


}
