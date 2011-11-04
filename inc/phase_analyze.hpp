#pragma once
#include <string>
#include <map>
#include <vector>
#include "node.hpp"
#include "copier.hpp"
#include "prelude/phase.hpp"
#include "environment.hpp"
#include "import/library.hpp"
#include "utility/isinstance.hpp"
#include "utility/up_get.hpp"
#include "utility/snippets.hpp"
#include "utility/markers.hpp"

namespace backend {

class phase_analyze
    : public copier {
private:
    const std::string m_entry_point;
    bool m_in_entry;
    std::map<std::string, std::shared_ptr<phase_t> > m_fns;
    environment<std::string, completion> m_completions;
    environment<std::string, std::shared_ptr<name> > m_substitutions;
    //Should this be implemented with std::stack<result_type>?
    std::vector<result_type> m_additionals;
    completion m_result_completion;
    void add_phase_boundary(const name& n);
public:
    using copier::operator();
    phase_analyze(const std::string& entry_point, const registry& reg);
    result_type operator()(const suite& n);
    result_type operator()(const procedure& n);
    result_type operator()(const apply& n);
    result_type operator()(const bind& n);
    result_type operator()(const ret& n);
    
};

}
