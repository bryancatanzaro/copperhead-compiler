#pragma once
#include <sstream>
#include <vector>

namespace backend {

enum struct iteration_structure {
    scalar,
    sequential,
    parallel,
    independent
};

//XXX
//Some compilers require this, others don't
bool operator<(iteration_structure a, iteration_structure b) {
    return (int)a < (int)b;
}



enum struct completion {
    invariant,
    local,
    total
};


class phase_t {
private:
    const std::vector<completion> m_args;
    const completion m_result;
public:
    phase_t(std::vector<completion>&& args, const completion& result);
    typedef std::vector<completion>::const_iterator iterator;
    iterator begin() const;
    iterator end() const;
    completion result() const;
};


        
}

std::ostream& operator<<(std::ostream& strm,
                         const backend::iteration_structure& is);

//To make completions print
std::ostream& operator<<(std::ostream& strm,
                         const backend::completion& cn);

std::ostream& operator<<(std::ostream& strm,
                         const backend::phase_t& ct);
