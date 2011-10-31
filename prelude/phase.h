#pragma once

enum struct iteration_structure {
    scalar,
    sequential,
    parallel,
    independent
};


//To make iteration_structures print
std::ostream& operator<<(std::ostream& strm,
                         const iteration_structure& is) {
    switch(is) {
    case iteration_structure::scalar:
        strm << "scalar";
        break;
    case iteration_structure::sequential:
        strm << "sequential";
        break;
    case iteration_structure::parallel:
        strm << "parallel";
        break;
    case iteration_structure::independent:
        strm << "independent";
        break;
    default:
        strm << "unknown";
    }
    return strm;
}


enum struct completion {
    invariant,
    local,
    total
};


//To make completions print
std::ostream& operator<<(std::ostream& strm,
                         const completion& cn) {
    switch(cn) {
    case completion::invariant:
        strm << "invariant";
        break;
    case completion::local:
        strm << "local";
        break;
    case completion::total:
        strm << "total";
        break;
    }
    return strm;
}

class phase_t {
private:
    const std::vector<completion> m_args;
    const completion m_result;
public:
    phase_t(std::vector<completion>&& args, const completion& result)
        : m_args(args), m_result(result) {}
    typedef std::vector<completion>::const_iterator iterator;
    iterator begin() const {
        return m_args.cbegin();
    }
    iterator end() const {
        return m_args.cend();
    }
    completion result() const {
        return m_result;
    }
};

std::ostream& operator<<(std::ostream& strm,
                         const phase_t& ct) {
    strm << "(";
    for(auto i = ct.begin();
        i != ct.end();
        i++) {
        strm << *i;
        if (std::next(i) != ct.end()) {
            strm << ", ";
        }
    }
    strm << ") -> ";
    strm << ct.result();
    return strm;
}
        
