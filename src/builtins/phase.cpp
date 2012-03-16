#include "import/phase.hpp"

namespace backend {

phase_t::phase_t(std::vector<completion>&& args, const completion& result)
    : m_args(args), m_result(result) {}

phase_t::iterator phase_t::begin() const {
    return m_args.cbegin();
}

phase_t::iterator phase_t::end() const {
    return m_args.cend();
}

completion phase_t::result() const {
    return m_result;
}

int phase_t::size() const {
    return m_args.size();
}

}

//XXX
//Some compilers require this, others don't
bool operator<(backend::iteration_structure a, backend::iteration_structure b) {
    return (int)a < (int)b;
}


//To make iteration_structures print
std::ostream& operator<<(std::ostream& strm,
                         const backend::iteration_structure& is) {
    switch(is) {
    case backend::iteration_structure::scalar:
        strm << "scalar";
        break;
    case backend::iteration_structure::sequential:
        strm << "sequential";
        break;
    case backend::iteration_structure::parallel:
        strm << "parallel";
        break;
    case backend::iteration_structure::independent:
        strm << "independent";
        break;
    default:
        strm << "unknown";
    }
    return strm;
}

//To make completions print
std::ostream& operator<<(std::ostream& strm,
                         const backend::completion& cn) {
    switch(cn) {
    case backend::completion::none:
        strm << "none";
        break;
    case backend::completion::local:
        strm << "local";
        break;
    case backend::completion::total:
        strm << "total";
        break;
    case backend::completion::invariant:
        strm << "invariant";
        break;
    }
    return strm;
}

std::ostream& operator<<(std::ostream& strm,
                         const backend::phase_t& ct) {
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
