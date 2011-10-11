#pragma once

enum struct iteration_structure {
    scalar,
    sequential,
    parallel,
    independent
};


//To make iteration_structures comparable
//And thus work as part of a key to a map
bool operator<(iteration_structure l, iteration_structure r) {
    return (int)l < int(r);
}

//To make iteration_structures print
std::ostream& operator<<(std::ostream& strm, iteration_structure is) {
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
