#pragma once
#include <string>

namespace backend {

template<typename O, typename I>
void print_iterable(O& o, I& i,
                    std::string sep=std::string(","),
                    std::string el=std::string(""),
                    std::string open=std::string(""),
                    std::string close=std::string(""))
{
    o << open;
    for(auto it = i.begin();
        it != i.end();
        it++) {
        o << el << *it;
        if (std::next(it) != i.end())
            o << sep;
    }
    o << close;
}

}
