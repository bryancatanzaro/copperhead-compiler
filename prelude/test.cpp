#include <iostream>
#include "decl.hpp"
#include "../utility/tuple_io.hpp"
#include "../toolchain/toolchain.hpp"

using namespace std;
using namespace backend;


template<typename T>
void print_iterable(ostream& strm, const T& in) {
    for(auto i = in.begin(); i != in.end(); i++) {
        strm << *i;
        if (next(i) != in.end()) {
            strm << ", ";
        }
    }
    strm << endl;
}

template<typename T>
void print_keys(ostream& strm, const T &in) {
    for(auto i = in.begin(); i != in.end(); i++) {
        strm << i->first;
        if (next(i) != in.end()) {
            strm << ", ";
        }
    }
    strm << endl;
}

int main() {
    shared_ptr<library> builtins = get_builtins();
    shared_ptr<registry> table(new registry());
    table->add_library(builtins);
    cout << "Fns: ";
    print_keys(cout, table->fns());
    cout << "Includes: ";
    print_iterable(cout, table->includes());
    cout << "Include paths: ";
    print_iterable(cout, table->include_dirs());
    gcc_nvcc compiler(table);
    cout << compiler.command_line() << endl;
}
