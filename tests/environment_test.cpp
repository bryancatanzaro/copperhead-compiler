#include "environment.hpp"

#include <iostream>

template<typename T>
void inspect(T& in) {
    for(int i = 1; i < 10; i++) {
        typename T::citer iter = in.find(i);
        if(iter != in.end()) {
            std::cout << i << " found!" << std::endl;
        } else {
            std::cout << i << " NOT found." << std::endl;
        }
    }
}

int main() {
    backend::environment<int> x;
    x.insert(1);
    x.insert(2);
    x.begin_scope();
    x.insert(3);
    x.insert(4);
    inspect(x);
    x.end_scope();
    inspect(x);

    typedef std::pair<int, float> value_type;
    backend::environment<int, float> y;
    y.insert(value_type(1, 2.78f));
    y.begin_scope();
    y.insert(value_type(2, 3.14f));
    inspect(y);
    y.end_scope();
    inspect(y);
    y.end_scope();
}
