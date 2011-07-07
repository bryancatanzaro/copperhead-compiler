#include <iostream>
#include "../backend.hpp"
#include "repr_printer.hpp"
#include "type_printer.hpp"
#include <boost/variant.hpp>

using namespace backend;
using std::shared_ptr;
using std::vector;
int main() {
    // shared_ptr<name> n(new name("n"));
    // shared_ptr<number> o(new number("0"));
    // shared_ptr<tuple> no(new tuple(vector<shared_ptr<expression>>{n, o}));
    // shared_ptr<apply> app(new apply(n, no));
    // repr_printer rp(std::cout);
    // rp(*n);
    // std::cout << std::endl;
    // rp(*o);
    // std::cout << std::endl;
    // rp(*no);
    // std::cout << std::endl;
    // rp(*app);
    // std::cout << std::endl;
    shared_ptr<name> xpy_id(new name("xpy"));
    shared_ptr<name> op_add_id(new name("op_add"));
    shared_ptr<name> x_id(new name("x"));
    shared_ptr<name> y_id(new name("y"));
    shared_ptr<name> xi_id(new name("xi"));
    shared_ptr<name> yi_id(new name("yi"));
    shared_ptr<tuple> xiyi(new tuple(vector<shared_ptr<expression>>{xi_id, yi_id}));
    shared_ptr<apply> addxiyi(new apply(op_add_id, xiyi)); 
    shared_ptr<lambda> lambda0(new lambda(xiyi, addxiyi));
    shared_ptr<name> map_id(new name("map"));
    shared_ptr<tuple> map_appl_args(new tuple(vector<shared_ptr<expression> >{lambda0, x_id, y_id}));
    shared_ptr<apply> map_apply(new apply(map_id, map_appl_args));
    shared_ptr<ret> map_ret(new ret(map_apply));
    shared_ptr<tuple> xy(new tuple(vector<shared_ptr<expression>>{x_id, y_id}));                                          
                                              
    shared_ptr<procedure> xpy(new procedure(xpy_id, xy, vector<shared_ptr<statement>>{map_ret}));
    repr_printer rp(std::cout);
    rp(*xpy);
    std::cout << std::endl;
    shared_ptr<type_t> seq_int32(new sequence_t(int32_mt));
    repr_type_printer tp(std::cout);
    boost::apply_visitor(tp, *seq_int32);
    std::cout << std::endl;
}
