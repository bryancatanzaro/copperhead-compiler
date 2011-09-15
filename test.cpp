#include <iostream>
#include "node.hpp"
#include "expression.hpp"
#include "statement.hpp"
#include "type.hpp"
#include "monotype.hpp"
#include "repr_printer.hpp"
#include "type_printer.hpp"
#include "py_printer.hpp"
#include "compiler.hpp"
#include "cuda_printer.hpp"
#include <boost/variant.hpp>

using namespace backend;
using std::shared_ptr;
using std::vector;


int main() {
   
    shared_ptr<name> xpy_id(new name("xpy"));
    shared_ptr<name> op_add_id(new name("op_add"));
    shared_ptr<name> x_id(new name("x"));
    shared_ptr<name> y_id(new name("y"));
    shared_ptr<name> xi_id(new name("xi"));
    shared_ptr<name> yi_id(new name("yi"));
    shared_ptr<tuple> xiyi(new tuple(vector<shared_ptr<expression>>{xi_id, yi_id}));
    shared_ptr<apply> addxiyi(new apply(op_add_id, xiyi));
    shared_ptr<ret> add_ret(new ret(addxiyi));
    shared_ptr<suite> lambda0_body(new suite(vector<shared_ptr<statement>>{add_ret}));
    shared_ptr<name> lambda0_id(new name("lambda0"));
    shared_ptr<procedure> lambda0(new procedure(lambda0_id, xiyi, lambda0_body));
    //shared_ptr<lambda> lambda0(new lambda(xiyi, addxiyi));
    shared_ptr<name> map_id(new name("map"));
    shared_ptr<tuple> map_appl_args(new tuple(vector<shared_ptr<expression> >{lambda0_id, x_id, y_id}));
    shared_ptr<apply> map_apply(new apply(map_id, map_appl_args));
    shared_ptr<ret> map_ret(new ret(map_apply));
    shared_ptr<tuple> xy(new tuple(vector<shared_ptr<expression>>{x_id, y_id}));                                          
    shared_ptr<suite> xpy_body(new suite(vector<shared_ptr<statement>>{map_ret}));
    shared_ptr<procedure> xpy(new procedure(xpy_id, xy, xpy_body));
    shared_ptr<suite> source(new suite(vector<shared_ptr<statement>>{lambda0, xpy}));
    repr_printer rp(std::cout);
    rp(*source);
    std::cout << std::endl;
    //shared_ptr<type_t> seq_int32(new sequence_t(int32_mt));
    //repr_type_printer tp(std::cout);
    //boost::apply_visitor(tp, *seq_int32);
    //std::cout << std::endl;
    py_printer pp(std::cout);
    pp(*source);

    std::string entry_point("xpy");
    compiler t_compiler(entry_point);
    
    shared_ptr<suite> functorized = t_compiler(*source);
    pp(*functorized);

    std::cout << "---------------------" << std::endl;
    cuda_printer cp(entry_point, std::cout);
    cp(*functorized);
}
