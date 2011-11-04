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
using std::make_shared;
using std::vector;

int main(void)
{
  // types
  shared_ptr<monotype_t> Int32_t = int32_mt;
  shared_ptr<monotype_t> vecInt32_t(new sequence_t(Int32_t));

  // make op_add
  shared_ptr<tuple_t> int_tuple = make_shared<tuple_t>(vector<shared_ptr<type_t> >{Int32_t, Int32_t});
  shared_ptr<monotype_t> binop_t = make_shared<fn_t>(int_tuple, Int32_t);
  shared_ptr<name> op_add_scalar(new name("op_add", binop_t));

  // make map2
  shared_ptr<tuple_t> intvec_tuple = make_shared<tuple_t>(vector<shared_ptr<type_t> >{binop_t, vecInt32_t, vecInt32_t});
  shared_ptr<monotype_t> vecop_t = make_shared<fn_t>(intvec_tuple, vecInt32_t);
  shared_ptr<name> op_add_vector(new name("map2", vecop_t));

  // a, b
  shared_ptr<name> a(new name("a", vecInt32_t));
  shared_ptr<name> b(new name("b", vecInt32_t));
  
  // addproc(a,b)
  //   result = map2(op_add, a, b)
  //   return result
  shared_ptr<tuple> args(new tuple(vector<shared_ptr<expression> >{op_add_scalar, a, b}));
  shared_ptr<name> result(new name("result", vecInt32_t));
  shared_ptr<bind> res =
    make_shared<bind>(result,
                      make_shared<apply>(op_add_vector, args));
  shared_ptr<ret> addret(new ret(result));
  shared_ptr<name> addprocname(new name("addproc", vecop_t));
  shared_ptr<suite> funcbody(new suite(vector<shared_ptr<statement> >{res, addret}));
  shared_ptr<tuple> funcargs(new tuple(vector<shared_ptr<expression> >{a, b}));
  shared_ptr<procedure> addproc(new procedure(addprocname, funcargs, funcbody, vecop_t));

  // the whole program
  shared_ptr<suite> program(new suite(vector<shared_ptr<statement> >{addproc}));


  // print the representation
  std::cout << *program << std::endl;
  
  // compile
  std::string entry("addproc");
  compiler comp(entry);
  shared_ptr<suite> functorized = comp(*program);
 
  // print the representation
  py_printer pp(std::cout);
  pp(*program); std::cout << "\n";

  // cuda printer
  std::ostringstream os;
  backend::cuda_printer p(entry, comp.reg(), os);
  p(*functorized);
  std::cout << os.str() << "\n";

  return 0;
}
