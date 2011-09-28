from coresyntax import *
import coretypes

arg_type = coretypes.Sequence(coretypes.Float32())
arg_types = coretypes.Tuple([arg_type])
return_type = coretypes.Sequence(coretypes.Float32())
proc_type = coretypes.Fn(arg_types, return_type)


res = Name("_result")
res.type = return_type
arg = Name("x")
arg.type = arg_type
fn_call = Bind(res, Apply(Name("adjacent_difference"),
                                      Tuple([arg])))
ret = Return(res)
body = Suite([fn_call, ret])

args = Tuple([arg])
proc = Procedure(Name("adj"), args, body)

proc.type = proc_type
args.type = arg_types



print(proc)



compiler = Compiler("adj")
compiled = compiler(Suite([proc]))
print(compiled)

