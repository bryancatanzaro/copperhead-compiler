from coresyntax import *
import coretypes

a = Return(Apply(Name("adjacent_difference"), Tuple([Name("x")])))
body = Suite([a])
arg = Name("x")
args = Tuple([arg])
proc = Procedure(Name("adj"), args, body)

arg_type = coretypes.Sequence(coretypes.Float32())
arg_types = coretypes.Tuple([arg_type])
return_type = coretypes.Sequence(coretypes.Float32())
proc_type = coretypes.Fn(arg_types, return_type)
proc.type = proc_type
args.type = arg_types
arg.type = arg_type


print(proc)



compiler = Compiler("adj")
compiled = compiler(Suite([proc]))
print(compiled)

