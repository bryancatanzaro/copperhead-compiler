from coresyntax import *
import coretypes

a = Return(Apply(Name("adjacent_difference"), Tuple([Name("x")])))
body = Suite([a])
args = Tuple([Name("x")])
proc = Procedure(Name("adj"), args, body)

arg_types = coretypes.Tuple([coretypes.Sequence(coretypes.Float32())])
return_type = coretypes.Sequence(coretypes.Float32())
proc_type = coretypes.Fn(arg_types, return_type)
proc.type = proc_type

print(proc)
print(proc.type)



compiler = Compiler("adj")
compiled = compiler(Suite([proc]))
print(compiled)

#functorized = functorize_pass(Suite([inner]))
#print(functorized)
