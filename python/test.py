from bindings import *
a = Return(Apply(Name("adjacent_difference"), Tuple([Name("x"), Name("y")])))
body = Suite([a])
args = Tuple([Name("x"), Name("y")])
proc = Procedure(Name("adj"), args, body)
print(proc)

compiler = Compiler("adj")
compiled = compiler(Suite([proc]))
print(compiled)

#functorized = functorize_pass(Suite([inner]))
#print(functorized)
