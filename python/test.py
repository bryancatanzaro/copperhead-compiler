from coresyntax import *
import coretypes

float32_t = coretypes.Float32()
sub_arg_types = coretypes.Tuple([float32_t, float32_t])
sub_fn_type = coretypes.Fn(sub_arg_types, float32_t)
sub_res = Name("_result")
sub_res.type = float32_t
sub_arg_a = Name('a')
sub_arg_a.type = float32_t
sub_arg_b = Name('b')
sub_arg_b.type = float32_t
sub_args = Tuple([sub_arg_a, sub_arg_b])
sub_args.type = sub_arg_types
sub_call = Bind(sub_res, Apply(Name("op_sub"), sub_args))
sub_ret = Return(sub_res)
sub_body = Suite([sub_call, sub_ret])
sub_proc_name = Name('_lambda0')
sub_proc = Procedure(sub_proc_name,
                     sub_args,
                     sub_body)
sub_proc.type = sub_fn_type


arg_type = coretypes.Sequence(coretypes.Float32())
arg_types = coretypes.Tuple([arg_type])
return_type = coretypes.Sequence(coretypes.Float32())
proc_type = coretypes.Fn(arg_types, return_type)


res = Name("_result")
res.type = return_type
arg = Name("x")
arg.type = arg_type
fn_call = Bind(res, Apply(Name("adjacent_difference"),
                                     Tuple([sub_proc_name, arg])))
ret = Return(res)
body = Suite([fn_call, ret])

args = Tuple([arg])
proc = Procedure(Name("adj"), args, body)

proc.type = proc_type
args.type = arg_types


assembly = Suite([sub_proc, proc])

print(assembly)



compiler = Compiler("adj")
compiled = compiler(assembly)
print(compiled)



