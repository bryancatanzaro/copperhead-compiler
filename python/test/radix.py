from coresyntax import *
import coretypes
import compiler

a_t = coretypes.Monotype("a")
bin_op_t = coretypes.Polytype([a_t],
                              coretypes.Fn(coretypes.Tuple([a_t, a_t]),
                                           a_t))
un_op_t = coretypes.Polytype([a_t],
                             coretypes.Fn(coretypes.Tuple([a_t]),
                                          a_t))

op_neg = Name('op_neg')
op_neg.type = un_op_t
op_pos = Name('op_pos')
op_pos.type = un_op_t
op_add = Name('op_add')
op_add.type = bin_op_t

cmp_eq = Name('cmp_eq')
cmp_eq.type = coretypes.Polytype([a_t],
                                 coretypes.Fn(coretypes.Tuple([a_t, a_t]),
                                              coretypes.Bool()))

op_rshift = Name('op_rshift')
op_rshift.type = bin_op_t

op_and = Name('op_and')
op_and.type = bin_op_t

op_xor = Name('op_xor')
op_xor.type = bin_op_t

int32_t = coretypes.Int32()

delta_arg_types = coretypes.Tuple([int32_t, int32_t, int32_t])
flag_name = Name('_flag')
ones_before_name = Name('_ones_before')
zeros_after_name = Name('_zeros_after')
flag_name.type = int32_t
ones_before_name.type = int32_t
zeros_after_name.type = int32_t
cond = Apply(cmp_eq,
             Tuple([flag_name,
                    Literal('0')]))
delta_result = Name('result')
delta_result.type = int32_t
then_suite = Suite([Bind(delta_result,
                    Apply(op_neg,
                          Tuple([ones_before_name]))),
                    Return(delta_result)])
else_suite = Suite([Bind(delta_result,
                    Apply(op_pos,
                          Tuple([zeros_after_name]))),
                    Return(delta_result)])
delta_cond = Conditional(cond, then_suite, else_suite)
delta_name = Name('_delta')
delta_proc = Procedure(delta_name,
                       Tuple([flag_name, ones_before_name, zeros_after_name]),
                       Suite([delta_cond]))
delta_proc.type = coretypes.Fn(delta_arg_types, int32_t)
delta_name.type = delta_proc.type        

lambda0_arg_types = coretypes.Tuple([int32_t, int32_t])
x_name = Name('_x')
x_name.type = int32_t
k0_name = Name('k0')
k0_name.type = int32_t
lambda0_args = Tuple([x_name, k0_name])
lambda0_args.type = lambda0_arg_types
e0_name = Name('e0')
e0_name.type = int32_t
e0_bind = Bind(e0_name,
               Apply(op_rshift,
                     Tuple([x_name, k0_name])))
e1_name = Name('e1')
e1_name.type = int32_t
e1_bind = Bind(e1_name,
               Apply(op_and,
                     Tuple([e0_name, Literal('1')])))
lambda0_suite = Suite([e0_bind,
                       e1_bind,
                       Return(e1_name)])
lambda0_name = Name('lambda0')
lambda0_proc = Procedure(lambda0_name,
                         lambda0_args,
                         lambda0_suite)
lambda0_proc_type = coretypes.Fn(lambda0_arg_types, int32_t)
lambda0_proc.type = lambda0_proc_type
lambda0_name.type = lambda0_proc_type

lambda1_arg_types = coretypes.Tuple([int32_t])
f_name = Name('_f')
f_name.type = int32_t
lambda1_args = Tuple([f_name])
lambda1_args.type = lambda1_arg_types
l1_e0_name = Name('e0')
l1_e0_name.type = int32_t
l1_e0_bind = Bind(l1_e0_name,
                  Apply(op_xor,
                        Tuple([f_name,
                               Literal('1')])))
lambda1_suite = Suite([l1_e0_bind,
                       Return(l1_e0_name)])

lambda1_name = Name('lambda1')
lambda1_proc = Procedure(lambda1_name,
                         lambda1_args,
                         lambda1_suite)
lambda1_proc.type = coretypes.Fn(lambda1_arg_types, int32_t)
lambda1_name.type = lambda1_proc.type


rs_iteration_name = Name('_radix_sort_iteration')
seq_int32_t = coretypes.Sequence(int32_t)
A_name = Name('_A')
A_name.type = seq_int32_t
lsb_name = Name('_lsb')
lsb_name.type = int32_t
rs_iteration_arg_types = coretypes.Tuple([seq_int32_t, int32_t])
rs_iteration_args = Tuple([A_name, lsb_name])
rs_iteration_args.type = rs_iteration_arg_types
flags_name = Name('_flags')
flags_name.type = seq_int32_t
ones_name = Name('_ones')
ones_name.type = seq_int32_t
zeros_name = Name('_zeros')
zeros_name.type = seq_int32_t
offsets_name = Name('_offsets')
offsets_name.type = seq_int32_t
flags_bind = Bind(flags_name,
                  Apply(Name('map'),
                        Tuple([Closure(Tuple([lsb_name]),
                                       lambda0_name),
                               A_name])))
ones_bind = Bind(ones_name,
                 Apply(Name('scan'),
                       Tuple([Name('op_add'), flags_name])))
rs_e0_name = Name('e0')
rs_e0_name.type = seq_int32_t
rs_e0_bind = Bind(rs_e0_name,
                  Apply(Name('map'),
                        Tuple([lambda1_name,
                               flags_name])))
zeros_bind = Bind(zeros_name,
                  Apply(Name('rscan'),
                        Tuple([Name('op_add'), rs_e0_name])))
offsets_bind = Bind(offsets_name,
                    Apply(Name('map'),
                          Tuple([delta_name, flags_name, ones_name, zeros_name])))
rs_e1_name = Name('e1')
rs_e1_name.type = seq_int32_t
rs_e1_bind = Bind(rs_e1_name,
                  Apply(Name('indices'),
                         Tuple([A_name])))
rs_e2_name = Name('e2')
rs_e2_name.type = seq_int32_t
rs_e2_bind = Bind(rs_e2_name,
                  Apply(Name('map'),
                         Tuple([op_add, offsets_name, rs_e1_name])))
rs_result_name = Name('result')
rs_result_name.type = seq_int32_t
rs_result_bind = Bind(rs_result_name,
                      Apply(Name('permute'),
                             Tuple([A_name, rs_e2_name])))
                                   
                                   

rs_suite = Suite([flags_bind,
                  ones_bind,
                  rs_e0_bind,
                  zeros_bind,
                  offsets_bind,
                  rs_e1_bind,
                  rs_e2_bind,
                  rs_result_bind,
                  Return(rs_result_name)])
rs_proc = Procedure(rs_iteration_name,
                    rs_iteration_args,
                    rs_suite)

rs_proc.type = coretypes.Fn(rs_iteration_arg_types,
                            seq_int32_t)
rs_iteration_name.type = rs_proc.type
                    
assembly = Suite([delta_proc, lambda0_proc, lambda1_proc, rs_proc])
print(assembly)




t_compiler = compiler.Compiler("_radix_sort_iteration")
compiled = t_compiler(assembly)
print(compiled)



