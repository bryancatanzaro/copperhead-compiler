from coretypes import *
a = Monotype('a')
b = Monotype('b')
var_b = Var(b)
vartuple_b = Vartuple(b)
map_fn = Fn(vartuple_b, a)
seq_a = Sequence(a)
seq_b = Sequence(b)
var_seq_b = Var(seq_b)
map_args = Tuple([map_fn, var_seq_b])
map_mono_t = Fn(map_args, seq_a)
map_t = Polytype([a, var_b], map_mono_t)
print(map_t)


var_seq_a = Var(seq_a)
seq_vartuple_a = Sequence(Vartuple(a))
zip_mono_t = Fn(Tuple([var_seq_a]),
                seq_vartuple_a)
zip_t = Polytype([Var(a)], zip_mono_t)
print(zip_t)

unzip_mono_t = Fn(Tuple([seq_vartuple_a]),
                         Tuple([var_seq_a]))
unzip_t = Polytype([Var(a)], unzip_mono_t)
print(unzip_t)
