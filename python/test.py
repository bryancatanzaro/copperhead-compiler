from bindings import *
a = Return(Name('asdf'))
print(repr(a))
print(str(a))
b = a.val()
print(repr(b))
print(str(b))
c = Return(Name('fdas'))
d = Suite([a, c])
print(repr(d))
print(str(d))
print("Iterating")
for x in d:
    print x
e = Tuple([Name('a'), Name('b')])
print(repr(e))
print(str(e))
print("Iterating")
for x in e:
    print x
