import numpy as np
a = [1, 2, 3]
b = [4, 5, 6]
print(a.pop(-1))
print(a)

c = np.array(a) + np.array(b)

print(c)
a = 3
print(a)
b = a
print(b)
b = 5
print(a)