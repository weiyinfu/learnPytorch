import torch as t
"""
torch的max函数=np.max+np.argmax
"""
a=t.rand(3,2)
b=t.max(a,1)
print(a)
print(b)
print(b[0])
print(b[1])
print(t.argmax(a,1))