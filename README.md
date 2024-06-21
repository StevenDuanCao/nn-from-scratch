# nn-from-scratch

An auto-differentiation engine and neural network architecture built from the ground up. 

### Install
```
pip install numpy
```
### Usage
The Node object stores a value and gradient. The value can be manipulated through common operations. The gradient is automatically updated when backpropagation is called. The following is an example:
```
from nn_scratch.node import Node

a = Node(2.0)
b = Node(-3.0)
c = Node(1.0)
d = a * b
e = d - c
print(e.value) # prints -7.0, the result of the forward pass
e.backprop()
print(a.grad) # prints -3.0, the value of de/da
print(b.grad) # prints 2.0, the value of de/db
print(c.grad) # prints -1.0, the value of de/dc
print(d.grad) # prints 1.0, the value of de/dd
```
