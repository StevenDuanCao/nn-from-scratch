import math

# node that automatically stores and calculates gradients for backpropagation
class Node:
  def __init__(self, value, _children=()):
    self.value = value
    self.grad = 0.0
    self._backprop = lambda: None # derivative function according to operation done
    self._prev = set(_children) # track previous node

  def __repr__(self):
    return f"Node({self.value})"

  # addition
  def __add__(self, other):
    other = other if isinstance(other, Node) else Node(other) # allow for simple addition of scalars
    out = Node(self.value + other.value, (self, other))
    def _backprop():
      self.grad += out.grad
      other.grad += out.grad
    out._backprop = _backprop
    return out

  def __radd__(self, other): # allow for addition of scalars in reverse order
    return self + other

  # multiplication
  def __mul__(self, other):
    other = other if isinstance(other, Node) else Node(other)
    out = Node(self.value * other.value, (self, other))
    def _backprop():
      self.grad += other.value * out.grad
      other.grad += self.value * out.grad
    out._backprop = _backprop
    return out

  def __rmul__(self, other):
    return self * other

  # subtraction
  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self, other):
    return (-self) + other

  # power
  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only support scalar powers"
    out = Node(self.value ** other, (self,))
    def _backprop():
      self.grad += (other * self.value ** (other - 1)) * out.grad
    out._backprop = _backprop
    return out

  # division
  def __truediv__(self, other):
    return self * other**-1

  def __rtruediv__(self, other):
    return other * self**-1

  # expontential
  def exp(self):
    out = Node(math.exp(self.value), (self,))
    def _backprop():
      self.grad += out.value * out.grad
    out._backprop = _backprop
    return out
  
  # hyperbolic tangent
  def tanh(self):
    t = (math.exp(2*self.value)-1) / (math.exp(2*self.value)+1)
    out = Node(t, (self,))
    def _backprop():
      self.grad += (1-t**2) * out.grad
    out._backprop = _backprop
    return out

  # backpropagation
  def backprop(self):
    ordered = []
    visited = set()
    def topo(n): # topological search to order nodes
      if n not in visited:
        visited.add(n)
        for child in n._prev:
          topo(child)
        ordered.append(n)
    topo(self)
    self.grad = 1.0 # for final node
    for node in reversed(ordered):
      node._backprop()