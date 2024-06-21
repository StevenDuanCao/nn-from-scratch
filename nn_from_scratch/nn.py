import random
from node import Node

class Neuron:
  def __init__(self, n_in):
    # initalizing weights and bias
    self.w = [Node(random.uniform(-1, 1)) for _ in range(n_in)]
    self.b = Node(random.uniform(-1, 1))
  
  def __call__(self, x):
    z = sum((w_i * x_i for w_i, x_i in zip(self.w, x)), self.b) # linear function
    out = z.tanh() # activation function
    return out
  
  def parameters(self):
    return self.w + [self.b]

class Layer:
  def __init__(self, n_in, n_out):
    self.neurons = [Neuron(n_in) for _ in range(n_out)] 
  
  def __call__(self, x):
    out = [n(x) for n in self.neurons]
    return out[0] if len(out)==1 else out # single value for final output
  
  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]

class MLP:
  def __init__(self, layer_dim):
    self.layers = [Layer(layer_dim[i], layer_dim[i+1]) for i in range(len(layer_dim)-1)]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for l in self.layers for p in l.parameters()]