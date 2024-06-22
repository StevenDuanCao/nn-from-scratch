# nn_from_scratch

An auto-differentiation engine and neural network architecture built with only a single dependency `numpy` to demonstrate foundational knowledge of ML. 

### Auto-differentiation
The `Node` class stores a value and gradient. The value can be manipulated through common operations. The gradient is automatically updated when backpropagation is called. Below is an example:
```python
from nn_from_scratch.node import Node

a = Node(2.0)
b = Node(-3.0)
c = Node(1.0)
d = a * b
e = d - c
print(e.value) # prints -7.0 (result of the forward pass)
e.backprop()
print(a.grad) # prints -3.0 (de/da)
print(b.grad) # prints 2.0 (de/db)
print(c.grad) # prints -1.0 (de/dc)
print(d.grad) # prints 1.0 (de/dd)
```

### Training a neural network
The `MLP` class provides the architecture to build a multi-layer perception model. Below is an example:
```python
# Sample of binary data with non-linear boundary
def boundary(x):
    return 7 * np.sin(0.5 * x)
n_samples = 200
X = np.random.uniform(-10, 10, (n_samples, 2))
Y = (X[:, 1] > boundary(X[:, 0]))

plt.figure()
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], color='red')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], color='blue')
```
```python
# initialize and train model
model = MLP([2, 16, 16, 1])
num_iter = 200
learning_rate = 1e-3
for k in range(num_iter):
    # forward pass
    ypred = [model(x) for x in X]
    loss = sum((y_a - y_p)**2 for y_a, y_p in zip(Y, ypred))
    # backpropagation
    for p in model.parameters():
        p.grad = 0 
    loss.backprop()
    # update parameters
    for p in model.parameters():
        p.value += -learning_rate * p.grad
    # track training loss
    if k % 50 == 0:
        print(f"Iter {k} | Loss {loss.value:.4f}")

# prints Iter 0: Loss 82.9024
# prints Iter 50: Loss 22.4701
# prints Iter 100: Loss 18.0410
# prints Iter 150: Loss 15.5451
```
```python
# visualize decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = 0.25
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.c_[xx.ravel(), yy.ravel()]
Z = np.array([model(z).value for z in Z])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', marker='o', s=20, cmap=plt.cm.coolwarm)
```
