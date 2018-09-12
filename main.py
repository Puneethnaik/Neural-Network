import nn
import numpy as np
obj = nn.NeuralNetwork(input_dim=2, layer_size=[2, 3, 5, 1], num_classes=2, num_layers=4)
obj.initialize()
X = []
x = np.array([0, 0])
x = np.reshape(x,newshape=[2, 1])
X.append(x)

x = np.array([1, 0])
x = np.reshape(x,newshape=[2, 1])
X.append(x)

x = np.array([0, 1])
x = np.reshape(x,newshape=[2, 1])
X.append(x)

x = np.array([1, 1])
x = np.reshape(x,newshape=[2, 1])
X.append(x)

X = np.array(X)

y = np.array([0, 0, 0, 1])

# print(X.shape, X)
# obj.forward_prop(X)
# obj.backprop(X)
obj.train(X, y, 0.1, 0.2, 800)
print(obj.predict(X))

print(obj.sigmoid(np.array([10**-20])))