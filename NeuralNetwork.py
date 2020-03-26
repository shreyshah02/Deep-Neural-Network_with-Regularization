import numpy as np
import matplotlib.pyplot as plt

class DeepNeuralNetwork:
    def __init__(self, x_train, y_train, x_test, y_test, alpha, layers=2, neurons=(8, 1),
                 activations=('ReLU', 'sigmoid'), val=False, reg=False, lmbda=0, drop=False, kp=1):
        if val:
            split = int(0.8*x_train.shape[1])
            self.x_val = x_train[:, split:]
            self.y_val = y_train[:, split:]
            
        else:
            split = x_train.shape[1]
            self.x_val = None
            self.y_val = None
        self.val = val
        self.x_train = x_train[:, :split]
        self.y_train = y_train[:, :split]
        self.x_test = x_test
        self.y_test = y_test
        self.alpha = alpha
        self.neurons = neurons
        self.W = None
        self.b = None
#         self.Z = None
#         self.A = None
        self.activations = activations
        self.layers = layers
        self.m = self.x_train.shape[1]
        self.reg = reg
        self. lmbda = lmbda
        self.drop = drop
        self.kp = kp
    
    
    def act_sigmoid(self, z):
        A = 1/(1+np.exp(-z))
        return A


    def act_tanh(self, z):
        A = np.tanh(z)
        return A


    def act_relu(self, z):
        A = np.maximum(0, z)
        return A
    
    
    def Activation_function(self, z, act):
        if act == 'ReLU':
            A = self.act_relu(z)
        elif act == 'tanh':
            A = self.act_tanh(z)
        elif act == 'sigmoid':
            A = self.act_sigmoid(z)
        return A

    
    def act_gradient(self, A, act = 'ReLU'):
        dg = np.zeros_like(A)
        if act == 'ReLU':
            dg[A>0] = 1
        elif act == 'sigmoid':
            dg = A*(1 - A)
        elif act == 'tanh':
            dg = 1 - np.square(A)

        return dg
    
    
    def initialization(self):
        W = []
        b = []
        n = self.x_train.shape[0]
        for i in range(self.layers):
            w = np.random.randn(self.neurons[i], n) / n**0.5 # *0.01
#             bias = np.random.randn(self.neurons[i], 1)* 0.1
            bias = np.zeros((self.neurons[i], 1))
#             w = np.random.random((self.neurons[i], n))*0.01
#             bias = np.random.random((self.neurons[i], 1)) * 0.01
            W.append(w)
            b.append(bias)
#             print('bias[{}] = '.format(i), bias)
            n = self.neurons[i]
        self.W = W
        self.b = b
    
    
    def ForwardProp(self, x):#(self, x, y)
        Z = []
        A = []
        a = x
        # activation of the 0th layer is the training example
        # z of 1st layer is at Z[0]
        # activation of 1st layer is at A[1]
        A.append(a)
        for i in range(self.layers):
            z_l = np.dot(self.W[i], a) + self.b[i]
            a_l = self.Activation_function(z_l, self.activations[i])
            Z.append(z_l)
            A.append(a_l)
            a = a_l
#         self.Z = Z
#         self.A = A
        return Z, A
    
    def ForwardProp_Drop(self, x):  # (self, x, y)
        Z = []
        A = []
        D = []
        a = x
        # activation of the 0th layer is the training example
        # z of 1st layer is at Z[0]
        # activation of 1st layer is at A[1]
        A.append(a)
        for i in range(self.layers):
            z_l = np.dot(self.W[i], a) + self.b[i]
            a_l = self.Activation_function(z_l, self.activations[i])
            if i != self.layers - 1:
                d_l = np.random.randn(a_l.shape[0], a_l.shape[1])
                d_l = (d_l<self.kp)
                a_l = a_l * d_l
                a_l = a_l/self.kp
                D.append(d_l)
            Z.append(z_l)
            A.append(a_l)

            a = a_l
        #         self.Z = Z
        #         self.A = A
        return Z, A, D
    
    def BackProp(self, Z, A):
#         dA = - self.y_train/A[-1] + (1 - self.y_train)/(1 - A[-1])
        dZ = Z.copy()
        dW = self.W.copy()
        db = self.b.copy()
        a = A[-1]
        dZ[-1] = A[-1] - self.y_train
        for i in range(len(dZ) - 1, -1, -1):
#             dZ[i] = dA*self.act_gradient(a, self.activations[i])
            dW[i] = dZ[i].dot(A[i].T) / self.m
            if self.reg:
                dW[i] += self.lmbda*self.W[i] / self.m
            db[i] = np.sum(dZ[i], axis = 1) / self.m
            db[i] = np.reshape(db[i], (-1, 1))
#             print('Shape of db[{}] = '.format(i),db[i].shape)
            dA = np.dot(self.W[i].T, dZ[i])
            a = A[i]
            if i != 0:
                dZ[i - 1] = dA*self.act_gradient(a, self.activations[i - 1])
            
        return dW, db
    
    def BackProp_Drop(self, Z, A, D):
        #         dA = - self.y_train/A[-1] + (1 - self.y_train)/(1 - A[-1])
        dZ = Z.copy()
        dW = self.W.copy()
        db = self.b.copy()
        a = A[-1]
        dZ[-1] = A[-1] - self.y_train
        for i in range(len(dZ) - 1, -1, -1):
            #             dZ[i] = dA*self.act_gradient(a, self.activations[i])
            dW[i] = dZ[i].dot(A[i].T) / self.m
            if self.reg:
                dW[i] += self.lmbda * self.W[i] / self.m
            db[i] = np.sum(dZ[i], axis=1) / self.m
            db[i] = np.reshape(db[i], (-1, 1))
            #             print('Shape of db[{}] = '.format(i),db[i].shape)
            dA = np.dot(self.W[i].T, dZ[i])
            a = A[i]
            if i != 0:
                dZ[i - 1] = dA * self.act_gradient(a, self.activations[i - 1])
                dA = dA*D[i-1]
                dA = dA / self.kp
        return dW, db

    def train_model(self, epochs):
        self.initialization()
        costs = []
        for epoch in range(epochs):
            if self.drop:
                if epoch == 1:
                    print('In drop')
                Z, A, D = self.ForwardProp_Drop(self.x_train)
                dW, db = self.BackProp_Drop(Z, A, D)
            else:
                if epoch == 1:
                    print('Not In drop')
                Z, A = self.ForwardProp(self.x_train)
                dW, db = self.BackProp(Z, A)
            cost = self.ComputeCost(A[-1], self.y_train)
            costs.append(cost)
#             plt.plot(epoch, cost, 'bx')
#             plt.title('Learning Curve')
#             plt.xlabel('iterations')
#             plt.ylabel('Cost')
            
            for i in range(self.layers):
#                 print('Update Iteration {}'.format(i))
                self.W[i] -= self.alpha*dW[i]
                self.b[i] -= self.alpha*db[i]
        e = range(epochs)
        plt.figure()
        plt.plot(e, costs)
        plt.xlabel('epochs')
        plt.ylabel('Cost function')
        plt.title('Learning rate')
        Z_f, A_f = self.ForwardProp(self.x_train)
        cost_f = self.ComputeCost(A_f[-1], self.y_train)
        print('Cost after Training = ', cost_f)
        A_t = np.around(A_f[-1])
        accuracy = np.mean(A_t == self.y_train)
        print('Accuracy on Training Data = ', accuracy)
        return self.W[i], self.b[i], A_f[-1]
    
    def test_model(self, x):#(self, x, y)
        Z, A = self.ForwardProp(x)
#         cost = self.ComputeCost(A[-1], y)
#         A_t = np.around(A[-1])
#         accuracy = np.mean(A_t == y)
#         print('The cost on test set = {}'.format(cost))
#         print('The accuracy on test set = {}'.format(accuracy))
        return np.ravel(np.around(A[-1]))
    
    def ComputeCost(self, A, y):
        cost = - np.nansum(y*np.log(A) + (1-y)*np.log(1 - A)) / y.shape[1]
        if self.reg:
            regularization = 0
            for i in range(self.layers):
                regularization += np.sum(np.square(self.W[i]))
            regularization = (self.lmbda*regularization)/(2 * y.shape[1])
            cost += regularization
        return cost
    
    def Validate_model(self):
        Z, A = self.ForwardProp(self.x_val)
        cost = self.ComputeCost(A[-1], self.y_val)
        A_v = np.around(A[-1])
        accuracy = np.mean(A_v == self.y_val)
        print('The cost on Validation set for alpha:{} = {}'.format(self.alpha, cost))
        print('The accuracy on Validation set = {}'.format(accuracy))
        return cost


# class DeepNeuralNetwork_Dropout:
#     def __init__(self, x_train, y_train, x_test, y_test, alpha, layers=2, neurons=(8, 1),
#                  activations=('ReLU', 'sigmoid'), val=False, reg=False, lmbda=0, drop=False, kp=1):
#         if val:
#             split = int(0.8 * x_train.shape[1])
#             self.x_val = x_train[:, split:]
#             self.y_val = y_train[:, split:]

#         else:
#             split = x_train.shape[1]
#             self.x_val = None
#             self.y_val = None
#         self.val = val
#         self.x_train = x_train[:, :split]
#         self.y_train = y_train[:, :split]
#         self.x_test = x_test
#         self.y_test = y_test
#         self.alpha = alpha
#         self.neurons = neurons
#         self.W = None
#         self.b = None
#         #         self.Z = None
#         #         self.A = None
#         self.activations = activations
#         self.layers = layers
#         self.m = self.x_train.shape[1]
#         self.reg = reg
#         self.lmbda = lmbda
#         self.drop = drop
#         self.kp = kp

#     def act_sigmoid(self, z):
#         A = 1 / (1 + np.exp(-z))
#         return A

#     def act_tanh(self, z):
#         A = np.tanh(z)
#         return A

#     def act_relu(self, z):
#         A = np.maximum(0, z)
#         return A

#     def Activation_function(self, z, act):
#         if act == 'ReLU':
#             A = self.act_relu(z)
#         elif act == 'tanh':
#             A = self.act_tanh(z)
#         elif act == 'sigmoid':
#             A = self.act_sigmoid(z)
#         return A

#     def act_gradient(self, A, act='ReLU'):
#         dg = np.zeros_like(A)
#         if act == 'ReLU':
#             dg[A > 0] = 1
#         elif act == 'sigmoid':
#             dg = A * (1 - A)
#         elif act == 'tanh':
#             dg = 1 - np.square(A)

#         return dg

#     def initialization(self):
#         W = []
#         b = []
#         n = self.x_train.shape[0]
#         for i in range(self.layers):
#             w = np.random.randn(self.neurons[i], n) / n**0.5  # *0.01
# #             bias = np.random.randn(self.neurons[i], 1) * 0.1
#             bias = np.zeros((self.neurons[i], 1))
#             #             w = np.random.random((self.neurons[i], n))*0.01
#             #             bias = np.random.random((self.neurons[i], 1)) * 0.01
#             W.append(w)
#             b.append(bias)
#             #             print('bias[{}] = '.format(i), bias)
#             n = self.neurons[i]
#         self.W = W
#         self.b = b

#     def ForwardProp(self, x):  # (self, x, y)
#         Z = []
#         A = []
#         a = x
#         # activation of the 0th layer is the training example
#         # z of 1st layer is at Z[0]
#         # activation of 1st layer is at A[1]
#         A.append(a)
#         for i in range(self.layers):
#             z_l = np.dot(self.W[i], a) + self.b[i]
#             a_l = self.Activation_function(z_l, self.activations[i])
#             Z.append(z_l)
#             A.append(a_l)
#             a = a_l
#         #         self.Z = Z
#         #         self.A = A
#         return Z, A

#     def ForwardProp_Drop(self, x):  # (self, x, y)
#         Z = []
#         A = []
#         D = []
#         a = x
#         # activation of the 0th layer is the training example
#         # z of 1st layer is at Z[0]
#         # activation of 1st layer is at A[1]
#         A.append(a)
#         for i in range(self.layers):
#             z_l = np.dot(self.W[i], a) + self.b[i]
#             a_l = self.Activation_function(z_l, self.activations[i])
#             if i != self.layers - 1:
#                 d_l = np.random.randn(a_l.shape[0], a_l.shape[1])
#                 d_l = (d_l<self.kp)
#                 a_l = a_l * d_l
#                 a_l = a_l/self.kp
#                 D.append(d_l)
#             Z.append(z_l)
#             A.append(a_l)

#             a = a_l
#         #         self.Z = Z
#         #         self.A = A
#         return Z, A, D

#     def BackProp(self, Z, A, D):
#         #         dA = - self.y_train/A[-1] + (1 - self.y_train)/(1 - A[-1])
#         dZ = Z.copy()
#         dW = self.W.copy()
#         db = self.b.copy()
#         a = A[-1]
#         dZ[-1] = A[-1] - self.y_train
#         for i in range(len(dZ) - 1, -1, -1):
#             #             dZ[i] = dA*self.act_gradient(a, self.activations[i])
#             dW[i] = dZ[i].dot(A[i].T) / self.m
#             if self.reg:
#                 dW[i] += self.lmbda * self.W[i] / self.m
#             db[i] = np.sum(dZ[i], axis=1) / self.m
#             db[i] = np.reshape(db[i], (-1, 1))
#             #             print('Shape of db[{}] = '.format(i),db[i].shape)
#             dA = np.dot(self.W[i].T, dZ[i])
#             a = A[i]
#             if i != 0:
#                 dZ[i - 1] = dA * self.act_gradient(a, self.activations[i - 1])
#                 dA = dA*D[i-1]
#                 dA = dA / self.kp
#         return dW, db

#     def train_model(self, epochs):
#         self.initialization()
#         costs = []
#         for epoch in range(epochs):
#             Z, A, D = self.ForwardProp_Drop(self.x_train)
#             cost = self.ComputeCost(A[-1], self.y_train)
#             costs.append(cost)
# #             plt.plot(epoch, cost, 'bx')
# #             plt.title('Learning Curve')
# #             plt.xlabel('iterations')
# #             plt.ylabel('Cost')
#             dW, db = self.BackProp(Z, A, D)
#             for i in range(self.layers):
#                 #                 print('Update Iteration {}'.format(i))
#                 self.W[i] -= self.alpha * dW[i]
#                 self.b[i] -= self.alpha * db[i]
#         e = range(epochs)
#         plt.figure()
#         plt.plot(e, costs)
#         plt.xlabel('epochs')
#         plt.ylabel('Cost function')
#         plt.title('Learning rate')
#         Z_f, A_f = self.ForwardProp(self.x_train)
#         cost_f = self.ComputeCost(A_f[-1], self.y_train)
#         print('Cost after Training = ', cost_f)
#         A_t = np.around(A_f[-1])
#         accuracy = np.mean(A_t == self.y_train)
#         print('Accuracy on Training Data = ', accuracy)
#         return self.W[i], self.b[i], A_f[-1]

#     def test_model(self, x):  # (self, x, y)
#         Z, A = self.ForwardProp(x)
#         #         cost = self.ComputeCost(A[-1], y)
#         #         A_t = np.around(A[-1])
#         #         accuracy = np.mean(A_t == y)
#         #         print('The cost on test set = {}'.format(cost))
#         #         print('The accuracy on test set = {}'.format(accuracy))
#         return np.ravel(A[-1])

#     def ComputeCost(self, A, y):
#         cost = - np.nansum(y * np.log(A) + (1 - y) * np.log(1 - A)) / y.shape[1]
#         if self.reg:
#             regularization = 0
#             for i in range(self.layers):
#                 regularization += np.sum(np.square(self.W[i]))
#             regularization = (self.lmbda * regularization) / (2 * y.shape[1])
#             cost += regularization
#         return cost

#     def Validate_model(self):
#         Z, A = self.ForwardProp(self.x_val)
#         cost = self.ComputeCost(A[-1], self.y_val)
#         A_v = np.around(A[-1])
#         accuracy = np.mean(A_v == self.y_val)
#         print('The cost on Validation set for alpha:{} = {}'.format(self.alpha, cost))
#         print('The accuracy on Validation set = {}'.format(accuracy))
#         return cost