import numpy as np
import pandas as pd
import random
import time
import math
import matplotlib.pyplot as plt

def sigmoid(x, derivative=False):
    if derivative:
        sig = sigmoid(x)
        return sig * (1.0 - sig)
    else:
        return 1.0 / (1.0 + np.exp(-x))

def tanh(x, derivative=False):
    if derivative:
        return 1 - tanh(x) ** 2
    else:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def leaky_relu(x, derivative=False):
    a = 0.01
    if derivative:
        result = np.ones(x.shape)
        result[x < 0] = a
        return result
    else:
        result = x[:]
        result[x < 0] = a
        return result

def elu(x, derivative=False):
    a = 0.1
    if derivative:
        result = np.ones(x.shape)
        result[x < 0] = a * np.exp(x[x < 0])
        return result
    else:
        x[x < 0] = a * (np.exp(x[x < 0]) - 1)
        return x

def relu(x, derivative=False):
    if derivative:
        result = np.ones(x.shape) 
        result[x < 0] = 0
        return result
    else:
        result = x
        result[x < 0] = 0
        return result

def softmax(x, derivative=False):
    if derivative:
        s = softmax(x).reshape(-1,1)
        return np.diag(np.diagflat(s) - np.dot(s, s.T)).reshape(x.shape)
    else:
        exps = np.exp(x + 1e-8)
        return exps / (np.sum(exps, axis=0) + 1e-8)

def mse_cost(predicted, target, derivative=False):
    if derivative:
        return -(target - predicted)
    else:
        return 0.5 * (target - predicted) ** 2

def cross_entropy(predicted, target, derivative=False):
    if derivative:
        result = -target * (1.0 / (predicted + 1e-8)) + (1 - target) * (1.0 / (1.0 - predicted + 1e-8))
        return result
    else:
        result = np.array(target)
        result[result == 1.0] = -np.log(predicted[result == 1.0] + 1e-8)
        result[result == 0.0] = -np.log(1.0 - predicted[result == 0.0] + 1e-8)
        return result

def split_data(data, test_ratio):
        np.random.shuffle(data)
        test_size = int(len(data) * test_ratio)
        train_data, test_data = np.split(data, [len(data) - test_size])
        return train_data, test_data

class NeuralNetwork:

    layers = []
    mini_batch_size = 2

    curr_epoch = 0

    accuracy_arr = []
    loss_arr = []
    curr_epoch_accuracy = 0
    curr_epoch_loss = 0

    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def add_layer(self, n, activation_func, w = None, b = None):
        if not len(self.layers):
            self.layers.append(NeuralLayer(n, self.n_inputs, activation_func, w, b))
        else:
            self.layers.append(NeuralLayer(n, self.layers[-1].n, activation_func, w, b))

    def build_network(self, activation_func, cost_func, w = None, b = None):
        self.add_layer(self.n_outputs, activation_func, w, b)
        self.layers[-1].cost = cost_func
        for layer in self.layers:
            layer.set_wb()
        
    
    def forward(self, input):
        new_input = np.array(input)
        for layer in self.layers:
            layer.prev = new_input
            layer.z = np.dot(layer.weights.T, new_input) + layer.bias
            new_input = layer.activate_neurons(layer.z)
            layer.a = new_input
        return new_input

    def backpropagate(self, predicted, target):
        if self.n_outputs > 1:
            predicted_map = np.zeros(predicted.shape)
            predicted_map[predicted == np.max(predicted, axis=0)] = 1
            self.curr_epoch_accuracy += np.sum(predicted_map * target)
        self.curr_epoch_loss += np.sum(self.layers[-1].cost(predicted, target))

        for i in reversed(range(len(self.layers))):
            if i == (len(self.layers) - 1):
                self.layers[i].error = self.layers[i].cost(predicted, target, derivative=True) * \
                self.layers[i].activation_func(self.layers[i].z, derivative=True)
            else:
                if self.gd_optimizer == 'nag':
                    self.layers[i].error = np.dot(self.layers[i + 1].weights - self.gamma1 * self.momentum_w[i + 1], self.layers[i + 1].error) * \
                    self.layers[i].activation_func(self.layers[i].z, derivative=True)
                else:
                    self.layers[i].error = np.dot(self.layers[i + 1].weights, self.layers[i + 1].error) * \
                    self.layers[i].activation_func(self.layers[i].z, derivative=True)
            self.layers[i].dw = np.dot(self.layers[i].prev, self.layers[i].error.T) / predicted.shape[1]
            self.layers[i].db = np.sum(self.layers[i].error) / predicted.shape[1]

    def update_model(self):
        for i in range(len(self.layers)):
            if self.gd_optimizer == None:
                self.delta_w[i] = self.lr * self.layers[i].dw
                self.delta_b[i] = self.lr * self.layers[i].db
            elif self.gd_optimizer == 'momentum' or self.gd_optimizer == 'nag':
                self.momentum_w[i] = self.gamma1 * self.momentum_w[i] + self.lr * self.layers[i].dw
                self.momentum_b[i] = self.gamma1 * self.momentum_b[i] + self.lr * self.layers[i].db
                self.delta_w[i] = self.momentum_w[i]
                self.delta_b[i] = self.momentum_b[i]
            elif self.gd_optimizer == 'adagrad':
                self.cache_w[i] += self.layers[i].dw ** 2
                self.delta_w[i] = self.lr * self.layers[i].dw / (np.sqrt(self.cache_w[i]) + 1e-8)
                self.cache_b[i] += self.layers[i].db ** 2
                self.delta_b[i] = self.lr * self.layers[i].db / (np.sqrt(self.cache_b[i]) + 1e-8)
            elif self.gd_optimizer == 'rmsprop':
                self.cache_w[i] = self.gamma1 * self.cache_w[i] + (1 - self.gamma1) * self.layers[i].dw ** 2
                self.delta_w[i] = self.lr * self.layers[i].dw / (np.sqrt(self.cache_w[i]) + 1e-8)
                self.cache_b[i] = self.gamma1 * self.cache_b[i] + (1 - self.gamma1) * self.layers[i].db ** 2
                self.delta_b[i] = self.lr * self.layers[i].db / (np.sqrt(self.cache_b[i]) + 1e-8)
            elif self.gd_optimizer == 'adadelta':
                self.cache_w[i] = self.gamma1 * self.cache_w[i] + (1 - self.gamma1) * self.layers[i].dw ** 2
                self.x_w[i] = self.gamma1 * self.x_w[i] + (1 - self.gamma1) * self.delta_w[i] ** 2
                self.delta_w[i] = np.sqrt(self.x_w[i] + 1e-8) * self.layers[i].dw / np.sqrt(self.cache_w[i] + 1e-8)
                self.cache_b[i] = self.gamma1 * self.cache_b[i] + (1 - self.gamma1) * self.layers[i].db ** 2
                self.x_b[i] = self.gamma1 * self.x_b[i] + (1 - self.gamma1) * self.delta_b[i] ** 2
                self.delta_b[i] = np.sqrt(self.x_b[i] + 1e-8) * self.layers[i].db / np.sqrt(self.cache_b[i] + 1e-8)
            elif self.gd_optimizer == 'adam':
                self.m_w[i] = self.gamma1 * self.m_w[i] + (1 - self.gamma1) * self.layers[i].dw 
                self.u_w[i] = self.gamma2 * self.u_w[i] + (1 - self.gamma2) * self.layers[i].dw ** 2
                self.m_ws[i] = self.m_w[i] / (1 - self.gamma1 ** (self.curr_epoch + 1))
                self.u_ws[i] = self.u_w[i] / (1 - self.gamma2 ** (self.curr_epoch + 1))
                self.delta_w[i] = self.lr * self.m_ws[i] / np.sqrt(self.u_ws[i] + 1e-8)

                self.m_b[i] = self.gamma1 * self.m_b[i] + (1 - self.gamma1) * self.layers[i].db
                self.u_b[i] = self.gamma2 * self.u_b[i] + (1 - self.gamma2) * self.layers[i].db ** 2
                self.m_bs[i] = self.m_b[i] / (1 - self.gamma1 ** (self.curr_epoch + 1))
                self.u_bs[i] = self.u_b[i] / (1 - self.gamma2 ** (self.curr_epoch + 1))
                self.delta_b[i] = self.lr * self.m_bs[i] / np.sqrt(self.u_bs[i] + 1e-8)

            self.layers[i].weights -= self.delta_w[i]
            self.layers[i].bias -= self.delta_b[i]

    def sgd(self, inputs, outputs):
        time1 = time.time()
        for epoch in range(self.n_epoch):
            self.curr_epoch_accuracy = 0
            self.curr_epoch_loss = 0
            self.curr_epoch = epoch
            for i in np.random.choice(len(inputs), len(inputs), replace=False):
                if self.n_outputs > 1:
                    output_label = np.zeros((self.n_outputs, 1))
                    output_label[int(outputs[i])] = 1
                else:
                    output_label = outputs[i]
                predicted = self.forward(inputs[i].reshape(inputs[i].shape[0], -1))
                self.backpropagate(predicted, output_label)
                self.update_model()
            self.accuracy_arr.append(self.curr_epoch_accuracy / inputs.shape[0])
            self.loss_arr.append(self.curr_epoch_loss / inputs.shape[0])
        time2 = time.time()
        print('Stochastic GD {} ---- {}'.format(self.n_epoch, (time2 - time1) * 1000.0))

    def bgd(self, inputs, outputs):
        time1 = time.time()
        for epoch in range(self.n_epoch):
            self.curr_epoch_accuracy = 0
            self.curr_epoch_loss = 0
            self.curr_epoch = epoch
            if self.n_outputs > 1:
                output_label = np.zeros((self.n_outputs, len(outputs)))
                for i in range(len(outputs)):
                    output_label[int(outputs[i])][i] = 1
            else:
                output_label = outputs
            predicted = self.forward(inputs.T)
            self.backpropagate(predicted, output_label)
            self.update_model()
            self.accuracy_arr.append(self.curr_epoch_accuracy / inputs.shape[0])
            self.loss_arr.append(self.curr_epoch_loss / inputs.shape[0])
        time2 = time.time()
        print('Batch GD {} ---- {}'.format(self.n_epoch, (time2 - time1) * 1000.0))

    def mbgd(self, inputs, outputs, mini_batch_size=2):
        time1 = time.time()
        mini_batches = self.get_mini_batches(inputs, outputs)
        for epoch in range(self.n_epoch):    
            self.curr_epoch_accuracy = 0
            self.curr_epoch_loss = 0
            self.curr_epoch = epoch
            for batch in mini_batches:
                batch_inputs = batch[0]
                batch_outputs = batch[1]
                if self.n_outputs > 1:
                    output_label = np.zeros((self.n_outputs, len(batch[1])))
                    for i in range(len(batch[1])):
                        output_label[int(batch[1][i])][i] = 1
                else:
                    output_label = batch_outputs
                predicted = self.forward(batch_inputs.T)
                self.backpropagate(predicted, output_label)
                self.update_model()
            self.accuracy_arr.append(self.curr_epoch_accuracy / inputs.shape[0])
            self.loss_arr.append(self.curr_epoch_loss / inputs.shape[0])
        time2 = time.time()
        print('Mini-Batch GD {} ---- {}'.format(self.n_epoch, (time2 - time1) * 1000.0))

    def get_mini_batches(self, inputs, outputs):
        np.random.seed(1)
        m = inputs.shape[0]
        mini_batches = []
        permutation = list(np.random.permutation(m))
        shuffled_inputs = inputs[permutation]
        shuffled_outputs = outputs[permutation]
        n_complete_minibatches = math.floor(m / self.mini_batch_size)

        for i in range(n_complete_minibatches):
            mini_batch_inputs = shuffled_inputs[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
            mini_batch_outputs = shuffled_outputs[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
            mini_batch = (mini_batch_inputs, mini_batch_outputs)
            mini_batches.append(mini_batch)

        if m % self.mini_batch_size != 0:
            mini_batch_inputs = shuffled_inputs[n_complete_minibatches * self.mini_batch_size]
            mini_batch_outputs = shuffled_outputs[n_complete_minibatches * self.mini_batch_size]
            mini_batch = (mini_batch_inputs.reshape(1, -1), mini_batch_outputs.reshape(1, -1))
            mini_batches.append(mini_batch)

        return mini_batches

    def train(self, inputs, outputs, gd_func, n_epoch=1, lr=0.001, gd_optimizer=None):
        self.n_epoch = n_epoch
        self.lr = lr
        self.gd_optimizer = gd_optimizer

        self.delta_w = np.array([np.zeros_like(self.layers[i].weights) for i in range(len(self.layers))])
        self.delta_b = np.zeros((len(self.layers), 1))

        if gd_optimizer == 'momentum' or gd_optimizer == 'nag':
            self.momentum_w = np.array([np.zeros_like(self.layers[i].weights) for i in range(len(self.layers))])
            self.momentum_b = np.zeros((len(self.layers), 1))
        elif gd_optimizer == 'adagrad' or gd_optimizer == 'rmsprop' or gd_optimizer == 'adadelta':
            self.cache_w = np.array([np.zeros_like(self.layers[i].weights) for i in range(len(self.layers))])
            self.cache_b = np.zeros((len(self.layers), 1))
            if gd_optimizer == 'adadelta':
                self.x_w = np.array([np.zeros_like(self.layers[i].weights) for i in range(len(self.layers))])
                self.x_b = np.zeros((len(self.layers), 1))
        elif gd_optimizer == 'adam':
            self.m_w =np.array([np.zeros_like(self.layers[i].weights) for i in range(len(self.layers))])
            self.u_ws = np.array([np.zeros_like(self.layers[i].weights) for i in range(len(self.layers))])
            self.m_ws =np.array([np.zeros_like(self.layers[i].weights) for i in range(len(self.layers))])
            self.u_w =np.array([np.zeros_like(self.layers[i].weights) for i in range(len(self.layers))])
            self.m_b = np.zeros((len(self.layers), 1))
            self.u_bs = np.zeros((len(self.layers), 1))
            self.m_bs = np.zeros((len(self.layers), 1))
            self.u_b = np.zeros((len(self.layers), 1))
            self.t = 0

        self.gamma1 = .9
        self.gamma2 = .999
        gd_func(inputs, outputs)

    def test(self, inputs, outputs):
        predicted = self.forward(inputs.T)
        if self.n_outputs > 1:
            output_label = np.zeros((self.n_outputs, len(outputs)))
            for i in range(len(outputs)):
                output_label[int(outputs[i])][i] = 1
        else:
            output_label = outputs
        if self.n_outputs > 1:
            predicted_map = np.zeros(predicted.shape)
            predicted_map[predicted == np.max(predicted, axis=0)] = 1
            self.test_accuracy = np.sum(predicted_map * output_label) / inputs.shape[0]
            for i in range(predicted.T.shape[0]):
                print('Target: {} ----- Predicted: {} {}'.format(outputs[i], predicted.T[i], predicted_map.T[i]))
            print(self.test_accuracy)


class NeuralLayer:

    weights = np.array([])
    bias = 1
    
    def __init__(self, n, n_prev, activation_func, w = None, b = None):
        self.n = n
        self.n_prev = n_prev
        self.activation_func = activation_func

    def set_wb(self, w = None, b = None):
        #np.random.seed(1)
        self.weights = w if w else np.random.uniform(low=-1.0, high=1.0, size=(self.n_prev, self.n))
        self.bias = b if b else 1

    def activate_neurons(self, input):
        z = self.activation_func(input)
        return z


def run():
    dataset = pd.read_csv('C:\Development\PythonProjects\ML\my_nn\seeds.csv')
    df = pd.DataFrame(dataset)
    dv = df.values
    dv[:,:-1] = (dv[:,:-1] - dv[:,:-1].min()) / (dv[:,:-1].max() - dv[:,:-1].min())
    dv[:,-1] -= 1
    train_set, test_set = split_data(dv, 0.2)
    train_inputs = train_set[:,:-1]
    train_outputs = train_set[:,-1:]

    network = NeuralNetwork(train_inputs.shape[1], len(np.unique(train_outputs)))
    network.add_layer(25, relu)
    network.add_layer(15, relu)
    network.build_network(softmax, cross_entropy)

    network.mini_batch_size = 10
    network.train(train_inputs, train_outputs, network.sgd, 100, 0.001, gd_optimizer='adam')
    #plt.plot(network.loss_arr)
    plt.plot(network.accuracy_arr)
    plt.show()

    test_inputs = test_set[:,:-1]
    test_outputs = test_set[:,-1:]

    network.test(test_inputs, test_outputs)

if __name__ == '__main__':
    run()