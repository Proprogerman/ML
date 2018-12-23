import neural_network
from neural_network import NeuralNetwork, NNLayer, relu, softmax, cross_entropy, sigmoid, split_data, tanh
import numpy as np
import pandas as pd
import random
import time
import math
import random
import matplotlib.pyplot as plt


def map_data(data):
    if type(data).__module__ == 'numpy':
        vocab = np.unique(data)
    else:
        vocab = set(data)
    global vocab_size
    vocab_size = len(vocab)
    global data_to_int
    data_to_int = {o:i for i, o in enumerate(vocab)}
    global int_to_data
    int_to_data = {i:o for i, o in enumerate(vocab)}


class RNN(NeuralNetwork):

    def __init__(self, n_inputs, n_outputs):
        super().__init__(n_inputs, n_outputs)

    def add_layer(self, n, activation_func, w = None, b = None, t=None):
        if not len(self.layers):
            self.layers.append(RNNLayer(n, self.n_inputs, activation_func, w, b, t))
        else:
            self.layers.append(RNNLayer(n, self.layers[-1].n, activation_func, w, b, t))

    def prepare_for_train(self):
        for layer in self.layers:
            layer.prepare_for_train('layer_params')
            if type(layer) is RNNLayer:
                for timestep in layer.timesteps:
                    timestep.prepare_for_train('layer_params', 'timestep_params', timestep_params=(layer.n, 1))

    def update_model(self):
        for layer in self.layers:
            if type(layer) is RNNLayer:
                for i in range(len(layer.timesteps)):
                    timestep_dw, timestep_db = layer.timesteps[i].get_optimized_diff('timestep_params')
                    layer.timesteps_conn[math.floor(abs(i - 0.5))] -= timestep_dw
                layer_dw, layer_db = layer.timesteps[-1].get_optimized_diff('layer_params')
                layer.timesteps[-1].weights -= layer_dw
                layer.timesteps[-1].bias -= layer_db
            else:
                layer_dw, layer_db = layer.get_optimized_diff('layer_params')
                layer.weights -= layer_dw
                layer.bias -= layer_db

    def sgd(self, inputs, outputs):
        time1 = time.time()
        global curr_epoch, curr_data_point
        for epoch in range(self.n_epoch):
            self.curr_epoch_accuracy = 0
            self.curr_epoch_loss = 0
            curr_epoch = epoch
            for i in range(len(inputs)):
                curr_data_point = i
                predicted = self.forward(inputs[i].reshape(inputs[i].shape[0], -1))
                self.backpropagate(predicted, outputs[i].reshape(outputs[i].shape[0], -1))
                self.update_model()
            self.accuracy_arr.append(self.curr_epoch_accuracy / inputs.shape[0])
            self.loss_arr.append(self.curr_epoch_loss / inputs.shape[0])
        curr_data_point = len(inputs) - 1
        time2 = time.time()
        print('Stochastic GD {} ---- {}'.format(self.n_epoch, (time2 - time1) * 1000.0))



class RNNLayer(NNLayer):

    def __init__(self, n, n_prev, activation_func, w=None, b=None, t=None):
        super().__init__(n, n_prev, activation_func, w=w, b=b)
        self.t = t

    def set_wb(self, w=None, b=None, t=None):
        #np.random.seed(1)
        self.weights = w if w else np.random.uniform(low=-1.0, high=1.0, size=(self.n_prev, self.n))
        self.timesteps = t if t else []
        self.timesteps_conn = []
        self.bias = b if b else 1
    
    def forward(self, input):
        global curr_data_point
        prev_input = np.zeros((self.n, 1))
        self.prev = np.copy(input)
        if len(self.timesteps_conn) and curr_data_point > 0:
            prev_input = self.timesteps_conn[math.floor(abs(curr_data_point - 0.5))] * self.timesteps[curr_data_point - 1].a
        if len(self.timesteps) < neural_network.data_size:
            self.timesteps.append(NNLayer(self.n, self.n_prev, self.activation_func))
            self.timesteps[-1].set_wb()
            self.timesteps[-1].prepare_for_train('layer_params', 'timestep_params', timestep_params=(self.n, 1))
            self.weights = self.timesteps[-1].weights
            if len(self.timesteps_conn) < neural_network.data_size - 1:
                self.timesteps_conn.append(np.random.uniform(low=-1.0, high=1.0, size=(self.n, 1)))
        self.timesteps[curr_data_point].prev = self.prev
        self.timesteps[curr_data_point].prev_timestep = prev_input
        self.timesteps[curr_data_point].z = np.dot(self.timesteps[curr_data_point].weights.T, input) + self.bias
        self.z = self.timesteps[curr_data_point].z + prev_input
        self.timesteps[curr_data_point].a = self.activate_neurons(self.z) 
        return self.timesteps[curr_data_point].a

    def backward(self, prev_err):
        curr_timestep_params = getattr(self.timesteps[-1], 'layer_params')
        layer_params = getattr(self, 'layer_params')
        curr_timestep_params['error'] = np.zeros_like(self.z)
        curr_timestep_params['db'] = 0
        for i in reversed(range(len(self.timesteps))):
            timestep_params = getattr(self.timesteps[i], 'timestep_params')
            timestep_params['error'] = prev_err * self.activation_func(self.timesteps[i].z, derivative=True)
            curr_timestep_params['error'] += timestep_params['error']
            timestep_params['dw'] = self.timesteps[i].prev_timestep * timestep_params['error'] / prev_err.shape[1]
            timestep_params['db'] = np.sum(timestep_params['error']) / prev_err.shape[1]
            curr_timestep_params['db'] += timestep_params['db']
        curr_timestep_params['dw'] = np.dot(self.timesteps[-1].prev, curr_timestep_params['error'].T) / prev_err.shape[1]
        layer_params['error'] = curr_timestep_params['error']
        #curr_timestep_params['db'] = np.sum(curr_timestep_params['error']) / prev_err.shape[1]


def run():

    data = '123456'

    neural_network.map_data(data)
    vocab_size = neural_network.vocab_size
    data_to_int = neural_network.data_to_int
    int_to_data = neural_network.int_to_data

    network = RNN(vocab_size, vocab_size)
    #network.add_layer(25, relu)
    network.add_layer(10, tanh)
    network.add_layer(10, tanh)
    network.build_network(softmax, cross_entropy)

    train_inputs = neural_network.data_to_label(data[:-1])
    train_outputs = neural_network.data_to_label(data[1:])

    network.train(train_inputs, train_outputs, network.sgd, 100, 0.0001, gd_optimizer='adadelta')

    test_data = '1'
    test_label = neural_network.data_to_label(test_data)
    result = network.forward(test_label.T)
    print(int_to_data[int(np.argmax(result.T[-1]))])
    """for i in range(len(data) - 1):
        print(neural_network.int_to_data[i])"""
    #plt.plot(network.loss_arr)
    plt.plot(network.loss_arr)
    plt.show()

    """test_inputs = test_set[:,:-1]
    test_outputs = test_set[:,-1:]

    network.test(test_inputs, test_outputs)"""

if __name__ == '__main__':
    run()