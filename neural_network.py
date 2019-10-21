import numpy as np
import pandas as pd
import random
import time
import math


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
        s = softmax(x).reshape(-1, 1)
        return np.diag(np.diagflat(s) - np.dot(s, s.T)).reshape(x.shape)
    else:
        exps = np.exp(x + 1e-8)
        return exps / (np.sum(exps, axis=1).reshape(x.shape[0], -1) + 1e-8)

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
        result = np.copy(target)
        result[target == 1.0] = -np.log(predicted[target == 1.0] + 1e-8)
        result[target == 0.0] = -np.log(1.0 - predicted[target == 0.0] + 1e-8)
        return result

def split_data(data, test_ratio):
    np.random.shuffle(data)
    test_size = int(len(data) * test_ratio)
    train_data, test_data = np.split(data, [len(data) - test_size])
    return train_data, test_data

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

def data_to_label(data):
    if type(data) == str:
        result = np.zeros((len(data), vocab_size))
    else:
        result = np.zeros((data.shape[0], vocab_size))
    for i in range(len(data)):
        if type(data) == str:
            result[i][data_to_int[str(data[i])]] = 1.0
        else:
            result[i][data_to_int[int(data[i])]] = 1.0
    return result

def label_to_data(label):   
    data_id = np.argmax(label)
    return int_to_data[data_id]


class NeuralNetwork:

    layers = []
    mini_batch_size = 2

    accuracy_arr = []
    loss_arr = []
    curr_epoch_accuracy = 0
    curr_epoch_loss = 0

    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def add_layer(self, n, activation_func, w = None, b = None):
        if not len(self.layers):
            self.layers.append(NNLayer(n, self.n_inputs, activation_func, w, b))
        else:
            self.layers.append(NNLayer(n, self.layers[-1].n, activation_func, w, b))

    def build_network(self, activation_func, cost_func, w = None, b = None):
        self.layers.append(NNLayer(self.n_outputs, self.layers[-1].n, activation_func, w, b))
        self.layers[-1].cost = cost_func
        for layer in self.layers:
            layer.set_wb(w, b)
    
    def forward(self, input):
        input = np.atleast_2d(input)
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backpropagate(self, predicted, target):
        target = np.atleast_2d(target)
        if self.n_outputs > 1:
            predicted_map = np.zeros(predicted.shape)
            predicted_map[predicted == np.max(predicted, axis=1).reshape(predicted.shape[0], -1)] = 1
            self.curr_epoch_accuracy += np.sum(predicted_map * target)
        self.curr_epoch_loss += np.sum(self.layers[-1].cost(predicted, target))

        for i in reversed(range(len(self.layers))):
            if i == (len(self.layers) - 1):
                cost = self.layers[i].cost(predicted, target, derivative=True)
                self.layers[i].backward(cost)
            else:
                global optimizer, gamma1
                next_params = getattr(self.layers[i + 1], 'layer_params')
                nag_opt = -gamma1 * next_params['momentum_w'] if optimizer == 'nag' else np.zeros(self.layers[i+1].weights.shape)
                error = np.dot(next_params['error'], self.layers[i + 1].weights.T + nag_opt.T)
                self.layers[i].backward(error)

    def update_model(self):
        for layer in self.layers:
            dw, db = layer.get_optimized_diff('layer_params')
            layer.weights -= dw
            layer.bias -= db

    def sgd(self, inputs, outputs):
        time1 = time.time()
        global curr_epoch
        for epoch in range(self.n_epoch):
            self.curr_epoch_accuracy = 0
            self.curr_epoch_loss = 0
            curr_epoch = epoch
            for i in np.random.choice(len(inputs), len(inputs), replace=False):
                predicted = self.forward(inputs[i])
                self.backpropagate(predicted, outputs[i])
                self.update_model()
            self.accuracy_arr.append(self.curr_epoch_accuracy / inputs.shape[0])
            self.loss_arr.append(self.curr_epoch_loss / inputs.shape[0])
        time2 = time.time()
        print('Stochastic GD {} ---- {}'.format(self.n_epoch, (time2 - time1) * 1000.0))

    def bgd(self, inputs, outputs):
        time1 = time.time()
        global curr_epoch
        for epoch in range(self.n_epoch):
            self.curr_epoch_accuracy = 0
            self.curr_epoch_loss = 0
            curr_epoch = epoch
            predicted = self.forward(inputs)
            self.backpropagate(predicted, outputs)
            self.update_model()
            self.accuracy_arr.append(self.curr_epoch_accuracy / inputs.shape[0])
            self.loss_arr.append(self.curr_epoch_loss / inputs.shape[0])
        time2 = time.time()
        print('Batch GD {} ---- {}'.format(self.n_epoch, (time2 - time1) * 1000.0))

    def mbgd(self, inputs, outputs, mini_batch_size=2):
        time1 = time.time()
        mini_batches = self.get_mini_batches(inputs, outputs)
        global curr_epoch
        for epoch in range(self.n_epoch):    
            self.curr_epoch_accuracy = 0
            self.curr_epoch_loss = 0
            curr_epoch = epoch
            for batch in mini_batches:
                batch_inputs = batch[0]
                batch_outputs = batch[1]
                predicted = self.forward(batch_inputs)
                self.backpropagate(predicted, batch_outputs)
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

    def train(self, inputs, outputs, gd_func, n_epoch=1, l_r=0.001, gd_optimizer=None):
        self.n_epoch = n_epoch
        global lr, optimizer, data_size
        lr = l_r
        optimizer = gd_optimizer
        data_size = inputs.shape[0]

        inputs = np.atleast_2d(inputs)
        outputs = np.atleast_2d(outputs)

        self.prepare_for_train()

        global gamma1, gamma2 
        gamma1, gamma2 =.9, .999
        gd_func(inputs, outputs)

    def prepare_for_train(self):
        for layer in self.layers:
            layer.prepare_for_train('layer_params')

    def test(self, inputs, outputs):
        predicted = self.forward(inputs)
        if self.n_outputs > 1:
            output_label = np.zeros((len(outputs), self.n_outputs))
            for i in range(len(outputs)):
                output_label[i][data_to_int[int(outputs[i])]] = 1.0
        else:
            output_label = outputs
        if self.n_outputs > 1:
            predicted_map = np.zeros(predicted.shape)
            predicted_map[predicted == np.max(predicted, axis=1).reshape(predicted.shape[0],-1)] = 1.0

            self.test_accuracy = np.sum(predicted_map * output_label) / inputs.shape[0]
            for i in range(predicted.shape[0]):
                print('Target: {} ----- Predicted: {} {}'.format(outputs[i], predicted[i], predicted_map[i]))
            print(self.test_accuracy)


class NNLayer:

    weights = np.array([])
    bias = 1
    
    def __init__(self, n, n_prev, activation_func, w=None, b=None):
        self.n = n
        self.n_prev = n_prev
        self.activation_func = activation_func

    def set_wb(self, w=None, b=None):
        #np.random.seed(1)
        self.weights = w if w else np.random.uniform(low=-1.0, high=1.0, size=(self.n_prev, self.n))
        self.bias = b if b else np.random.uniform(low=-1.0, high=1.0, size=(1,self.n))

    def activate_neurons(self, input):
        z = self.activation_func(input)
        return z
    
    def forward(self, input):
        self.prev = input
        self.z = np.dot(input, self.weights) + self.bias
        self.a = self.activate_neurons(self.z)
        return self.a

    def backward(self, prev_err):
        item_params = getattr(self, 'layer_params')
        item_params['error'] = prev_err * self.activation_func(self.z, derivative=True)
        item_params['dw'] = np.dot(self.prev.T, item_params['error']) / prev_err.shape[1]
        item_params['db'] = item_params['error'] / prev_err.shape[1]

    def prepare_for_train(self, *items, **item_weights):
        global optimizer

        for item in items:
            setattr(self, item, dict())
            params = getattr(self, item)
            item_weights_shape = item_weights[item] if item in item_weights else self.weights.shape
            params['delta_w'] = np.zeros(item_weights_shape)
            params['delta_b'] = 0

            if optimizer == 'momentum' or optimizer == 'nag':
                params['momentum_w'] = np.zeros(item_weights_shape)
                params['momentum_b'] = 0
            elif optimizer == 'adagrad' or optimizer == 'rmsprop' or optimizer == 'adadelta':
                params['cache_w'] = np.zeros(item_weights_shape)
                params['cache_b'] = 0
                if optimizer == 'adadelta':
                    params['x_w'] = np.zeros(item_weights_shape)
                    params['x_b'] = 0
            elif optimizer == 'adam':
                params['m_w'], params['u_w'] = np.zeros(item_weights_shape), np.zeros(item_weights_shape)
                params['m_ws'], params['u_ws'] = np.zeros(item_weights_shape), np.zeros(item_weights_shape)
                params['m_b'] , params['u_b'], params['m_bs'], params['u_bs'] = 0, 0, 0, 0


    def get_optimized_diff(self, params_name):
        global optimizer, gamma1, gamma2, lr, curr_epoch
        params = getattr(self, params_name)
        if optimizer == None:
            params['delta_w'] = lr * params['dw']
            params['delta_b'] = lr * params['db']
        elif optimizer == 'momentum' or optimizer == 'nag':
            params['momentum_w'] = gamma1 * params['momentum_w'] + lr * params['dw']
            params['momentum_b'] = gamma1 * params['momentum_b'] + lr * params['db']
            params['delta_w'] = params['momentum_w']
            params['delta_b'] = params['momentum_b']
        elif optimizer == 'adagrad':
            params['cache_w'] += params['dw'] ** 2
            params['delta_w'] = lr * params['dw'] / (np.sqrt(params['cache_w']) + 1e-8)
            params['cache_b'] += params['db'] ** 2
            params['delta_b'] = lr * params['db'] / (np.sqrt(params['cache_b']) + 1e-8)
        elif optimizer == 'rmsprop':
            params['cache_w'] = gamma1 * params['cache_w'] + (1 - gamma1) * params['dw'] ** 2
            params['delta_w'] = lr * params['dw'] / (np.sqrt(params['cache_w']) + 1e-8)
            params['cache_b'] = gamma1 * params['cache_b'] + (1 - gamma1) * params['db'] ** 2
            params['delta_b'] = lr * params['db'] / (np.sqrt(params['cache_b']) + 1e-8)
        elif optimizer == 'adadelta':
            params['cache_w'] = gamma1 * params['cache_w'] + (1 - gamma1) * params['dw'] ** 2
            params['x_w'] = gamma1 * params['x_w'] + (1 - gamma1) * params['delta_w'] ** 2
            params['delta_w'] = np.sqrt(params['x_w'] + 1e-8) * params['dw'] / np.sqrt(params['cache_w'] + 1e-8)
            params['cache_b'] = gamma1 * params['cache_b'] + (1 - gamma1) * params['db'] ** 2
            params['x_b'] = gamma1 * params['x_b'] + (1 - gamma1) * params['delta_b'] ** 2
            params['delta_b'] = np.sqrt(params['x_b'] + 1e-8) * params['db'] / np.sqrt(params['cache_b'] + 1e-8)
        elif optimizer == 'adam':
            params['m_w'] = gamma1 * params['m_w'] + (1 - gamma1) * params['dw']
            params['u_w'] = gamma2 * params['u_w'] + (1 - gamma2) * params['dw'] ** 2
            params['m_ws'] = params['m_w'] / (1 - gamma1 ** (curr_epoch + 1))
            params['u_ws'] = params['u_w'] / (1 - gamma2 ** (curr_epoch + 1))
            params['delta_w'] = lr * params['m_ws'] / np.sqrt(params['u_ws'] + 1e-8)

            params['m_b'] = gamma1 * params['m_b'] + (1 - gamma1) * params['db']
            params['u_b'] = gamma2 * params['u_b'] + (1 - gamma2) * params['db'] ** 2
            params['m_bs'] = params['m_b'] / (1 - gamma1 ** (curr_epoch + 1))
            params['u_bs'] = params['u_b'] / (1 - gamma2 ** (curr_epoch + 1))
            params['delta_b'] = lr * params['m_bs'] / np.sqrt(params['u_bs'] + 1e-8)

        return np.sum(params['delta_w'], axis=0), np.sum(params['delta_b'], axis=0)
