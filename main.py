import numpy as np
import scipy.special


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        # input -> hidden
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes,  self.inodes))
        # hidden -> output
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learning_rate

        # sigmoid function
        self.activation_function = lambda x:scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # change list to 2d
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # input -> hidden
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # hidden -> output
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # error of result
        output_errors = targets - final_outputs

        # T는 전치연산
        # error of hidden
        hidden_errors = np.dot(self.who.T, output_errors)

        # fix hidden to output w
        self.who += self.lr * np.dot((output_errors * final_outputs*(1.0-final_outputs)), np.transpose(hidden_outputs))

        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs*(1.0 - hidden_outputs)), np.transpose(inputs))

        
        # cal to hidden
        pass

    # ask
    def query(self, inputs_list):
        # change inputs_list to 2d
        inputs = np.array(inputs_list, ndmin=2).T

        # input(in -> hidden)
        hidden_inputs = np.dot(self.wih, inputs)

        # ouput(hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        # input(final)
        final_inputs = np.dot(self.who, hidden_outputs)

        # output(final)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs