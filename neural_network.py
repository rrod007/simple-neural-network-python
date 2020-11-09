"""
This project consists of a 3 - layer neural network.
An input layer, a hidden layer, and an output layer.
"""
import random
import math


class NeuralNetwork:

    def __init__(self, num_inputs, num_hidden, num_outputs, activation, bias):
        """

        :param num_inputs: Number of input neurons
        :param num_hidden: Number of neurons in hidden layer
        :param num_outputs: Number of output neurons
        :param activation: String representing the activation function
        :param bias: Bias for all neurons
        """

        # Run checks on input data
        if num_inputs < 1 or num_hidden < 1 or num_outputs < 1:
            raise Exception("All layers must have at least one neuron")

        available_activations = Activation.available_activations()
        if activation not in available_activations:
            raise Exception(f"Available activation functions:\n{available_activations}")

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.bias = bias
        self.links = None  # Set later by structure
        self.nodes = None  # Set later by structure

        # call structure

    # Public
    def train(self, data, labels, epochs, rate=0.1):
        """
        trains NN

        :param data: list of lists, each of these represents a row of a csv file with
        numerical training data. All should be the same size.
        :param labels: List of labels, should represent each label for each training row.
        They must be in order so they match their training rows.
        :param epochs: Number of epochs to train the NN for.
        :param rate: Learning rate, 0.1 by default.
        """

        raise NotImplementedError

    def predict(self, data):
        """
        Makes prediction

        :param data: list of lists, each of these represents a row of a csv file with
        numerical data to make predictions on. All lists should be the same size.

        :return List of predictions for each neuron.
        Ordered from the top to the bottom of the layer.
        """

        raise NotImplementedError

    # Private
    def structure(self):
        """
        Defines structure

        does just that really...
        """

        raise NotImplementedError


class Neuron:

    count = 0

    def __init__(self, layer, bias, input_links, output_links):
        """

        :param layer: The layer the neuron belongs to
        :param bias: Neuron's bias
        :param input_links: A list with IDs of all input links
        :param output_links: A list with IDs of all output links links
        """

        self.id = self.count
        self.layer = layer
        self.bias = bias
        self.input_links = input_links
        self.output_links = output_links

        # Increment object count
        self.count += 1

    def output(self):
        """
        Calculates node's output
        (passes bias / weights / values through activation and gets output)
        """

        raise NotImplementedError


class Links:

    count = 0

    def __init__(self, input_neuron, output_neuron):
        """

        :param input_neuron: the ID of the input neuron
        :param output_neuron: the ID of the output neuron
        """

        self.id = self.count
        self.input_neuron = input_neuron
        self.output_neuron = output_neuron
        self.weight = random.uniform(0, 1)

        # increment object count
        self.count += 1


class Activation:

    """
    Provides activation functions for the neurons.
    """

    @staticmethod
    def logistic_sigmoid(value):
        return 1 / (1 + math.exp(-value))

    @staticmethod
    def relu(value):
        return max(0, value)

    @staticmethod
    def available_activations():
        return "logistic_sigmoid", "relu"
