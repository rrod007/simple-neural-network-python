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
        self.links = []  # Set later by structure
        self.neurons = []  # Set later by structure

        # call structure to populate neurons and create links
        self.__structure()

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
    def __structure(self):
        """
        Defines structure

        does just that really...
        """

        # Populate neurons

        # For each num inputs add a new neuron on layer 0 (input layer)
        for i in range(self.num_inputs):
            self.neurons.append(Neuron(0, self.bias))

        # For each num hidden neurons add a new neuron on layer 1 (hidden layer)
        for i in range(self.num_hidden):
            self.neurons.append(Neuron(1, self.bias))

        # For each num output neurons add a new neuron on layer 3 (output layer)
        for i in range(self.num_outputs):
            self.neurons.append(Neuron(2, self.bias))

        # Populate links

        # For each neuron:
        # Check if it has next layer
        # If so, add link with input as the neuron itself and output all neurons from the next layer

        for i, neuron_in in enumerate(self.neurons):
            # Check if it has next layer
            if neuron_in.layer == 2:
                break
            # If so, add links with this neuron's as input,
            # for each of the next layer's node as output.
            for j, neuron_out in enumerate(self.neurons):
                if neuron_out.id == neuron_in.id:
                    continue

                if neuron_in.layer + 1 != neuron_out.layer:
                    continue

                # Append new link to links list
                link = Link(neuron_in.id, neuron_out.id)
                self.links.append(link)

                # Add link id to input neuron's output links
                self.neurons[i].output_links.append(link.id)

                # Add link id to output neuron's input links
                self.neurons[j].input_links.append(link.id)


class Neuron:

    count = 0

    def __init__(self, layer, bias):
        """

        :param layer: The layer the neuron belongs to
        :param bias: Neuron's bias
        """

        self.id = self.count
        self.layer = layer
        self.bias = bias
        self.input_links = []  # List of all links which go in the neuron
        self.output_links = []  # List of all links which go out of the neuron

        # Increment object count
        self.count += 1

    def output(self):
        """
        Calculates neurons's output
        (passes bias / weights / values through activation and gets output)
        """

        raise NotImplementedError


class Link:

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
