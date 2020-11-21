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
        self.data = None
        self.labels = None
        self.training_input_sample = None  # Training sample for current training iteration
        self.prediction_input_sample = None  # Prediction sample for current prediction iteration
        self.activation = getattr(Activation(), activation)

        # call structure to populate neurons and create links
        self.__structure()

    # Public
    def data_input(self, data, labels=None):
        """
        Sets data for training the NN or making predictions.
        You must re-insert data before each training or prediction session


        :param data: list of lists, each of these represents a row of a csv file with
        numerical training data. All should be the same size. All values must be numerical
        :param (set to None by default, do not set if you wish to predict) labels: List of lists.
        Each should represent the expected values for the output neurons, from the top to the bottom
        All must be ordered so that they match their training rows. All values must be numerical
        """
        # Do checks on inputs

        # All data and label rows should have the same size
        for row in data:
            if len(row) != len(data[0]):
                raise Exception("All data rows must have the same number of values")

        if labels is not None:
            for label_sample in labels:
                if len(label_sample) != len(labels[0]):
                    raise Exception("All label rows must have the same number of values")

        # Number of attributes in each data / label sample must match number of input / output neurons
        if len(data[0]) != self.num_inputs:
            raise Exception("Number of data attributes on each data row must match number of input neurons")
        if labels is not None:
            if len(labels[0]) != self.num_outputs:
                raise Exception("Number of labels on each labels row must match number of output neurons")

        self.data = data
        if labels is not None:
            self.labels = labels

    def train(self, epochs, rate=0.1):
        """
        trains NN
        :param epochs: Number of epochs to train the NN for.
        :param rate: Learning rate, 0.1 by default.
        """

        # Do checks if labels are set.
        if self.labels is None:
            raise Exception("Labels parameter is not set. Re-insert data with data_input")

        # Get training samples together
        training_set = []
        for i in range(len(self.data)):
            training_sample = [self.data[i], self.labels[i]]
            training_set.append(training_sample)

        # Update weights x amount of times,
        # With x being the length of the training sample * num. of epochs
        # Training samples are randomly extracted from the training set
        for i in range(epochs * len(training_set)):
            # Get a random training sample
            self.training_input_sample = random.choice(training_set)
            # print(self.training_input_sample)
            # TESTED WORKING

            # Update weights on each output neuron's input links
            label_count = 0
            for neuron in self.neurons:
                if neuron.layer != 2:
                    continue

                # Get current neuron's predicted output
                output = self.output(neuron)
                # print(output)
                # TESTED WORKING

                # Get current neuron's expected output
                label = self.training_input_sample[1][label_count]
                label_count += 1

                # Set current neuron's error
                neuron.error = (label - output) * self.__transfer_derivative(output)

                # Find each input link object
                for link in self.links:
                    if link.id not in neuron.input_links:
                        continue
                    # Update link's weight:
                    # Get output from the link's input neuron
                    for hidden_neuron in self.neurons:
                        if hidden_neuron.id != link.input_neuron:
                            continue
                        # Get input value for output neuron's input link
                        link_input_value = self.output(hidden_neuron)
                        link.weight = link.weight + rate * neuron.error * link_input_value

            # Update weights on each hidden neuron's input links
            for neuron in self.neurons:
                if neuron.layer != 1:
                    continue

                # Increment the hidden neuron's error:
                # Find each output neuron
                for output_neuron in self.neurons:
                    if output_neuron.layer != 2:
                        continue
                    output_derivative = self.__transfer_derivative(self.output(output_neuron))
                    # output_derivative = self.output(output_neuron)
                    error = output_neuron.error

                    # Get weight from link between hidden neuron and output neuron
                    for link in self.links:
                        if (link.id not in neuron.output_links) or (link.id not in output_neuron.input_links):
                            continue
                        weight = link.weight

                        # Increment hidden neuron's error
                        neuron.error = neuron.error + weight * error * output_derivative

                # Update the hidden neuron's input links' weights
                for link in self.links:
                    if link.id not in neuron.input_links:
                        continue
                    # Get link's input value and then update weight
                    for input_neuron in self.neurons:
                        if input_neuron.id != link.input_neuron:
                            continue
                        link_input = self.output(input_neuron)
                        # update weight
                        link.weight = link.weight + rate * neuron.error * link_input

            # Reset all neuron's errors for the next training iteration
            for neuron in self.neurons:
                neuron.error = 0

        # Reset input data and data related to training after training session is complete
        self.data = None
        self.labels = None
        self.training_input_sample = None

        # print("TRAINING SUCCESSFUL!!!")

        # raise NotImplementedError

    def predict(self):
        """
        Makes prediction

        :return List of lists of predictions for each output neuron.
        Ordered from the top to the bottom of the layer.
        """

        # Check if self.labels not is set
        if self.labels is not None:
            Exception("Call the input function again and do *not* set the labels parameter")

        prediction_output_samples = []

        # Iterate through every row on the inserted data
        for prediction_input_sample in self.data:
            # Set current prediction sample
            self.prediction_input_sample = prediction_input_sample

            predictions = []
            for output_neuron in self.neurons:
                if output_neuron.layer != 2:
                    continue
                predictions.append(self.output(output_neuron))

            prediction_output_samples.append(predictions)

        self.data = None
        self.labels = None
        self.prediction_input_sample = None

        return prediction_output_samples

        # raise NotImplementedError

    def output(self, neuron):
        """
        Calculates neuron's output

        :param neuron: Neuron object to get the output from

        (passes bias / weights / values through activation and gets output)
        """

        weights = []
        values = []

        # Loop through all links
        # If the link is in the neuron's input links list, then:
        # 1. Get the link's weight, and right after
        # 2. Get that link's input neuron's output.
        # If this neuron doesn't have an input

        # If neuron is in first layer, return corresponding training / prediction sample input
        if neuron.layer == 0:
            if self.labels is None:
                # neuron.id corresponds to prediction sample's list index
                # print(self.prediction_input_sample)
                return self.prediction_input_sample[neuron.id]

            # neuron.id corresponds to training sample's list index
            # print(self.training_input_sample)
            return self.training_input_sample[0][neuron.id]

        for link in self.links:
            # Move on if link is not in input links for the neuron
            if link.id not in neuron.input_links:
                continue

            # Append link's weight
            weights.append(link.weight)

            # Go through NNs neurons and find the neuron who is the link's input
            # Append that neuron's output to values
            for neuron_in in self.neurons:
                if neuron_in.id == link.input_neuron:
                    # print("Got HERE")
                    values.append(self.output(neuron_in))

        output = neuron.bias
        for (weight, value) in zip(weights, values):
            output += weight * value

        return self.activation(output)

        # raise NotImplementedError

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

        # BUG: Neurons being added all with id set to 0

        # Populate links

        # For each neuron:
        # Check if it has next layer
        # If so, add link with input as the neuron itself
        # and output all neurons from the next layer

        for i, neuron_in in enumerate(self.neurons):
            # Check if it has next layer
            # print(neuron_in.layer)
            # TESTED WORKS
            if neuron_in.layer == 2:
                break

            # If so, add links with this neuron as input,
            # for each of the next layer's node as output.
            for j, neuron_out in enumerate(self.neurons):
                if neuron_out.id == neuron_in.id:
                    continue

                if neuron_in.layer + 1 != neuron_out.layer:
                    continue

                # Append new link to links list
                link = Link(neuron_in.id, neuron_out.id)
                # print(str(link) + "LINK ADDED!!!")
                # TEST PASSED
                self.links.append(link)

                # Add link id to input neuron's output links
                self.neurons[i].output_links.append(link.id)

                # Add link id to output neuron's input links
                self.neurons[j].input_links.append(link.id)

    @staticmethod
    def __transfer_derivative(output):
        """
        Returns the transfer derivative for neuron's outputs
        """
        return output * (1.0 - output)

    def print_nn(self):
        """
        Displays the weights of each neuron to neuron connection
        """
        print("\n\nNeural Network structure and weights:\n-------------------------------------\n")
        for link in self.links:
            # Print each connection

            neuron_in = None
            neuron_out = None
            for neuron in self.neurons:
                # find input neuron
                if neuron.id == link.input_neuron:
                    neuron_in = neuron
                # find output neuron
                if neuron.id == link.output_neuron:
                    neuron_out = neuron

            connection = f"LINK: Neuron IN --layer: {neuron_in.layer}   --id: {neuron_in.id}   *** Link WEIGHT: {link.weight}" \
                         f" ***   Neuron OUT: --layer: {neuron_out.layer}   ---id: {neuron_out.id}"

            print(connection)


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
        self.error = 0
        self.input_links = []  # List of all links which go in the neuron
        self.output_links = []  # List of all links which go out of the neuron

        # Increment object count
        Neuron.count += 1


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
        Link.count += 1


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
