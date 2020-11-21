from neural_network import *


def main():

    # Gather data
    data = [[1, 1], [1, 0], [0, 1], [0, 0]]

    labels = [[0], [1], [1], [0]]

    # Create model
    model = NeuralNetwork(2, 2, 1, "relu", 0)

    # Insert data to train
    model.data_input(data, labels)

    # Train model
    model.train(20)

    # Display NN
    model.print_nn()

    # Insert prediction data
    model.data_input(data)

    # Predict
    predictions = model.predict()

    # Display predictions
    print("\n\nPREDICTIONS:\n------------\n")
    for i, prediction in enumerate(predictions):
        print(str(i) + ": " + str(prediction))

    # raise NotImplementedError


if __name__ == "__main__":
    main()
