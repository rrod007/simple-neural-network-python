from neural_network import *
import csv
import sys


def main():
    """
    Train and run predictions on the shopping.csv data set
    """
    # Check command-line arguments
    if len(sys.argv) != 3:
        sys.exit("Usage: python test_shopping.py data_for_training data_for_prediction")

    # Load data set from csv.
    training_data, labels = load_training_data(sys.argv[1])
    print(len(training_data[0]))

    # Create AI model
    model = NeuralNetwork(17, 4, 1, "logistic_sigmoid", 0)

    # Insert data to train
    model.data_input(training_data, labels)

    # Train model
    model.train(1)

    # Insert data to predict
    prediction_data = load_prediction_data(sys.argv[2])
    model.data_input(prediction_data)

    # Run predictions
    predictions = model.predict()

    # Display predictions
    print("PREDICTIONS:\n------------\n")
    for i, prediction in enumerate(predictions):
        print(str(i) + ": " + str(prediction))

    # raise NotImplementedError


def load_training_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    # set one list for all evidence lists, and another for all label values
    evidence = []
    labels = []

    # get each row from csv file into a *list of rows*
    with open(filename) as f:
        users = csv.reader(f, delimiter=',')

        # dictionaries to translate csv values to numerical values
        months = {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5, "Jul": 6,
                  "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11}
        user_type = {"New_Visitor": 0, "Returning_Visitor": 1, "Other": random.choice([0, 1])}
        weekend_revenue = {"TRUE": 1, "FALSE": 0}

        # iterate through each row
        first_row = True
        for user in users:
            # skip header row
            if first_row:
                first_row = False
                continue

            local_evidence = []

            # iterate through each evidence value and append it to local_evidence
            for i in range(len(user) - 1):
                # change val to numerical before appending if needed
                if i == 10:
                    local_evidence.append(months[user[i]])
                elif i == 15:
                    local_evidence.append(user_type[user[i]])
                elif i == 16:
                    local_evidence.append(weekend_revenue[user[i]])

                # change val type before appending if needed
                elif i == 0 or i == 2 or i == 4 or i == 11 or i == 12 or i == 13 or i == 14:
                    local_evidence.append(int(user[i]))
                elif i == 1 or i == 3 or i == 5 or i == 6 or i == 7 or i == 8 or i == 9:
                    local_evidence.append(float(user[i]))

            # append the complete list of evidence for the current user
            evidence.append(local_evidence)

            # append appropriate int value to labels for current user's label
            labels.append([weekend_revenue[user[-1]]])

    return evidence, labels

    # raise NotImplementedError


def load_prediction_data(filename):
    users_data = []

    with open(sys.argv[2]) as f:
        users = csv.reader(f, delimiter=',')

        # dictionaries to translate csv values to numerical values
        months = {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5, "Jul": 6,
                  "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11}
        user_type = {"New_Visitor": 0, "Returning_Visitor": 1, "Other": random.choice([0, 1])}
        weekend_revenue = {"TRUE": 1, "FALSE": 0}

        for user in users:

            local_data = []

            # iterate through each user data value and append it to local_data
            for i in range(len(user)):
                # change val to numerical before appending if needed
                if i == 10:
                    local_data.append(months[user[i]])
                elif i == 15:
                    local_data.append(user_type[user[i]])
                elif i == 16:
                    local_data.append(weekend_revenue[user[i]])

                # change val type before appending if needed
                elif i == 0 or i == 2 or i == 4 or i == 11 or i == 12 or i == 13 or i == 14:
                    local_data.append(int(user[i]))
                elif i == 1 or i == 3 or i == 5 or i == 6 or i == 7 or i == 8 or i == 9:
                    local_data.append(float(user[i]))

            # append the complete list of data for the current user
            users_data.append(local_data)

    return users_data


if __name__ == "__main__":
    main()