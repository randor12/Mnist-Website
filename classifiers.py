from naive_bayes import *
from kernelsvm import *
import pickle


def classify_naive_bayes(data):
    """
    Classify the number using naive bayes
    :param data: this is the data being predicted on (shape=(784, n)
                - input values between 0 to 1)
    :return: returns the prediction for the data (shape=(n))
    """
    model = pickle.load(open('naive_bayes.h5', 'rb'))

    predictions = naive_bayes_predict(data, model)
    return predictions

def classify_svm_one_vs_one(data):
    """
    Classify the number using SVM with One vs One
    :param data: this is the data being predicted on (shape=(784, n))
                - input values between 0 to 1
    :return: returns the prediction to the data (shape=(n))
    """
    # load the models for the SVM
    model = pickle.load(open('svm.h5', 'rb'))

    one_to_one_models, labels = model[0], model[1]

    vals = []
    for i in range(len(one_to_one_models)):
        # get the predictions for each model
        predictions, s, l = kernel_svm_predict(
            data, model=one_to_one_models[i], fLabels=labels[i])
        vals.append(predictions)

    vals = np.array(vals)
    d, n = vals.shape

    counts = []
    for i in range(n):
        # determine the most common class for each input
        t, c = np.unique(vals[:, i], return_counts=True)

        counts.append(t[np.argmax(c)])

    counts = np.array(counts)

    f_vals = counts
    return f_vals
