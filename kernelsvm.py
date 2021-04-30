"""Functions for training kernel support vector machines."""
import numpy as np
from quadprog_wrapper import solve_quadprog

"""Please include your name and vt email id here as a comment."""

"""
Name: Ryan Nicholas
Email: rynicholas@vt.edu
"""


def polynomial_kernel(row_data, col_data, order):
    """
    Compute the Gram matrix between row_data and col_data for the polynomial kernel.

    :param row_data: ndarray of shape (2, m), where each column is a data example
    :type row_data: ndarray
    :param col_data: ndarray of shape (2, n), where each column is a data example
    :type col_data: ndarray
    :param order: scalar quantity is the order of the polynomial (the maximum exponent)
    :type order: float
    :return: a matrix whose (i, j) entry is the kernel value between row_data[:, i] and col_data[:, j]
    :rtype: ndarray
    """
    #############################################
    # TODO: Insert your code below to implement the polynomial kernel.
    # This computation should take around 1--3 lines of code if you use matrix operations.
    #############################################
    # Get the Gram Matrix -> no for loop, +Bonus Here
    K = (row_data.T.dot(col_data) + 1) ** order
    return K



def kernel_svm_predict(data, model, fLabels):
    """
    Predict binary-class labels for a batch of test points

    :param data: ndarray of shape (2, n), where each column is a data example
    :type data: ndarray
    :param model: learned model from kernel_svm_train
    :type model: dict
    :return: array of +1 or -1 labels
    :rtype: array
    """
    gram_matrix = polynomial_kernel(
        data, model['support_vectors'], model['params']['order'])

    scores = gram_matrix.dot(
        model['alphas'] * model['sv_labels']) + model['bias']
    scores = scores.ravel()

    labels = 2 * (scores > 0) - 1  # threshold and map to {-1, 1}

    f_labels = np.array(
        list(map(lambda x: fLabels[0] if x == 1 else fLabels[1], labels)))

    return f_labels, scores, labels
