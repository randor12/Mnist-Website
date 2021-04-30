"""This module includes methods for training and predicting using naive Bayes."""
import numpy as np


class NaiveModel():
    def __init__(self):
        self.parameters = None
        self.prob_output = []
        self.prob_inputs = []
        self.input_given_outputs = []
        self.input_prime = []
        self.labels = []
        self.num_classes = 0
        self.one_output = False
        self.cond = []
        self.n_size = 0
        self.d_size = 0
        self.max_x_g_y = []
        self.min_x_g_y = []

    def set_n_size(self, sz):
        self.n_size = sz

    def set_max_x_g_y(self, m):
        self.max_x_g_y = m

    def get_max_x_g_y(self):

        return self.max_x_g_y

    def set_min_x_g_y(self, m):
        self.min_x_g_y = m

    def get_min_x_g_y(self):
        return self.min_x_g_y

    def set_num_classes(self, c):
        self.num_classes = c

    def get_num_classes(self):
        return self.num_classes

    def get_n_size(self):
        return self.n_size

    def set_d_size(self, sz):
        self.d_size = sz

    def get_d_size(self):
        return self.d_size

    def set_labels(self, l):
        self.labels = l

    def get_labels(self):
        return self.labels

    def set_parameters(self, p):
        self.parameters = p

    def get_parameters(self):
        return self.parameters

    def set_prob_output(self, p_o):
        self.prob_output = p_o

    def get_prob_output(self):
        return self.prob_output

    def set_prob_input(self, p_i):
        self.prob_inputs = p_i

    def get_prob_input(self):
        return self.prob_inputs

    def set_input_given_outputs(self, i_g_o):
        self.cond = i_g_o
        self.input_given_outputs = i_g_o
        input_output_prime = [[1 - m for m in f]
                              for f in self.input_given_outputs]
        self.input_given_outputs = np.log(self.input_given_outputs)
        self.input_prime = np.log(input_output_prime)

    def get_input_given_output_prime(self):
        return self.input_prime

    def get_input_give_outputs(self):
        return self.input_given_outputs

    def get_priors(self):
        return self.prob_output

    def get_conditional_prob(self):
        return self.cond

    def predict(self, x):
        """
        :param x: input that is size (d, n)
        :rtype: ndarray
        :return: ndarry that is size n which returns the predicted output
        :rtype: numpy.ndarray
        """
        (d, n) = x.shape

        if (self.one_output):
            # if there is only 1 label value
            arr = np.full(fill_value=self.labels[0], shape=n)
            return arr
        else:
            # P(X=1 | Y)
            x_g_y = self.get_input_give_outputs()
            # P(X=0 | Y)
            x_g_y_prime = self.get_input_given_output_prime()
            # P(Y)
            prior_prob = self.get_prob_output()
            vals = np.full(shape=(self.num_classes, n),
                           fill_value={}, dtype=list)
            x = np.transpose(x)
            loc_val = np.where(x == 0)
            inverse_mult_val = np.where(x != 0)

            for i in range(self.num_classes):
                # gather posterior probability
                probs = np.multiply(x, x_g_y[i])
                prob_cpy = probs.copy()
                prob_cpy[loc_val] = 1
                prob_cpy = np.multiply(prob_cpy, x_g_y_prime[i])
                prob_cpy[inverse_mult_val] = probs[inverse_mult_val]
                # prob_cpy = [sum(m) + prior_prob[i]
                #             for m in prob_cpy]

                prob_cpy = prob_cpy.sum(axis=1)
                prob_cpy = prob_cpy + prior_prob[i]
                # collect the probabilities
                vals[i] = prob_cpy
            # compare what labels had the highest probability
            arr = np.argmax(vals, axis=0)

            return arr


def naive_bayes_predict(data, model):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.
    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    # INSERT YOUR CODE HERE FOR USING THE LEARNED NAIVE BAYES PARAMETERS
    # TO CLASSIFY THE DATA
    arr = model.predict(data)
    return arr
