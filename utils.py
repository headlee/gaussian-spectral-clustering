import numpy as np


def random_choice_prob_index(prob_mat, axis=1):
    """
    Vectorized method to obtain indices for 2D array of probabilities obtained from
     https://stackoverflow.com/questions/47722005/vectorizing-numpy-random-choice-for-given-2d-array-of-probabilities-along-an-a
    :param prob_mat: (num_samples, num_choices) Numpy array containing probabilities for num_samples data samples
    :param axis: Axis value to compute over
    :return: (num_samples) Numpy array containing the random choice outputs
    """
    r = np.expand_dims(np.random.rand(prob_mat.shape[1 - axis]), axis=axis)
    return (prob_mat.cumsum(axis=axis) > r).argmax(axis=axis)


def calc_stats(data_mat):
    """
    Calculates the covariance to the minus one-half power and mean of the given data matrix
    :param data_mat: (num_features, num_samples) Input data matrix
    :return: Covariance to the minus one-half power and mean of the data matrix
    """
    data_mat_mean = np.mean(data_mat, axis=1)
    data_mat_cov = np.cov(data_mat)

    U, s, Vh = np.linalg.svd(data_mat_cov)
    gamma_minushalf = U @ (np.diag(s ** (-0.5))) @ (Vh)

    return gamma_minushalf, data_mat_mean


def whiten_data(data_mat):
    """
    Whitens the given data matrix
    :param data_mat: (num_features, num_samples) Input data matrix
    :return: Whitened version of the data matrix
    """
    gamma_minushalf, data_mat_mean = calc_stats(data_mat)
    data_mat_w = gamma_minushalf @ ((data_mat.transpose() - data_mat_mean).transpose())

    return data_mat_w


# TODO: K_y is divided by N_y in the paper (ddof=0 I think)
def rx(data_mat):
    """
    Performs RX anomaly detection on the given data. Code adapted from MATLAB examples given in class
    :param data_mat: (num_features, num_samples) Input data matrix
    :return: (num_samples) Numpy array containing anomaly detection results
    """
    data_mat_w = whiten_data(data_mat)
    rx_vec = np.sum(data_mat_w ** 2, axis=0)
    return rx_vec
