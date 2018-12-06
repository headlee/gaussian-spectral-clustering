import numpy as np
from scipy.stats import multivariate_normal

from utils import random_choice_prob_index, rx


def compute_pcs_and_project(data_mat):
    """
    Implements the "Compute PCS and Project' block from the Beaven paper

    Uses SVD to compute the PCA representation of the given data
    :param data_mat: Input data to project into PCA coordinates
    :return: Tuple containing the projected data and the eigenvectors used
    """
    # Step 1: Compute covariance matrix
    gamma = np.cov(data_mat)

    # Step 2: Compute the svd
    U, S, Vh = np.linalg.svd(gamma)

    # Step 3: Retain the eigenvectors of the covariance
    eig_vecs = U.copy()

    # Step 4: Project the demeaned hsi_data_mat onto U
    mean_vec = np.mean(data_mat, axis=1)
    x_pca = -U.transpose().dot((data_mat.transpose() - mean_vec).transpose())

    return x_pca, eig_vecs


def compute_multivariate_gaussian_statistics(data_mat, threshold=None):
    """
    Implements the 'Compute MV Gaussian Statistics' block from the Beaven paper

    Uses the multivariate stats to do RX anomaly detection for culling pixels
    :param data_mat: (num_features, num_samples) data matrix (PC bands N+1-end, trailing order PCs)
    :param threshold: Optional threshold value for RX algorithm for classifying outliers. Default is to classify
                      anything greater than 2 std. deviations away from the mean as an outlier
    :return: Indices of the outliers in the data_mat
    """
    rx_vec = rx(data_mat)

    if threshold is None:
        # Define outlier threshold as anything greater than 2 std. deviations away from the mean
        threshold = rx_vec.mean() + 2 * rx_vec.std()

    outlier_ixs = np.nonzero(rx_vec > threshold)[0]

    return outlier_ixs


def initial_class_assignment(data_mat, num_classes, method='rand', init_indices=None):
    """
    Implements the 'Initial Class Assignment' block from the Beaven paper

    Either randomly assigns all samples to classes, or takes in a spectral vector for each class to use as a starting
    point for class means.
    :param data_mat: (num_features, num_samples) data matrix (PC bands 1-N, leading order PCs)
    :param num_classes: Number of classes to cluster into
    :param method: ('rand', 'select') String specifying which initialization method to choose
    :param init_indices: (num_classes,) List-type containing one sample index into data_mat per class to use
                         with the 'select' initialization method
    :return: ((num_samples,), (num_features, num_samples), (num_features, num_features, num_samples)) Tuple containing
              Numpy arrays for initial class membership indices (ranging from 0 - num_classes-1), class means, and class
              covariances, respectively
    """

    num_features, num_samples = data_mat.shape
    class_mean = np.zeros([num_features, num_classes])
    class_cov = np.zeros([num_features, num_features, num_classes])

    if method == 'rand':
        # Option 1 - Random assignment
        cluster_member_ix = np.random.randint(0, high=num_classes, size=num_samples)

        # For each class, find all members of that class and calculate a class mean and covariance
        for class_idx in range(num_classes):
            ixs = np.nonzero(cluster_member_ix == class_idx)[0]
            class_mean[:, class_idx] = data_mat[:, ixs].mean(axis=1)
            class_cov[:, :, class_idx] = np.cov(data_mat[:, ixs])
    elif method == 'select' and np.array(init_indices).shape[0] == num_classes:
        # Option 2 - Selective sampling
        # Use the given indices to initialize the class means
        for class_idx in range(num_classes):
            class_mean[:, class_idx] = data_mat[:, init_indices[class_idx]]

        # Calculate global covariance and use that as starting point for each class
        # Create num_classes copies of the covariance matrix along the third dimension
        # TODO: Check this
        class_cov = np.dstack([np.cov(data_mat)] * num_classes)

        # TODO: How do I actually initially assign samples in this case? For now, just doing it randomly
        cluster_member_ix = np.random.randint(0, high=num_classes, size=num_samples)
    else:
        print('Invalid method given - did you pick "select" with invalid init_indices param?')
        raise ValueError

    return cluster_member_ix, class_mean, class_cov


def compute_class_statistics(data_mat, class_ixs, num_classes):
    """
    Implements the 'Compute Class Statistics' block from the Beaven paper

    Computes class means, covariances, and prior probabilities.
    :param data_mat: (num_features, num_samples) data matrix (PC bands 1-N, leading order PCs)
    :param class_ixs: (num_samples) Numpy array containing class membership indices
    :param num_classes: Number of classes to cluster into
    :return: ((num_features, num_samples), (num_features, num_features, num_samples), (num_classes) Tuple containing
             calculated class means, covariances, and prior probabilities, respectively
    """
    num_features, num_samples = data_mat.shape
    class_mean = np.zeros([num_features, num_classes])
    class_cov = np.zeros([num_features, num_features, num_classes])

    for class_idx in range(num_classes):
        ixs = np.nonzero(class_ixs == class_idx)[0]
        class_mean[:, class_idx] = data_mat[:, ixs].mean(axis=1)
        class_cov[:, :, class_idx] = np.cov(data_mat[:, ixs])

    # TODO: Is this correct? Or should the prior probabilities be set to the posterior probs from last iteration?
    class_priors = np.bincount(class_ixs)
    class_priors = class_priors / np.sum(class_priors)

    return class_mean, class_cov, class_priors


def compute_posterior_probability_and_assign(data_mat, class_ixs, class_mean, class_cov, class_priors,
                                             dead_class_threshold=0):
    """
    Implements the 'Compute Posterior Probability & Assign to Class' block from the Beaven paper
    :param data_mat: (num_features,num_samples) data matrix (PC bands 1-N, leading order PCs)
    :param class_ixs: (num_samples,) array containing prior class membership indices
    :param class_mean: (num_features, num_classes) numpy matrix containing prior class means
    :param class_cov: (num_features, num_features, num_classes) numpy matrix containing prior class covariance matrices
    :param class_priors: (num_classes,) array containing prior class probabilities
    :param dead_class_threshold: Integer specifying the number of minimum members a class may have before it is
                                 considered dead and adaptive class management takes over. Default = 0 = no adaptive
                                 management occurs (may have issues with singular matrices with this default!)
    :return: updated_class_ixs: (num_samples,) array containing updated class indices
    """

    K, num_samples = data_mat.shape
    num_classes = class_mean.shape[1]
    # print('K, nc:', K, num_classes)

    posterior_probabilities = np.zeros([num_samples, num_classes])
    # posterior_probabilities = np.zeros(num_classes)
    p_c = np.zeros([num_samples, num_classes])
    # p_c = np.zeros(num_classes)
    for class_idx in range(num_classes):
        #         # TODO: Create p_c calc function
        #         # TODO: Is x_bar supposed to be all data or only class? I think all...
        #         # ixs = np.nonzero(class_ixs == class_idx)[0]
        #         # x_bar = data_mat[:, ixs]
        #         x_bar = data_mat.copy()
        m_c = class_mean[:, class_idx]
        K_c = class_cov[:, :, class_idx]
        #         term1 = ((2 * np.pi) ** (-1 * K / 2))
        #         term2 = np.linalg.det(K_c) ** (-1 / 2)
        #         # term3 = np.exp(-1/2 * (x_bar.transpose() - m_c).transpose().transpose() * np.linalg.inv(K_c) * (x_bar.transpose() - m_c).transpose())
        #         term3 = np.exp(-1 / 2 * (x_bar.transpose() - m_c).transpose().transpose().dot(np.linalg.inv(K_c)).dot(
        #             (x_bar.transpose() - m_c).transpose()))
        #         p_c[class_idx] = term1 * term2 * term3

        # SCIPY built-in: from scipy.stats import multivariate_normal
        # TODO: Transpose covariance? Don't give data_mat at all?
        p_c[:, class_idx] = multivariate_normal.pdf(data_mat.transpose(), mean=m_c, cov=K_c.transpose())

        posterior_probabilities[:, class_idx] = class_priors[class_idx] * p_c[:, class_idx]

    denom = np.sum(posterior_probabilities, axis=1)
    # print(denom.size)

    # Divide by total value in denom
    posterior_probabilities = posterior_probabilities / np.sum(posterior_probabilities, axis=1, keepdims=True)

    # print(posterior_probabilities)

    # Randomly assign pixels to classes with the calculated posterior probabilities
    # updated_class_ixs = np.zeros(num_samples)

    # for sample_idx in range(num_samples):
    #    updated_class_ixs[sample_idx] = np.random.choice(num_classes, size=1, p=posterior_probabilities[sample_idx, :])

    updated_class_ixs = random_choice_prob_index(posterior_probabilities)

    # updated_class_ixs = np.random.choice(num_classes, size=num_samples, p=posterior_probabilities)
    # TODO: Check vs. original ixs?
    updated_class_ixs = updated_class_ixs.astype('int64')
    # Get number of samples in each class
    class_counts = np.bincount(updated_class_ixs)
    print('class counts', class_counts)

    # If any class is less than a certain threshold, consider it "dead" and perform adaptive class management
    dead_class_ixs = np.nonzero(class_counts[class_counts < dead_class_threshold])[0]
    for dead_class in dead_class_ixs:
        print('Dead class (idx {}), less members than threshold ({})'.format(dead_class, dead_class_threshold))
        updated_class_ixs = adaptive_class_management(data_mat, updated_class_ixs, dead_class, num_classes)

    return updated_class_ixs


def adaptive_class_management(data_mat, class_ixs, dead_class_ix, num_classes):
    """
    Performs adaptive class management as described in the Beaven paper
    :param data_mat: (num_features, num_samples) data matrix (PC bands 1-N, leading order PCs)
    :param class_ixs: (num_samples,) array containing class membership indices
    :param dead_class_ix: Index of the empty class that is to be removed
    :param num_classes: Number of total possible cluster classes
    :return: updated_class_ixs: (num_samples,) array containing class indices after adaptive class management
    """
    # Step 1 - Nominate a dominant class
    class_vars = np.zeros(num_classes)
    # Find the class with the most scatter (variance?) in the first principal component dimension
    for class_idx in range(num_classes):
        ixs = np.nonzero(class_ixs == class_idx)[0]
        class_vars[class_idx] = np.var(data_mat[0, ixs])

    dominant_class_ix = np.argmax(class_vars)
    dominant_class_ixs = np.nonzero(class_ixs == dominant_class_ix)[0]

    # Step 2 - Split dominant class into 2 subclasses
    tmp = data_mat[0, dominant_class_ixs] - np.mean(data_mat[0, dominant_class_ixs])

    # Data that has a first principal component value less than the class mean will be transferred to the dead class
    neg_ixs = np.nonzero(tmp[tmp < 0])
    class_ixs[neg_ixs] = dead_class_ix

    # Step 3 - Continue iterating with the new class indices
    return class_ixs


def iterate_clustering(data_mat, class_ixs, num_classes, N=100, dead_class_threshold=0):
    """
    Iterates through the Compute Class Statistics and Compute Posterior Probability & Assign to Class blocks
    :param data_mat: (num_features, num_samples) data matrix (PC bands 1-N, leading order PCs)
    :param class_ixs: (num_samples,) array containing class membership indices
    :param num_classes: Number of classes to cluster into
    :param N: Number of iterations
    :param dead_class_threshold: Integer specifying the number of minimum members a class may have before it is
                                 considered dead and adaptive class management takes over. Default = 0 = no adaptive
                                 management occurs (may have issues with singular matrices with this default!)
    :return: (num_samples,) Numpy array containing final class membership indices
    """
    updated_class_ixs = class_ixs
    for iter_idx in range(N):
        # TODO: Need to return/save off the posterior probabilities and make them priors?
        class_mean, class_cov, class_priors = compute_class_statistics(data_mat, updated_class_ixs, num_classes)
        updated_class_ixs = compute_posterior_probability_and_assign(data_mat,
                                                                     class_ixs,
                                                                     class_mean,
                                                                     class_cov,
                                                                     class_priors,
                                                                     dead_class_threshold)
        print('Finished iteration #', iter_idx)

    return updated_class_ixs
