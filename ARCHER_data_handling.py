import numpy as np
import spectral.io.envi as envi # install with "pip install spectral"
from spectral import open_image


def load_archer_data(img_filename, img_truth_filename=None):
    """
    Loads a given ARCHER image, and optionally the truth file that goes along with it.
    :param img_filename: Full path and filename to ARCHER HSI data on disk
    :param img_truth_filename: (Optional) Full path and filename to ARCHER truth image on disk
    :return: Tuple containing the HSI data mat, size of the data, and optionally the truth data mat
    """
    # Use third party spectral library to read in ENVI formatted data
    img_metadata = img_filename + '.hdr'
    img = envi.open(img_metadata, img_filename)

    # Get data into a ImageArray, basically a numpy ndarray with some extra features
    hsi_data = img.load()

    nl, ns, nb = hsi_data.shape
    hsi_data_mat = hsi_data.reshape([nl * ns, nb], order='F').transpose()

    truth_data_mat = None

    if img_truth_filename is not None:
        img_truth_metadata = img_truth_filename + '.hdr'

        # Read in image truth file
        img_truth = envi.open(img_truth_metadata, img_truth_filename)
        truth_data = img_truth.load()

        # Reshape truth data
        truth_data_mat = truth_data.reshape([nl*ns, 1], order='F').transpose()

    return hsi_data_mat, (nl, ns, nb), truth_data_mat


def load_aviris_data(img_filename, img_truth_filename=None):
    """
    Loads a given AVIRIS image
    :param img_filename: Full path and filename to AVIRIS HSI data on disk
    :param img_truth_filename: (Optional) Full path and filename to ARCHER truth image on disk
    :return: Tuple containing the HSI data mat, size of the data, and optionally the truth data mat
    """
    # Use third party spectral library to read in AVIRIS data
    img = open_image(img_filename)

    # Get data into a ImageArray, basically a numpy ndarray with some extra features
    hsi_data = img.load()

    nl, ns, nb = hsi_data.shape
    hsi_data_mat = hsi_data.reshape([nl * ns, nb], order='F').transpose()

    truth_data_mat = None

    if img_truth_filename is not None:
        # Read in image truth file
        img_truth = open_image(img_truth_filename)
        truth_data = img_truth.load()

        # Reshape truth data
        truth_data_mat = truth_data.reshape([nl*ns, 1], order='F').transpose()

    return hsi_data_mat, (nl, ns, nb), truth_data_mat


def serials_to_idxs(serials, truth_data_mat):
    """
    Given a list of serial numbers, returns corresponding indices into the ARCHER data
    :param serials: List of serial numbers
    :param truth_data_mat: ARCHER truth data
    :return: Numpy array containing the indices in the ARCHER image for the given serial number(s)
    """
    indices = []
    for idx in range(len(serials)):
        ix = np.nonzero(truth_data_mat.squeeze() == serials[idx])[0]
        indices = indices + list(ix)
    return np.array(indices)
