import os

from scipy.io import loadmat

from astropy.utils import data

__all__ = ['load_mat']

def load_mat(url_or_filepath):
    if os.path.exists(url_or_filepath):
        return loadmat(url_or_filepath)

    return loadmat(data.download_file(url_or_filepath, cache=True))
