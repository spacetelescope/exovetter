import typing

from scipy.io import loadmat
from astropy.utils import data

ENCODING = 'utf-8'

def load_mat(url: str) -> typing.Dict[str, typing.Any]:
    if url.startswith('http'):
        return loadmat(data.download_file(url, cache=True))

    raise NotImplementedError(f'URL Format not implemented[{url}]')

