from pytest_utils import mat_url

def test__load_mat(mat_url):
    import numpy

    from exovetter.utils import load_mat

    mat_data = load_mat(mat_url)
    assert mat_data['map'].__class__ is numpy.ndarray
