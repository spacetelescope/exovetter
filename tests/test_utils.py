from pytest_utils import mat_url  # noqa: F401


def test__load_mat(mat_url):  # noqa: F811
    import numpy

    from exovetter.utils import load_mat

    mat_data = load_mat(mat_url)
    assert isinstance(mat_data['map'], np.ndarray)
