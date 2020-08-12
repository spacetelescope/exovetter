from pytest_utils import mat_url, mat_filepath  # noqa: F401


def test__load_mat__remote(mat_url):  # noqa: F811
    import numpy as np

    from exovetter.utils import load_mat

    mat_data = load_mat(mat_url)
    assert isinstance(mat_data['map'], np.ndarray)

def test__load_mat__local(mat_filepath):
    import numpy as np

    from exovetter.utils import load_mat

    mat_data = load_mat(mat_filepath)
    assert isinstance(mat_data['map'], np.ndarray)
