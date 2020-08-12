import os
import pytest
import requests

MAT_URL = 'https://sourceforge.net/p/lpptransitlikemetric/code/HEAD/tree/data/maps/mapQ1Q17DR24-DVMed6084.mat?format=raw'  # noqa: E501
MAT_LOCAL_PATH = '/tmp/mapQ1Q17DR24-DVMed6084.mat'

@pytest.fixture
def mat_url():
    return MAT_URL

@pytest.fixture
def mat_filepath():
    if not os.path.exists(MAT_LOCAL_PATH):
        with open(MAT_LOCAL_PATH, 'wb') as stream:
            response = requests.get(MAT_URL, stream=True)
            for chunk in response.iter_content(1024):
                stream.write(chunk)

    return MAT_LOCAL_PATH

