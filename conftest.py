# This file is used to configure the behavior of pytest.

from astropy.tests.helper import enable_deprecations_as_exceptions

try:
    from pytest_astropy_header.display import (PYTEST_HEADER_MODULES,
                                               TESTED_VERSIONS)
except ImportError:
    PYTEST_HEADER_MODULES = {}
    TESTED_VERSIONS = {}

try:
    from exovetter import __version__
except ImportError:
    __version__ = ''


PYTEST_HEADER_MODULES['Astropy'] = 'astropy'

# We can add these back to test header if they become a test dependency later.
PYTEST_HEADER_MODULES.pop('Matplotlib', None)
PYTEST_HEADER_MODULES.pop('h5py', None)
PYTEST_HEADER_MODULES.pop('Pandas', None)

TESTED_VERSIONS['exovetter'] = __version__

# Turn all deprecation warnings into exceptions.
enable_deprecations_as_exceptions()
