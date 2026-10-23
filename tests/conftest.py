from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.make_synthetic import make_synthetic


@pytest.fixture(scope="session")
def synthetic_export(tmp_path_factory):
    return make_synthetic(tmp_path_factory.mktemp("allgcl"))


def pytest_addoption(parser):
    parser.addoption(
        "--allgcl-export",
        default=None,
        help="Real SPEC 4.1 export for integration tests",
    )


@pytest.fixture(scope="session")
def smoke_export(request):
    supplied = request.config.getoption("--allgcl-export")
    return Path(supplied) if supplied else request.getfixturevalue("synthetic_export")
