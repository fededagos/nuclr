from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.make_synthetic import make_synthetic


@pytest.fixture(scope="session")
def synthetic_export(tmp_path_factory):
    return make_synthetic(tmp_path_factory.mktemp("allgcl"))
