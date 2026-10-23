import h5py
import numpy as np
from temporaldata import Data
from src.transforms.unit_dropout import UnitDropout


def test_calcium_dropout_consistent(synthetic_export):
    with h5py.File(synthetic_export / 'processed/euler_allgcl_2025/synthetic_train.h5') as f:
        original = Data.from_hdf5(f, lazy=False).slice(0., 32.768)
    np.random.seed(1)
    raw = original.calcium_traces.df_over_f.copy()
    normalized = original.calcium_traces.df_over_f_norm.copy()
    ids = original.units.id.copy()
    dropped = UnitDropout(min_units=12)(original)
    assert 12 <= len(dropped.units.id) < len(ids)
    kept = np.array([np.where(ids == x)[0][0] for x in dropped.units.id])
    np.testing.assert_array_equal(dropped.calcium_traces.df_over_f, raw[:, kept])
    np.testing.assert_array_equal(dropped.calcium_traces.df_over_f_norm, normalized[:, kept])
