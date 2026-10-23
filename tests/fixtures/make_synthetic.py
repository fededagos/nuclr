"""Deterministic SPEC 4.1 fixture. Generated binaries stay outside the checkout."""
import argparse
import datetime
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries
from brainsets.core import serialize_fn_map

BRAINSET = 'euler_allgcl_2025'
RATE = 7.8125


def make_synthetic(output: Path):
    output = Path(output)
    processed = output / 'processed' / BRAINSET
    processed.mkdir(parents=True, exist_ok=True)
    rows = []
    sessions = {'train': ['synthetic_train'], 'valid': ['synthetic_valid'], 'test': []}
    rng = np.random.default_rng(42)
    for split in ('train', 'valid'):
        sid = sessions[split][0]
        ids = np.array([f'{sid}_roi{i}' for i in range(40)])
        # Every condition has both stimuli; each 120 s block is separated by 60 s.
        start = np.arange(4, dtype=float) * 180
        end = start + 120
        conditions = np.array(['c1', 'c1', 'd', 'd'])
        blocks = Interval(start=start, end=end,
                          condition=conditions, condition_raw=np.array(['C1', 'C1', 'D', 'D']),
                          condition_class=np.array(['control', 'control', 'drug', 'drug']),
                          stimulus=np.array(['chirp', 'movingbar', 'chirp', 'movingbar']))
        timestamps = np.concatenate([s + np.arange(938) / RATE for s in start])
        values = rng.normal(size=(len(timestamps), 40)).astype('float32')
        # Distinct signal per unit makes ordering bugs visible.
        values += np.arange(40, dtype='float32')[None, :] / 40
        meta = pd.DataFrame(dict(id=ids, unit_id=ids, session_id=sid, split=split,
                                 group_id=np.arange(40) % 2 + 1,
                                 supergroup=np.where(np.arange(40) % 2, 'OFF', 'ON'),
                                 chirp_qidx=0.8, bar_qidx=0.8,
                                 label_condition='C1', label_condition_is_drug=False))
        rows.append(meta)
        units = ArrayDict(**{c: meta[c].to_numpy() for c in meta.columns})
        data = Data(
            brainset=Data(id=BRAINSET, origin_version='synthetic',
                                        derived_version='0.0.0', source='synthetic', description='SPEC 4.1 test fixture'),
            subject=Data(id=f'retina_{split}', species='MUS_MUSCULUS'),
            session=Data(id=sid, recording_date=datetime.datetime(2026, 1, 1), split=split),
            device=Data(id='synthetic_2p', imaging_rate=RATE),
            units=units, blocks=blocks,
            calcium_traces=IrregularTimeSeries(timestamps=timestamps, df_over_f=values,
                                              df_over_f_norm=(values / np.percentile(values, 99)).astype('float32'),
                                              domain=Interval(start=start, end=end)),
            domain=Interval(start=start, end=end),
        )
        with h5py.File(processed / f'{sid}.h5', 'w') as f:
            data.to_hdf5(f, serialize_fn_map=serialize_fn_map)
    pd.concat(rows).to_parquet(output / 'units.parquet', index=False)
    (output / 'splits.json').write_text(json.dumps(dict(seed=42, fractions=[.7, .15, .15],
        group_key=['experimenter', 'date'], groups={'retina_train': 'train', 'retina_valid': 'valid'}, sessions=sessions)))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    make_synthetic(parser.parse_args().output)
