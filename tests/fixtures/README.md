`make_synthetic.py OUTPUT` builds the SPEC 4.1 schema with the pinned temporaldata
serializer. It creates two disjoint retina/session groups (train and valid), each
with 40 units, two conditions, chirp and movingbar blocks of 120 seconds, and
60-second gaps. Sampling rate is 7.8125 Hz. Outputs include processed HDF5,
`units.parquet` and `splits.json`; generated data belongs in a temporary directory.

The test suite always keeps this deterministic fixture for pairing edge cases.
Integration smoke commands also accept a real export once W1 provides one.
