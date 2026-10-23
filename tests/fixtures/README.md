`make_synthetic.py OUTPUT` builds the SPEC 4.1 schema with the pinned temporaldata
serializer. It creates two disjoint retina/session groups (train and valid), each
with 40 units, two conditions, chirp and movingbar blocks of 120 seconds, and
60-second gaps. Sampling rate is 7.8125 Hz. Outputs include processed HDF5,
`units.parquet`, `unit_conditions.parquet` and `splits.json`; generated data belongs
in a temporary directory. Missing synthetic geometry is NaN. Description fields
use plain temporaldata Data objects, compatible with NuCLR's frozen loader and
the string-valued descriptions specified for the real export.

The test suite keeps this deterministic fixture for pairing edge cases. The
Python builder also accepts a block duration and a `valid_conditions` sequence;
the latter exercises absent conditions and `wo`/`washout` blocks. Integration
smoke commands can use a real export once W1 provides one.
