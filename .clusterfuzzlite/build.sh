#!/bin/bash -eu

# Install nethical
pip3 install --no-cache-dir --require-hashes -r requirements-hashed.txt
pip3 install --no-deps .

# Build fuzz targets with the OSS-Fuzz Python launcher.
for fuzzer in "$SRC"/nethical/.clusterfuzzlite/*_fuzzer.py; do
    compile_python_fuzzer "$fuzzer"
done
