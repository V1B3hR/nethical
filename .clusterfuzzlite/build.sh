#!/bin/bash -eu

# Install nethical dependencies (hashes enforced) and stage local package without unhashed pip install
pip3 install --no-cache-dir --require-hashes -r requirements-hashed.txt
SITE_PKG="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cp -r "$SRC/nethical/nethical" "$SITE_PKG/nethical"
cp "$SRC/nethical/nethical_cli.py" "$SITE_PKG/"

# Build fuzz targets with the OSS-Fuzz Python launcher.
for fuzzer in "$SRC"/nethical/.clusterfuzzlite/*_fuzzer.py; do
    compile_python_fuzzer "$fuzzer"
done
