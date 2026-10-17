#!/bin/bash -eu

# Install nethical
pip3 install --no-cache-dir --require-hashes -r requirements-hashed.txt
pip3 install --no-deps -e .

# Build fuzz target
find .clusterfuzzlite/ -name "*_fuzzer.py" -exec cp {} $OUT/ \;
for fuzzer in $(find $SRC/nethical/.clusterfuzzlite -name '*_fuzzer.py'); do
    fuzzer_name=$(basename -s .py "$fuzzer")
    cat << 'EOF' > "$OUT/$fuzzer_name"
#!/bin/sh
# Wrapper for python fuzzer
exec python3 "$0.py" "$@"
EOF
    cp "$fuzzer" "$OUT/$fuzzer_name.py"
    chmod +x "$OUT/$fuzzer_name"
done
