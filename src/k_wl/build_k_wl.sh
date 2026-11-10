#!/usr/bin/env bash
set -euo pipefail

# Can be invoked from ANY directory; put the compiled module beside this script.
MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
CXX="${CXX:-g++}"

# Install once in your chosen Python environment:
#   python -m pip install pybind11
"$PYTHON" -c 'import pybind11' >/dev/null || {
    echo "pybind11 is missing. Install it with: $PYTHON -m pip install pybind11" >&2
    exit 1
}

EXT="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
# Allow CXX to be overridden, including a custom compiler path.
"$CXX" -O3 -std=c++17 -fPIC -shared \
    $("$PYTHON" -m pybind11 --includes) \
    "$MODULE_DIR/k_wl_bindings.cpp" \
    -o "$MODULE_DIR/_k_wl${EXT}"

echo "Built: $MODULE_DIR/_k_wl${EXT}"
