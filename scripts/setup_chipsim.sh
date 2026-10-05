#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHIPSIM_DIR="${ROOT_DIR}/third_party/CHIPSIM"
VENV_DIR="${ROOT_DIR}/.chipsim-venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILD_JOBS="${CHIPSIM_BUILD_JOBS:-12}"

has_shared_python() {
    "$1" -c '
import pathlib
import sysconfig

libdir = pathlib.Path(sysconfig.get_config_var("LIBDIR") or "")
names = {
    sysconfig.get_config_var("LDLIBRARY"),
    sysconfig.get_config_var("INSTSONAME"),
}
raise SystemExit(not any(name and (libdir / name).exists() for name in names))
' >/dev/null 2>&1
}

if ! has_shared_python "${PYTHON_BIN}"; then
    if [[ "${PYTHON_BIN}" == "python3" ]] && \
       [[ -x /usr/bin/python3 ]] && has_shared_python /usr/bin/python3; then
        echo "Default python3 has no shared library; using /usr/bin/python3 for gem5."
        PYTHON_BIN=/usr/bin/python3
    else
        echo "${PYTHON_BIN} does not provide the shared Python library required by gem5."
        echo "Set PYTHON_BIN to a Python installation with development libraries."
        exit 1
    fi
fi

echo "Checking out the pinned CHIPSIM submodule..."
git -C "${ROOT_DIR}" submodule update --init --recursive third_party/CHIPSIM

for command in c++ m4; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "Missing build tool: ${command}"
        echo "See integrations/chipsim/README.md for system package requirements."
        exit 1
    fi
done

echo "Creating the isolated CHIPSIM environment at ${VENV_DIR}..."
if [[ -x "${VENV_DIR}/bin/python" ]]; then
    CURRENT_BASE="$("${VENV_DIR}/bin/python" -c 'import os, sys; print(os.path.realpath(sys._base_executable))')"
    REQUESTED_BASE="$("${PYTHON_BIN}" -c 'import os, sys; print(os.path.realpath(sys.executable))')"
    if [[ "${CURRENT_BASE}" != "${REQUESTED_BASE}" ]]; then
        "${PYTHON_BIN}" -m venv --clear "${VENV_DIR}"
    fi
else
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${CHIPSIM_DIR}/docs/requirements.txt"

echo "Building gem5 Garnet with ${BUILD_JOBS} jobs..."
(
    cd "${CHIPSIM_DIR}/integrations/gem5"
    PYTHON_REAL="$("${PYTHON_BIN}" -c 'import os, sys; print(os.path.realpath(sys.executable))')"
    PYTHON_CONFIG_BIN="${PYTHON_REAL}-config"
    PYTHON_LIBDIR="$("${VENV_DIR}/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
    if [[ ! -x "${PYTHON_CONFIG_BIN}" ]]; then
        echo "Cannot find python-config for ${PYTHON_REAL}."
        exit 1
    fi

    export PATH="${VENV_DIR}/bin:$(dirname "${PYTHON_REAL}"):/usr/local/bin:/usr/bin:/bin"
    export PYTHON_CONFIG="${PYTHON_CONFIG_BIN}"
    if [[ -x /usr/bin/protoc ]]; then
        export PROTOC=/usr/bin/protoc
    fi
    export LD_LIBRARY_PATH="${PYTHON_LIBDIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    "${VENV_DIR}/bin/scons" build/Garnet_standalone/gem5.opt -j"${BUILD_JOBS}"
)

echo "CHIPSIM is ready. Run: bash scripts/test_chipsim_integration.sh"
