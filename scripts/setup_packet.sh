#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/.hydra-packet-build"
mkdir -p "${BUILD_DIR}"
TEMP_LIBRARY="$(mktemp "${BUILD_DIR}/libhydra_packet.XXXXXX.so")"
trap 'rm -f "${TEMP_LIBRARY}"' EXIT
"${CXX:-c++}" -std=c++17 -O3 -Wall -Wextra -Werror -fPIC -shared \
    "${ROOT_DIR}/integrations/packet/packet_network.cc" \
    "${ROOT_DIR}/integrations/packet/c_api.cc" -o "${TEMP_LIBRARY}"
mv "${TEMP_LIBRARY}" "${BUILD_DIR}/libhydra_packet.so"
echo "HYDRA-Packet built: ${BUILD_DIR}/libhydra_packet.so"
