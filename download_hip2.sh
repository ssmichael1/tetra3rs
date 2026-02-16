#!/usr/bin/env bash

# This script downloads the Hipparcos catalog (hip2.dat.gz) from the CDS archive,
# uncompresses it, and saves it to the data directory where it can be used to
# build the Tetra3 star catalog (as used in tests).

set -euo pipefail

DATA_DIR="data"
URL="http://cdsarc.u-strasbg.fr/ftp/I/311/hip2.dat.gz"
ARCHIVE_PATH="$DATA_DIR/hip2.dat.gz"
OUTPUT_PATH="$DATA_DIR/hip2.dat"

mkdir -p "$DATA_DIR"

echo "Downloading hip2.dat.gz..."
curl --fail --location --output "$ARCHIVE_PATH" "$URL"

echo "Uncompressing to hip2.dat..."
gunzip --force "$ARCHIVE_PATH"

echo "Done: $OUTPUT_PATH"
