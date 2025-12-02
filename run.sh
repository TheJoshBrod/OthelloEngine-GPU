#!/bin/bash

# Build both serial and parallel binaries (parallel via nvcc if available)
set -e

INCLUDE_DIR="-I include"
TIME_MS=5000

# Names for the two builds
OUT_SERIAL=othello_serial
OUT_PARALLEL=othello_parallel

echo "Building serial binary with g++..."
if g++ $INCLUDE_DIR -std=c++17 -O2 -o $OUT_SERIAL \
    src/main.cpp src/othello.cpp src/negamax.cpp src/serial/serial.cpp src/heuristic.cpp src/parallel/parallel_stub.cpp; then
    echo "Serial build succeeded: $OUT_SERIAL"
else
    echo "Serial build failed." >&2
    exit 1
fi

# Try to build the parallel (CUDA) binary
PARALLEL_AVAILABLE=0
if command -v nvcc >/dev/null 2>&1; then
    echo "nvcc found — building parallel CUDA binary..."
    if nvcc $INCLUDE_DIR -O3 -std=c++11 -arch=sm_70 \
        src/main.cpp src/othello.cpp src/negamax.cpp src/serial/serial.cpp \
        src/parallel/parallel.cu src/heuristic.cpp \
        -o $OUT_PARALLEL; then
        echo "Parallel build succeeded: $OUT_PARALLEL"
        PARALLEL_AVAILABLE=1
    else
        echo "Parallel build failed with nvcc." >&2
    fi
else
    echo "nvcc not found — skipping parallel build."
fi

# Run the parallel binary if it was built, otherwise run the serial one
if [ "$PARALLEL_AVAILABLE" -eq 1 ]; then
    echo "Running parallel binary..."
    ./$OUT_PARALLEL --parallel --time $TIME_MS
else
    echo "Running serial binary..."
    ./$OUT_SERIAL --serial --time $TIME_MS
fi