#!/bin/bash

# Build script for Othello with serial, parallel, and optimized versions
set -e

INCLUDE_DIR="-I include"
TIME_MS=15000

# Names for the three builds
OUT_SERIAL=othello_serial
OUT_PARALLEL=othello_parallel
OUT_O1=othello_o1

# Parse command line arguments
VERSION="o1"  # default
while [[ $# -gt 0 ]]; do
    case $1 in
        --serial)
            VERSION="serial"
            shift
            ;;
        --parallel)
            VERSION="parallel"
            shift
            ;;
        --o1)
            VERSION="o1"
            shift
            ;;
        --time)
            TIME_MS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--serial|--parallel|--o1] [--time TIME_MS]"
            exit 1
            ;;
    esac
done

echo "Selected version: $VERSION"
echo "Time per move: ${TIME_MS}ms"
echo ""

# Build serial binary
echo "Building serial binary with g++..."
if g++ $INCLUDE_DIR -std=c++17 -O3 -o $OUT_SERIAL \
    src/main.cpp src/othello.cpp src/serial/serial.cpp \
    src/heuristic.cpp; then
    echo "✓ Serial build succeeded: $OUT_SERIAL"
else
    echo "✗ Serial build failed." >&2
    exit 1
fi

# Try to build parallel (CUDA) binary
PARALLEL_AVAILABLE=0
if command -v nvcc >/dev/null 2>&1; then
    echo "Building parallel CUDA binary..."
    if nvcc $INCLUDE_DIR -O3 -std=c++17 -arch=sm_70 -Xptxas -w -DENABLE_CUDA -DENABLE_PARALLEL_BASE -DENABLE_NAIVE \
        src/main.cpp src/othello.cpp src/serial/serial.cpp \
        src/parallel/parallel.cu src/naive_parallel/naive.cu src/heuristic.cpp \
        -o $OUT_PARALLEL; then
        echo "✓ Parallel build succeeded: $OUT_PARALLEL"
        PARALLEL_AVAILABLE=1
    else
        echo "✗ Parallel build failed with nvcc." >&2
    fi
else
    echo "⊘ nvcc not found — skipping parallel build."
fi

# Try to build optimized (CUDA) binary
O1_AVAILABLE=0
if command -v nvcc >/dev/null 2>&1; then
    echo "Building optimized CUDA binary (o1)..."
    if nvcc $INCLUDE_DIR -O3 -std=c++17 -arch=sm_70 -Xptxas -w -DENABLE_CUDA -DENABLE_PARALLEL_OPT1 -DENABLE_NAIVE \
        src/main.cpp src/othello.cpp src/serial/serial.cpp \
        src/parallel_optimized_1/parallel.cu src/naive_parallel/naive.cu src/heuristic.cpp \
        -o $OUT_O1; then
        echo "✓ Optimized build succeeded: $OUT_O1"
        O1_AVAILABLE=1
    else
        echo "✗ Optimized build failed with nvcc." >&2
    fi
else
    echo "⊘ nvcc not found — skipping optimized build."
fi

echo ""
echo "=========================================="
echo "Build Summary:"
echo "  Serial:    ✓ Available"
echo "  Parallel:  $([ $PARALLEL_AVAILABLE -eq 1 ] && echo '✓ Available' || echo '✗ Not available')"
echo "  Optimized: $([ $O1_AVAILABLE -eq 1 ] && echo '✓ Available' || echo '✗ Not available')"
echo "=========================================="
echo ""

# Run the requested version
case $VERSION in
    serial)
        echo "Running serial binary..."
        ./$OUT_SERIAL --serial --time $TIME_MS
        ;;
    parallel)
        if [ "$PARALLEL_AVAILABLE" -eq 1 ]; then
            echo "Running parallel binary..."
            ./$OUT_PARALLEL --parallel --time $TIME_MS
        else
            echo "Error: Parallel binary not available. Falling back to serial." >&2
            ./$OUT_SERIAL --serial --time $TIME_MS
        fi
        ;;
    o1)
        if [ "$O1_AVAILABLE" -eq 1 ]; then
            echo "Running optimized binary (o1)..."
            ./$OUT_O1 --parallel --time $TIME_MS
        elif [ "$PARALLEL_AVAILABLE" -eq 1 ]; then
            echo "Warning: Optimized binary not available. Falling back to parallel." >&2
            ./$OUT_PARALLEL --parallel --time $TIME_MS
        else
            echo "Warning: No CUDA binaries available. Falling back to serial." >&2
            ./$OUT_SERIAL --serial --time $TIME_MS
        fi
        ;;
esac
