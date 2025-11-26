#!/bin/bash
if g++ -I include -o othello src/main.cpp src/othello.cpp src/negamax.cpp src/serial/serial.cpp src/parallel/parallel.cpp src/heuristic.cpp; then
    ./othello --serial --time 5000
fi