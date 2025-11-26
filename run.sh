#!/bin/bash
if g++ -I include -o othello src/main.cpp src/othello.cpp src/serial/serial.cpp src/parallel/parallel.cpp; then
    ./othello 1
fi