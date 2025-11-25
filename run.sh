#!/bin/bash
if g++ -o othello main.cpp othello.cpp serial.cpp; then
    ./othello 1
fi