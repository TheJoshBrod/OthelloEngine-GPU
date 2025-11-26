#ifndef PARALLEL_H
#define PARALLEL_H

#include "serial.h"

// Forward declaration (serial.h defines GameState)
class Othello;

// Unified parallel negamax entrypoint (time limit in ms)
// Implemented as a stub that currently delegates to serial code if parallel
GameState negamax_parallel(Othello* game, int time_limit_ms);

#endif

