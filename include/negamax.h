#ifndef NEGAMAX_H
#define NEGAMAX_H

#include "serial.h"
#include "parallel.h"
#include "pvsplit.h"

// Unified negamax function pointer. Signature: (Othello*, time_limit_ms)
extern GameState (*negamax_fn)(Othello* game, int time_limit_ms);

// Set the implementation to parallel or serial. time_limit_ms is stored by caller
void set_negamax_mode(bool use_parallel);

#endif
