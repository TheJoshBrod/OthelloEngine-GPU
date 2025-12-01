// Stub implementation of negamax_parallel for builds without CUDA
// Delegates to serial negamax implementation so the program links and runs
#include "pvsplit.h"
#include "serial.h"

GameState negamax_parallel(Othello* game, int time_limit_ms) {
    return negamax_serial(game, time_limit_ms);
}
