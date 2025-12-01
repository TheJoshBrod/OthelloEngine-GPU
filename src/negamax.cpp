#include "negamax.h"
#include "serial.h"
// #include "parallel.h"
#include "pvsplit.h"

// Default points to serial implementation
GameState (*negamax_fn)(Othello* game, int time_limit_ms) = nullptr;

void set_negamax_mode(bool use_parallel){
    if (use_parallel){
        negamax_fn = &negamax_parallel;
    } else {
        negamax_fn = &negamax_serial;
    }
}
