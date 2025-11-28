#ifndef HEURISTIC_H
#define HEURISTIC_H
// Heuristic evaluation and helpers
// grid: 8x8 chars: 'X' for X, 'O' for O, '-' for empty
#include "serial.h"
// Bitboard-based evaluator: evaluate a GameState from perspective of is_X
double dynamic_heuristic_evaluation(const GameState& state, bool is_x);
// Count valid moves using bitboard move generator
int num_valid_moves(const GameState& state, bool for_X);

#endif
