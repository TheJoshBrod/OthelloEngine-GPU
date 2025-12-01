#ifndef SERIAL_H
#define SERIAL_H

#include "gamestate.h"
#include <vector>

// Forward declaration
class Othello;

// Unified serial negamax entrypoint (time limit in ms, currently unused)
GameState negamax_serial(Othello* game, int time_limit_ms);

// Expose bitboard move generator for other modules
std::vector<GameState> find_all_moves(GameState state);

#endif
