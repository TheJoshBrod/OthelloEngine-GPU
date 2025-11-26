#include "serial.h"
#include "othello.h"
#include "heuristic.h"
#include <cmath>
#include <vector>

using namespace std;

GameState make_move(GameState state, uint64_t move_bit) {
    GameState new_state;
    uint64_t my_pieces = state.black_turn ? state.black : state.white;
    uint64_t opp_pieces = state.black_turn ? state.white : state.black;
    uint64_t flipped = 0;
    
    // Check all 8 directions and flip pieces
    const int shifts[8] = {8, 9, 1, -7, -8, -9, -1, 7};
    
    for (int dir = 0; dir < 8; dir++) {
        int shift = shifts[dir];
        uint64_t line = 0;
        uint64_t pos = move_bit;
        
        // Walk in direction collecting opponent pieces
        while (true) {
            if (shift > 0) pos <<= shift;
            else pos >>= -shift;
            
            if (!pos || !(pos & opp_pieces)) break;
            line |= pos;
        }
        
        // If we ended on our piece, these are valid flips
        if (pos & my_pieces) flipped |= line;
    }
    
    // Apply the move
    my_pieces |= move_bit | flipped;
    opp_pieces &= ~flipped;
    
    new_state.black_turn = !state.black_turn;
    if (state.black_turn) {
        new_state.black = my_pieces;
        new_state.white = opp_pieces;
    } else {
        new_state.white = my_pieces;
        new_state.black = opp_pieces;
    }
    
    return new_state;
}

vector<GameState> find_all_moves(GameState state) {
    vector<GameState> moves;
    
    uint64_t my_pieces = state.black_turn ? state.black : state.white;
    uint64_t opp_pieces = state.black_turn ? state.white : state.black;
    uint64_t empty = ~(my_pieces | opp_pieces);
    
    // Directions: N, NE, E, SE, S, SW, W, NW
    const int shifts[8] = {8, 9, 1, -7, -8, -9, -1, 7};
    const uint64_t masks[8] = {
        0xFFFFFFFFFFFFFF00ULL,
        0xFEFEFEFEFEFEFE00ULL,
        0xFEFEFEFEFEFEFEFEULL,
        0xFEFEFEFEFEFEFE00ULL,
        0x00FFFFFFFFFFFFFFULL,
        0x007F7F7F7F7F7F7FULL,
        0x7F7F7F7F7F7F7F7FULL,
        0x7F7F7F7F7F7F7F00ULL
    };
    
    uint64_t valid_moves = 0;
    
    // For each direction, find pieces that can be flipped
    for (int dir = 0; dir < 8; dir++) {
        int shift = shifts[dir];
        uint64_t mask = masks[dir];
        
        // Find opponent pieces adjacent to my pieces in this direction
        uint64_t candidates = opp_pieces & mask;
        if (shift > 0)
            candidates &= (my_pieces << shift);
        else
            candidates &= (my_pieces >> -shift);
        
        // Extend along the direction through opponent pieces
        for (int i = 0; i < 5; i++) {  // Max 6 pieces can be flipped
            if (shift > 0)
                candidates |= (candidates << shift) & opp_pieces & mask;
            else
                candidates |= (candidates >> -shift) & opp_pieces & mask;
        }
        
        // Valid moves are empty squares adjacent to the line of opponent pieces
        if (shift > 0)
            valid_moves |= (candidates << shift) & empty & mask;
        else
            valid_moves |= (candidates >> -shift) & empty & mask;
    }
    
    // Generate a GameState for each valid move
    while (valid_moves) {
        // Get lowest bit and remove it
        uint64_t move_bit = valid_moves & -valid_moves;
        valid_moves ^= move_bit;
        
        GameState new_state = make_move(state, move_bit);
        moves.push_back(new_state);
    }
    
    return moves;
}

int8_t score_board(GameState state, bool is_black){
    double h = dynamic_heuristic_evaluation(state, is_black);
    if (std::isnan(h) || std::isinf(h)) h = 0.0;
    if (h > 127.0) h = 127.0;
    if (h < -127.0) h = -127.0;
    return static_cast<int8_t>(std::round(h));
}

// Removed old minimax / transposition table and best_move functions.
// This file now exposes move generation, scoring, and the iterative-deepening negamax entrypoint.

// Iterative deepening negamax with simple alpha-beta and time limit
// Starts at depth 3 and increases until time expires (time_limit_ms==0 means no limit)
#include <chrono>

static std::chrono::steady_clock::time_point g_start_time;
static int g_time_limit_ms = 0;
static bool time_exceeded() {
    if (g_time_limit_ms <= 0) return false;
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_start_time).count();
    return elapsed >= g_time_limit_ms;
}

// Depth-limited negamax with alpha-beta. Returns score from perspective of 'is_black'.
int negamax_dfs(const GameState& state, int depth, int alpha, int beta, bool is_black) {
    if (time_exceeded()) return 0;

    vector<GameState> children = find_all_moves(state);
    if (children.empty()){
        // try skip turn
        GameState skip = state;
        skip.black_turn = !skip.black_turn;
        children = find_all_moves(skip);
        if (children.empty()){
            return score_board(state, is_black);
        }
    }

    if (depth == 0) {
        return score_board(state, is_black);
    }

    int best = -128;
    for (const GameState& child : children){
        int val = -negamax_dfs(child, depth - 1, -beta, -alpha, is_black);
        if (time_exceeded()) return 0;
        if (val > best) best = val;
        if (best > alpha) alpha = best;
        if (alpha >= beta) break; // beta cutoff
    }
    return best;
}

GameState negamax_serial(Othello* game, int time_limit_ms) {
    GameState root = game->get_board();
    bool is_black = game->getCurrentPlayer() == 'X';

    g_time_limit_ms = time_limit_ms;
    g_start_time = std::chrono::steady_clock::now();

    // Get initial moves
    vector<GameState> children = find_all_moves(root);
    if (children.empty()) return root; // no move

    GameState best_state = children[0];

    int depth = 3;
    while (true) {
        if (time_exceeded()) break;

        int16_t best_score = -127;
        GameState best_at_this_depth = best_state;

        for (const GameState& child : children) {
            if (time_exceeded()) break;
            int score = negamax_dfs(child, depth - 1, -128, 128, is_black);
            if (time_exceeded()) break;
            if (score > best_score) {
                best_score = score;
                best_at_this_depth = child;
            }
        }

        if (!time_exceeded()) {
            // commit results of this completed depth
            best_state = best_at_this_depth;
            depth++;
            // safety cap to avoid infinite loop
            if (depth > 64) break;
        } else {
            break; // time ran out during this depth
        }
    }

    return best_state;
}
