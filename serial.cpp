#include <unordered_map>
#include <stack>
#include <math.h>
#include "serial.h"
#include "othello.h"

using namespace std;

// Define the globals declared in the header
static std::unordered_map<GameState, TTEntry, GameStateHash> table;

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
    int8_t black_score = __builtin_popcountll(state.black) - __builtin_popcountll(state.white);
    return (is_black) ? black_score : -black_score;
}

int8_t minimax(GameState state, bool is_black){
    if (table.find(state) != table.end()){
        return table[state].score;
    }

    vector<GameState> children = find_all_moves(state);
    if (children.empty()){
        // If no children, skip turn
        GameState skip_turn = state;
        skip_turn.black_turn = !skip_turn.black_turn;
        children = find_all_moves(skip_turn);
        
        // If after skip STILL no valid moves gameover
        if (children.empty()){
            return score_board(state, is_black);
        }
    }

    bool is_my_turn = (state.black_turn && is_black) || (!state.black_turn && !is_black);
    int16_t best_score = is_my_turn ? -127 : 127;
    uint8_t best_move = 0;
    for (GameState child : children){
        // Recursively calculate score
        int8_t score = minimax(child, is_black);
        
        // Convert board state diff to tile selected
        // Note to self: __builtin_ctzll = how many 0s before first 1
        // AKA compiler function that is faster than log_2(child.white & ~state.white) 
        uint8_t move;
        if (state.black_turn)
            move = __builtin_ctzll(child.black & ~state.black);
        else
            move = __builtin_ctzll(child.white & ~state.white);

        // Calculate which score is optimal (min v max)
        if (is_my_turn && best_score < score){
            best_score = score;
            best_move = move;
        }
        else if (!is_my_turn && score < best_score){
            best_score = score;
            best_move = move;
        }
    }

    table[state] = {best_score, best_move};

    return best_score;
}

GameState best_move(Othello* game){
    
    // Find children
    GameState state = game->get_board();
    vector<GameState> children = find_all_moves(state);
    
    // Determine which player we're finding the best move for
    bool is_black = game->getCurrentPlayer() == 'X';
    
    // If no valid moves, return current state (game will handle skipping turn)
    if (children.empty()) {
        return state;
    }
    
    // Evaluate each child and find the best one
    // minimax returns score from is_black player's perspective, so we always maximize
    int16_t best_score = -127;
    GameState best_state = children[0];
    
    for(auto child : children){
        
        cout << is_black << endl;
        int8_t score = minimax(child, is_black);
        
        cout << score << endl;
        // Always maximize since score is from current player's perspective
        if (score > best_score) {
            best_score = score;
            best_state = child;
        }
    }
    return best_state;
}
