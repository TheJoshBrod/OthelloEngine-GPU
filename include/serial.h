#ifndef SERIAL_H
#define SERIAL_H

#include <cstdint>
#include <functional>

// Forward declaration
class Othello;

// Usage:
// is_tile_X(x=0,y=1) = x && (x*8 + y) 
struct GameState {
    uint64_t o;
    uint64_t x;
    bool x_turn;
    
    // Operators for map/set usage
    bool operator<(const GameState& other) const {
        if (o != other.o) return o < other.o;
        if (x != other.x) return x < other.x;
        return x_turn < other.x_turn;
    }
    
    bool operator==(const GameState& other) const {
        return o == other.o && x == other.x && x_turn == other.x_turn;
    }
};

// Quick hash for maps/sets
struct GameStateHash {
    size_t operator()(const GameState& p) const {
        // Start with the O pieces
        size_t seed = std::hash<uint64_t>()(p.o);

        seed ^= std::hash<uint64_t>()(p.x) + 0x9E3779B97F4A7C15ULL + (seed << 6) + (seed >> 2);
        seed ^= std::hash<bool>()(p.x_turn) + 0x9E3779B97F4A7C15ULL + (seed << 6) + (seed >> 2);

        return seed;
    }
};

// Value of map representing best move given state
// score: Represents final score [-64, 64], if unknown assume WORST possible score for us
// bestmove: Bitmap on where to go to find score, if unknown
struct TTEntry {
    int16_t score; 
    uint8_t bestMove; 
};

// Unified serial negamax entrypoint (time limit in ms, currently unused)
GameState negamax_serial(Othello* game, int time_limit_ms);

// Expose bitboard move generator for other modules
std::vector<GameState> find_all_moves(GameState state);

#endif
