#ifndef SERIAL_H
#define SERIAL_H

#include <cstdint>
#include <functional>

// Forward declaration
class Othello;

// Usage:
// is_tile_black(x=0,y=1) = black && (x*8 + y) 
struct GameState {
    uint64_t white;
    uint64_t black;
    bool black_turn;
    
    // Operators for map/set usage
    bool operator<(const GameState& other) const {
        if (white != other.white) return white < other.white;
        if (black != other.black) return black < other.black;
        return black_turn < other.black_turn;
    }
    
    bool operator==(const GameState& other) const {
        return white == other.white && black == other.black && black_turn == other.black_turn;
    }
};

// Quick hash for maps/sets
struct GameStateHash {
    size_t operator()(const GameState& p) const {
        // Start with the white pieces
        size_t seed = std::hash<uint64_t>()(p.white);

        seed ^= std::hash<uint64_t>()(p.black) + 0x9E3779B97F4A7C15ULL + (seed << 6) + (seed >> 2);
        seed ^= std::hash<bool>()(p.black_turn) + 0x9E3779B97F4A7C15ULL + (seed << 6) + (seed >> 2);

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
