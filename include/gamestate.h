#ifndef GAMESTATE_H
#define GAMESTATE_H

#include <cstdint>
#include <functional>
#include <cstddef>

// Usage:
// is_tile_X(x=0,y=1) = x && (x*8 + y) 
struct GameState {
    uint64_t o;
    uint64_t x;
    bool x_turn;
    double score;
    
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

#endif



