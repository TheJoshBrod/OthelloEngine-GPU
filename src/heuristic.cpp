#include "heuristic.h"
#include "serial.h"
#include <cstdint>
#include <cmath>

// File masks to avoid shifts wrapping rows
static const uint64_t FILE_A = 0x0101010101010101ULL;
static const uint64_t FILE_H = 0x8080808080808080ULL;

static inline int popcount(uint64_t x){ return __builtin_popcountll(x); }

int num_valid_moves(const GameState& state, bool for_black){
    GameState s = state;
    s.x_turn = for_black;
    auto moves = find_all_moves(s);
    return static_cast<int>(moves.size());
}

// Pure bitboard evaluator implementing the provided heuristic
double dynamic_heuristic_evaluation(const GameState& state, bool is_black){
    uint64_t x = state.x;
    uint64_t o = state.o;
    uint64_t my_pieces = is_black ? x : o;
    uint64_t opp_pieces = is_black ? o : x;
    uint64_t occupied = x | o;
    uint64_t empty = ~occupied;

    double p = 0, c = 0, l = 0, m = 0, f = 0, d = 0;

    // positional weights
    const int V[8][8] = {
        {20, -3, 11, 8, 8, 11, -3, 20},
        {-3, -7, -4, 1, 1, -4, -7, -3},
        {11, -4, 2, 2, 2, 2, -4, 11},
        {8, 1, 2, -3, -3, 2, 1, 8},
        {8, 1, 2, -3, -3, 2, 1, 8},
        {11, -4, 2, 2, 2, 2, -4, 11},
        {-3, -7, -4, 1, 1, -4, -7, -3},
        {20, -3, 11, 8, 8, 11, -3, 20}
    };

    // Piece counts
    int my_tiles = popcount(my_pieces);
    int opp_tiles = popcount(opp_pieces);

    // d: positional score
    for (int pos = 0; pos < 64; ++pos){
        int r = pos / 8, c = pos % 8;
        uint64_t bit = (1ULL << pos);
        if (my_pieces & bit) d += V[r][c];
        else if (opp_pieces & bit) d -= V[r][c];
    }

    // frontier tiles: pieces adjacent to an empty square
    uint64_t e = empty;
    uint64_t adj_empty = 0ULL;
    // north
    adj_empty |= (e << 8);
    // south
    adj_empty |= (e >> 8);
    // east
    adj_empty |= ((e & ~FILE_H) << 1);
    // west
    adj_empty |= ((e & ~FILE_A) >> 1);
    // NE
    adj_empty |= ((e & ~FILE_H) << 9);
    // NW
    adj_empty |= ((e & ~FILE_A) << 7);
    // SE
    adj_empty |= ((e & ~FILE_H) >> 7);
    // SW
    adj_empty |= ((e & ~FILE_A) >> 9);

    int my_front_tiles = popcount(my_pieces & adj_empty);
    int opp_front_tiles = popcount(opp_pieces & adj_empty);

    if (my_tiles + opp_tiles > 0){
        if (my_tiles > opp_tiles) p = (100.0 * my_tiles) / (my_tiles + opp_tiles);
        else if (my_tiles < opp_tiles) p = -(100.0 * opp_tiles) / (my_tiles + opp_tiles);
        else p = 0;
    } else p = 0;

    if (my_front_tiles + opp_front_tiles > 0){
        if (my_front_tiles > opp_front_tiles) f = -(100.0 * my_front_tiles) / (my_front_tiles + opp_front_tiles);
        else if (my_front_tiles < opp_front_tiles) f = (100.0 * opp_front_tiles) / (my_front_tiles + opp_front_tiles);
        else f = 0;
    } else f = 0;

    // Corner occupancy
    int my_corners = 0, opp_corners = 0;
    auto corner = [&](int pos){
        uint64_t bit = 1ULL << pos;
        if (my_pieces & bit) my_corners++;
        else if (opp_pieces & bit) opp_corners++;
    };
    corner(0); corner(7); corner(56); corner(63);
    c = 25.0 * (my_corners - opp_corners);

    // Corner closeness
    int my_close = 0, opp_close = 0;
    // top-left corner neighbors: (0,1)=1, (1,1)=9, (1,0)=8
    if (!(occupied & (1ULL << 0))){
        if (my_pieces & (1ULL << 1)) my_close++; else if (opp_pieces & (1ULL << 1)) opp_close++;
        if (my_pieces & (1ULL << 9)) my_close++; else if (opp_pieces & (1ULL << 9)) opp_close++;
        if (my_pieces & (1ULL << 8)) my_close++; else if (opp_pieces & (1ULL << 8)) opp_close++;
    }
    // top-right (0,7)=7 -> neighbors (0,6)=6, (1,6)=14, (1,7)=15
    if (!(occupied & (1ULL << 7))){
        if (my_pieces & (1ULL << 6)) my_close++; else if (opp_pieces & (1ULL << 6)) opp_close++;
        if (my_pieces & (1ULL << 14)) my_close++; else if (opp_pieces & (1ULL << 14)) opp_close++;
        if (my_pieces & (1ULL << 15)) my_close++; else if (opp_pieces & (1ULL << 15)) opp_close++;
    }
    // bottom-left (7,0)=56 neighbors (6,0)=48, (6,1)=49, (7,1)=57
    if (!(occupied & (1ULL << 56))){
        if (my_pieces & (1ULL << 57)) my_close++; else if (opp_pieces & (1ULL << 57)) opp_close++;
        if (my_pieces & (1ULL << 49)) my_close++; else if (opp_pieces & (1ULL << 49)) opp_close++;
        if (my_pieces & (1ULL << 48)) my_close++; else if (opp_pieces & (1ULL << 48)) opp_close++;
    }
    // bottom-right (7,7)=63 neighbors (6,7)=55, (6,6)=54, (7,6)=62
    if (!(occupied & (1ULL << 63))){
        if (my_pieces & (1ULL << 55)) my_close++; else if (opp_pieces & (1ULL << 55)) opp_close++;
        if (my_pieces & (1ULL << 54)) my_close++; else if (opp_pieces & (1ULL << 54)) opp_close++;
        if (my_pieces & (1ULL << 62)) my_close++; else if (opp_pieces & (1ULL << 62)) opp_close++;
    }
    l = -12.5 * (my_close - opp_close);

    // Mobility using bitboard move generator
    int my_moves = num_valid_moves(state, is_black);
    int opp_moves = num_valid_moves(state, !is_black);
    if (my_moves + opp_moves > 0){
        if (my_moves > opp_moves) m = (100.0 * my_moves) / (my_moves + opp_moves);
        else if (my_moves < opp_moves) m = -(100.0 * opp_moves) / (my_moves + opp_moves);
        else m = 0;
    } else m = 0;

    double score = (10.0 * p) + (801.724 * c) + (382.026 * l) + (78.922 * m) + (74.396 * f) + (10.0 * d);
    return score;
}
