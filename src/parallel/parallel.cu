// Simple parallel stub: currently delegates to serial implementation.
#include "parallel.h"
#include "serial.h"
#include "othello.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <math_constants.h>



// Find heuristic score of each value
__global__ void heuristic(const uint64_t* x, const uint64_t* o, const uint8_t* x_turn, float* score, int n) {
    
    // CUDA math
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i++){
        uint64_t my_pieces  = (x_turn[i]) ? x[i] : o[i];
        uint64_t opp_pieces = (x_turn[i]) ? o[i] : x[i];
        uint64_t occupied = x[i] | o[i];
        uint64_t empty = ~(occupied);

        // File masks to avoid shifts wrapping rows
        static const uint64_t FILE_A = 0x0101010101010101ULL;
        static const uint64_t FILE_H = 0x8080808080808080ULL;

        float p = 0, c = 0, l = 0, m = 0, f = 0, d = 0;

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
        int my_tiles = __popcll(my_pieces);
        int opp_tiles = __popcll(opp_pieces);

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

        int my_front_tiles = __popcll(my_pieces & adj_empty);
        int opp_front_tiles = __popcll(opp_pieces & adj_empty);

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
        
        uint8_t my_moves = 0;
        uint8_t opp_moves = 0;

        for (int i = 0; i < 2; i++){
            if(i != 0){
                int64_t temp = opp_pieces; 
                opp_pieces = my_pieces;
                my_pieces = temp;
            }
            
            uint8_t valid_moves = 0;
            
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
                valid_moves++;
            }
            
            my_moves = (i == 0) ? valid_moves : my_moves;
            opp_moves = (i == 0) ? opp_moves : valid_moves ;
        }

        if (my_moves + opp_moves > 0){
            if (my_moves > opp_moves) m = (100.0 * my_moves) / (my_moves + opp_moves);
            else if (my_moves < opp_moves) m = -(100.0 * opp_moves) / (my_moves + opp_moves);
            else m = 0;
        } else m = 0;

        score[i] = (10.0 * p) + (801.724 * c) + (382.026 * l) + (78.922 * m) + (74.396 * f) + (10.0 * d);

    }
}

// Find the maximum value of each block in 1D array
__global__ void findMaxPerBlock(const float* arr, int size, float* blockVals, int* blockIdxs) {
    __shared__ float sharedVal[256];
    __shared__ int sharedIdx[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize
    if (idx < size) {
        sharedVal[threadIdx.x] = arr[idx];
        sharedIdx[threadIdx.x] = idx;
    } else {
        sharedVal[threadIdx.x] = -CUDART_INF_F;
        sharedIdx[threadIdx.x] = -1;
    }
    __syncthreads();

    // Reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (sharedVal[threadIdx.x + stride] > sharedVal[threadIdx.x]) {
                sharedVal[threadIdx.x] = sharedVal[threadIdx.x + stride];
                sharedIdx[threadIdx.x] = sharedIdx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    // Store block result
    if (threadIdx.x == 0) {
        blockVals[blockIdx.x] = sharedVal[0];
        blockIdxs[blockIdx.x] = sharedIdx[0];
    }
}

GameState negamax_parallel(Othello* game, int time_limit_ms){
    
    // Get all possible moves from current board state
    std::vector<GameState> states = find_all_moves(game->get_board());
    
    // Handle edge case
    if (states.empty()) {
        return game->get_board();
    }
    // If can't saturate the GPU use serial CPU
    if (states.size() < 500){
        return negamax_serial(game, time_limit_ms);
    }
    
    int n = states.size();
    
    // Allocate host memory
    uint64_t* h_x_states = (uint64_t*) malloc(sizeof(uint64_t) * n);
    uint64_t* h_o_states = (uint64_t*) malloc(sizeof(uint64_t) * n);
    uint8_t* h_x_turn = (uint8_t*) malloc(sizeof(uint8_t) * n);
    float* h_scores = (float*) malloc(sizeof(float) * n);
    
    // Copy game states to host arrays
    for (int i = 0; i < n; i++) {
        h_x_states[i] = states[i].x;
        h_o_states[i] = states[i].o;
        h_x_turn[i] = states[i].x_turn;
    }
    
    // Allocate memory
    uint64_t* d_x_states;
    uint64_t* d_o_states;
    uint8_t* d_x_turn;
    float* d_scores;
    
    cudaMalloc(&d_x_states, sizeof(uint64_t) * n);
    cudaMalloc(&d_o_states, sizeof(uint64_t) * n);
    cudaMalloc(&d_x_turn, sizeof(uint8_t) * n);
    cudaMalloc(&d_scores, sizeof(float) * n);
    
    cudaMemcpy(d_x_states, h_x_states, sizeof(uint64_t) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_o_states, h_o_states, sizeof(uint64_t) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_turn, h_x_turn, sizeof(uint8_t) * n, cudaMemcpyHostToDevice);
    
    // Launch heuristic kernel on all games found
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int threadsPerBlock = prop.maxThreadsPerBlock;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    heuristic<<<blocksPerGrid, threadsPerBlock>>>(d_x_states, d_o_states, d_x_turn, d_scores, n);

    
    // Find max per block
    float* d_blockVals;
    int* d_blockIdxs;
    cudaMalloc(&d_blockVals, sizeof(float) * blocksPerGrid);
    cudaMalloc(&d_blockIdxs, sizeof(int) * blocksPerGrid);
    
    findMaxPerBlock<<<blocksPerGrid, threadsPerBlock>>>(d_scores, n, d_blockVals, d_blockIdxs);
    
    float* h_blockVals = (float*) malloc(sizeof(float) * blocksPerGrid);
    int* h_blockIdxs = (int*) malloc(sizeof(int) * blocksPerGrid);
    
    cudaMemcpy(h_blockVals, d_blockVals, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blockIdxs, d_blockIdxs, sizeof(int) * blocksPerGrid, cudaMemcpyDeviceToHost);
    
    // Final reduction on CPU
    float maxVal = h_blockVals[0];
    int maxIdx = h_blockIdxs[0];
    for (int i = 1; i < blocksPerGrid; i++) {
        if (h_blockVals[i] > maxVal) {
            maxVal = h_blockVals[i];
            maxIdx = h_blockIdxs[i];
        }
    }
    GameState bestMove = states[maxIdx];
    
    // Clean up memory
    cudaFree(d_x_states);
    cudaFree(d_o_states);
    cudaFree(d_x_turn);
    cudaFree(d_scores);
    cudaFree(d_blockVals);
    cudaFree(d_blockIdxs);
    free(h_x_states);
    free(h_o_states);
    free(h_x_turn);
    free(h_scores);
    free(h_blockVals);
    free(h_blockIdxs);
    
    return bestMove;
}
