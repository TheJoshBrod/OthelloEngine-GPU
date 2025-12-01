// Parallelizes work for negamax
#include "parallel.h"
#include "othello.h"
#include "heuristic.h"
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cfloat>


// Find next generation of board states
__global__ void find_all_moves_kernel(const uint64_t* x_in, const uint64_t* o_in, const uint8_t* x_turn_in,
                                    int num_states,
                                    uint64_t* x_out, uint64_t* o_out, uint8_t* x_turn_out,
                                    int* parent_idx_out,
                                    int* out_count,
                                    int max_output) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;

    uint64_t state_x = x_in[idx];
    uint64_t state_o = o_in[idx];
    uint8_t turn = x_turn_in[idx];

    uint64_t my_pieces = turn ? state_x : state_o;
    uint64_t opp_pieces = turn ? state_o : state_x;
    uint64_t empty = ~(my_pieces | opp_pieces);

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

    // --- FIND VALID MOVES ---
    uint64_t valid_moves = 0;

    for (int dir = 0; dir < 8; dir++) {
        int shift = shifts[dir];
        uint64_t mask = masks[dir];

        uint64_t candidates = opp_pieces & mask;

        if (shift > 0)
            candidates &= (my_pieces << shift);
        else
            candidates &= (my_pieces >> -shift);

        for (int i = 0; i < 5; i++) {
            if (shift > 0)
                candidates |= (candidates << shift) & opp_pieces & mask;
            else
                candidates |= (candidates >> -shift) & opp_pieces & mask;
        }

        if (shift > 0)
            valid_moves |= (candidates << shift) & empty & mask;
        else
            valid_moves |= (candidates >> -shift) & empty & mask;
    }
    if (!valid_moves){
        // No valid moves for current player - need to pass
        // First check if opponent has any valid moves
        uint64_t opp_valid_moves = 0;
        
        for (int dir = 0; dir < 8; dir++) {
            int shift = shifts[dir];
            uint64_t mask = masks[dir];
            uint64_t candidates = my_pieces & mask;
            
            if (shift > 0)
                candidates &= (opp_pieces << shift);
            else
                candidates &= (opp_pieces >> -shift);
            
            for (int i = 0; i < 5; i++) {
                if (shift > 0)
                    candidates |= (candidates << shift) & my_pieces & mask;
                else
                    candidates |= (candidates >> -shift) & my_pieces & mask;
            }
            
            if (shift > 0)
                opp_valid_moves |= (candidates << shift) & empty & mask;
            else
                opp_valid_moves |= (candidates >> -shift) & empty & mask;
        }
        
        // If Opponent can move, just flip turn
        if (opp_valid_moves) {
            int out_idx = atomicAdd(out_count, 1);
            if (out_idx < max_output) {
                parent_idx_out[out_idx] = idx;
                x_turn_out[out_idx] = !turn;
                x_out[out_idx] = state_x;
                o_out[out_idx] = state_o;
            }
        }
        // else: both players have no moves aka game over
    }

    // --- GENERATE RESULTING STATES ---
    while (valid_moves) {
        uint64_t move_bit = valid_moves & -valid_moves;
        valid_moves ^= move_bit;

        uint64_t new_my = my_pieces;
        uint64_t new_opp = opp_pieces;
        uint64_t flipped = 0;

        // Perform flipping
        for (int dir = 0; dir < 8; dir++) {
            int shift = shifts[dir];
            uint64_t pos = move_bit;
            uint64_t line = 0;

            while (true) {
                if (shift > 0) pos <<= shift;
                else pos >>= -shift;

                if (!pos || !(pos & new_opp)) break;
                line |= pos;
            }

            if (pos & new_my)
                flipped |= line;
        }

        new_my |= move_bit | flipped;
        new_opp &= ~flipped;

        // Atomically get output slot
        int out_idx = atomicAdd(out_count, 1);
        if (out_idx < max_output) {
            parent_idx_out[out_idx] = idx;
            x_turn_out[out_idx] = !turn;
            if (turn) {
                x_out[out_idx] = new_my;
                o_out[out_idx] = new_opp;
            } else {
                o_out[out_idx] = new_my;
                x_out[out_idx] = new_opp;
            }
        }
    }
}


static int g_time_limit_ms = 0;
static auto g_start_time = std::chrono::steady_clock::now();
static bool time_exceeded() {
    if (g_time_limit_ms <= 0) return false;
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_start_time).count();
    return elapsed >= g_time_limit_ms;
}

std::unordered_set<GameState, GameStateHash> serial_find_all_moves(GameState state) {
    std::unordered_set<GameState, GameStateHash> moves;
    
    uint64_t my_pieces  = state.x_turn ? state.x : state.o;
    uint64_t opp_pieces = state.x_turn ? state.o : state.x;
    uint64_t empty      = ~(my_pieces | opp_pieces);
    
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

    // --- FIND VALID MOVE POSITIONS ---
    for (int dir = 0; dir < 8; dir++) {
        int shift = shifts[dir];
        uint64_t mask = masks[dir];
        
        uint64_t candidates = opp_pieces & mask;
        
        if (shift > 0)
            candidates &= (my_pieces << shift);
        else
            candidates &= (my_pieces >> -shift);

        for (int i = 0; i < 5; i++) {
            if (shift > 0)
                candidates |= (candidates << shift) & opp_pieces & mask;
            else
                candidates |= (candidates >> -shift) & opp_pieces & mask;
        }

        if (shift > 0)
            valid_moves |= (candidates << shift) & empty & mask;
        else
            valid_moves |= (candidates >> -shift) & empty & mask;
    }

    // --- GENERATE RESULTING STATES ---
    while (valid_moves) {
        uint64_t move_bit = valid_moves & -valid_moves;
        valid_moves ^= move_bit;

        uint64_t new_my = my_pieces;
        uint64_t new_opp = opp_pieces;
        uint64_t flipped = 0;

        // Perform flipping logic (from old make_move)
        for (int dir = 0; dir < 8; dir++) {
            int shift = shifts[dir];
            uint64_t pos = move_bit;
            uint64_t line = 0;

            while (true) {
                if (shift > 0) pos <<= shift;
                else pos >>= -shift;

                if (!pos || !(pos & new_opp)) break;
                line |= pos;
            }

            if (pos & new_my)
                flipped |= line;
        }

        new_my |= move_bit | flipped;
        new_opp &= ~flipped;

        GameState next;
        next.x_turn = !state.x_turn;

        if (state.x_turn) {
            next.x = new_my;
            next.o = new_opp;
        } else {
            next.o = new_my;
            next.x = new_opp;
        }
        next.score = dynamic_heuristic_evaluation(next, next.x_turn);
        moves.insert(next);
    }

    return moves;
}

class GPUBatchProcessor {
private:
    
    // GameState info
    uint64_t *d_x, *d_o;
    uint8_t *d_turns;
    
    // Find All Moves info
    uint64_t *d_new_x, *d_new_o;
    uint8_t *d_new_turns;
    int *d_out_count;
    int *d_parent_indices;

    int capacity;
    int max_output_capacity;


public:
    GPUBatchProcessor(int max_capacity) : capacity(max_capacity) {
        max_output_capacity = max_capacity * 15;
       
        cudaMalloc(&d_x, capacity * sizeof(uint64_t));
        cudaMalloc(&d_o, capacity * sizeof(uint64_t));
        cudaMalloc(&d_turns, capacity * sizeof(uint8_t));
        cudaMalloc(&d_parent_indices, capacity * sizeof(int));
        
        // Add these:
        cudaMalloc(&d_new_x, max_output_capacity * sizeof(uint64_t));
        cudaMalloc(&d_new_o, max_output_capacity * sizeof(uint64_t));
        cudaMalloc(&d_new_turns, max_output_capacity * sizeof(uint8_t));
        cudaMalloc(&d_out_count, sizeof(int));
    }

    ~GPUBatchProcessor() {
        cudaFree(d_x);
        cudaFree(d_o);
        cudaFree(d_turns);
        cudaFree(d_parent_indices);
        cudaFree(d_new_x);
        cudaFree(d_new_o);
        cudaFree(d_new_turns);
        cudaFree(d_out_count);
    }

    std::vector<std::pair<GameState, int>> find_all_moves(const std::vector<GameState>& states) {
        int num_states = states.size();
        
        // Prepare host data
        std::vector<uint64_t> h_x(num_states);
        std::vector<uint64_t> h_o(num_states);
        std::vector<uint8_t> h_turns(num_states);
        
        for (int i = 0; i < num_states; i++) {
            h_x[i] = states[i].x;
            h_o[i] = states[i].o;
            h_turns[i] = states[i].x_turn;
        }

        // Upload input states
        cudaMemcpy(d_x, h_x.data(), num_states * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_o, h_o.data(), num_states * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_turns, h_turns.data(), num_states * sizeof(uint8_t), cudaMemcpyHostToDevice);

        // Reset output counter
        int zero = 0;
        cudaMemcpy(d_out_count, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // Execute kernel (finds all moves directly, no need to count first)
        int blockSize = 256;
        int gridSize = (num_states + blockSize - 1) / blockSize;
        
        find_all_moves_kernel<<<gridSize, blockSize>>>(
            d_x, d_o, d_turns,              // Input states
            num_states,                      // Number of input states
            d_new_x, d_new_o, d_new_turns,  // Output states
            d_parent_indices,               // Stores parent's location
            d_out_count,                     // Output counter
            max_output_capacity              // Max outputs
        );
        
        cudaDeviceSynchronize();

        // Get the actual number of states generated
        int actual_count;
        cudaMemcpy(&actual_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Download results
        std::vector<uint64_t> h_new_x(actual_count);
        std::vector<uint64_t> h_new_o(actual_count);
        std::vector<uint8_t> h_new_turns(actual_count);
        std::vector<int> h_parent_indices(actual_count);
        
        cudaMemcpy(h_new_x.data(), d_new_x, actual_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_new_o.data(), d_new_o, actual_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_new_turns.data(), d_new_turns, actual_count * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_parent_indices.data(), d_parent_indices, actual_count * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Convert to GameState vector
        std::vector<std::pair<GameState, int>> result;
        result.reserve(actual_count);
        for (int i = 0; i < actual_count; i++) {
            GameState state;
            state.x = h_new_x[i];
            state.o = h_new_o[i];
            state.x_turn = h_new_turns[i];
            result.push_back({state, h_parent_indices[i]});
        }
        
        return result;
    }

};

GameState negamax_parallel(Othello* game, int time_limit_ms){
    // Calculates next best move based on current game state in parallel
    // Two phases:
    // - Expansion (Generate tree of future states and estimate "potential" given a heuristic)
    // - Reduction (Shrink list of future states using children's score to return best next move)

    int MAX_GPU_CAPACITY = 1e6;
    int MIN_GPU_CAPACITY = 500;

    // Saturate GPU work by getting initial batch of gamestates for 
    std::unordered_map<GameState, GameState, GameStateHash> state_map;
    std::unordered_set<GameState, GameStateHash> current_gen = {game->get_board()};
    while(current_gen.size() < MIN_GPU_CAPACITY && !current_gen.empty()){
        std::unordered_set<GameState, GameStateHash> next_gen;

        for(const auto& old_state : current_gen){
            std::unordered_set<GameState, GameStateHash> new_states = serial_find_all_moves(old_state);

            for (const auto& new_state : new_states){
                if (state_map.find(new_state) == state_map.end()) {
                    state_map[new_state] = old_state;
                    next_gen.insert(new_state);
                }
            }
        }
        current_gen = std::move(next_gen);
    }
    if (current_gen.empty()){
        return game->get_board();
    }


    // Get all possible moves from above serial pre-work game states
    g_time_limit_ms = time_limit_ms;
    g_start_time = std::chrono::steady_clock::now(); 
    GPUBatchProcessor gpu(MAX_GPU_CAPACITY);
    while(!time_exceeded() && !current_gen.empty()){

        // ***************
        // Expansion Phase
        // ***************
        std::vector<GameState> current_gen_vec(current_gen.begin(), current_gen.end());
        std::vector<std::pair<GameState, int>> results = gpu.find_all_moves(current_gen_vec);

        current_gen.clear();
        int new_states_added = 0;
        for(auto state : results){
            if (state_map.find(state.first) != state_map.end()) {
                continue;
            }
            state_map[state.first] = current_gen_vec[state.second];
            current_gen.insert(state.first);
            new_states_added++;
        }
        if (new_states_added == 0){
            break;
        }

        // ***************
        // Depth Phase
        // ***************
    }
    
    // TODO: This is wrong fix it
    return game->get_board();
}
