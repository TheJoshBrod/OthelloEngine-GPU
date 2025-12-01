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

// Evaluate board state (simple piece count heuristic)
// -------- CONSTANT POSITIONAL TABLE --------
__constant__ int V[64] = {
    20,-3,11,8,8,11,-3,20,
    -3,-7,-4,1,1,-4,-7,-3,
    11,-4,2,2,2,2,-4,11,
    8,1,2,-3,-3,2,1,8,
    8,1,2,-3,-3,2,1,8,
    11,-4,2,2,2,2,-4,11,
    -3,-7,-4,1,1,-4,-7,-3,
    20,-3,11,8,8,11,-3,20
};

// File masks for frontier detection
__device__ constexpr uint64_t FILE_A = 0x0101010101010101ULL;
__device__ constexpr uint64_t FILE_H = 0x8080808080808080ULL;

// ---------------- DYNAMIC HEURISTIC EVALUATION ----------------
__device__ int evaluate_board(uint64_t x, uint64_t o, bool is_x) {

    uint64_t my = is_x ? x : o;
    uint64_t opp = is_x ? o : x;
    uint64_t occupied = x | o;
    uint64_t empty = ~occupied;

    double p=0,c=0,l=0,m=0,f=0,d=0;

    // --- Tile counts ---
    int my_tiles  = __popcll(my);
    int opp_tiles = __popcll(opp);

    // --- Positional weights (lookup) ---
    for (int pos = 0; pos < 64; pos++) {
        uint64_t bit = (1ULL << pos);
        if (my & bit)      d += V[pos];
        else if (opp & bit) d -= V[pos];
    }

    // --- Frontier tiles ---
    uint64_t adj = 0;
    adj |= (empty << 8);
    adj |= (empty >> 8);
    adj |= ((empty & ~FILE_H) << 1);
    adj |= ((empty & ~FILE_A) >> 1);
    adj |= ((empty & ~FILE_H) << 9);
    adj |= ((empty & ~FILE_A) << 7);
    adj |= ((empty & ~FILE_H) >> 7);
    adj |= ((empty & ~FILE_A) >> 9);

    int my_front  = __popcll(my  & adj);
    int opp_front = __popcll(opp & adj);

    // --- Piece differential ---
    if (my_tiles + opp_tiles)
        p = (my_tiles > opp_tiles) ?
            (100.0 * my_tiles) / (my_tiles + opp_tiles) :
            (my_tiles < opp_tiles) ?
                -(100.0 * opp_tiles) / (my_tiles + opp_tiles) : 0;

    // --- Frontier pressure ---
    if (my_front + opp_front)
        f = (my_front < opp_front) ?
            (100.0 * opp_front)/(my_front + opp_front) :
            (my_front > opp_front) ?
                -(100.0 * my_front)/(my_front + opp_front) : 0;

    // --- Corners ---
    const int corners[4]={0,7,56,63};
    int my_c=0, opp_c=0;
    for(int i=0;i<4;i++){
        uint64_t b=1ULL<<corners[i];
        if(my & b) my_c++;
        else if(opp & b) opp_c++;
    }
    c = 25.0*(my_c - opp_c);

    // Corner adjacency
    const int close[12]={1,9,8,6,14,15,57,49,48,55,54,62};
    int my_close=0,opp_close=0;
    for(int i=0;i<4;i++){
        if(!(occupied & (1ULL<<corners[i]))){
            for(int j=0;j<3;j++){
                int sq = close[i*3+j];
                if(my & (1ULL<<sq)) my_close++;
                else if(opp & (1ULL<<sq)) opp_close++;
            }
        }
    }
    l = -12.5*(my_close - opp_close);

    // --- Mobility (using your existing move generator) ---
    int my_moves  = __popcll(get_valid_moves(my, opp));
    int opp_moves = __popcll(get_valid_moves(opp, my));

    if(my_moves + opp_moves)
        m = (my_moves > opp_moves) ?
            (100.0 * my_moves)/(my_moves + opp_moves) :
            (my_moves < opp_moves) ?
                -(100.0 * opp_moves)/(my_moves + opp_moves) : 0;

    // --- Final Weighted Score (exact formula) ---
    double score =
        (10.0 * p) + (801.724 * c) + (382.026 * l) +
        (78.922 * m) + (74.396 * f) + (10.0 * d);

    return (int)score;
}



// Get valid moves for a position
__device__ uint64_t get_valid_moves(uint64_t my_pieces, uint64_t opp_pieces) {
    uint64_t empty = ~(my_pieces | opp_pieces);
    uint64_t valid_moves = 0;
    
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
    
    return valid_moves;
}

// Apply a move and return new board state
__device__ void apply_move(uint64_t move_bit, uint64_t my_pieces, uint64_t opp_pieces,
                           uint64_t* new_my, uint64_t* new_opp) {
    const int shifts[8] = {8, 9, 1, -7, -8, -9, -1, 7};
    
    *new_my = my_pieces;
    *new_opp = opp_pieces;
    uint64_t flipped = 0;

    for (int dir = 0; dir < 8; dir++) {
        int shift = shifts[dir];
        uint64_t pos = move_bit;
        uint64_t line = 0;

        while (true) {
            if (shift > 0) pos <<= shift;
            else pos >>= -shift;

            if (!pos || !(pos & *new_opp)) break;
            line |= pos;
        }

        if (pos & *new_my)
            flipped |= line;
    }

    *new_my |= move_bit | flipped;
    *new_opp &= ~flipped;
}

// Recursive alpha-beta search
__device__ int alphabeta(uint64_t x_pieces, uint64_t o_pieces, bool x_turn,
                        int depth, int alpha, int beta, bool maximizing) {
    if (depth == 0) {
        return evaluate_board(x_pieces, o_pieces, true);
    }
    
    uint64_t my_pieces = x_turn ? x_pieces : o_pieces;
    uint64_t opp_pieces = x_turn ? o_pieces : x_pieces;
    uint64_t valid_moves = get_valid_moves(my_pieces, opp_pieces);
    
    // No valid moves - check if game over or pass
    if (!valid_moves) {
        uint64_t opp_valid = get_valid_moves(opp_pieces, my_pieces);
        if (!opp_valid) {
            // Game over - return final score
            return evaluate_board(x_pieces, o_pieces, true);
        }
        // Pass turn
        return alphabeta(x_pieces, o_pieces, !x_turn, depth - 1, alpha, beta, !maximizing);
    }
    
    if (maximizing) {
        int max_eval = -10000;
        while (valid_moves) {
            uint64_t move = valid_moves & -valid_moves;
            valid_moves ^= move;
            
            uint64_t new_my, new_opp;
            apply_move(move, my_pieces, opp_pieces, &new_my, &new_opp);
            
            uint64_t new_x = x_turn ? new_my : new_opp;
            uint64_t new_o = x_turn ? new_opp : new_my;
            
            int eval = alphabeta(new_x, new_o, !x_turn, depth - 1, alpha, beta, false);
            max_eval = max(max_eval, eval);
            alpha = max(alpha, eval);
            
            if (beta <= alpha)
                break; // Beta cutoff
        }
        return max_eval;
    } else {
        int min_eval = 10000;
        while (valid_moves) {
            uint64_t move = valid_moves & -valid_moves;
            valid_moves ^= move;
            
            uint64_t new_my, new_opp;
            apply_move(move, my_pieces, opp_pieces, &new_my, &new_opp);
            
            uint64_t new_x = x_turn ? new_my : new_opp;
            uint64_t new_o = x_turn ? new_opp : new_my;
            
            int eval = alphabeta(new_x, new_o, !x_turn, depth - 1, alpha, beta, true);
            min_eval = min(min_eval, eval);
            beta = min(beta, eval);
            
            if (beta <= alpha)
                break; // Alpha cutoff
        }
        return min_eval;
    }
}

// Kernel: Each thread evaluates one root move
__global__ void alphabeta_search_kernel(uint64_t x_in, uint64_t o_in, uint8_t x_turn_in,
                                       int max_depth,
                                       uint64_t* moves_out,
                                       int* scores_out,
                                       int* num_moves_out) {
    __shared__ int move_count;
    __shared__ uint64_t shared_moves;
    
    int tid = threadIdx.x;
    
    // First thread counts and stores valid moves
    if (tid == 0) {
        uint64_t my_pieces = x_turn_in ? x_in : o_in;
        uint64_t opp_pieces = x_turn_in ? o_in : x_in;
        shared_moves = get_valid_moves(my_pieces, opp_pieces);
        move_count = __popcll(shared_moves);
        *num_moves_out = move_count;
    }
    __syncthreads();
    
    // Each thread handles one move
    if (tid >= move_count) return;
    
    uint64_t my_pieces = x_turn_in ? x_in : o_in;
    uint64_t opp_pieces = x_turn_in ? o_in : x_in;
    
    // Extract the tid-th move
    uint64_t moves_copy = shared_moves;
    uint64_t my_move = 0;
    for (int i = 0; i <= tid; i++) {
        my_move = moves_copy & -moves_copy;
        moves_copy ^= my_move;
    }
    
    // Apply move
    uint64_t new_my, new_opp;
    apply_move(my_move, my_pieces, opp_pieces, &new_my, &new_opp);
    
    uint64_t new_x = x_turn_in ? new_my : new_opp;
    uint64_t new_o = x_turn_in ? new_opp : new_my;
    
    // Search with alpha-beta
    const int NEG_INF = -10000;
    const int POS_INF = 10000;
    int score = alphabeta(new_x, new_o, !x_turn_in, max_depth - 1, 
                        NEG_INF, POS_INF, !x_turn_in);
    
    moves_out[tid] = my_move;
    scores_out[tid] = score;
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
    uint64_t *d_x, *d_o;
    uint8_t *d_turns;
    
    uint64_t *d_new_x, *d_new_o;
    uint8_t *d_new_turns;
    int *d_out_count;
    int *d_parent_indices;
    
    // For alpha-beta
    int *d_scores;
    uint64_t *d_moves;
    int *d_num_moves;

    int capacity;
    int max_output_capacity;

public:
    GPUBatchProcessor(int max_capacity) : capacity(max_capacity) {
        max_output_capacity = max_capacity * 15;
       
        cudaMalloc(&d_x, capacity * sizeof(uint64_t));
        cudaMalloc(&d_o, capacity * sizeof(uint64_t));
        cudaMalloc(&d_turns, capacity * sizeof(uint8_t));
        cudaMalloc(&d_parent_indices, capacity * sizeof(int));
        cudaMalloc(&d_scores, capacity * sizeof(int));
        
        cudaMalloc(&d_new_x, max_output_capacity * sizeof(uint64_t));
        cudaMalloc(&d_new_o, max_output_capacity * sizeof(uint64_t));
        cudaMalloc(&d_new_turns, max_output_capacity * sizeof(uint8_t));
        cudaMalloc(&d_out_count, sizeof(int));
        
        cudaMalloc(&d_moves, 64 * sizeof(uint64_t));
        cudaMalloc(&d_num_moves, sizeof(int));
    }

    ~GPUBatchProcessor() {
        cudaFree(d_x);
        cudaFree(d_o);
        cudaFree(d_turns);
        cudaFree(d_parent_indices);
        cudaFree(d_scores);
        cudaFree(d_new_x);
        cudaFree(d_new_o);
        cudaFree(d_new_turns);
        cudaFree(d_out_count);
        cudaFree(d_moves);
        cudaFree(d_num_moves);
    }

    std::vector<std::pair<GameState, int>> find_all_moves(const std::vector<GameState>& states) {
        int num_states = states.size();
        
        std::vector<uint64_t> h_x(num_states);
        std::vector<uint64_t> h_o(num_states);
        std::vector<uint8_t> h_turns(num_states);
        
        for (int i = 0; i < num_states; i++) {
            h_x[i] = states[i].x;
            h_o[i] = states[i].o;
            h_turns[i] = states[i].x_turn;
        }

        cudaMemcpy(d_x, h_x.data(), num_states * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_o, h_o.data(), num_states * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_turns, h_turns.data(), num_states * sizeof(uint8_t), cudaMemcpyHostToDevice);

        int zero = 0;
        cudaMemcpy(d_out_count, &zero, sizeof(int), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int gridSize = (num_states + blockSize - 1) / blockSize;
        

        for(int i = 0; i < states.size(); i++){
            alphabeta_search_kernel<<<gridSize, blockSize>>>(
                d_x, d_o, d_turns,
                num_states,
                d_new_x, d_new_o, d_new_turns,
                d_parent_indices,
                d_out_count,
                max_output_capacity
            );
        }
        
        
        cudaDeviceSynchronize();

        int actual_count;
        cudaMemcpy(&actual_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        std::vector<uint64_t> h_new_x(actual_count);
        std::vector<uint64_t> h_new_o(actual_count);
        std::vector<uint8_t> h_new_turns(actual_count);
        std::vector<int> h_parent_indices(actual_count);
        
        cudaMemcpy(h_new_x.data(), d_new_x, actual_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_new_o.data(), d_new_o, actual_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_new_turns.data(), d_new_turns, actual_count * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_parent_indices.data(), d_parent_indices, actual_count * sizeof(int), cudaMemcpyDeviceToHost);
        
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
    
    // Evaluate multiple states with alpha-beta
    std::vector<int> batch_alphabeta(const std::vector<GameState>& states, int depth) {
        int num_states = states.size();
        
        std::vector<uint64_t> h_x(num_states);
        std::vector<uint64_t> h_o(num_states);
        std::vector<uint8_t> h_turns(num_states);
        
        for (int i = 0; i < num_states; i++) {
            h_x[i] = states[i].x;
            h_o[i] = states[i].o;
            h_turns[i] = states[i].x_turn;
        }

        cudaMemcpy(d_x, h_x.data(), num_states * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_o, h_o.data(), num_states * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_turns, h_turns.data(), num_states * sizeof(uint8_t), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int gridSize = (num_states + blockSize - 1) / blockSize;
        
        batch_alphabeta_kernel<<<gridSize, blockSize>>>(
            d_x, d_o, d_turns,
            num_states,
            depth,
            d_scores
        );
        
        cudaDeviceSynchronize();

        std::vector<int> scores(num_states);
        cudaMemcpy(scores.data(), d_scores, num_states * sizeof(int), cudaMemcpyDeviceToHost);
        
        return scores;
    }
    
    // Solve single position to given depth
    std::pair<uint64_t, int> solve_position(GameState state, int depth) {
        alphabeta_search_kernel<<<1, 64>>>(
            state.x, state.o, state.x_turn,
            depth,
            d_moves,
            d_scores,
            d_num_moves
        );
        
        cudaDeviceSynchronize();
        
        int num_moves;
        cudaMemcpy(&num_moves, d_num_moves, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (num_moves == 0) {
            return {0, 0};
        }
        
        std::vector<uint64_t> moves(num_moves);
        std::vector<int> scores(num_moves);
        
        cudaMemcpy(moves.data(), d_moves, num_moves * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(scores.data(), d_scores, num_moves * sizeof(int), cudaMemcpyDeviceToHost);
        
        int best_idx = 0;
        int best_score = scores[0];
        for (int i = 1; i < num_moves; i++) {
            if (scores[i] > best_score) {
                best_score = scores[i];
                best_idx = i;
            }
        }
        
        return {moves[best_idx], best_score};
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
    int current_depth = 1;
    GameState best_move = game->get_board();
    
    while (!time_exceeded()) {
        // Evaluate all current states at current depth
        std::vector<GameState> state_vec(current_gen.begin(), current_gen.end());
        
        if (state_vec.empty()) break;
        
        std::vector<int> scores = gpu.batch_alphabeta(state_vec, current_depth);
        
        // Find promising states (top 50% or states above threshold)
        std::vector<std::pair<int, GameState>> scored_states;
        for (size_t i = 0; i < state_vec.size(); i++) {
            scored_states.push_back({scores[i], state_vec[i]});
        }
        
        // Sort by score descending
        std::sort(scored_states.begin(), scored_states.end(),
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Keep top states or expand if we have capacity
        size_t keep_count = std::min(scored_states.size(), 
                                    (size_t)(MAX_GPU_CAPACITY / 20));
        
        current_gen.clear();
        for (size_t i = 0; i < keep_count && !time_exceeded(); i++) {
            current_gen.insert(scored_states[i].second);
        }
        
        // Track best first move
        GameState temp = scored_states[0].second;
        while (state_map.find(temp) != state_map.end() && 
               state_map[temp].x != game->get_board().x) {
            temp = state_map[temp];
        }
        best_move = temp;
        
        // Expand next generation if time permits
        if (!time_exceeded() && current_gen.size() < MAX_GPU_CAPACITY / 2) {
            std::vector<GameState> expand_vec(current_gen.begin(), current_gen.end());
            auto results = gpu.find_all_moves(expand_vec);
            
            current_gen.clear();
            for (const auto& [state, parent_idx] : results) {
                if (state_map.find(state) == state_map.end()) {
                    state_map[state] = expand_vec[parent_idx];
                    current_gen.insert(state);
                }
            }
        }
        
        current_depth++;
        if (current_depth > 12) break;
    }
    
    return best_move;
}
