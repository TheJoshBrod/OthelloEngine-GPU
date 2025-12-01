// negamax_parallel.cu
// PV-Split GPU-enabled Negamax connector for your Othello project
// - plugs into existing Othello class via negamax_fn
// - uses host heuristic (dynamic_heuristic_evaluation) for CPU PV search
// - runs sibling subtree evaluations in parallel on GPU
//
// Build with nvcc and link with your heuristic.o and other project objects.
// Example: nvcc -O3 -arch=sm_70 negamax_parallel.cu heuristic.cpp -o othello_gpu

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <cstring>
#include <cassert>
#include <unordered_map>
#include <cmath>

// include your headers
#include "pvsplit.h"
#include "GameState.h"
#include "othello.h"
#include "heuristic.h"   // provides dynamic_heuristic_evaluation(const GameState&, bool)
#include "serial.h"      // if you have serial find_all_moves etc; otherwise we use GameState helpers

// -------------------------------
// Tunable parameters
// -------------------------------
#define HOST_TT_MB 32           // Host transposition table size (MB)
#define DEV_TT_SIZE 256         // entries per-block device TT (shared mem)
#define MAX_GPU_CHILDREN 1024   // max number of sibling children we'll send to GPU in one batch
#define WORKER_BLOCK_THREADS 64 // threads per block (we only use thread 0 for control + others init shared mem)
#define GPU_STACK_MAX_DEPTH 20  // maximum ply / stack depth per worker (tune down if stack big)
#define DEVICE_MAX_DEPTH 8      // maximum depth worker will search (device-level limit)

// -------------------------------
// Utility: CUDA check
// -------------------------------
#define CUDA_CALL(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

// -------------------------------
// Device-side simple positional weights (must mirror CPU table for consistency)
// -------------------------------
__constant__ int d_pos_weights[64] = {
    20,-3,11,8,8,11,-3,20,
    -3,-7,-4,1,1,-4,-7,-3,
    11,-4,2,2,2,2,-4,11,
    8,1,2,-3,-3,2,1,8,
    8,1,2,-3,-3,2,1,8,
    11,-4,2,2,2,2,-4,11,
    -3,-7,-4,1,1,-4,-7,-3,
    20,-3,11,8,8,11,-3,20
};

// -------------------------------
// Device-friendly state
// -------------------------------
struct DeviceState {
    uint64_t o;
    uint64_t x;
    uint8_t x_turn; // 0 or 1
};

// -------------------------------
// Device move generation helpers (copy of your bitboard logic)
// -------------------------------
__device__ inline int d_popcount(uint64_t x) { return __popcll(x); }
__device__ constexpr uint64_t D_FILE_A = 0x0101010101010101ULL;
__device__ constexpr uint64_t D_FILE_H = 0x8080808080808080ULL;

__device__ inline uint64_t d_gen_moves_for(uint64_t P, uint64_t O) {
    uint64_t empty = ~(P|O);
    uint64_t moves = 0ULL;
    // East
    {
        uint64_t mask = O & ~D_FILE_H;
        uint64_t t = mask & (P << 1);
        t |= mask & (t << 1); t |= mask & (t << 1); t |= mask & (t << 1); t |= mask & (t << 1);
        moves |= empty & (t << 1);
    }
    // West
    {
        uint64_t mask = O & ~D_FILE_A;
        uint64_t t = mask & (P >> 1);
        t |= mask & (t >> 1); t |= mask & (t >> 1); t |= mask & (t >> 1); t |= mask & (t >> 1);
        moves |= empty & (t >> 1);
    }
    // North (+8)
    {
        uint64_t mask = O;
        uint64_t t = mask & (P << 8);
        t |= mask & (t << 8); t |= mask & (t << 8); t |= mask & (t << 8); t |= mask & (t << 8);
        moves |= empty & (t << 8);
    }
    // South (-8)
    {
        uint64_t mask = O;
        uint64_t t = mask & (P >> 8);
        t |= mask & (t >> 8); t |= mask & (t >> 8); t |= mask & (t >> 8); t |= mask & (t >> 8);
        moves |= empty & (t >> 8);
    }
    // NE (+9)
    {
        uint64_t mask = O & ~D_FILE_H;
        uint64_t t = mask & (P << 9);
        t |= mask & (t << 9); t |= mask & (t << 9); t |= mask & (t << 9); t |= mask & (t << 9);
        moves |= empty & (t << 9);
    }
    // SW (-9)
    {
        uint64_t mask = O & ~D_FILE_A;
        uint64_t t = mask & (P >> 9);
        t |= mask & (t >> 9); t |= mask & (t >> 9); t |= mask & (t >> 9); t |= mask & (t >> 9);
        moves |= empty & (t >> 9);
    }
    // NW (+7)
    {
        uint64_t mask = O & ~D_FILE_A;
        uint64_t t = mask & (P << 7);
        t |= mask & (t << 7); t |= mask & (t << 7); t |= mask & (t << 7); t |= mask & (t << 7);
        moves |= empty & (t << 7);
    }
    // SE (-7)
    {
        uint64_t mask = O & ~D_FILE_H;
        uint64_t t = mask & (P >> 7);
        t |= mask & (t >> 7); t |= mask & (t >> 7); t |= mask & (t >> 7); t |= mask & (t >> 7);
        moves |= empty & (t >> 7);
    }
    return moves;
}

__device__ inline void d_make_move_and_flip(uint64_t move_bit, uint64_t *P, uint64_t *O) {
    uint64_t flips = 0ULL;
    // East
    {
        uint64_t t = 0ULL;
        uint64_t x = move_bit << 1;
        while ((x & ~D_FILE_A) && (x & *O)) { t |= x; x <<= 1; }
        if (x & *P) flips |= t;
    }
    // West
    {
        uint64_t t = 0ULL;
        uint64_t x = move_bit >> 1;
        while ((x & ~D_FILE_H) && (x & *O)) { t |= x; x >>= 1; }
        if (x & *P) flips |= t;
    }
    // North
    {
        uint64_t t = 0ULL;
        uint64_t x = move_bit << 8;
        while (x && (x & *O)) { t |= x; x <<= 8; }
        if (x & *P) flips |= t;
    }
    // South
    {
        uint64_t t = 0ULL;
        uint64_t x = move_bit >> 8;
        while (x && (x & *O)) { t |= x; x >>= 8; }
        if (x & *P) flips |= t;
    }
    // NE
    {
        uint64_t t = 0ULL;
        uint64_t x = move_bit << 9;
        while ((x & ~D_FILE_H) && (x & *O)) { t |= x; x <<= 9; }
        if (x & *P) flips |= t;
    }
    // SW
    {
        uint64_t t = 0ULL;
        uint64_t x = move_bit >> 9;
        while ((x & ~D_FILE_A) && (x & *O)) { t |= x; x >>= 9; }
        if (x & *P) flips |= t;
    }
    // NW
    {
        uint64_t t = 0ULL;
        uint64_t x = move_bit << 7;
        while ((x & ~D_FILE_H) && (x & *O)) { t |= x; x <<= 7; }
        if (x & *P) flips |= t;
    }
    // SE
    {
        uint64_t t = 0ULL;
        uint64_t x = move_bit >> 7;
        while ((x & ~D_FILE_A) && (x & *O)) { t |= x; x >>= 7; }
        if (x & *P) flips |= t;
    }

    *P ^= flips;
    *P |= move_bit;
    *O &= ~flips;
}

// -------------------------------
// Device evaluation (simple but aligned with host heuristic)
// -------------------------------
__device__ inline int d_eval(const DeviceState &s) {
    uint64_t my = s.x_turn ? s.x : s.o;
    uint64_t opp = s.x_turn ? s.o : s.x;
    int score = 0;
    // positional
    for (int i=0;i<64;i++){
        uint64_t bit = (1ULL<<i);
        if (my & bit) score += d_pos_weights[i];
        else if (opp & bit) score -= d_pos_weights[i];
    }
    // mobility
    int my_moves = d_popcount(d_gen_moves_for(my, opp));
    int opp_moves = d_popcount(d_gen_moves_for(opp, my));
    if (my_moves + opp_moves) {
        score += (my_moves - opp_moves) * 10;
    }
    // frontier approx (count pieces adjacent to empties)
    uint64_t occupied = my | opp, empty = ~occupied;
    uint64_t adj = 0;
    adj |= (empty << 8);
    adj |= (empty >> 8);
    adj |= ((empty & ~D_FILE_H) << 1);
    adj |= ((empty & ~D_FILE_A) >> 1);
    adj |= ((empty & ~D_FILE_H) << 9);
    adj |= ((empty & ~D_FILE_A) << 7);
    adj |= ((empty & ~D_FILE_H) >> 7);
    adj |= ((empty & ~D_FILE_A) >> 9);
    int my_front = d_popcount(my & adj);
    int opp_front = d_popcount(opp & adj);
    score += (opp_front - my_front) * 5; // want fewer frontier tiles
    return s.x_turn ? score : -score;
}

// -------------------------------
// Device small TT entry (shared memory per block)
// -------------------------------
struct DevTTEntry {
    uint64_t key;
    int depth;
    int score;
    int flag; // 0 exact, 1 lower, 2 upper
};

// simple device hash
__device__ inline uint64_t d_hash_state(const DeviceState &s) {
    // mix x and o and turn
    uint64_t h = s.x * 0x9e3779b97f4a7c15ULL ^ (s.o + 0x9e3779b97f4a7c15ULL) ^ (uint64_t)s.x_turn;
    return h;
}

// probe dev-tt in shared memory (simple linear probe)
__device__ inline bool d_tt_probe(DevTTEntry *dtt, uint64_t key, int depth, int alpha, int beta, int *out_score) {
    int idx = (int)(key) & (DEV_TT_SIZE - 1);
    for (int i=0;i<8;i++){
        int j = (idx + i) & (DEV_TT_SIZE - 1);
        DevTTEntry e = dtt[j];
        if (e.key != 0 && e.key == key && e.depth >= depth) {
            if (e.flag == 0) { *out_score = e.score; return true; }
            if (e.flag == 1 && e.score >= beta) { *out_score = e.score; return true; }
            if (e.flag == 2 && e.score <= alpha) { *out_score = e.score; return true; }
        }
    }
    return false;
}

__device__ inline void d_tt_store(DevTTEntry *dtt, uint64_t key, int depth, int score, int flag) {
    int idx = (int)(key) & (DEV_TT_SIZE - 1);
    // pick small window to replace
    int best = idx;
    for (int i=0;i<4;i++){
        int j = (idx + i) & (DEV_TT_SIZE - 1);
        if (dtt[j].key == 0 || dtt[j].depth <= depth) { best = j; break; }
    }
    dtt[best].key = key;
    dtt[best].depth = depth;
    dtt[best].score = score;
    dtt[best].flag = flag;
}

// -------------------------------
// Device iterative negamax (stack-based) used by each worker
// - We implement a simple depth-first search using explicit stack frames
// - Node frame stores state, iterator over moves, alpha/beta, best so far
// Note: This is intentionally conservative to be portable & avoid recursion.
// -------------------------------
struct StackFrame {
    DeviceState state;
    int depth;
    int alpha;
    int beta;
    int best;
    uint64_t moves;   // bitboard of remaining moves
    uint64_t last_move; // last considered move
    int stage; // 0 = new, 1 = children in progress
};

__device__ int device_negamax_iter(DeviceState root, int max_depth, int orig_alpha, int orig_beta, DevTTEntry *dtt) {
    // quick terminal/leaf
    if (max_depth <= 0) return d_eval(root);

    // use simple manual stack on local memory
    StackFrame stack[GPU_STACK_MAX_DEPTH];
    int sp = 0;

    // push root
    stack[sp++] = { root, max_depth, orig_alpha, orig_beta, -100000000, d_gen_moves_for(root.x_turn ? root.x : root.o, root.x_turn ? root.o : root.x), 0ULL, 0 };

    int return_score = -100000000;

    while (sp > 0) {
        StackFrame &f = stack[sp-1];

        // terminal or depth 0
        if (f.depth == 0 || f.moves == 0) {
            // if no moves -> check pass
            if (f.moves == 0) {
                // pass: swap turn and check opponent moves
                DeviceState nb = f.state;
                nb.x_turn = nb.x_turn ? 0 : 1;
                uint64_t opp_moves = d_gen_moves_for(nb.x_turn ? nb.x : nb.o, nb.x_turn ? nb.o : nb.x);
                if (opp_moves == 0) {
                    // game over
                    int val = d_eval(f.state) * 1000;
                    // pop
                    sp--;
                    if (sp == 0) return val;
                    // parent processing: negate etc -- we handle below by writing last_move and reading returned val
                    // We simulate returning by setting return_score to val and letting parent handle it
                    return_score = val;
                    continue;
                } else {
                    // pass but opponent has moves: recurse with depth-1
                    f.depth -= 1;
                    f.state.x_turn = nb.x_turn;
                    f.moves = opp_moves;
                    continue;
                }
            } else {
                // leaf evaluate
                int val = d_eval(f.state);
                sp--;
                return_score = val;
                continue;
            }
        }

        // check TT
        uint64_t key = d_hash_state(f.state);
        int ttval;
        if (d_tt_probe(dtt, key, f.depth, f.alpha, f.beta, &ttval)) {
            sp--;
            return_score = ttval;
            continue;
        }

        // pick next move
        uint64_t move = f.moves & -f.moves;
        f.moves ^= move;

        // apply move
        DeviceState nb = f.state;
        uint64_t P = nb.x_turn ? nb.x : nb.o;
        uint64_t O = nb.x_turn ? nb.o : nb.x;
        d_make_move_and_flip(move, &P, &O);
        if (nb.x_turn) { nb.x = P; nb.o = O; } else { nb.o = P; nb.x = O; }
        nb.x_turn = nb.x_turn ? 0 : 1;

        // if we have room, push child frame
        if (sp < GPU_STACK_MAX_DEPTH) {
            // push child
            stack[sp++] = { nb, f.depth - 1, -f.beta, -f.alpha, -100000000, d_gen_moves_for(nb.x_turn ? nb.x : nb.o, nb.x_turn ? nb.o : nb.x), 0ULL, 0 };
            continue;
        } else {
            // stack overflow guard: evaluate child with shallow quick eval
            int val = d_eval(nb);
            val = -val;
            if (val > f.best) f.best = val;
            if (val > f.alpha) f.alpha = val;
            if (f.alpha >= f.beta) {
                // cutoff
                // store into TT as lowerbound
                d_tt_store(dtt, key, f.depth, f.best, 1);
                sp--;
                return_score = f.best;
                continue;
            }
            // continue with current frame
            continue;
        }

        // unreachable
    } // while

    return return_score;
}

// -------------------------------
// Kernel: each block evaluates one child subtree.
// - shared memory used for device TT
// - only thread 0 executes the worker loop (others init shared arrays quickly)
// -------------------------------
struct KernelJob {
    DeviceState root;
    int depth;
    int alpha;
    int beta;
};

__global__ void pvsplit_worker_kernel(const KernelJob *jobs, int n_jobs, int *out_scores) {
    int bid = blockIdx.x;
    if (bid >= n_jobs) return;

    extern __shared__ DevTTEntry s_dtt[]; // size DEV_TT_SIZE

    // parallel init shared TT
    for (int i = threadIdx.x; i < DEV_TT_SIZE; i += blockDim.x) {
        s_dtt[i].key = 0ULL;
        s_dtt[i].depth = 0;
        s_dtt[i].score = 0;
        s_dtt[i].flag = 0;
    }
    __syncthreads();

    if (threadIdx.x != 0) return; // single lane does the DFS (keeps things simple)

    KernelJob job = jobs[bid];
    DeviceState root = job.root;
    int depth = job.depth;
    int alpha = job.alpha;
    int beta = job.beta;

    // zero-window probe
    int probe = -device_negamax_iter(root, depth, -alpha - 1, -alpha, s_dtt);
    int final_score;
    if (probe > alpha) {
        final_score = -device_negamax_iter(root, depth, -beta, -alpha, s_dtt);
    } else {
        final_score = probe;
    }
    out_scores[bid] = final_score;
}

// -------------------------------
// Host transposition table (simple replacement / direct mapped)
// -------------------------------
struct TTEntryHost { uint64_t key; int depth; int score; uint8_t flag; uint64_t best_move; };
struct HostTT {
    TTEntryHost *table;
    size_t size;
    size_t mask;
    HostTT(size_t mb = HOST_TT_MB) {
        size_t bytes = mb << 20;
        size = bytes / sizeof(TTEntryHost);
        size_t p2 = 1;
        while (p2 * 2 <= size) p2 <<= 1;
        size = p2;
        mask = size - 1;
        table = (TTEntryHost*)malloc(sizeof(TTEntryHost)*size);
        memset(table, 0, sizeof(TTEntryHost)*size);
    }
    ~HostTT(){ free(table); }
    void store(uint64_t key, int depth, int score, uint8_t flag, uint64_t best_move) {
        size_t idx = key & mask;
        if (table[idx].key == 0 || table[idx].depth <= depth) {
            table[idx].key = key; table[idx].depth = depth; table[idx].score = score; table[idx].flag = flag; table[idx].best_move = best_move;
        } else {
            // occasional replacement
            table[idx].key = key; table[idx].depth = depth; table[idx].score = score; table[idx].flag = flag; table[idx].best_move = best_move;
        }
    }
    bool probe(uint64_t key, TTEntryHost &out) {
        size_t idx = key & mask;
        if (table[idx].key == key && table[idx].key != 0) { out = table[idx]; return true; }
        return false;
    }
};

// Zobrist on host
static uint64_t HOST_ZOBRIST[64][2];
static uint64_t HOST_ZOBRIST_SIDE;

static void init_host_zobrist(uint64_t seed = 0xC0FFEE123456789ULL) {
    std::mt19937_64 rng(seed);
    for (int i=0;i<64;i++){ HOST_ZOBRIST[i][0] = rng(); HOST_ZOBRIST[i][1] = rng(); }
    HOST_ZOBRIST_SIDE = rng();
}
static inline uint64_t host_zobrist_hash(const GameState &s) {
    uint64_t h = 0;
    for (int i=0;i<64;i++){
        uint64_t bit = (1ULL<<i);
        if (s.x & bit) h ^= HOST_ZOBRIST[i][0];
        else if (s.o & bit) h ^= HOST_ZOBRIST[i][1];
    }
    if (s.x_turn) h ^= HOST_ZOBRIST_SIDE;
    return h;
}

// -------------------------------
// CPU Serial Negamax + TT (using host heuristic dynamic_heuristic_evaluation)
// -------------------------------
int negamax_cpu_host(HostTT &tt, const GameState &gs, int depth, int alpha, int beta) {
    if (depth <= 0) {
        // use provided host heuristic
        double h = dynamic_heuristic_evaluation(gs, gs.x_turn);
        return (int)round(h);
    }
    // probe TT
    uint64_t key = host_zobrist_hash(gs);
    TTEntryHost e;
    if (tt.probe(key, e) && e.depth >= depth) {
        if (e.flag == 0) return e.score;
        if (e.flag == 1 && e.score >= beta) return e.score;
        if (e.flag == 2 && e.score <= alpha) return e.score;
    }
    // generate moves using provided serial move generator
    GameState scopy = gs;
    scopy.x_turn = gs.x_turn;
    // user-provided function find_all_moves should exist (serial). We'll call it.
    auto moves = find_all_moves(gs); // returns set<GameState>
    if (moves.empty()) {
        // pass
        GameState passed = gs; passed.x_turn = !gs.x_turn;
        auto opp_moves = find_all_moves(passed);
        if (opp_moves.empty()) {
            // terminal
            double hf = dynamic_heuristic_evaluation(gs, gs.x_turn);
            return (int)round(hf) * 1000;
        } else {
            int val = -negamax_cpu_host(tt, passed, depth-1, -beta, -alpha);
            tt.store(key, depth, val, 0, 0);
            return val;
        }
    }
    // order moves: try TT best_move first if present
    std::vector<GameState> mvvec(moves.begin(), moves.end());
    if (tt.probe(key, e) && e.best_move) {
        // find and move best to front - best_move stores a bitmask we don't have here easily
        // skip this optimization if we can't map easily
    }
    int best = -100000000;
    uint64_t best_move_mask = 0;
    int orig_alpha = alpha;
    for (auto &child : mvvec) {
        int val = -negamax_cpu_host(tt, child, depth-1, -beta, -alpha);
        if (val > best) {
            best = val;
            // we can record child board's move mask by deducing difference, but skip for simplicity
        }
        if (val > alpha) alpha = val;
        if (alpha >= beta) break;
    }
    uint8_t flag = 0;
    if (best <= orig_alpha) flag = 2; // upper
    else if (best >= beta) flag = 1; // lower
    else flag = 0;
    tt.store(key, depth, best, flag, best_move_mask);
    return best;
}

// -------------------------------
// PV-Split root: CPU searches PV child then sends remaining children to GPU
// returns best child as GameState
// -------------------------------
GameState pv_split_root_gpu(Othello *game, HostTT &tt, const GameState &root_state, int depth) {
    // generate children
    auto children_set = find_all_moves(root_state);
    if (children_set.empty()) return root_state;
    std::vector<GameState> children(children_set.begin(), children_set.end());

    // move ordering: no fancy ordering here (could use TT/hist heuristics)
    // put one child as PV: choose first (or maybe best by quick heuristic)
    // Let's pick the child that maximizes quick host heuristic as PV
    int best_quick_idx = 0;
    double best_q = -1e300;
    for (size_t i=0;i<children.size();i++){
        double h = dynamic_heuristic_evaluation(children[i], children[i].x_turn);
        if (h > best_q) { best_q = h; best_quick_idx = (int)i; }
    }
    // move chosen PV to front
    if (best_quick_idx != 0) std::swap(children[0], children[best_quick_idx]);

    // CPU PV search on first child (full-window)
    GameState pv_child = children[0];
    int pv_score = -negamax_cpu_host(tt, pv_child, depth-1, -100000000, 100000000);
    int best_score = pv_score;
    GameState best_state = children[0];

    // prepare GPU jobs for remaining children
    int n_remain = (int)children.size() - 1;
    if (n_remain <= 0) return best_state;

    if (n_remain > MAX_GPU_CHILDREN) n_remain = MAX_GPU_CHILDREN; // cap
    int n_jobs = n_remain;

    // allocate host arrays
    std::vector<KernelJob> jobs(n_jobs);
    for (int i=0;i<n_jobs;i++){
        GameState &gs = children[i+1];
        DeviceState ds; ds.o = gs.o; ds.x = gs.x; ds.x_turn = gs.x_turn ? 1 : 0;
        jobs[i].root = ds;
        jobs[i].depth = std::min(depth-1, DEVICE_MAX_DEPTH);
        jobs[i].alpha = best_score;
        jobs[i].beta = 100000000;
    }

    // copy to device
    KernelJob *d_jobs = nullptr;
    int *d_out = nullptr;
    CUDA_CALL(cudaMalloc((void**)&d_jobs, sizeof(KernelJob) * n_jobs));
    CUDA_CALL(cudaMalloc((void**)&d_out, sizeof(int) * n_jobs));
    CUDA_CALL(cudaMemcpy(d_jobs, jobs.data(), sizeof(KernelJob) * n_jobs, cudaMemcpyHostToDevice));

    // launch kernel: one block per job
    int blocks = n_jobs;
    int threads = WORKER_BLOCK_THREADS;
    size_t shared_bytes = sizeof(DevTTEntry) * DEV_TT_SIZE;
    pvsplit_worker_kernel<<<blocks, threads, shared_bytes>>>(d_jobs, n_jobs, d_out);
    CUDA_CALL(cudaDeviceSynchronize());

    // get results
    std::vector<int> h_out(n_jobs);
    CUDA_CALL(cudaMemcpy(h_out.data(), d_out, sizeof(int)*n_jobs, cudaMemcpyDeviceToHost));

    // integrate results
    for (int i=0;i<n_jobs;i++){
        int sc = h_out[i];
        if (sc > best_score) { best_score = sc; best_state = children[i+1]; }
    }

    CUDA_CALL(cudaFree(d_jobs));
    CUDA_CALL(cudaFree(d_out));

    // store into host TT
    uint64_t keyroot = host_zobrist_hash(root_state);
    // for simplicity store exact
    tt.store(keyroot, depth, best_score, 0, 0);

    return best_state;
}

// -------------------------------
// Top-level iterative deepening controller callable from Othello::computer_turn()
// -------------------------------
GameState negamax_parallel(Othello *game, int time_limit_ms) {
    GameState root = game->get_board();
    HostTT tt(HOST_TT_MB);
    init_host_zobrist();

    // time control
    auto start = std::chrono::steady_clock::now();
    auto time_exceeded = [&](int ms)->bool {
        if (ms <= 0) return false;
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() >= ms;
    };

    GameState best = root;
    int maxDepth = 8; // tune: CPU deeper; device depth limited
    for (int depth = 1; depth <= maxDepth; depth++) {
        if (time_exceeded(time_limit_ms)) break;
        best = pv_split_root_gpu(game, tt, root, depth);
        // optional: report
        // printf("[ID] depth %d -> best score (approx)\n", depth);
    }
    return best;
}