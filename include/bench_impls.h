#ifndef BENCH_IMPLS_H
#define BENCH_IMPLS_H

#include "gamestate.h"
class Othello;

// serial
GameState negamax_serial(Othello* game, int time_limit_ms);
int get_last_depth_serial();

// naive CUDA
GameState negamax_naive_cuda(Othello* game, int time_limit_ms);
int get_last_depth_naive();

// base parallel (kept as negamax_parallel in parallel/parallel.cu)
GameState negamax_parallel_base_cuda(Othello* game, int time_limit_ms);
int get_last_depth_parallel();

// optimized parallel
GameState negamax_parallel_opt1_cuda(Othello* game, int time_limit_ms);
int get_last_depth_opt1();

#endif
