#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <string>
#include <numeric>

#include "../include/bench_impls.h"
#include "../include/othello.h"
#include "../include/gamestate.h"

using namespace std;

static inline int popcount64(uint64_t x){ return __builtin_popcountll(x); }

enum Impl { SERIAL=0, NAIVE=1, PARALLEL_BASE=2, PARALLEL_OPT1=3 };

struct PhaseStats {
    int moves = 0;
    long long depth_sum = 0;
    void add(int depth){ moves++; depth_sum += depth; }
    double avg() const { return moves ? double(depth_sum)/moves : 0.0; }
};

// Play a self-play game where both sides use 'impl', with per-move time_limit_ms.
// Returns stats per phase (early/mid/late), where early: pieces < 16, mid: [16,40), late: >=40

map<string, PhaseStats> play_one_game(Impl impl, int time_limit_ms){
    map<string, PhaseStats> stats;
    stats["early"] = PhaseStats();
    stats["mid"] = PhaseStats();
    stats["late"] = PhaseStats();

    // create game with 0 human players
    Othello game(0, best_move_serial, 'X', time_limit_ms);

    while (true){
        auto score = game.getScore();
        // check terminal
        if (!game.hasValidMoves('X') && !game.hasValidMoves('O')) break;

        if (!game.hasValidMoves(game.getCurrentPlayer())){
            // skip turn
            char cur = game.getCurrentPlayer();
            game.setBoard(game.get_board().x, game.get_board().o, (cur=='X')? 'O' : 'X');
            continue;
        }

        // call selected implementation directly
        GameState next;
        int last_depth = 0;
        switch (impl){
            case SERIAL:
                next = negamax_serial(&game, time_limit_ms);
                last_depth = get_last_depth_serial();
                break;
            case NAIVE:
                next = negamax_naive_cuda(&game, time_limit_ms);
                last_depth = get_last_depth_naive();
                break;
            case PARALLEL_BASE:
                next = negamax_parallel_base_cuda(&game, time_limit_ms);
                last_depth = get_last_depth_parallel();
                break;
            case PARALLEL_OPT1:
                next = negamax_parallel_opt1_cuda(&game, time_limit_ms);
                last_depth = get_last_depth_opt1();
                break;
        }

        // Determine phase based on resulting piece count
        int pieces = popcount64(next.x) + popcount64(next.o);
        string phase = "early";
        if (pieces >= 40) phase = "late";
        else if (pieces >= 16) phase = "mid";

        // If depth unknown (0), we still record as 0
        stats[phase].add(last_depth);

        // update board
        char cur = next.x_turn ? 'X' : 'O';
        game.setBoard(next.x, next.o, cur);
    }

    return stats;
}

int main(int argc, char** argv){
    vector<int> times_sec = {5,10,20,30};
    int games_per_config = 2;
    if (argc > 1) games_per_config = stoi(argv[1]);

    vector<pair<Impl,string>> impls = {
        {SERIAL, "serial"},
        {NAIVE, "naive_cuda"},
        {PARALLEL_BASE, "parallel_base"},
        {PARALLEL_OPT1, "parallel_opt1"}
    };

    for (auto &impl_pair : impls){
        cout << "\n=== Implementation: " << impl_pair.second << " ===\n";
        for (int tsec : times_sec){
            cout << "Time per move: " << tsec << "s (" << tsec*1000 << " ms)\n";
            PhaseStats agg_early, agg_mid, agg_late;
            for (int g = 0; g < games_per_config; ++g){
                auto stats = play_one_game(impl_pair.first, tsec*1000);
                agg_early.moves += stats["early"].moves;
                agg_early.depth_sum += stats["early"].depth_sum;
                agg_mid.moves += stats["mid"].moves;
                agg_mid.depth_sum += stats["mid"].depth_sum;
                agg_late.moves += stats["late"].moves;
                agg_late.depth_sum += stats["late"].depth_sum;
                cout << "  Game "<< (g+1) <<": early avg="<<(stats["early"].avg())<<" mid avg="<<(stats["mid"].avg())<<" late avg="<<(stats["late"].avg())<<"\n";
            }

            cout << "  Aggregated over "<<games_per_config<<" games:\n";
            cout << "    Early (pieces<16): avg depth = " << (agg_early.avg()) << " ("<<agg_early.moves<<" moves)\n";
            cout << "    Mid   (16<=p<40): avg depth = " << (agg_mid.avg()) << " ("<<agg_mid.moves<<" moves)\n";
            cout << "    Late  (p>=40):    avg depth = " << (agg_late.avg()) << " ("<<agg_late.moves<<" moves)\n";
        }
    }

    return 0;
}
