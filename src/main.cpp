#include "othello.h"
#include <iostream>
#include <string>
#include <limits>
#include "negamax.h"

using namespace std;

int main(int argc, char* argv[]) {
    // Default settings
    bool use_parallel = false;
    int time_limit_ms = 0;

    // Simple CLI parsing for --parallel/--serial and --time <ms>
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--parallel") {
            use_parallel = true;
        } else if (arg == "--serial") {
            use_parallel = false;
        } else if (arg == "--time" && i + 1 < argc) {
            time_limit_ms = std::stoi(argv[++i]);
        } else {
            cout << "Unknown arg: " << arg << "\n";
        }
    }

    // Set negamax implementation
    set_negamax_mode(use_parallel);

    // Ask user for role
    cout << "Choose mode:\n";
    cout << "  1) Play as \033[31mX\033[0m\n";
    cout << "  2) Play as \033[34mO\033[0m\n";
    cout << "  3) AI vs AI\n";
    int choice = 0;
    while (!(cin >> choice) || choice < 1 || choice > 3) {
        cin.clear();
        cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        cout << "Please enter 1, 2 or 3: ";
    }

    int num_players = 0;
    char human_side = 'X';
    if (choice == 1) { num_players = 1; human_side = 'X'; }
    else if (choice == 2) { num_players = 1; human_side = 'O'; }
    else { num_players = 0; }

    Othello game(num_players, best_move_serial, human_side, time_limit_ms);
    game.play();
    return 0;
}