#ifndef OTHELLO_H
#define OTHELLO_H

#include "serial.h"
#include "parallel.h"
#include <iostream>
#include <vector>
#include <utility>
#include <cstdint>

enum ai_type{
    first_move,
    best_move_serial,
    naive_cuda,
    parallel_base,
    parallel_opt1
};


class Othello {
private:
    static const int SIZE = 8;
    uint64_t x;
    uint64_t o;
    int human_players;
    char human_side; // 'X' or 'O' when human_players == 1
    char currentPlayer;

    ai_type computer_mode; 
    int time_limit_ms; // per-move time limit passed to negamax

    const int dx[8] = { -1,-1,-1,0,0,1,1,1 };
    const int dy[8] = { -1,0,1,-1,1,-1,0,1 };

    // Bitmap helper functions
    void setBit(uint64_t& board, int row, int col);
    void clearBit(uint64_t& board, int row, int col);
    bool getBit(uint64_t board, int row, int col);

public:
    // Constructor
    Othello(int num_players, ai_type ai_mode, char human_side_in = 'X', int time_limit_ms_in = 0);

    // Display / UI helpers
    void displayBoard();
    void printValidMoves(char player);

    // Core game logic
    bool isValidMove(int row, int col, char player);
    void flipPieces(int row, int col, char player);
    bool hasValidMoves(char player);
    std::vector<std::pair<int, int>> retrieveValidMoves(char player);
    std::pair<int, int> getScore();

    // Helper functions
    char getCurrentPlayer();
    GameState get_board();
    // Debug/testing helper: set internal board directly
    void setBoard(uint64_t x_in, uint64_t o_in, char cur);

    // Turn logic
    void human_turn();
    void computer_turn();

    // Gameplay loop
    void play();

};

#endif