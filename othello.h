#ifndef OTHELLO_H
#define OTHELLO_H

#include <iostream>
#include <vector>
#include <string>
#include <limits>
using namespace std;

class Othello {
private:
    static const int SIZE = 8;
    vector<vector<char>> board;
    int human_players;
    char currentPlayer;

    const int dx[8] = { -1,-1,-1,0,0,1,1,1 };
    const int dy[8] = { -1,0,1,-1,1,-1,0,1 };

public:
    // Constructor
    Othello(int num_players);

    // Display / UI helpers
    void displayBoard();
    void printValidMoves(char player);

    // Core game logic
    bool isValidMove(int row, int col, char player);
    void flipPieces(int row, int col, char player);
    bool hasValidMoves(char player);
    vector<pair<int, int>> retrieveValidMoves(char player);
    pair<int, int> getScore();

    // Turn logic
    void human_turn();
    void computer_turn();

    // Gameplay loop
    void play();

};

#endif
