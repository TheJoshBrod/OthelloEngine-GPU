#include "othello.h"
#include <iostream>
#include <limits>
#include <vector>
using namespace std;

// ************************
// Constructor Functions
// ************************
Othello::Othello(int num_players) : black(0), white(0), currentPlayer('X'), human_players(num_players) {
    // Initial board setup
    setBit(white, 3, 3);
    setBit(black, 3, 4);
    setBit(black, 4, 3);
    setBit(white, 4, 4);
}

// ************************
// Display / UI helpers
// ************************
void Othello::displayBoard() {
    cout << "\n   ";
    for (int i = 0; i < SIZE; i++)
        cout << i << " ";
    cout << "\n  ----------------\n";
    for (int i = 0; i < SIZE; i++) {
        cout << i << "|";
        for (int j = 0; j < SIZE; j++) {
            if (getBit(black, i, j))
                cout << "X|";
            else if (getBit(white, i, j))
                cout << "O|";
            else
                cout << " |";
        }
        cout << "\n  ----------------\n";
    }
}

void Othello::printValidMoves(char player) {
    cout << "\nValid moves for player " << player << ": ";
    bool found = false;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (isValidMove(i, j, player)) {
                cout << "(" << i << "," << j << ") ";
                found = true;
            }
    if (!found)
        cout << "None";
    cout << "\n";
}

// ************************
// Core game logic
// ************************
bool Othello::isValidMove(int row, int col, char player) {
    if (row < 0 || row >= SIZE || col < 0 || col >= SIZE)
        return false;
    if (getBit(black, row, col) || getBit(white, row, col))
        return false;

    uint64_t& myPieces = (player == 'X') ? black : white;
    uint64_t& opponentPieces = (player == 'X') ? white : black;

    for (int dir = 0; dir < 8; dir++) {
        int r = row + dx[dir];
        int c = col + dy[dir];
        bool foundOpponent = false;

        while (r >= 0 && r < SIZE && c >= 0 && c < SIZE) {
            if (!getBit(black, r, c) && !getBit(white, r, c))
                break;
            if (getBit(opponentPieces, r, c))
                foundOpponent = true;
            else if (getBit(myPieces, r, c))
                return foundOpponent;
            r += dx[dir];
            c += dy[dir];
        }
    }
    return false;
}

void Othello::flipPieces(int row, int col, char player) {
    uint64_t& myPieces = (player == 'X') ? black : white;
    uint64_t& opponentPieces = (player == 'X') ? white : black;

    setBit(myPieces, row, col);

    for (int dir = 0; dir < 8; dir++) {
        int r = row + dx[dir];
        int c = col + dy[dir];
        vector<pair<int, int>> toFlip;

        while (r >= 0 && r < SIZE && c >= 0 && c < SIZE) {
            if (!getBit(black, r, c) && !getBit(white, r, c))
                break;
            if (getBit(opponentPieces, r, c))
                toFlip.push_back({r, c});
            else if (getBit(myPieces, r, c)) {
                for (auto& p : toFlip) {
                    clearBit(opponentPieces, p.first, p.second);
                    setBit(myPieces, p.first, p.second);
                }
                break;
            }
            r += dx[dir];
            c += dy[dir];
        }
    }
}

bool Othello::hasValidMoves(char player) {
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (isValidMove(i, j, player))
                return true;
    return false;
}

vector<pair<int, int>> Othello::retrieveValidMoves(char player) {
    vector<pair<int, int>> moves;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (isValidMove(i, j, player))
                moves.push_back({i, j});
    return moves;
}

pair<int, int> Othello::getScore() {
    int x = 0, o = 0;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++) {
            if (getBit(black, i, j))
                x++;
            else if (getBit(white, i, j))
                o++;
        }
    return {x, o};
}

// ************************
// Helper Functions
// ************************
char Othello::getCurrentPlayer(){
    return currentPlayer;
}

void Othello::setBit(uint64_t& board, int row, int col) {
    board |= (1ULL << (row * SIZE + col));
}

void Othello::clearBit(uint64_t& board, int row, int col) {
    board &= ~(1ULL << (row * SIZE + col));
}

bool Othello::getBit(uint64_t board, int row, int col) {
    return (board & (1ULL << (row * SIZE + col))) != 0;
}

// ************************
// Turn Logic Functions
// ************************
void Othello::human_turn(){
    while (true){
        int row, col;
        if (!(cin >> row >> col)) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Invalid input. Please enter two numbers.\n";
        }
        if (isValidMove(row, col, currentPlayer)) {
            flipPieces(row, col, currentPlayer);
            currentPlayer = (currentPlayer == 'X') ? 'O' : 'X';
            return;
        } else {
            cout << "Invalid move! Try again.\n";
        }
        displayBoard();
    }
}

void Othello::computer_turn(){
    vector<pair<int, int>> moves = retrieveValidMoves(currentPlayer);
    if (!moves.empty())
        flipPieces(moves[0].first, moves[0].second, currentPlayer);
    currentPlayer = (currentPlayer == 'X') ? 'O' : 'X';
}

// ************************
// Gameplay Loop Functions
// ************************
void Othello::play() {
    cout << "=== OTHELLO GAME ===\n";
    while (true) {
        displayBoard();
        auto [xScore, oScore] = getScore();
        cout << "\nScore - X: " << xScore << " | O: " << oScore << "\n";

        if (!hasValidMoves('X') && !hasValidMoves('O')) {
            cout << "\n=== GAME OVER ===\n";
            cout << ((xScore > oScore) ? "Player X wins!" : (oScore > xScore ? "Player O wins!" : "Tie!")) << "\n";
            break;
        }

        if (!hasValidMoves(currentPlayer)) {
            cout << "\nPlayer " << currentPlayer << " has no valid moves. Skipping turn.\n";
            currentPlayer = (currentPlayer == 'X') ? 'O' : 'X';
            continue;
        }

        printValidMoves(currentPlayer);

        if (human_players == 2 || (human_players == 1 && currentPlayer == 'X'))
            human_turn();
        else
            computer_turn();
    }
}