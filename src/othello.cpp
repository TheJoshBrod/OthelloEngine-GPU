#include "othello.h"
#include <iostream>
#include <limits>
#include <vector>
#include "negamax.h"

using namespace std;

// ************************
// Constructor Functions
// ************************
Othello::Othello(int num_players, ai_type ai_mode, char human_side_in, int time_limit_ms_in) : x(0), o(0), currentPlayer('X'), human_players(num_players), human_side(human_side_in), computer_mode(ai_mode), time_limit_ms(time_limit_ms_in) {
    // Initial board setup
    setBit(o, 3, 3);
    setBit(x, 3, 4);
    setBit(x, 4, 3);
    setBit(o, 4, 4);
}

// ************************
// Display / UI helpers
// ************************
void Othello::displayBoard() {
    cout << "\n   ";
    for (int i = 0; i < SIZE; i++)
        cout << i << "   ";
    cout << "\n  --------------------------------\n";
    for (int i = 0; i < SIZE; i++) {
        cout << i << "|";
        for (int j = 0; j < SIZE; j++) {
            if (getBit(x, i, j))
                cout << " \033[31mX\033[0m |";
            else if (getBit(o, i, j))
                cout << " \033[34mO\033[0m |";
            else
                cout << "   |";
        }
        cout << "\n  --------------------------------\n";
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
    if (getBit(x, row, col) || getBit(o, row, col))
        return false;
    uint64_t& myPieces = (player == 'X') ? x : o;
    uint64_t& opponentPieces = (player == 'X') ? o : x;

    for (int dir = 0; dir < 8; dir++) {
        int r = row + dx[dir];
        int c = col + dy[dir];
        bool foundOpponent = false;

        while (r >= 0 && r < SIZE && c >= 0 && c < SIZE) {
            if (getBit(opponentPieces, r, c)) {
                foundOpponent = true;
            } else if (getBit(myPieces, r, c)) {
                if (foundOpponent) return true;
                break;
            } else {
                break;
            }
            r += dx[dir];
            c += dy[dir];
        }
    }
    return false;
}

void Othello::flipPieces(int row, int col, char player) {
    uint64_t& myPieces = (player == 'X') ? x : o;
    uint64_t& opponentPieces = (player == 'X') ? o : x;

    setBit(myPieces, row, col);

    for (int dir = 0; dir < 8; dir++) {
        int r = row + dx[dir];
        int c = col + dy[dir];
        vector<pair<int, int>> toFlip;

        while (r >= 0 && r < SIZE && c >= 0 && c < SIZE) {
            if (!getBit(x, r, c) && !getBit(o, r, c))
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
    int xCount = 0, oCount = 0;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++) {
            if (getBit(this->x, i, j))
                xCount++;
            else if (getBit(this->o, i, j))
                oCount++;
        }
    return {xCount, oCount};
}

// ************************
// Helper Functions
// ************************
char Othello::getCurrentPlayer(){
    return currentPlayer;
}

GameState Othello::get_board(){
    return {o, x, currentPlayer == 'X'};
}

// Debug/testing helper
void Othello::setBoard(uint64_t x_in, uint64_t o_in, char cur) {
    x = x_in;
    o = o_in;
    currentPlayer = cur;
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
    displayBoard();    
    printValidMoves(currentPlayer);

    while (true){
        int row, col;
        if (!(cin >> row >> col)) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Invalid input. Please enter two numbers.\n";
            continue;
        }
        if (isValidMove(row, col, currentPlayer)) {
            flipPieces(row, col, currentPlayer);
            return;
        } else {
            cout << "Invalid move! Try again.\n";
        }
        displayBoard();        
        printValidMoves(currentPlayer);

    }
}

void Othello::computer_turn(){
    if (computer_mode == first_move){
        vector<pair<int, int>> moves = retrieveValidMoves(currentPlayer);
        if (!moves.empty())
            flipPieces(moves[0].first, moves[0].second, currentPlayer);
    }
    else{
        // Use the unified negamax entrypoint (serial or parallel) with the configured time limit
        extern GameState (*negamax_fn)(Othello*, int);
        GameState new_board = negamax_fn(this, time_limit_ms);
        o = new_board.o;
        x = new_board.x;
    }
}

// ************************
// Gameplay Loop Functions
// ************************
void Othello::play() {
    cout << "=== OTHELLO GAME ===\n";
    while (true) {
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

        if (human_players == 2 || (human_players == 1 && currentPlayer == human_side))
            human_turn();
        else
            computer_turn();

        // Print board after the move was made
        displayBoard();

        // Flip turns
        currentPlayer = (currentPlayer == 'X') ? 'O' : 'X';
    }
}