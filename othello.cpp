#include "othello.h"
#include <string>
#include <limits>
#include <ios>
using namespace std;

// ************************
// Constructor Functions
// ************************

Othello::Othello(int num_players)
    : board(SIZE, vector<char>(SIZE, ' ')), currentPlayer('X'), human_players(num_players)
{
    board[3][3] = 'O';
    board[3][4] = 'X';
    board[4][3] = 'X';
    board[4][4] = 'O';
}


// ************************
// Display / UI helpers
// ************************

void Othello::displayBoard() {
    cout << "\n  ";
    for (int i = 0; i < SIZE; i++) cout << i << " ";
    cout << "\n  ----------------\n";

    for (int i = 0; i < SIZE; i++) {
        cout << i << "|";
        for (int j = 0; j < SIZE; j++) cout << board[i][j] << "|";
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
    if (!found) cout << "None";
    cout << "\n";
}


// ************************
// Core game logic
// ************************

bool Othello::isValidMove(int row, int col, char player) {
    if (row < 0 || row >= SIZE || col < 0 || col >= SIZE) return false;
    if (board[row][col] != ' ') return false;

    char opponent = (player == 'X') ? 'O' : 'X';

    for (int dir = 0; dir < 8; dir++) {
        int r = row + dx[dir];
        int c = col + dy[dir];
        bool foundOpponent = false;

        while (r >= 0 && r < SIZE && c >= 0 && c < SIZE) {
            if (board[r][c] == ' ') break;
            if (board[r][c] == opponent) foundOpponent = true;
            else if (board[r][c] == player) return foundOpponent;
            r += dx[dir];
            c += dy[dir];
        }
    }
    return false;
}

void Othello::flipPieces(int row, int col, char player) {
    char opponent = (player == 'X') ? 'O' : 'X';
    board[row][col] = player;

    for (int dir = 0; dir < 8; dir++) {
        int r = row + dx[dir];
        int c = col + dy[dir];
        vector<pair<int, int>> toFlip;

        while (r >= 0 && r < SIZE && c >= 0 && c < SIZE) {
            if (board[r][c] == ' ') break;
            if (board[r][c] == opponent) toFlip.push_back({r, c});
            else if (board[r][c] == player) {
                for (auto& p : toFlip) board[p.first][p.second] = player;
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
            if (isValidMove(i, j, player)) return true;
    return false;
}


vector<pair<int, int>> Othello::retrieveValidMoves(char player) {
    vector<pair<int, int>> moves;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (isValidMove(i, j, player)) moves.push_back({i, j});
    return moves;
}

pair<int, int> Othello::getScore() {
    int x = 0, o = 0;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++) {
            if (board[i][j] == 'X') x++;
            else if (board[i][j] == 'O') o++;
        }
    return {x, o};
}

// ************************
// Helper Functions
// ************************

char Othello::getCurrentPlayer(){
    return currentPlayer;
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
    vector<pair<int,int>> moves = retrieveValidMoves(currentPlayer);
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
