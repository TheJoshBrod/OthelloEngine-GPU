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
    
    const int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    
public:
    Othello(int num_players) : board(SIZE, vector<char>(SIZE, ' ')), currentPlayer('X'), human_players(num_players) {
        // Initialize starting position
        board[3][3] = 'O';
        board[3][4] = 'X';
        board[4][3] = 'X';
        board[4][4] = 'O';
    }
    
    void displayBoard() {
        cout << "\n  ";
        for (int i = 0; i < SIZE; i++) {
            cout << i << " ";
        }
        cout << "\n  ----------------\n";
        
        for (int i = 0; i < SIZE; i++) {
            cout << i << "|";
            for (int j = 0; j < SIZE; j++) {
                cout << board[i][j] << "|";
            }
            cout << "\n  ----------------\n";
        }
    }
    
    bool isValidMove(int row, int col, char player) {
        if (row < 0 || row >= SIZE || col < 0 || col >= SIZE) return false;
        if (board[row][col] != ' ') return false;
        
        char opponent = (player == 'X') ? 'O' : 'X';
        
        for (int dir = 0; dir < 8; dir++) {
            int r = row + dx[dir];
            int c = col + dy[dir];
            bool foundOpponent = false;
            
            while (r >= 0 && r < SIZE && c >= 0 && c < SIZE) {
                if (board[r][c] == ' ') break;
                if (board[r][c] == opponent) {
                    foundOpponent = true;
                } else if (board[r][c] == player) {
                    if (foundOpponent) return true;
                    break;
                }
                r += dx[dir];
                c += dy[dir];
            }
        }
        return false;
    }
    
    void flipPieces(int row, int col, char player) {
        char opponent = (player == 'X') ? 'O' : 'X';
        board[row][col] = player;
        
        for (int dir = 0; dir < 8; dir++) {
            int r = row + dx[dir];
            int c = col + dy[dir];
            vector<pair<int, int>> toFlip;
            
            while (r >= 0 && r < SIZE && c >= 0 && c < SIZE) {
                if (board[r][c] == ' ') break;
                if (board[r][c] == opponent) {
                    toFlip.push_back({r, c});
                } else if (board[r][c] == player) {
                    for (auto& p : toFlip) {
                        board[p.first][p.second] = player;
                    }
                    break;
                }
                r += dx[dir];
                c += dy[dir];
            }
        }
    }
    
    bool hasValidMoves(char player) {
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                if (isValidMove(i, j, player)) return true;
            }
        }
        return false;
    }
    
    void printValidMoves(char player) {
        cout << "\nValid moves for player " << player << ": ";
        bool found = false;
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                if (isValidMove(i, j, player)) {
                    cout << "(" << i << "," << j << ") ";
                    found = true;
                }
            }
        }
        if (!found) cout << "None";
        cout << "\n";
    }
    
    vector<pair<int, int>> retrieveValidMoves(char player) {
        vector<pair<int, int>> moves;
        bool found = false;
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                if (isValidMove(i, j, player)) {
                    moves.push_back({i,j});
                }
            }
        }
        return moves;
    }
    
    pair<int, int> getScore() {
        int x = 0, o = 0;
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                if (board[i][j] == 'X') x++;
                else if (board[i][j] == 'O') o++;
            }
        }
        return {x, o};
    }
    
    void human_turn(){
        while (true){
            int row, col;
            if (!(cin >> row >> col)) {
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                cout << "Invalid input. Please enter two numbers.\n";
            }

            bool valid_move = isValidMove(row, col, currentPlayer);
            if (valid_move) {
                flipPieces(row, col, currentPlayer);
                currentPlayer = (currentPlayer == 'X') ? 'O' : 'X';
                return;
            } 
            else {
                cout << "Invalid move! Try again.\n";
            }

            displayBoard();
        }
    }

    void computer_turn(){
        vector<pair<int,int>> moves = retrieveValidMoves(currentPlayer);
        flipPieces(moves[0].first, moves[1].second, currentPlayer);
    }

    void play() {
        cout << "=== OTHELLO GAME ===\n";
        cout << "Player X: You\nPlayer O: Opponent\n";
        cout << "Enter moves as: row col (e.g., 2 3)\n";
        
        while (true) {
            displayBoard();
            auto [xScore, oScore] = getScore();
            cout << "\nScore - X: " << xScore << " | O: " << oScore << "\n";
            
            if (!hasValidMoves('X') && !hasValidMoves('O')) {
                cout << "\n=== GAME OVER ===\n";
                if (xScore > oScore) cout << "Player X wins!\n";
                else if (oScore > xScore) cout << "Player O wins!\n";
                else cout << "It's a tie!\n";
                break;
            }
            
            if (!hasValidMoves(currentPlayer)) {
                cout << "\nPlayer " << currentPlayer << " has no valid moves. Skipping turn.\n";
                currentPlayer = (currentPlayer == 'X') ? 'O' : 'X';
                cout << "Press Enter to continue...";
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                continue;
            }
            
            printValidMoves(currentPlayer);
            cout << "Player " << currentPlayer << "'s turn. Enter move: ";
            
            if (human_players == 2){
                human_turn();
            }
            else if (human_players == 1){
                if (currentPlayer == 'X')
                    human_turn();
                else
                    computer_turn();
            }
            else{
                computer_turn();
            }
        }
    }
};

int main(int argc, char* argv[]) {
    
    // Handles args to determine number of players/computers
    int num_players = 1;
    if (argc >= 2){
        num_players = std::stoi(argv[1]);
        if (num_players < 0 || num_players > 2){
            cout << "Must have between 0-2 human players\n";
            return;
        }
    }

    Othello game(num_players);
    game.play();
    return 0;
}