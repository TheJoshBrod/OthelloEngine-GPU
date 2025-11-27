#include "othello.h"
#include <iostream>
using namespace std;

uint64_t bit(int r,int c){ return 1ULL << (r*8 + c); }

int main(){
    Othello game(1, first_move, 'O');
    uint64_t x=0,o=0;
    // Row 0: all O
    for(int c=0;c<8;c++) o |= bit(0,c);
    // Row1 all O
    for(int c=0;c<8;c++) o |= bit(1,c);
    // Row2 all O
    for(int c=0;c<8;c++) o |= bit(2,c);
    // Row3: X O O X X X X X
    x |= bit(3,0);
    o |= bit(3,1);
    o |= bit(3,2);
    x |= bit(3,3);
    x |= bit(3,4);
    x |= bit(3,5);
    x |= bit(3,6);
    x |= bit(3,7);
    // Row4:   X O O X X X  
    // col0 empty
    x |= bit(4,1);
    o |= bit(4,2);
    o |= bit(4,3);
    x |= bit(4,4);
    x |= bit(4,5);
    x |= bit(4,6);
    // col7 empty
    // Row5:   X X O O O    
    x |= bit(5,1);
    x |= bit(5,2);
    o |= bit(5,3);
    o |= bit(5,4);
    o |= bit(5,5);
    // Row6:     O O      
    o |= bit(6,4);
    o |= bit(6,5);
    // Row7: all empty

    // Set to player O's turn
    game.setBoard(x,o,'O');

    game.displayBoard();
    game.printValidMoves('O');
    cout << "isValidMove(4,7,'O') = " << game.isValidMove(4,7,'O') << "\n";
    return 0;
}
