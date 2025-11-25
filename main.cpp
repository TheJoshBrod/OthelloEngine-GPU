#include <iostream>
#include <vector>
#include <string>


#include "othello.h"
#include "serial.h"

using namespace std;

int main(int argc, char* argv[]) {
    
    // Handles args to determine number of players/computers
    int num_players = -1;
    if (argc >= 2){
        num_players = std::stoi(argv[1]);
    }
    if (num_players < 0 || num_players > 2){
        cout << "Must have between 0-2 human players\n";
        return 1;
    }


    Othello game(num_players);
    game.play();
    return 0;
}