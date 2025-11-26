// Simple parallel stub: currently delegates to serial implementation.
#include "parallel.h"
#include "serial.h"

GameState negamax_parallel(Othello* game, int time_limit_ms){
	// TODO: replace with actual parallel implementation
	return negamax_serial(game, time_limit_ms);
}
