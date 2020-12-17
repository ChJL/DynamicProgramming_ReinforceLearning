import numpy as np
import random
from math import sqrt, log
import time

from MCTS_w_Random import Board, Node, UCT, Random_Opponent

board = Board()
board.restricted_cond()
rd_opp = Random_Opponent()
#board.show()

print("==== start the simulation =====")
print(" it will take about 4 minutes.")
print(" it's the simulation for probability of victory part in the report")

players = {0: "MCTS O", 1: "Random X"}

turn = 0;

# how many games should play
count = 0

# To see the probability
win_time = 0
los_time = 0
tie_time = 0
# set the first step condition for MCTS
first_step = 1


while count < 500:
    if turn ==0 and first_step == 0 :
        ''' MCTS play '''

        current_state = board
        action = UCT(current_state,1000)
        board.move(action)

    elif first_step == 1:
        # restrict the first step of MCTS
        board.move([0,0])
        first_step = 0
    else:
        ''' Random Opponent Play'''
        current_state = board
        action = rd_opp.take_action(current_state)
        board.move(action)
        
    # judge the result
    is_over, winner = board.find_winner()
    if is_over:
        if winner != None:

            if winner == 0:
                win_time += 1
            if winner == 1:
                los_time += 1
            
        else:
            #print(" tie !")
            tie_time += 1
        count +=1
        
        board.restricted_cond()
        first_step = 1

        turn =1
        if count%25 == 0:
            expected_r = (win_time - los_time)/count
            print(count," Step")
            print(" win probability : ", win_time/count)
        

    turn ^= 1