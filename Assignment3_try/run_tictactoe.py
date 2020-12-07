#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : Administrator
# date   : 2018/6/28
# https://github.com/int8/monte-carlo-tree-search
# https://github.com/zhuliquan/tictactoe_mcts
# https://github.com/hayoung-kim/mcts-tic-tac-toe
from game import Game
from mcts import MCTS
from human import Human


if __name__ == '__main__':
    game = Game()
    human = Human()
    ai = MCTS()
    players = {0: ai, 1: human}
    game.render()
    print("=== start in restricted condition===")
    turn = 0
    while True:
        current_state = game.state
        action = players[turn].take_action(current_state)
        game.step(action)
        game.render()
        print("#=== {0}place at{1} ===#".format(players[turn], action))

        # judge the result
        is_over, winner = game.game_result()
        if is_over:
            if winner:
                print("winner {0}".format(players[turn]))
            else:
                print("tie")
            game.reset()
            game.render()
            print("==== Restart the Game ====")
            turn = 1
            #break

        # change the player 
        turn = (turn + 1) % 2


