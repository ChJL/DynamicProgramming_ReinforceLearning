#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : Administrator
# date   : 2018/6/29


class Human:
    def __init__(self):
        pass

    def __str__(self):
        return "human"

    def take_action(self, current_state):
        '''
        The action by human player
        parameter: current_state
        return action
        '''
        while True:
            while True:
                command = input("please type in the format of i,j ")
                try:
                    i, j = [int(index) for index in command.split(",")]
                    break
                except:
                    print("input format are not acceptable, please re-type")
            action = i, j
            if action not in current_state.get_available_actions():
                print("action is not acceptable, please re-type")
                print("available states: ", current_state.get_available_actions())
            else:
                break
        return action
