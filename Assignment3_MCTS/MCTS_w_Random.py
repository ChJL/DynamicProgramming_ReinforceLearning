import numpy as np
import random
from math import sqrt, log
import time
class Board:
 
    def __init__(self):
        self.state = np.zeros([2,3,3])
        self.player = 0 # current player's turn
    
    def reset(self):
        # new game
        self.state = np.zeros([2,3,3])
        self.player = 0

    def restricted_cond(self):
        # set the initialization to restricted condition
        self.state = np.zeros([2,3,3])
        self.state[0,0,1] = 1
        self.state[0,2,0] = 1
        self.state[1,0,2] = 1
        self.state[1,1,1] = 1

        self.player = 0

    def copy(self):
        #make copy of the board

        copy = Board()
        copy.player = self.player
        copy.state = np.copy(self.state)
        return copy

    def move(self, move):
        #take move of form [x,y] and play the move for the current player
        
        if np.any(self.state[:,move[0],move[1]]): return
        if move not in self.get_moves():
            print("invalid move")
            return
        self.state[self.player][move[0],move[1]] = 1
        self.player ^= 1

    def get_moves(self):
        #return remaining possible board moves (where there are no O's or X's)
      
        return np.argwhere(self.state[0]+self.state[1]==0).tolist()

    def result(self):
        #check rows, columns, and diagonals for sequence of 3 X's or 3 O's
        
        board = self.state[self.player^1]
        col_sum = np.any(np.sum(board,axis=0)==3)
        row_sum = np.any(np.sum(board,axis=1)==3)
        d1_sum  = np.any(np.trace(board)==3)
        d2_sum  = np.any(np.trace(np.flip(board,1))==3)
        return col_sum or row_sum or d1_sum or d2_sum
      
    def find_winner(self):
        board = self.state[self.player^1]
        col_sum = np.any(np.sum(board,axis=0)==3)
        row_sum = np.any(np.sum(board,axis=1)==3)
        d1_sum  = np.any(np.trace(board)==3)
        d2_sum  = np.any(np.trace(np.flip(board,1))==3)
        temp_board = self.state
        if col_sum or row_sum or d1_sum or d2_sum:
            #print(self.player^1)
            return True, self.player^1
        elif np.sum(temp_board) == 9:
            return True, None
        else:
            return False, None
            
    def show(self):
        board = self.state
        for i in range(3):
            for j in range(3):
                if board[0,i,j] ==1:
                    print(" o ", end = "")
                elif board[1,i,j] ==1:
                    print(" x ", end = "")
                else:
                    print(" . ", end = "")
            print()

class Node:

  def __init__(self, parent=None, action=None, board=None):
    self.parent = parent
    self.board = board
    self.children = []
    self.wins = 0
    self.visits = 0
    self.untried_actions = board.get_moves()
    self.action = action

  def select(self):
    #select child of node with  highest UCB1 value
    
    s = sorted(self.children, key=lambda c:c.wins/c.visits+0.2*sqrt(2*log(self.visits)/c.visits))
    return s[-1]

  def expand(self, action, board):
    #expand parent node (self) by adding child node with given action and state

    child = Node(parent=self, action=action, board=board)
    self.untried_actions.remove(action)
    self.children.append(child)
    return child

  def update(self, result):
    self.visits += 1
    self.wins += result

def UCT(rootstate, maxiters):

  root = Node(board=rootstate)

  for i in range(maxiters):
    node = root
    board = rootstate.copy()

    # selection - select best child if parent fully expanded and not terminal
    while node.untried_actions == [] and node.children != []:
      node = node.select()
      board.move(node.action)

    # expansion - expand parent to a random untried action
    if node.untried_actions != []:
      a = random.choice(node.untried_actions)
      board.move(a)
      node = node.expand(a, board.copy())

    # simulation - rollout to terminal state from current state using random actions
    while board.get_moves() != [] and not board.result():
      board.move(random.choice(board.get_moves()))

    # backpropagation - propagate result of rollout game up the tree
    # reverse the result if player at the node lost the rollout game
    while node != None:
      result = board.result()
      if result:
        if node.board.player==board.player:
          result = 1
        else: result = -1
      else: result = 0
      node.update(result)
      node = node.parent

  s = sorted(root.children, key=lambda c:c.wins/c.visits)
  return s[-1].action


class Random_Opponent:
    def __init__(self):
        pass

    def __str__(self):
        return "random opponent"
    def take_action(self,current_state):
        random_action = random.choice(current_state.get_moves())

        return random_action




class Human:
    def __init__(self):
        pass

    def __str__(self):
        return "human"
    def take_action(self,current_state):
        while True:
            while True:
                command = input("please type in the format of i,j ")
                try:
                    i, j = [int(index) for index in command.split(",")]
                    break
                except:
                    print("input format are not acceptable, please re-type")
            action = []
            action.append(i)
            action.append(j)
            
            if action not in current_state.get_moves():
                print("action is not acceptable, please re-type")
                print("available states: ", current_state.get_moves())
            else:
                break
        return action

if __name__ == '__main__':

	board = Board()
	# the game can also start in restricted case
	#board.restricted_cond()

	# the game can also play by human vs MCTS
	#human = Human()

	# The Oppenent take random actions
	rd_opp = Random_Opponent()

	print(" This Pyfile would show you 5 games that are played by: MCTS O vs Random agent X")
	time.sleep(6)
	print("====== Start the Game 1=======")
	# board in initial condition
	board.show()


	players = {0: "MCTS O", 1: "Random X"}


	# determine who start first
	# but for O and X, it still need to revise the self.player in the class
	turn = 0;

	# how many games should play
	count = 0

	# To see the probability
	win_time = 0
	los_time = 0
	tie_time = 0


	while count < 5:
	    if turn ==0:
	        ''' MCTS play '''

	        print("=== Player {0} turn ====".format(players[turn]))
	        current_state = board
	        action = UCT(current_state,1000)
	        board.move(action)
	        board.show()
	        print("#==={0} move {1} ===#".format(players[turn], action))
	    else:
	        ''' Random Opponent Play'''

	        print("=== Player {0} turn ====".format(players[turn]))
	        current_state = board
	        action = rd_opp.take_action(current_state)

	        board.move(action)
	        board.show()
	        print("#==={0} move {1} ===#".format(players[turn], action))
	        
	    # judge the result
	    is_over, winner = board.find_winner()
	    if is_over:
	        print("#========== End game ==========#")
	        if winner != None:
	            print("winner is : player {0}".format(players[winner]))
	            #print("winner is:", winner)
	            if winner == 0:
	                win_time += 1
	            if winner == 1:
	                los_time += 1
	            
	        else:
	            print(" tie !")
	            tie_time += 1

	        count +=1
	        

	        if count == 5:
	            expected_r = (win_time - los_time)/count
	            print("In ",count," games, MCTS O win:", win_time," times")

	        else:
	        	board.reset()
	        	print()
	        	#board.restricted_cond()
	        	print("====== Start the Game ",count+1,"=======")
	        	board.show()

	        	turn =1
	        

	    turn ^= 1