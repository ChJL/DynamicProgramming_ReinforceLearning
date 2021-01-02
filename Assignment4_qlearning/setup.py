import numpy as np
from math import exp
from random import random, uniform, randrange
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(200)
# grid world length
N = 50

# goal_x, goal_y: coordinate for final termination
goal_x = 0
goal_y = 9

# states and congestion probability matrix
states = np.zeros((N,N))
# up down left right
conge_prob = np.zeros((N,N,4))
conge_weight = np.zeros((N,N,4))

occur_prob = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
for i in range(N):
	for j in range(N):
		conge_prob[i,j] =  np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], size=4)

for i in range(N):
	for j in range(N):
		for k in range(4):
			conge_weight[i,j,k] = 1/conge_prob[i,j,k]
			# set the boundaries to 0
			if i == 0 and k == 0 :
				conge_weight[i,j,k] = 0
			if j == 0 and k == 2 :
				conge_weight[i,j,k] = 0
			if i == 49 and k == 1 :
				conge_weight[i,j,k] = 0
			if j == 49 and k == 3 :
				conge_weight[i,j,k] = 0

# Calculate the minimum cost in each state, and record the next action

def min_cost(Conge_weight, n, Goal_x, Goal_y):
	# (Cost to goal coordinate, next action - 0:x-1 up , 1:x+1 down, 2: y-1 left, 3:y+1 right
  mincost_matrix = np.zeros((n,n,2))
  mincost_matrix[0,9,0] = 0
  loop = 0
  not_change = 0
  while loop < 2502:
    matrix_tmp = np.array(mincost_matrix)
    loop +=1
    for i in range (n):
      for j in range (n):
        if i == Goal_x and j == Goal_y:
          mincost_matrix[i,j,0] = 0
        else:
          cost = np.array([])
          next_action = np.array([])
          if i != 0 :
            cost = np.append(cost, matrix_tmp[i-1,j,0] + Conge_weight[i-1,j,0])
            next_action = np.append(next_action,0)
          if i != N-1 :
            cost = np.append(cost, matrix_tmp[i+1,j,0] + Conge_weight[i+1,j,1])
            next_action = np.append(next_action, 1)
          if j != 0 :
            cost = np.append(cost, matrix_tmp[i,j-1,0] + Conge_weight[i,j-1,2])
            next_action = np.append(next_action, 2)
          if j != N-1 : 
            cost = np.append(cost, matrix_tmp[i,j+1,0] + Conge_weight[i,j+1,3])
            next_action = np.append(next_action, 3)
          
          mincost_matrix[i,j,0] = np.min(cost)
          key = np.argmin(cost)
          try:
            mincost_matrix[i,j,1] = next_action[key]
          except:
            print("len of cost", len(cost))
            print("error")
            break


    #print("step: ", loop)
    if np.array_equal(matrix_tmp , mincost_matrix):
      not_change +=1
      if not_change == 2:
        break
  return mincost_matrix

# simple Heuristic, this function would return an array of Costs
def Heuristic(Goal_x, Goal_y, Simu_vec, Simu_t, Conge_cost):
  H_cost = np.array([])
  for t in range(Simu_t):
    i = int(Simu_vec[t,0])
    j = int(Simu_vec[t,1])
    total_cost = 0
    while i != Goal_x or j != Goal_y:
      moves = np.array([])
      move_cost = np.array([])
      # pick the min move from all possible moves
      if i < Goal_x and i != N-1:
        moves = np.append(moves, 1) # down
        move_cost = np.append(move_cost, Conge_cost[i,j,1])
      if i > Goal_x and i != 0:
        moves = np.append(moves, 0) # up
        move_cost = np.append(move_cost, Conge_cost[i,j,0])
      if j < Goal_y and j != N-1:
        moves = np.append(moves, 3) # right
        move_cost = np.append(move_cost, Conge_cost[i,j,3])
      if j > Goal_y and j != 0:
        moves = np.append(moves, 2) # left
        move_cost = np.append(move_cost, Conge_cost[i,j,2])
      
      min_move_cost = np.min(move_cost)
      min_key = np.argmin(move_cost)
      # calculate the cost so far
      total_cost += min_move_cost
      
      # update the location
      move = moves[min_key]
      if move == 1:
        i = i+1
      elif move == 0:
        i = i-1
      elif move == 3:
        j = j+1
      elif move == 2:
        j = j-1
    H_cost = np.append(H_cost, total_cost)
  return H_cost


# Q learning 

def QLearning(Goal_x, Goal_y, Q, Lr=0.01, Eps=0.3, Eps_decay=0.00005, Gamma=0.99, Simu_t, Simu_vec, Conge_cost):
  Q_cost = np.array([])

  for t in range(Simu_t):
    i = int(Simu_vec[t,0])
    j = int(Simu_vec[t,1])
    total_cost = 0
    
    if Eps > 0.01:
      Eps -= Eps_decay

    while i != Goal_x or j != Goal_y:
      Qtmp = np.array(Q)
      moves = np.array([])
      move_cost = np.array([])
      # possible moves:
      if i != N-1:
        moves = np.append(moves, 1) # down
        move_cost = np.append(move_cost, Conge_cost[i,j,1])
      if i != 0:
        moves = np.append(moves, 0) # up
        move_cost = np.append(move_cost, Conge_cost[i,j,0])
      if j != N-1:
        moves = np.append(moves, 3) # right
        move_cost = np.append(move_cost, Conge_cost[i,j,3])
      if j != 0:
        moves = np.append(moves, 2) # left
        move_cost = np.append(move_cost, Conge_cost[i,j,2])
      
      # explore
      random_pick = np.random.rand()
      if Eps > random_pick:
        next_move = np.randeom.choice(moves)
      
      #exploit
      else:


# this row would take about 37 seconds
mincost = min_cost(conge_weight, N, goal_x, goal_y)

# Heuristic simulation parameter:
simulation_size = 2
simulation_array = np.zeros((simulation_size,2))
for i in range(simulation_size):
  simulation_array[i] = np.random.choice(49,2)

# Simple Heuristic 
h_cost = Heuristic(goal_x, goal_y, simulation_array, simulation_size, conge_weight)

# Q learning
Q_vec = np.zeros((N,N,4))



