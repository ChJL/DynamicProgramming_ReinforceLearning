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

# Calculate the minimum cost in each state, and record the last state (from which state)

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
          pre_action = np.array([])
          if i != 0 :
            cost = np.append(cost, matrix_tmp[i-1,j,0] + Conge_weight[i-1,j,0])
            pre_action = np.append(pre_action,0)
          if i != N-1 :
            cost = np.append(cost, matrix_tmp[i+1,j,0] + Conge_weight[i+1,j,1])
            pre_action = np.append(pre_action, 1)
          if j != 0 :
            cost = np.append(cost, matrix_tmp[i,j-1,0] + Conge_weight[i,j-1,2])
            pre_action = np.append(pre_action, 2)
          if j != N-1 : 
            cost = np.append(cost, matrix_tmp[i,j+1,0] + Conge_weight[i,j+1,3])
            pre_action = np.append(pre_action, 3)
          
          mincost_matrix[i,j,0] = np.min(cost)
          key = np.argmin(cost)
          try:
            mincost_matrix[i,j,1] = pre_action[key]
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

# this row would take about 37 seconds
mincost = min_cost(conge_weight, N, goal_x, goal_y)

#print(conge_weight[0])
