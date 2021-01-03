import numpy as np
from math import exp
from random import random, uniform, randrange

np.random.seed(200)
# grid world length
N = 50

# goal_x, goal_y: coordinate for final termination
goal_x = 0
goal_y = 9

# states and congestion probability matrix
states = np.zeros((N, N))
# up down left right
conge_prob = np.zeros((N, N, 4))
conge_weight = np.zeros((N, N, 4))

occur_prob = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
for i in range(N):
    for j in range(N):
        conge_prob[i, j] = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], size=4)

for i in range(N):
    for j in range(N):
        for k in range(4):
            conge_weight[i, j, k] = 1 / conge_prob[i, j, k]
            # set the boundaries to 0
            if i == 0 and k == 0:
                conge_weight[i, j, k] = 0
            if j == 0 and k == 2:
                conge_weight[i, j, k] = 0
            if i == 49 and k == 1:
                conge_weight[i, j, k] = 0
            if j == 49 and k == 3:
                conge_weight[i, j, k] = 0


# Calculate the minimum cost in each state, and record the next action

def min_cost(Conge_weight, n, Goal_x, Goal_y):
	# (Cost to goal coordinate, previous action)
  mincost_matrix = np.zeros((n,n,2))
  mincost_matrix[0,9,0] = 0
  mincost_present = np.zeros(((n,n)))
  loop = 0
  not_change = 0
  while loop < 2502:
    matrix_tmp = np.array(mincost_matrix)
    loop +=1
    for i in range (n):
      for j in range (n):
        if i == Goal_x and j == Goal_y:
          mincost_matrix[i,j,0] = 0
          mincost_present[i,j] = 0
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
          mincost_present[i,j] = np.min(cost)
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
      if not_change == 50:
        break
  return mincost_matrix, mincost_present


# simple Heuristic, this function would return an array of Costs
def Heuristic(Goal_x, Goal_y, Simu_vec, Simu_t, Conge_cost):
    H_cost = np.array([])
    for t in range(Simu_t):
        i = int(Simu_vec[t, 0])
        j = int(Simu_vec[t, 1])
        total_cost = 0
        while i != Goal_x or j != Goal_y:
            moves = np.array([])
            move_cost = np.array([])
            # pick the min move from all possible moves
            if i < Goal_x and i != N - 1:
                moves = np.append(moves, 1)  # down
                move_cost = np.append(move_cost, Conge_cost[i, j, 1])
            if i > Goal_x and i != 0:
                moves = np.append(moves, 0)  # up
                move_cost = np.append(move_cost, Conge_cost[i, j, 0])
            if j < Goal_y and j != N - 1:
                moves = np.append(moves, 3)  # right
                move_cost = np.append(move_cost, Conge_cost[i, j, 3])
            if j > Goal_y and j != 0:
                moves = np.append(moves, 2)  # left
                move_cost = np.append(move_cost, Conge_cost[i, j, 2])

            min_move_cost = np.min(move_cost)
            min_key = np.argmin(move_cost)
            # calculate the cost so far
            total_cost += min_move_cost

            # update the location
            move = moves[min_key]
            if move == 1:
                i = i + 1
            elif move == 0:
                i = i - 1
            elif move == 3:
                j = j + 1
            elif move == 2:
                j = j - 1
        H_cost = np.append(H_cost, total_cost)
    return H_cost


# Q learning

def findMaxQ(Q, x, y, Moves):
    q_max = -1000000
    move_dec = 0
    # print("moves in function: ",Moves)
    for mv in Moves:
        mv_i = int(mv)
        # print(Q[x,y,mv_i])
        if Q[x, y, mv_i] >= q_max:
            q_max = Q[x, y, mv_i]
            move_dec = mv_i
    return [move_dec, q_max]


def QLearning(Goal_x, Goal_y, Q, Simu_t, Simu_vec, Conge_cost, Lr=0.1, Eps=0.99, Eps_decay=0.005, Gamma=0.99):
    Q_cost = np.array([])

    for t in range(Simu_t):
        print("simu time :", t)
        print("start coordinate: ", Simu_vec[t])
        i = int(Simu_vec[t, 0])
        j = int(Simu_vec[t, 1])
        total_cost = 0
        total_step = 0

        if Eps > 0.01:
            Eps -= Eps_decay
        if Lr > 0.01:
            Lr -= 0.0001

        while i != Goal_x or j != Goal_y:
            Qtmp = np.array(Q)
            moves = np.array([])
            move_cost = np.array([])

            flag = 0
            # possible moves:
            if i != N - 1:
                moves = np.append(moves, 1)  # down
                move_cost = np.append(move_cost, Conge_cost[i, j, 1])
            if i != 0:
                moves = np.append(moves, 0)  # up
                move_cost = np.append(move_cost, Conge_cost[i, j, 0])
            if j != N - 1:
                moves = np.append(moves, 3)  # right
                move_cost = np.append(move_cost, Conge_cost[i, j, 3])
            if j != 0:
                moves = np.append(moves, 2)  # left
                move_cost = np.append(move_cost, Conge_cost[i, j, 2])

            # explore
            random_pick = np.random.rand()
            if Eps > random_pick:
                next_move = np.random.choice(moves)

            # exploit
            else:
                # print("======= Exploit Q =======")
                Q_val = findMaxQ(Qtmp, i, j, moves)
                next_move = Q_val[0]
                flag = 1

            # update location by move
            next_loc = move(i, j, next_move)

            # possible moves for next state
            moves_next_state = np.array([])
            if next_loc[0] != N - 1:
                moves_next_state = np.append(moves_next_state, 1)  # down

            if next_loc[0] != 0:
                moves_next_state = np.append(moves_next_state, 0)  # up

            if next_loc[1] != N - 1:
                moves_next_state = np.append(moves_next_state, 3)  # right

            if next_loc[1] != 0:
                moves_next_state = np.append(moves_next_state, 2)  # left

            # the reward (cost) which would be use in the following formula
            next_move_i = int(next_move)

            # Q(t+1, a)
            # print("current loc", i,j)
            # print("possible moves ", moves)
            # print("next move: ", next_move_i)
            # print("0 for explore , 1 for exploit: ", flag)


            Q_val_next = findMaxQ(Qtmp, next_loc[0], next_loc[1], moves_next_state)
            r = next_loc[2]
            # Update Q

            Q[i, j, next_move_i] = (1 - Lr) * Qtmp[i, j, next_move_i] + Lr * (r + Gamma * Q_val_next[1])
            total_cost += 1

            i = next_loc[0]
            j = next_loc[1]

            total_step += 1

        Q_cost = np.append(Q_cost, total_cost)
        print("total cost: ", total_cost)
        print("total step: ", total_step)

    return Q_cost, Q
"""
limited bias:
    So, create some initial Q-values which steer your algorithm to the destination (this is the bias), 
    but make it small enough, that when learning about congestions, the algorithm will also choose other paths.
    我的理解是初始的q-table数值进行一些设置，比如j<9的时候，对于action：向左 以及action：向下（即向远离终点的方向移动时）加入penalty；
    同理j>9 的时候，对于action：向右 以及action：向下（即向远离终点的方向移动时）加入penalty；
    
    源代码中：
    Q_val_next = findMaxQ(Qtmp, next_loc[0], next_loc[1], moves_next_state)
    Q[i, j, next_move_i] = (1 - Lr) * Qtmp[i, j, next_move_i] + Lr * (r + Gamma * Q_val_next[1])
    修改为：
    penalty = -0.05 ???数值大小不知
    if (j < 9 and next_move == 2) or (j > 9 and next_move == 3) or (next_move == 1):
        Q[i, j, next_move_i] = (1 - Lr) * Qtmp[i, j, next_move_i] + Lr * (r + Gamma * Q_val_next[1] + penalty)
    else:
        Q[i, j, next_move_i] = (1 - Lr) * Qtmp[i, j, next_move_i] + Lr * (r + Gamma * Q_val_next[1])
    
"""

def move(i, j, action):
    indicator = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], size=1)
    r = 0
    if indicator <= conge_prob[i, j, action]:
        if action == 0:
            next_state_x = i - 1
            next_state_y = j
        elif action == 1:
            next_state_x = i + 1
            next_state_y = j
        elif action == 2:
            next_state_x = i
            next_state_y = j - 1
        elif action == 3:
            next_state_x = i
            next_state_y = j + 1
        if (next_state_y == 9) and (next_state_x == 0):
            r = 1
        return next_state_x, next_state_y, r
    else:
        return i, j, r



# this row would take about 37 seconds
mincost, mincost_pres = min_cost(conge_weight, N, goal_x, goal_y)

# Heuristic simulation parameter:
simulation_size = 100
simulation_array = np.zeros((simulation_size, 2))
for i in range(simulation_size):
    simulation_array[i] = np.random.choice(49, 2)

'''
# Simple Heuristic 
h_cost = Heuristic(goal_x, goal_y, simulation_array, simulation_size, conge_weight)
'''

# Q learning
Q_vec = np.zeros((N, N, 4))

simulation_size = 1000
simulation_array = np.zeros((simulation_size, 2))
for i in range(simulation_size):
  simulation_array[i,0] = 49
  simulation_array[i,1] = 49

qlearning_cost, New_Q1 = QLearning(goal_x, goal_y, Q_vec, simulation_size, simulation_array, conge_weight)

