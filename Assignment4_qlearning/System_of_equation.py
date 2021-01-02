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
#print(conge_weight)

# Calculate the minimum cost in each state, and record the last state (from which state)

def min_cost(Conge_weight, n, Goal_x, Goal_y):
    # (Cost to goal coordinate, next action - 0:x-1 up , 1:x+1 down, 2: y-1 left, 3:y+1 right
    mincost_matrix = np.zeros((n, n, 2))
    mincost_matrix[0, 9, 0] = 0
    loop = 0
    not_change = 0
    while loop < 2502:
        matrix_tmp = np.array(mincost_matrix)
        loop += 1
        for i in range(n):
            for j in range(n):
                if i == Goal_x and j == Goal_y:
                    mincost_matrix[i, j, 0] = 0
                else:
                    cost = np.array([])
                    pre_action = np.array([])
                    if i != 0:
                        cost = np.append(cost, matrix_tmp[i - 1, j, 0] + Conge_weight[i - 1, j, 0])
                        pre_action = np.append(pre_action, 0)
                    if i != N - 1:
                        cost = np.append(cost, matrix_tmp[i + 1, j, 0] + Conge_weight[i + 1, j, 1])
                        pre_action = np.append(pre_action, 1)
                    if j != 0:
                        cost = np.append(cost, matrix_tmp[i, j - 1, 0] + Conge_weight[i, j - 1, 2])
                        pre_action = np.append(pre_action, 2)
                    if j != N - 1:
                        cost = np.append(cost, matrix_tmp[i, j + 1, 0] + Conge_weight[i, j + 1, 3])
                        pre_action = np.append(pre_action, 3)

                    mincost_matrix[i, j, 0] = np.min(cost)
                    key = np.argmin(cost)
                    try:
                        mincost_matrix[i, j, 1] = pre_action[key]
                    except:
                        print("len of cost", len(cost))
                        print("error")
                        break

        # print("step: ", loop)
        if np.array_equal(matrix_tmp, mincost_matrix):
            not_change += 1
            if not_change == 2:
                break
    return mincost_matrix


# this row would take about 37 seconds
mincost = min_cost(conge_weight, N, goal_x, goal_y)

# print(conge_weight[0])


class Operation (object):
    def __init__(self, i, j) :
        self.up    = 0.25
        self.down  = 0.25
        self.left  = 0.25
        self.right = 0.25
        if i > 0:
            self.upState = (i - 1, j)
        else:
            self.upState = (i, j)
        if i < 49:
            self.downState = (i + 1, j)
        else:
            self.downState = (i, j)
        if j > 0:
            self.leftState = (i, j - 1)
        else:
            self.leftState = (i, j)
        if j < 49:
            self.rightState = (i, j + 1)
        else:
            self.rightState = (i, j)

rewardMatrix = np.zeros((N, N))
rewardMatrix[0, 9] = 0
tempMatrix = np.zeros((N, N))

stateMatrix = [[] for i in range(N)]

for i in range(N):
    for j in range(N):
        stateMatrix[i].append(Operation(i, j))

def updateReward(i, j):
    tempMatrix[i][j] = max((-1 + conge_prob[i, j, 0]*rewardMatrix[stateMatrix[i][j].upState[0]][stateMatrix[i][j].upState[1]] + (1 - conge_prob[i, j, 0])*rewardMatrix[i][j]),

                           (-1 + conge_prob[i, j, 1]*rewardMatrix[stateMatrix[i][j].downState[0]][stateMatrix[i][j].downState[1]] + (1 - conge_prob[i, j, 1])*rewardMatrix[i][j]),

                           (-1 + conge_prob[i, j, 2]*rewardMatrix[stateMatrix[i][j].leftState[0]][stateMatrix[i][j].leftState[1]] + (1 - conge_prob[i, j, 2])*rewardMatrix[i][j]),

                           (-1 + conge_prob[i, j, 3]*rewardMatrix[stateMatrix[i][j].rightState[0]][stateMatrix[i][j].rightState[1]] + (1 - conge_prob[i, j, 3])*rewardMatrix[i][j]))


threshold = 0.001
delta = 0
n = 0

for i in range(N):
    for j in range(N):
        v = rewardMatrix[i][j]
        updateReward(i, j)
        delta = max(delta, abs(v - tempMatrix[i][j]))

rewardMatrix = tempMatrix

while delta > threshold:
    delta = 0
    for i in range(N):
        for j in range(N):
            v = rewardMatrix[i][j]

            updateReward(i, j)

            delta = max(delta, abs(v - tempMatrix[i][j]))

    rewardMatrix = tempMatrix
    n += 1
    print(n)


# output

for i in range(0, N):
    for j in range(0, N):
        print(rewardMatrix[i][j])
        print(" ")
    print("\n")

