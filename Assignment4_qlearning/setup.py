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

occur_prob = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
for i in range(N):
	for j in range(N):
		conge_prob[i,j] =  np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], size=4)

print(conge_prob[0])
