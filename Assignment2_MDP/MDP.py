import numpy as np
from math import exp
from random import random, uniform, randrange
import seaborn as sns
import matplotlib.pyplot as plt

Dimension = 91
# Probability matrix
Prob_matrix = np.zeros((Dimension,Dimension))

# Distribution Pi, allocate the initial condition
Pi_init = np.zeros((Dimension))
Pi_init[0] =1

# allocate the value in Probability matrix
for i in range(Dimension):
	Prob_matrix[i][0] = 0.1 + i*0.01
	if i <90:
		Prob_matrix[i][i+1] = 0.9 - i*0.01

# Compare the Pi(n) Pi(n+1), see if they are almost the same
Pi_after = Pi_init
Pi_last_two = np.zeros((Dimension))
Pi_last_one = np.zeros((Dimension))
for i in range(300):
	Pi_after = Pi_after@Prob_matrix
	if i == 298:
		Pi_last_two = Pi_after
	if i == 299:
		Pi_last_one = Pi_after

print("====last two =====")
print(Pi_last_two)
print("====last one =====")
print(Pi_last_one)

Pi_diff = np.zeros((Dimension))
for i in range(Dimension):
	Pi_diff[i] = Pi_last_one[i] - Pi_last_two[i]

print("======difference=====")
print(Pi_diff)

Reward_vec = np.zeros((Dimension))
for i in range(Dimension):
	Reward_vec[i] = i*0.01 + 0.1
Average_cost = Pi_last_one@Reward_vec
print("average cost: ", Average_cost)


# ==================
# V + Phi = r + PV
# solve -> [I-P]V + Phi = r 
print("=====Poisson equation matrix =====")
print(" solve -> [I-P]V + Phi = r ")
Reward_vec_Poisson = np.zeros((Dimension+1))
Poisson_matrix = np.zeros((Dimension+1, Dimension+1))
for i in range(Dimension):
	for j in range(Dimension):
		if Prob_matrix[i][j] !=0:
			Poisson_matrix[i][j] = - Prob_matrix[i][j] 
for i in range(Dimension):
	Poisson_matrix[i][i] = Poisson_matrix[i][i] +1
	Poisson_matrix[i][91] = 1
	Reward_vec_Poisson[i] = 0.1 + i*0.01
Poisson_matrix[91][0] = 1
Poisson_matrix[91][91] = 0

print(Poisson_matrix)
print("======reward vector=====")
print(Reward_vec_Poisson)

C = np.linalg.solve(Poisson_matrix, Reward_vec_Poisson)

print("=====result of Poisson Equation=====")
print("phi is:", C[-1])
