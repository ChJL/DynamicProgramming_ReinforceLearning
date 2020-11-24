import numpy as np
from math import exp
from random import random, uniform, randrange
import seaborn as sns
import matplotlib.pyplot as plt

Dimension = 91

#pi*Probability = pi
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

#print("====last two =====")
#print(Pi_last_two)
#print("====last one =====")
#print(Pi_last_one)

Pi_diff = np.zeros((Dimension))
for i in range(Dimension):
	Pi_diff[i] = Pi_last_one[i] - Pi_last_two[i]

#print("======difference=====")
#print(Pi_diff)

Reward_vec = np.zeros((Dimension))
for i in range(Dimension):
	Reward_vec[i] = i*0.01 + 0.1
Average_cost = Pi_last_one@Reward_vec
#print(Reward_vec)
print("average cost: ", Average_cost,"\n")


# ==================
# V + Phi = r + PV
# solve -> [I-P]V + Phi = r 
print("=====Poisson equation=====")
print("Solve : [I-P]V + Phi = r ")

def PoissonEq (reward_v, prob_matrix):
	reward_vec = np.append(reward_v,0)
	poisson_matrix = np.zeros((92, 92))
	for i in range(91):
		for j in range(91):
			if prob_matrix[i][j] !=0:
				poisson_matrix[i][j] = - prob_matrix[i][j] 
	for i in range(91):
		poisson_matrix[i][i] = poisson_matrix[i][i] +1
		poisson_matrix[i][91] = 1
		#reward_vec[i] = 0.1 + i*0.01
	poisson_matrix[91][0] = 1
	poisson_matrix[91][91] = 0
	A = np.linalg.solve(poisson_matrix, reward_vec)
	return A


#print("=====result of Poisson Equation=====")
print("PE phi is:",PoissonEq(Reward_vec, Prob_matrix)[-1],"\n")

'''
# for plot stationary distribution
x = np.zeros((Dimension))
for i in range(Dimension):
	x[i] = i
fig, ax = plt.subplots()
plt.bar(x, Pi_last_one, color ="green",label='Probability' )
ax.annotate('0.1461', xy=(0, 0.1461), xytext=(20, 0.145),
            arrowprops=dict(facecolor='black', shrink=0.01, width = 0.1, headwidth=2),
            )
ax.annotate('0.1315', xy=(1, 0.1315), xytext=(20, 0.1315),
            arrowprops=dict(facecolor='black', shrink=0.01, width = 0.1, headwidth=2),
            )
ax.annotate('0.1170', xy=(2, 0.117), xytext=(20, 0.117),
            arrowprops=dict(facecolor='black', shrink=0.01, width = 0.1, headwidth=2),
            )
ax.legend(loc='upper right',fontsize = 12)
plt.ylabel('Probability in Stationary Distribution',fontsize = 13)
plt.xlabel('State',fontsize = 13)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.show()
'''
print( "======= value iteration ========")
def choosemin (array1, array2):
	array3 = np.zeros((len(array1)))
	for i in range(len(array1)):
		array3[i] = min(array1[i],array2[i])
	return array3
#========

V_init = np.zeros((Dimension))
V = V_init
Prob_first = Prob_matrix
Prob_second = np.zeros((Dimension,Dimension))
Reward_1 = Reward_vec
Reward_2 = np.zeros((Dimension))
for i in range(Dimension):
	Prob_second[i][0] =1
	Reward_2[i] = 0.5

V_diff = np.zeros((Dimension))
for i in range(Dimension):
	V_1_action = Reward_1 + Prob_first@V
	V_2_action = Reward_2 + Prob_second@V
	V_decision = choosemin(V_1_action,V_2_action)
	V_diff = V_decision - V
	V = V_decision
Value_phi = V_diff[0]
print("Value iteration phi: ", Value_phi,"\n")

# = ================================================ =
# = ================ Policy iteration method1 ====== =
# = ================================================ =
bestPolicy = np.zeros((Dimension))
for i in reversed(range(Dimension)):
    V = PoissonEq(Reward_vec, Prob_matrix) #solution
    # print("iteration nr: {}, new phi={}".format(i, V[-1]))
    if i == 90:
        continueCost = Reward_vec[i]
    else:
        continueCost = Reward_vec[i]+Prob_matrix[i][i+1]*V[i+1]
    repairCost = 0.5 + V[0]
    # print("continuecost: {}, repaircost={}".format(continueCost, repairCost))
    # print("previousReward: {}".format(Reward_vec[i]))
    if continueCost > repairCost:
        Reward_vec[i] = repairCost
        bestPolicy[i] = 1
        if i != 90:
            Prob_matrix[i][i+1]=0
        Prob_matrix[i][0]=1

print("poisson iteration new phi is:",PoissonEq(Reward_vec, Prob_matrix)[-1],"\n")
print(bestPolicy)
'''
# = ================================================ =
# = ================ Policy iteration method2 ====== =
# = ================================================ =

def choosemin_action (array1, array2):
	array3 = np.zeros((len(array1)))
	for i in range(len(array1)):
		if array1[i]<array2[i]:
			array3[i] = 0
		else:
			array3[i] = 1
	return array3

# =================================

#in Alpha,
#0 : action 1
#1 : action 2
#we set the initial Alpha [0,0,0,.....,0]

Alpha_init = np.zeros((Dimension))
Alpha = Alpha_init
for i in range(Dimension):
	reward_pol = np.zeros((Dimension))
	prob_pol = np.zeros((Dimension,Dimension))
	for j in range(Dimension):
		if Alpha[j] ==0:
			reward_pol[j] = Reward_1[j]
			prob_pol[j] = Prob_first[j]

		else:
			reward_pol[j] = Reward_2[j]
			prob_pol[j] = Prob_second[j]

	poisson_V = PoissonEq(reward_pol,prob_pol)[:-1]
	action1 = Reward_1 + Prob_first@poisson_V
	action2 = Reward_2 + Prob_second@poisson_V
	Alpha = choosemin_action(action1,action2)
	if i == 90:
		policy_phi = PoissonEq(reward_pol,prob_pol)[-1]
print("=======Poisson policy iteration")
print("Policy action:")
print(Alpha)
print("Policy iteration Phi: ",policy_phi)
'''




