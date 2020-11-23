import numpy as np

print("=====Poisson equation=====")
print("Solve : [I-P]V + Phi = r ")

Dimension = 91

Prob_matrix = np.zeros((Dimension,Dimension))

for i in range(Dimension):
	Prob_matrix[i][0] = 0.1 + i*0.01
	if i <90:
		Prob_matrix[i][i+1] = 0.9 - i*0.01

Reward_vec = np.zeros((Dimension))
for i in range(Dimension):
	Reward_vec[i] = i*0.01 + 0.1


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
print("standard phi is:",PoissonEq(Reward_vec, Prob_matrix)[-1],"\n")

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

print("new phi is:",PoissonEq(Reward_vec, Prob_matrix)[-1],"\n")
print(bestPolicy)