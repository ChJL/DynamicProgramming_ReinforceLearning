import numpy as np
from math import exp
from random import random, uniform, randrange
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(200)

f = np.array([500,300,200])
# capacity
C = 100
# time of iteration
T = 600	
# No. of class
classes = np.array([0,1,2])

mu = np.array([0.001,0.015,0.05])
vu = np.array([0.01,0.005,0.0025])
# value function & initialize
V = np.empty((C+1,T+1,3))

#print(V.shape)
# optimal policy matrix initialize
optimal = np.empty((C,T))

#initialize the value in the first state
for i in range(C+1):
	for j in range(3):
		V[i,T,j] = 0
V[0] = 0

# Set the policy for Price can not go down (1)
Price_Not_Go_Down = 0

#for each time level
for t in reversed(range(T)):
	#for each capacity level
	for x in range(1,C+1):
		#for each class
		for j in classes:

			prob_acc = 0
			prob_rej = 1
			prob_ini = 0
			max_revenue = 0

			# probibilities of accepting & rejecting each price of class
			# this could be a for loop since the probabilities may be accumulated
			# because when price is in class3, people in class1 and class2 are also
			# williing to pay

			for i in range(j+1):
				prob_ini = mu[i]*exp(vu[i]*t)
				prob_acc = prob_acc + prob_ini
				prob_rej = prob_rej - prob_ini
			total_revnue = 0

			# get revenue
			for k in range(0,3):
				total_revnue = prob_acc*(f[j]+V[x-1,t+1,k])  + prob_rej*V[x,t+1,k]
				if Price_Not_Go_Down == 0:
					if (total_revnue > max_revenue):
						max_revenue = total_revnue
						V[x,t,j] = max_revenue
						#optimal[x-1,t] = k+1
				if Price_Not_Go_Down == 1:
					if(total_revnue > max_revenue) and j<=k:
						max_revenue = total_revnue
						V[x,t,j] = max_revenue
						#optimal[x-1,t] = k+1

		# fill in the optimal policy matrix
		ini_max = 0
		for v in range(3):
			if V[x,t,v] > ini_max:
				ini_max = V[x,t,v]
				optimal[x-1,t] = v+1






expected_revenue = np.max(V)

print("expected revenue:",expected_revenue)
print("===============")


# ========== plot optimal policy =========
#sns.heatmap(optimal)
#plt.show()
'''
# =========== simulate 1000 times =========

simulation_profit_array = np.empty(1000)
# simulation and calculate the profit:
for simulation_t in range(1000):	
	capacity_left = 100
	total_profilt = 0
	random_demand_array = np.random.rand(600)
	for t in range(T):
		if capacity_left ==0:
			break
		demand = random_demand_array[t]*10
		if demand <= optimal[capacity_left-1,t]:
			j = optimal[capacity_left-1,t]-1
			k = int(j)
			total_profilt = f[k] + total_profilt
			capacity_left -= 1
		else:
			continue
	simulation_profit_array[simulation_t] = total_profilt

print("=====average of 1000 times=====")
print(np.average(simulation_profit_array))
print("========max of profit =======")
print(np.max(simulation_profit_array))
print("========min of profit =======")
print(np.min(simulation_profit_array))

'''
# ============ demand simulation plot ================

capacity_left = 100
total_profilt = 0
demand_choose = np.array([])
sell_price = np.array([])
capacity_record = np.array([])
random_demand_array = np.random.rand(600)
for t in range(T):
	capacity_record = np.append(capacity_record,capacity_left)
	# class that demand accept, 0: first class -> 2 thir
	if capacity_left ==0:
		break
	j = optimal[capacity_left-1,t]-1
	k = int(j)
	#print(capacity_left)
	#print(k)
	sell_price = np.append(sell_price,f[k])
	demand = random_demand_array[t]*10
	#if demand > optimal[capacity_left-1,t]:
	if demand > 3:
		demand_choose = np.append(demand_choose,0)
	if demand <=3:
		d = int(demand)
		demand_choose = np.append(demand_choose,f[d])
	if demand <= optimal[capacity_left-1,t]:

		j = optimal[capacity_left-1,t]-1
		k = int(j)
		total_profilt = f[k] + total_profilt
		capacity_left -= 1
	else:
		continue
shape = np.shape(demand_choose)

print("=====total_profit for simualting 1 time====")
print(total_profilt)
print("===========================================")
print("Ploting graph......")

t1 = np.arange(0,len(demand_choose),1)
ax1 = plt.subplot(211)
plt.ylabel('Price',fontsize = 15)
plt.xlabel('Time',fontsize = 15)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.plot(t1, demand_choose, 'o',ms=7, alpha=0.5, color='orange',label='demand')
plt.plot(t1, sell_price, '.', color='blue',label='price of policy')
ax1.legend(loc='lower right',fontsize = 12)

ax2 = plt.subplot(212)
plt.plot(t1, capacity_record, 'o-',ms=3, color = 'green', label = 'remain capacity')
plt.ylabel('Remain capacity',fontsize = 15)
plt.xlabel('Time',fontsize = 15)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
ax2.legend(loc = 'center right',fontsize = 12)

plt.show()

