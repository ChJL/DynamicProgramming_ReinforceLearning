import numpy as np
from math import exp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sns.set()

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

print("===")

# optimal policy matrix initialize
optimal = np.empty((C,T))

#initialize the value in the first state
for i in range(C+1):
	for j in range(3):
		V[i,T,j] = 0
V[0] = 0

# Set the price cannot go down policy
Price_Not_Go_Down = 0
#for each time level
for t in reversed(range(T)):
	#for each capacity level
	#print(t)s
	for x in range(1,C+1):
		#print(x)
		#print("======== T:",(t+1)," C: ",x," ========")
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

			# get revenue by given popularities
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

		ini_max = 0
		for v in range(3):
			if V[x,t,v] > ini_max:
				ini_max = V[x,t,v]
				optimal[x-1,t] = v+1


expected_revenue = np.max(V)
print(expected_revenue)
print("==============")
#print(optimal)
ax = sns.heatmap(optimal, vmin=1, vmax=3, cmap="RdYlGn", cbar_kws={'label': 'Class'})
#plt.xlabel("time")
ax.set_xlabel('Time', loc='left')
#plt.ylabel("capacity")
ax.set_ylabel('Capacity', loc='top')
plt.xticks([0,100,200,300,400,500,600],['0','100','200','300','400','500','600'],rotation=30)
plt.yticks([20, 40, 60, 80, 100],['20', '40', '60', '80','100'],rotation=30)
cbar = ax.collections[0].colorbar
cbar.set_ticks([1, 2, 3])
cbar.set_ticklabels(['1', '2', '3'])
ax.invert_yaxis()
#fig.figure(figsize=(20,5))
plt.savefig('Op_.png',dpi=300)
plt.show()
