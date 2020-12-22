# try the random function for simulation
from random import random, uniform, randrange
import numpy as np
np.random.seed(0)
'''
p = uniform(1,3)
for t in range(3):
	print("===========")
	#print(np.random.rand(10))
	x = np.random.rand(10)
	y = np.array([])
	for i in range(10):
		print(x[i]*10)
		y = np.append(y,x[i]*10)
		#print(randrange(1,20))
		#print(uniform(0,4))
		#print np.random.rand(10) 
	print(y)
print(int(0.5))
'''
occur_prob = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
x = np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], size=4)
print(x)