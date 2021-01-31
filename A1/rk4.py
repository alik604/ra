import numpy as np
from matplotlib import pyplot as plt 
def f(_, x):
	dx = np.zeros(4)
	dx[0] = x[1]
	dx[1] = x[3]**2 - k/x[0]**2 # u_1 is 0 
	dx[2] = x[3]
	dx[3] = -2*(x[1]*x[3])/(x[0]+0.001) #u_2 is 0
	# print(x," | " ,dx)
	return dx

if __name__ == '__main__':
	print("Start calculating")

	r = 410+6378 	# km
	r *=1000  		# meters
	dr = 0			# meters per secound  	
	theta  = 0		# rad?

	T = 92.68*60 	# secounds 
	dtheta  = 2*np.pi/T # rad per secound; Angular frequency

	G = 6.673*(10**-11)
	M = 5.97*(10**24)
	k = G*M

	x = np.array([r, dr, theta, dtheta])
	dx = f(None, x)
	print(f'X is      \t{x}')
	print(f'd/dx X is \t{dx}')


	list_of_dx = [dx]
	rk4 = [0, 0, 0, 0]
	dtime = 1
	time = np.arange(0, 3*1, step=dtime)
	for i in range(len(time)-1): # range(len(time))
		# x is a single step, half of which is 0. We only care about the y axis. 
		rk4[0] = dtime * f(None, list_of_dx[i])
		rk4[1] = dtime * f(None, list_of_dx[i] + rk4[0]/2) # my past way: list_of_dx[i] + rk4[0]/2
		rk4[2] = dtime * f(None, list_of_dx[i] + rk4[1]/2)
		rk4[3] = dtime * f(None, list_of_dx[i] + rk4[2])
		# print(f'd/dx X is \t{list_of_dx[i]}')
		# print(f'rk4 is    \t{rk4}')
		print("__________________________")

		next_dx = list_of_dx[i] + (rk4[0] + rk4[1]*2.0 + rk4[2]*2.0 + rk4[3])/6
		list_of_dx.append(next_dx)



	list_of_d 		= [i[0] for i in list_of_dx] # 0 
	list_of_dr 		= [i[1] for i in list_of_dx]
	list_of_theta 	= [i[2] for i in list_of_dx]
	list_of_dtheta	= [i[3] for i in list_of_dx] # 0 

	# print(f'r is     \t{list_of_d}')
	# print(f'theta is  \t{list_of_theta}')

	# plt.title("r")
	# plt.plot(list_of_d)
	# plt.show()

	# plt.title("d/dx r")
	# plt.plot(list_of_dr)
	# plt.show()

	# plt.title("theta")
	# plt.plot(list_of_theta)
	# plt.show()

	# plt.title("d/dx theta")
	# plt.plot(list_of_dtheta)
	# plt.show()

	# plt.title("sin and cos theta")
	# plt.plot(6378 * np.cos(np.linspace(0, 2*np.pi, 1000)), 6378 * np.sin(np.linspace(0, 2*np.pi, 1000)))
	# plt.plot(list_of_d*np.cos(list_of_theta), list_of_d*np.sin(list_of_theta))
	# plt.show()