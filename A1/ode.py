import numpy as np
import matplotlib.pyplot as plt
import sys

# watched: https://www.youtube.com/watch?v=_0mvWedqW7c

def backward_euler(a,y0,t):
    '''Approximate the solution of y'=ax by backward Euler's method.

    Parameters
    ----------
    f : function
    y0 : numpy array
    t : array
        1D NumPy array of t values where we approximate y values. Time step
        at each iteration is given by t[n+1] - t[n].

    Returns
    -------
    y : 2D NumPy array
        Approximation y[n] of the solution y(t_n) computed by Euler's method.

        𝑦^𝑘+1 = 𝑦^𝑘 + ℎ * 𝑓(𝑦_𝑘, 𝑢_𝑘)

    '''
    x = 1
    return x
    




def forward_euler(f, a, y0, t):
    '''Approximate the solution of y'=ax by forward Euler's method.

    Parameters
    ----------
    f : function
    y0 : numpy array
    t : array
        1D NumPy array of t values where we approximate y values. Time step
        at each iteration is given by t[n+1] - t[n].

    Returns
    -------
    y : 2D NumPy array
        Approximation y[n] of the solution y(t_n) computed by Euler's method.


        𝑦^𝑘+1 = 𝑦^𝑘 + ℎ * 𝑓(𝑦_𝑘+1, 𝑢_𝑘)
    '''

    m = len(y0)                     # Number of ODEs
    n = int((t[-1] - t[0])/delta_time) # Number of sub-intervals
    x = t[0]
    y = y0

    x_out = np.append(np.empty(0), t[0])
    y_out = np.append(np.empty(0), y0[0]) 
    print(x_out)
    print(y_out)

    for i in range(n):
        print(f'x {x}, y {y}')
        yprime = f(x, y) # Evaluates dy/dx
        
        for j in range(m):
            y[j] = y[j] + delta_time*yprime[j] # Eq. (8.2)
            
        x += delta_time # Increase x-step
        x_out = np.append(x_out, x) # Saves it in the x_out array
        
        for r in range(len(y)):
            y_out = np.append(y_out, y[r]) # Saves all new y's 
    print(x_out)
    print(y_out)       
    return [x_out, y_out]



def myFunc(x, y):
    '''
    We define our ODEs in this function
    '''
    dy = np.zeros((len(y)))
    dy[0] = 3*(1+x) - y[0]
    return dy

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print ("Delta time expected. You can use python ode.py 1/300")
    #     exit(1)

    delta_time = 0.5#eval(sys.argv[1])
    total_time = 2.

    f_euler_y_0, f_euler_y_1 = forward_euler(f=myFunc, a=None, y0 = np.array([4.0]) , t =np.array([1,1]))
    # b_euler_y = backward_euler(


    print("f_euler_y_0", f_euler_y_0)
    print("f_euler_y_1", f_euler_y_1)  
    plt.plot(f_euler_y_0, f_euler_y_1, "-",label='forward Euler')
    # plt.plot(b_euler_y[:,0], b_euler_y[:,1], "--", label='backward Euler')

    # plt.ylim(min_f[1], max_f[1])
    # plt.xlim(min_f[0], max_f[0])
    # plt.legend()

    # plt.grid(True)
    # plt.show()

    # print ("forward Euler: {}".format(f_euler_y))
    # print ("backward Euler: {}".format(b_euler_y))

    # print(np.linalg.eig(np.array([[0,1],[-500,-501]])))
