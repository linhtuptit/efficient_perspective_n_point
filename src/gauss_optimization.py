import numpy as np 
import linear_combination as lc

def gauss_newton_process(rho, L6x10, initBeta, iteration=10):
    '''
    Process Gauss-Newton optimization to find the efficient vector beta
    using Least Square method
    '''
    currBeta = initBeta

    for iter in range(iteration):
        J, r = compute_J_and_r_Gauss_Newton(currBeta, rho, L6x10)
        deltaBeta = np.matmul(np.linalg.inv(np.matmul(J.T, J)), np.matmul(J.T, r))
        currBeta += deltaBeta
        error = np.matmul(r.T, r)
    optBeta = currBeta

    return optBeta

def compute_J_and_r_Gauss_Newton(currBeta, rho, L):
    '''
    Compute Jacobian matrix J and error vector r
    '''
    J = np.zeros((6, 4))
    r = np.zeros((6, 1))

    # Define matrix Beta with 10 elements listed by: beta_00, beta_01
    # beta_11, beta_02, beta_12, beta_22, beta_03, beta_13, beta_23, beta_33
    # with relation: beta_ab = beta_a * beta_b
    Beta = [currBeta[0] * currBeta[0],
            currBeta[0] * currBeta[1],
            currBeta[1] * currBeta[1],
            currBeta[0] * currBeta[2],
            currBeta[1] * currBeta[2],
            currBeta[2] * currBeta[2],
            currBeta[0] * currBeta[3],
            currBeta[1] * currBeta[3],
            currBeta[2] * currBeta[3],
            currBeta[3] * currBeta[3]]
    
    for idx in range(6):
        J[idx, 0] = 2*currBeta[0]*L[idx, 0] + currBeta[1]*L[idx, 1] + currBeta[2]*L[idx, 3] + currBeta[3]*L[idx, 6]
        J[idx, 1] = currBeta[0]*L[idx, 1] + 2*currBeta[1]*L[idx, 2] + currBeta[2]*L[idx, 4] + currBeta[3]*L[idx, 7]
        J[idx, 2] = currBeta[0]*L[idx, 2] + currBeta[1]*L[idx, 4] + 2*currBeta[2]*L[idx, 5] + currBeta[3]*L[idx, 8]
        J[idx, 3] = currBeta[0]*L[idx, 3] + currBeta[1]*L[idx, 7] + currBeta[2]*L[idx, 8] + 2*currBeta[3]*L[idx, 9]
        r[idx] = rho[idx] - np.matmul(L[idx, :], Beta)
    
    return J, r