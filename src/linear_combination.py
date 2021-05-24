import numpy as np
import math

def squared_dist(vect1, vect2):
    '''
    Compute squared distance between two vector
    Return scale
    '''
    return np.sum((vect1 - vect2)**2, axis=0)

def compute_rho(cws):
    '''
    Compute rho_vector constructed by world coordinates cw
    Return vector with 6 elements
    '''
    rho = []
    for idx in range(3):
        for jdx in range(idx+1, 4):
            rho.append(squared_dist(cws[idx,:], cws[jdx,:]))
    del idx, jdx

    return rho

def compute_L6x10(K):
    '''
    Compute the 6x10-matrix L constructed by null-space K of matrix M
    Return the matrix with 6 rows and 10 columns
    '''
    L6x10 = np.zeros((6, 10))

    # Define vector v from null-space K of M
    v = []
    for idx in range(4):
        v.append(K[:, 11-idx])
    del idx

    # Define the difference vector of sub-vector in v
    dv = []
    for rdx in range(4):
        dv.append([])
        for idx in range(3):
            for jdx in range(idx+1, 4):
                diff = v[rdx][3*idx:3*(idx+1)]-v[rdx][3*jdx:3*(jdx+1)]
                dv[rdx].append(diff)
    del idx, jdx, rdx

    # Define list index representing index of 10-vector beta
    indexList = [
        (0, 0),     # beta_00
        (0, 1),     # beta_01
        (1, 1),     # beta_11
        (0, 2),     # beta_02
        (1, 2),     # beta_12
        (2, 2),     # beta_22
        (0, 3),     # beta_03
        (1, 3),     # beta_13
        (2, 3),     # beta_23
        (3, 3)]     # beta_33

    # Define matrix L
    for idx in range(6):
        jdx = 0
        for a, b in indexList:
            L6x10[idx, jdx] = np.matmul(dv[a][idx], dv[b][idx].T)
            if a != b:
                L6x10[idx, jdx] *= 2
            jdx += 1
    
    return L6x10

def compute_L6x3(L6x10):
    '''
    Choose a test linear combination to reduce computation time
    Set beta_2 = beta_3 = 0.0 and compute beta_0, beta_1 based relation:
    beta_00, beta_01, beta_11. Return a matrix L with 6 rows and 3 columns
    '''
    return L6x10[:, (0,1,2)]

def compute_L6x4(L6x10):
    '''
    Choose a test linear combination to reduce computation time
    Compute beta_0, beta_1, beta_2 and beta_3  based relation: 
    beta_00, beta_01, beta_02 and beta_03. Return a matrix L with 6 rows and 4 columns
    '''
    return L6x10[:, (0,1,3,6)]

def compute_L6x5(L6x10):
    '''
    Choose a test linear combination to reduce computation time
    Set beta_3 = 0 and compute beta_0, beta_1, beta_2 based relation: 
    beta_00, beta_01, beta_11, beta_02 and beta_12.
    Return a matrix L with 6 rows and 5 columns
    '''
    return L6x10[:, (0,1,2,3,4)]

def find_betas_approx_1(L6x10, rho):
    '''
    Solve linear equation: L*betas = rho with L is a 6x3-matrix
    rho is a 6x1-vector
    '''
    betas = np.zeros((4,1)).astype(float)

    L6x3 = compute_L6x3(L6x10)
    B = np.linalg.lstsq(L6x3, rho, rcond=None)[0]
    if(B[0] < 0):
        betas[0] = math.sqrt(-B[0])
        if(B[2] < 0):
            betas[1] = math.sqrt(-B[2])
    else:
        betas[0] = math.sqrt(B[0])
        if(B[2] > 0):
            betas[1] = math.sqrt(B[2])

    return betas

def find_betas_approx_2(L6x10, rho):
    '''
    Solve linear equation: L*betas = rho with L is a 6x4-matrix
    rho is a 6x1-vector
    '''
    betas = np.zeros((4,1)).astype(float)

    L6x4 = compute_L6x4(L6x10)
    B = np.linalg.lstsq(L6x4, rho, rcond=None)[0]
    if(B[0] < 0):
        betas[0] = math.sqrt(-B[0])
        betas[1] = -B[1] / betas[0]
        betas[2] = -B[2] / betas[0]
        betas[3] = -B[3] / betas[0]
    else:
        betas[0] = math.sqrt(B[0])
        betas[1] = B[1] / betas[0]
        betas[2] = B[2] / betas[0]
        betas[3] = B[3] / betas[0]

    return betas

def find_betas_approx_3(L6x10, rho):
    '''
    Solve linear equation: L*betas = rho with L is a 6x5-matrix
    rho is a 6x1-vector
    '''
    betas = np.zeros((4,1)).astype(float)

    L6x5 = compute_L6x5(L6x10)
    B = np.linalg.lstsq(L6x5, rho, rcond=None)[0]
    if(B[0] < 0):
        betas[0] = math.sqrt(-B[0])
        if(B[2] < 0):
            betas[1] = math.sqrt(-B[2])
    else:
        betas[0] = math.sqrt(B[0])
        if(B[2] > 0):
            betas[1] = math.sqrt(B[2])
    betas[2] = B[3] / betas[0]

    return betas
