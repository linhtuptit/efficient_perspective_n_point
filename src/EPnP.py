import math
import cv2 as cv
import numpy as np 
import gauss_optimization as go 
import linear_combination as lc

class EPnP:

    def __init__(self, uc, vc, fu, fv,
                 pws, uv,
                 number_of_correspondences):
        '''
        Set internal parameters
        '''
        self.uc = uc
        self.vc = vc
        self.fu = fu
        self.fv = fv
        self.pws = pws
        self.uv = uv
        self.number_of_correspondences = number_of_correspondences
        
    def choose_control_point(self):
        '''
        Chosse 4 control points for EPnP algorithm
        '''
        # Take control point C0 as a reference point
        cws = np.zeros((4, 3))
        cws[0, 0] = cws[0, 1] = cws[0, 2] = 0
        for idx in range(self.number_of_correspondences):
            for jdx in range(3):
                cws[0, jdx] += self.pws[idx, jdx]

        for jdx in range(3):
            cws[0, jdx] /= self.number_of_correspondences

        # Take C1, C2, C3 from PCA on the reference point
        # Compute correlation matrix between world points with reference points
        # Then, perform SVD to find out the direction of set of reference points
        PW0 = np.zeros((self.number_of_correspondences, 3))
        for idx in range(self.number_of_correspondences):
            for jdx in range(3):
                PW0[idx, jdx] = self.pws[idx, jdx] - cws[0, jdx]
        PW0tPW0 = np.matmul(PW0.T, PW0)
        w, U, Vt = cv.SVDecomp(PW0tPW0)
        for idx in range(1,4):
            scale = math.sqrt(w[idx-1] / self.number_of_correspondences)
            for jdx in range(3):
                cws[idx, jdx] = cws[0, jdx] + scale * Vt[idx-1, jdx]
        
        return cws

    def compute_barycentric_coordinates(self, cws):
        '''
        Compute barycentric coordinates based on control points
        '''
        cc = np.zeros((3, 3))
        for idx in range(3):
            for jdx in range(1, 4):
                cc[idx, jdx-1] = cws[jdx, idx] - cws[0, idx]
        
        ccInv = np.linalg.inv(cc)
        alphas = np.zeros((self.number_of_correspondences, 4))
        for idx in range(self.number_of_correspondences):
            for jdx in range(3):
                alphas[idx, jdx+1] = ccInv[jdx, 0] * (self.pws[idx, 0] - cws[0, 0]) \
                                   + ccInv[jdx, 1] * (self.pws[idx, 1] - cws[0, 1]) \
                                   + ccInv[jdx, 2] * (self.pws[idx, 2] - cws[0, 2])
            alphas[idx, 0] = 1.0 - alphas[idx, 1] - alphas[idx, 2] - alphas[idx, 3]

        return alphas

    def fill_M(self, alphas):
        '''
        Define matrix M representing for linear equations
        '''
        # Define parameters
        nP = self.number_of_correspondences

        M = np.zeros((2*nP, 12))
        for idx in range(nP):
            for jdx in range(4):
                M[idx, jdx] = alphas[idx, jdx] * self.fu
                M[idx, jdx+4] = 0.0
                M[idx, jdx+8] = alphas[idx, jdx] * (self.uc - self.uv[idx, 0])
                M[nP+idx, jdx] = 0.0
                M[nP+idx, jdx+4] = alphas[idx, jdx] * self.fv
                M[nP+idx, jdx+8] = alphas[idx, jdx] * (self.vc - self.uv[idx, 1])

        return M

    def compute_ccs(self, beta, Vt):
        '''
        Compute control point coordinates in camera system
        '''
        ccs = np.zeros((4, 3))
        for idx in range(4):
            ccs[idx, 0] = ccs[idx, 1] = ccs[idx, 2] = 0.0

        for idx in range(4):
            vt = Vt[:, 11-idx]  # Extract column vector 
            for jdx in range(4):
                for kdx in range(3):
                    ccs[jdx, kdx] += beta[idx] * vt[3*jdx+kdx]

        return ccs

    def compute_pcs(self, ccs, alphas):
        '''
        Computer reference point coordinates in camera system
        '''
        pcs = np.zeros((self.number_of_correspondences, 3))
        for idx in range(self.number_of_correspondences):
            for jdx in range(3):
                pcs[idx, jdx] = alphas[idx, 0] * ccs[0, jdx] \
                              + alphas[idx, 1] * ccs[1, jdx] \
                              + alphas[idx, 2] * ccs[2, jdx] \
                              + alphas[idx, 3] * ccs[3, jdx]

        return pcs

    def compute_R_and_t(self, optBeta, Vt, alphas):
        '''
        Compute rotation matrix and translation vector of camera
        and reprojection error as well
        '''
        # Compute ccs and pcs
        ccs = self.compute_ccs(optBeta, Vt)
        pcs = self.compute_pcs(ccs, alphas)

        pc0 = np.zeros((3))
        pw0 = np.zeros((3))

        for idx in range(self.number_of_correspondences):
            for jdx in range(3):
                pc0[jdx] += pcs[idx, jdx]
                pw0[jdx] += self.pws[idx, jdx]
        
        for jdx in range(3):
            pc0[jdx] = pc0[jdx] / self.number_of_correspondences
            pw0[jdx] = pw0[jdx] / self.number_of_correspondences

        H = np.zeros((3, 3))
        for idx in range(self.number_of_correspondences):
            for jdx in range(3):
                for kdx in range(3):
                    H[jdx, kdx] += (self.pws[idx, jdx] - pw0[jdx]) * (pcs[idx, kdx] - pc0[kdx])

        HD, HU, HVt = cv.SVDecomp(H)
        R = np.matmul(HU, HVt)
        if(np.linalg.det(R) < 0):
            R[2, 0] = -R[2, 0]
            R[2, 1] = -R[2, 1]
            R[2, 2] = -R[2, 2]
        t = pc0.T - np.matmul(R, pw0.T)

        # Compute reprojection error
        sumError = 0
        for idx in range(self.number_of_correspondences):
            Xc = np.matmul(R[0,:], self.pws[idx, :]) + t[0]
            Yc = np.matmul(R[1,:], self.pws[idx, :]) + t[1]
            ZcInv = 1.0 / (np.matmul(R[2,:], self.pws[idx, :]) + t[2])
            ue = self.uc + self.fu * Xc * ZcInv
            ve = self.vc + self.fv * Yc * ZcInv
            u = self.uv[idx, 0]
            v = self.uv[idx, 1]
            sumError += math.sqrt((u - ue)**2 + (v - ve)**2)
        sumError = sumError / self.number_of_correspondences

        return R, t, sumError

    def compute_pose(self):
        '''
        Estimate pose: compute rotation matrix R and translation vector t
        '''
        cws = self.choose_control_point()
        alphas = self.compute_barycentric_coordinates(cws)

        M = self.fill_M(alphas)
        MtM = np.matmul(M.T, M)
        D, U, Vt = cv.SVDecomp(MtM)

        L6x10 = lc.compute_L6x10(Vt)
        rho = lc.compute_rho(cws)

        # Initiate  output
        R = np.zeros((3, 3, 3))
        t = np.zeros((3, 3))
        error = np.zeros((3))

        initBeta1 = lc.find_betas_approx_1(L6x10, rho)
        optBeta1 = go.gauss_newton_process(rho, L6x10, initBeta1, iteration=40)
        R[0], t[0], error[0] = self.compute_R_and_t(optBeta1, Vt, alphas)

        initBeta2 = lc.find_betas_approx_2(L6x10, rho)
        optBeta2 = go.gauss_newton_process(rho, L6x10, initBeta2, iteration=40)
        R[1], t[1], error[1] = self.compute_R_and_t(optBeta2, Vt, alphas)

        initBeta3 = lc.find_betas_approx_3(L6x10, rho)
        optBeta3 = go.gauss_newton_process(rho, L6x10, initBeta3, iteration=40)
        R[2], t[2], error[2] = self.compute_R_and_t(optBeta3, Vt, alphas)

        # Choose and extract output
        N = 0
        if(error[1] < error[0]):    N = 1
        if(error[2] < error[N]):    N = 2
    
        return R[N], t[N], error[N]
