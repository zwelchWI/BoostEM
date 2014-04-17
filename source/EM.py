import sys
import numpy as np
import math


def Normal(Mu,Sigma,X):
    math.exp(-.5*(x-Mu)*Sigma.I*(x-Mu).T)/sqrt((2*math.pi)**len(X)*np.linalg.det(Sigma))    

def EM(L,U,Wl,Wu,Ys,maxIter,threshold):
    #L and U are lists of tuples L being labeled data and U being unlabeled data.
    #maxIter is the maximum number of iterations, -1 means dont stop
    #threshold is the log linear threshold difference at which to stop
    
    
    #Init using only labeled data
    Ycounts  = [0.0,0.0]

    for l in L:
        Ycounts[Ys.find(l[1])]= Ycounts[Ys.find(l[1])]+1


    Mus    = []
    Sigmas = []
    for yNdx in range(len(Ys)):
        Mu = mat(zeros(len(L[0][0].keys())))
        Sigma=mat(zeros([len(L[0][0].keys()),len(L[0][0].keys())]))
        for lNdx in range(len(L)):
            if Ys[yNdx] == L[lNdx][1]:        
                Mu = Mu + Wl[lNdx]*l[lNdx][0].values()
        Mu = Mu/Ycounts[yNdx]

        for lNdx in range(len(L)):
            if Ys[yNdx] == L[lNdx][1]:        
                Sigma = Sigma + (Wl[lNdx]*L[lNdx][0].values()-Mu).T*(Wl[lNdx]*L[lNdx][0].values()-Mu)
        Sigma = Sigma/Ycounts[yNdx]

        Mus.append(Mu)
        Sigmas.append(Sigma)

    l = Ycounts[0]+Ycounts[1]
    Pi = [Ycounts[0]/l,Ycounts[1]/l]

    ndx = -1
    diff = None
    while ndx < maxIter and threshold > diff:
        if maxIter>0:
            ndx=ndx+1

        #E step
        gammas = matrix(zeros(len(U),len(Ys)))
        for uNdx in range(len(U)):
            for yNdx in len(Range(Ys)):
                gammas[uNdx][yNdx] = Pi[yNdx]*Normal(Mus[yNdx],Sigmas[yNdx],U[uNdx][0].values())
            for yNdx in len(Range(Ys)):
                gammas[uNdx][yNdx] = gammas[uNdx][yNdx]/( gammas[uNdx][1]+ gammas[uNdx][2]) 
             
        #M step
        Ljs = []
        for yNdx in range(Ys):
            Lj = Ycount[yNdx]
            for uNdx in range(len(U)):
                Lj = Lj + gammas[uNdx][yNdx]
            Mu = mat(zeros(len(L[0][0].keys())))
            for lNdx in range(len(L)):
                if Ys[yNdx] == L[lNdx][1]:        
                    Mu = Mu + Wl[lNdx]*l[lNdx][0].values()
 
            for uNdx in range(len(U)):
                Mu = Mu + gammas[uNdx][yNdx]*Wu[uNdx]*U[uNdx][0].values()

            Mu = Mu/Lj

            Sigma=mat(zeros([len(L[0][0].keys()),len(L[0][0].keys())]))
            for lNdx in range(len(L)):
                if Ys[yNdx] == L[lNdx][1]:        
                    Sigma = Sigma + (Wl[lNdx]*L[lNdx][0].values()-Mu).T*(Wl[lNdx]*L[lNdx][0].values()-Mu)
 
            for uNdx in range(len(U)):
                Sigma = Sigma + gammas[uNdx][yNdx]*(Wu[uNdx]*U[uNdx][0].values()-Mu).T*(Wu[uNdx]*U[uNdx][0].values()-Mu)
            Sigma = Sigma/Lj

            Pi[yNdx] = Lj/float(len(L) + len(U))
            Sigmas[yNdx] = Sigma
            Mus[yNdx] = Mu
    return gammas.tolist()
           
        







