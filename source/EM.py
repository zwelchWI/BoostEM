import sys
import numpy as np
import math

# test

def Normal(Mu,Sigma,x):
    return math.exp(-.5*(x-Mu)*Sigma.I*(x-Mu).T)/math.sqrt((2*math.pi)**len(x)*np.linalg.det(Sigma))    

def log2(val):
    if val == 0.0:
        return 0.0
    return math.log(val,2)

def EM(L,U,Wl,Wu,Ys,maxIter,threshold):
    #L and U are lists of tuples L being labeled data and U being unlabeled data.
    #maxIter is the maximum number of iterations, -1 means dont stop
    #threshold is the log linear threshold difference at which to stop
    
    
    #Init using only labeled data
    Ycounts  = [0.0,0.0]
    Wlcounts = [0.0,0.0]
    for l in L:
        Ycounts[Ys.index(l[1])]= Ycounts[Ys.index(l[1])]+1
        Wlcounts[Ys.index(l[1])]=Wlcounts[Ys.index(l[1])]+Wl[L.index(l)]

    Mus    = []
    Sigmas = []
    for yNdx in range(len(Ys)):
        effW = Ycounts[yNdx]/Wlcounts[yNdx]
        Mu = np.mat(np.zeros(len(L[0][0].keys())))
        Sigma=np.mat(np.zeros([len(L[0][0].keys()),len(L[0][0].keys())]))
        for lNdx in range(len(L)):
            if Ys[yNdx] == L[lNdx][1]:
                X = np.matrix(L[lNdx][0].values())       
                Mu = Mu + Wl[lNdx]*effW*X
        Mu = Mu/Ycounts[yNdx]

        for lNdx in range(len(L)):
            if Ys[yNdx] == L[lNdx][1]:        
                X = np.matrix(L[lNdx][0].values())       
                Sigma = Sigma + (Wl[lNdx]*effW*X-Mu).T*(Wl[lNdx]*effW*X-Mu)
        Sigma = Sigma/Ycounts[yNdx]

        Mus.append(Mu)
        Sigmas.append(Sigma)


    l = Ycounts[0]+Ycounts[1]
    Pi = [Ycounts[0]/l,Ycounts[1]/l]


    ndx = -1
    diff = None
    condition = True
    while condition:
                
        #E step
        gammas = np.matrix(np.zeros([len(U),len(Ys)]))

        for uNdx in range(len(U)):
            for yNdx in range(len(Ys)):
                gammas[uNdx,yNdx] = Pi[yNdx]*Normal(Mus[yNdx],Sigmas[yNdx],U[uNdx][0].values())
            denom = gammas[uNdx,0]+ gammas[uNdx,1]
            for yNdx in range(len(Ys)):
                gammas[uNdx,yNdx] = gammas[uNdx,yNdx]/(denom) 
        #M step
        Ljs = []
        for yNdx in range(len(Ys)):
            effW = Ycounts[yNdx]/Wlcounts[yNdx]
            Lj = Ycounts[yNdx]
            for uNdx in range(len(U)):
                Lj = Lj + gammas[uNdx,yNdx]
            Mu = np.mat(np.zeros(len(L[0][0].keys())))
            for lNdx in range(len(L)):
                if Ys[yNdx] == L[lNdx][1]:        
                    X = np.matrix(L[lNdx][0].values())       
                    Mu = Mu + Wl[lNdx]*effW*X
 
            for uNdx in range(len(U)):
                X = np.matrix(U[uNdx][0].values())       
                Mu = Mu + gammas[uNdx,yNdx]*(Wu[uNdx]*len(U))*X

            Mu = Mu/Lj

            Sigma=np.mat(np.zeros([len(L[0][0].keys()),len(L[0][0].keys())]))
            for lNdx in range(len(L)):
                if Ys[yNdx] == L[lNdx][1]:
                    X = np.matrix(L[lNdx][0].values())           
                    Sigma = Sigma + (Wl[lNdx]*effW*X-Mu).T*(Wl[lNdx]*effW*X-Mu)
 
            for uNdx in range(len(U)):
                X = np.matrix(U[uNdx][0].values())       
                Sigma = Sigma + gammas[uNdx,yNdx]*(len(U)*Wu[uNdx]*X-Mu).T*(len(U)*Wu[uNdx]*X-Mu)
            Sigma = Sigma/Lj

            Pi[yNdx] = Lj/float(len(L) + len(U))
            Sigmas[yNdx] = Sigma
            Mus[yNdx] = Mu

        #need to calculate difference in log likelihood
        logLike = 0.0

        for l in L:
            yNdx = Ys.index(l[1])
            logLike = logLike + log2(Pi[yNdx]*Normal(Mus[yNdx],Sigmas[yNdx],l[0].values()))
        for u in U:
            hold = 0.0
            for yNdx in range(len(Ys)):
                hold = hold + Pi[yNdx]*Normal(Mus[yNdx],Sigmas[yNdx],u[0].values()) 
            logLike = logLike + log2(hold)
        if diff is None:
            #first iter
            diff = threshold
            newVal = logLike
        else:
            oldVal = newVal
            newVal= logLike
            diff = newVal-oldVal
            print diff
        
        if maxIter>0:
            ndx=ndx+1
            condition = ndx < maxIter and threshold <= diff
        else:
            condition = threshold <= diff
    return gammas.tolist()
           
        





L=[[{1:1},-1.0],[{1:0},-1.0],[{1:2},1.0],[{1:3},1.0]]
U=[[{1:1.5},None]]
Wl=[.6,.2,.1,.1]
Wu=[1]
Ys=[-1.0,1.0]


x=EM(L,U,Wl,Wu,Ys,-1,.0001)
print x









