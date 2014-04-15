import sys

def EM(L,U,maxIter,threshold):
    #L and U are lists of tuples L being labeled data and U being unlabeled data.
    #maxIter is the maximum number of iterations, -1 means dont stop
    #threshold is the log linear threshold difference at which to stop
    
    ndx = -1
    diff = None
    while ndx < maxIter and threshold > diff:
        if maxIter>0:
            ndx=ndx+1

        update = #some update step

        
        







