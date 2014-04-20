import sys
import numpy as np
import math
import getopt
from dtLearn import id3


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
           
def info():
    """ prints how to use this tool """
    print """
    ------------------------------------------------------------------
    here let me help you with that:
    -h or --help            prints this list of commands
    -input = file           arff input file
    -size = n               first n instances in input will be labeled
    -thresh =               log linear threshold difference for EM
    -maxiter =              max iterations for EM
    -tboost =               number of iterations for boost
    -learner                classifier to use
    ------------------------------------------------------------------
    """    

def readInputFile(fileName, labeledSize):
    f = open(fileName)

    # clear the relation line
    while True:
        l = f.readline().lower()
        if "@relation" in l:
            break

    # datasets
    L = []
    U = []
    attributes = []

    # read in the attributes, we only care about
    # the labels of the last one (the class attribute)
    while True:
        line = f.readline().lower()
        if "@data" in line:
            break
        line = line.strip()
        if len(line) == 0:
            continue # handles blank lines
        line = line.split('@attribute ')[1]
        line = line.replace(" ", "")
        line = line.replace("{", "")
        line = line.replace("}", "")
        line = line.split("'")

        attr = line[1]
        vals = line[2]
        vals = vals.split(",")
        attributes.append((attr, vals))
    
    Ys = vals

    count = 0

    # now get the dataset
    for line in f:
        dict = {}
        data = line.strip().replace(" ", "").split(",")
        for i in xrange(len(attributes) - 1):
            if len(attributes[i][1]) == 1:
                # real valued- data
                dict[attributes[i][0]] = float(data[i])
            else:
                dict[attributes[i][0]] = data[i]
        if count < labeledSize:
            # add the tuple of feature dictionary, labeled class value
            L.append((dict, data[len(attributes) - 1])) 
        else:
            # add the tuple of feature dictionary, None
            U.append((dict, None))
        count += 1

    f.close()
    print "attributes:", len(attributes), "  labeled:", len(L), "  unlabeled:", len(U)
    print "class values:", Ys[0], Ys[1]
    return (attributes, L, U, Ys)

def normalize(Wl, Wu):
    lSum = 0
    uSum = 0

    for weight in Wl:
        lSum += weight
    for weight in Wu:
        uSum += weight

    nWl = []
    nWu = []

    for weight in Wl:
        nWl.append(float(weight)/lSum)
    for weight in Wu:
        nWu.append(float(weight)/uSum)

    return (nWl, nWu)


def main():
    try:
        options, remainder = getopt.getopt(sys.argv[1:], 'h', ["help", "input=", 
            "size=", "thresh=", "maxiter=", "tboost=", "learner="])
    except getopt.GetoptError, e:
        print str(e)
        info()
        sys.exit(2)

    inputArff = None
    labeledSize = 0
    thresh = .0001
    maxIter = -1
    tboost = 200
    classifiers = ['bayes']
    learner = ""

    for o, a in options:
        if o in ("-h", "--help"):
            info()
            sys.exit(2);
        elif o in ("--input"):
            inputArff = a
        elif o in ("--size"):
            labeledSize = int(a)
        elif o in ("--thresh"):
            thresh = float(a)
        elif o in ("--maxiter"):
            maxIter = int(a)
        elif o in ("--tboost"):
            tboost = int(a)
        elif o in ("--learner"):
            learner = a
        else:
            print "unknown option: " + o
            info()
            sys.exit(2)


    if not inputArff:
        print "\ninput file needed. (use --input=)"
        sys.exit(2)

    if labeledSize < 1:
        print "\nsize argument needed. (use --size)"
        sys.exit(2)
    if not learner:
        print "\nlearner needed. (use --learner=)"
        sys.exit(2)

    (attributes, L, U, Ys) = readInputFile(inputArff, labeledSize)

    print "we are ready for the boosting, captain"
    # do boost stuff now
    Wl = []
    Wu = []

    Hs = []

    for i in xrange(len(L)):
        Wl.append(1)

    for i in xrange(len(U)):
        Wu.append(1)

    # da boost loop
    for t in xrange(tboost):
        print "boost cycle", t
        (Wl, Wu) = normalize(Wl, Wu)
        Pu = EM(L, U, Wl, Wu, Ys, maxIter, thresh)

        # learner part
        if(learner in 'dt'):
            # id3 decision tree, may need full instance counts (not weights)
            dataset = []

            for i in xrange(len(L)):
                pos = int(Pu[i][0]*100) # add frac of 100 positive instances
                neg = 100 - pos
                posinstance = L[i]
                neginstance = L[i]
                posinstance[attributes[-1][0]] = attributes[-1][1][0] # add in positve class value
                neginstance[attributes[-1][0]] = attributes[-1][1][1] # add in negative class value
                for i in xrange(pos):
                    dataset.append(posinstance)
                for i in xrange(neg):
                    dataset.append(neginstance)

            for i in xrange(len(U)):
                pos = int(Pu[i + len(L)][0]*100) # add frac of 100 positive instances
                neg = 100 - pos
                posinstance = U[i]
                neginstance = U[i]
                posinstance[attributes[-1][0]] = attributes[-1][1][0] # add in positve class value
                neginstance[attributes[-1][0]] = attributes[-1][1][1] # add in negative class value
                for i in xrange(pos):
                    dataset.append(posinstance)
                for i in xrange(neg):
                    dataset.append(neginstance)

            print "dataset length: ", len(dataset), "actual length: ", len(L) + len(U)

            tree = id3(attributes[:len(attributes)-1], dataset)
            tree.maketree()
            Hs.append(tree)
            print "\nGenerated ID3 Tree: " + tree.display()

            # add model to Hs list
        elif(learner in 'bayes'):
            # naive bayes
            print 'add in bayes code'
            # add model to Hs list

        # compute new weights
        Wl = []
        Wu = []

        for l in L:
            predY = predict(l[0], learner)
            






#L=[[{1:1},-1.0],[{1:0},-1.0],[{1:2},1.0],[{1:3},1.0]]
#U=[[{1:1.5},None]]
#Wl=[.6,.2,.1,.1]
#Wu=[1]
#Ys=[-1.0,1.0]


#x=EM(L,U,Wl,Wu,Ys,-1,.0001)
#print x,"!"

if __name__ == "__main__":
    main()