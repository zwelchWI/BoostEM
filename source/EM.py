import sys
import numpy as np
import math
import getopt
from dtLearn import id3
import copy

def Normal(Mu,Sigma,x):
    try:
        val= math.exp(-.5*(x-Mu)*Sigma.I*(x-Mu).T)/math.sqrt((2*math.pi)**len(x)*math.fabs(np.linalg.det(Sigma)))    
    except np.linalg.linalg.LinAlgError:
        print 'Warning, singular matrix'
        val =0.0
    return val

def log2(val):
    if val == 0.0:
        return 0.0
    return math.log(val,2)

def EM(L,U,Wl,Wu,Ys,maxIter,threshold):
    #L and U are lists of tuples L being labeled data and U being unlabeled data.
    #maxIter is the maximum number of iterations, -1 means dont stop
    #threshold is the log linear threshold difference at which to stop
    
    #Init using only labeled data

    if len(U) == 0:
        return []

    Ycounts  = [0.0,0.0]
    Wlcounts = [0.0,0.0]
    for l in L:
        Ycounts[Ys.index(l[1])]= Ycounts[Ys.index(l[1])]+1
        Wlcounts[Ys.index(l[1])]=Wlcounts[Ys.index(l[1])]+Wl[L.index(l)]

    Mus    = []
    Sigmas = []
    for yNdx in range(len(Ys)):
        effW = 0.0
        if Wlcounts[yNdx]:
            effW = Ycounts[yNdx]/Wlcounts[yNdx]
        Mu = np.mat(np.zeros(len(L[0][0].keys())))
        Sigma=np.mat(np.zeros([len(L[0][0].keys()),len(L[0][0].keys())]))
        for lNdx in range(len(L)):
            if Ys[yNdx] == L[lNdx][1]:
                X = np.matrix(L[lNdx][0].values())       
                Mu = Mu + (Wl[lNdx]*effW)*X
        if Ycounts[yNdx]:
            Mu = Mu/Ycounts[yNdx]

        for lNdx in range(len(L)):
            if Ys[yNdx] == L[lNdx][1]:        
                X = np.matrix(L[lNdx][0].values())       
                Sigma = Sigma + (Wl[lNdx]*effW*X-Mu).T*(Wl[lNdx]*effW*X-Mu)
        if Ycounts[yNdx]:
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
                if denom > 0:
                    gammas[uNdx,yNdx] = gammas[uNdx,yNdx]/(denom) 
        #M step
        Ljs = []
        for yNdx in range(len(Ys)):
            effW = 0.0
            if Wlcounts[yNdx]:
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
            if Lj:   
                Mu = Mu/Lj

            Sigma=np.mat(np.zeros([len(L[0][0].keys()),len(L[0][0].keys())]))
            for lNdx in range(len(L)):
                if Ys[yNdx] == L[lNdx][1]:
                    X = np.matrix(L[lNdx][0].values())           
                    Sigma = Sigma + (Wl[lNdx]*effW*X-Mu).T*(Wl[lNdx]*effW*X-Mu)
 
            for uNdx in range(len(U)):
                X = np.matrix(U[uNdx][0].values())       
                Sigma = Sigma + gammas[uNdx,yNdx]*(len(U)*Wu[uNdx]*X-Mu).T*(len(U)*Wu[uNdx]*X-Mu)
            if Lj:
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
    --dtM=		    Decision Tree cutoff
    --seed=                 random number seed
    --testFrac=             Fraction of dataset to be test in Test|Train split
    --labelFrac=            Fraction of trained data to be used as labeled data(rest unlabeled
    --verbose               Print more stuff
    --loseUnlabeled         Lose the unlabeled data

    ------------------------------------------------------------------
    """    

def readInputFile(fileName, labeledSize,testFrac,labelFrac):
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
    TotalData = []
    # now get the dataset
    for line in f:
        dict = {}
        data = line.lower().strip().replace(" ", "").split(",")
        for i in xrange(len(attributes) - 1):
            if len(attributes[i][1]) == 1:
                # real valued- data
                dict[attributes[i][0]] = float(data[i])
            else:
                dict[attributes[i][0]] = data[i]
        TotalData.append((dict, data[len(attributes) - 1]))
    #    if count < labeledSize:
            # add the tuple of feature dictionary, labeled class value
     #       L.append((dict, data[len(attributes) - 1])) 
      #  else:
            # add the tuple of feature dictionary, None
       #     U.append((dict, None))
        count += 1

    Test=[]
    Train=[]
    for datum in TotalData:
        if np.random.random() < testFrac:
            Test.append(datum)
        else:
            Train.append(datum)
    for datum in Train:
        if np.random.random() < labelFrac:
            L.append(datum)
        else:
            U.append((datum[0],None))    

    f.close()
    print "attributes:", len(attributes), "  labeled:", len(L), "  unlabeled:", len(U)
    print "class values:", Ys[0], Ys[1]
    return (attributes, L, U, Ys,Test)

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
            "size=", "thresh=", "maxiter=", "tboost=", "learner=","dtM=","seed=",
            "testFrac=","labelFrac=","verbose","loseUnlabeled"])
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
    testFrac  = .1
    labelFrac = .1
    seed = None
    dtM = 100
    verbose = False
    keepUnlabeled=True
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
        elif o in ("--seed"):
            seed = int(a)
        elif o in ("--testFrac"):
            testFrac = float(a) 
        elif o in ("--labelFrac"):
            labelFrac = float(a)
        elif o in ("--dtM"):
            dtM = int(a)
        elif o in ("--verbose"):
            verbose=True
        elif o in ("--loseUnlabeled"):
            keepUnlabeled=False
        else:
            print "unknown option: " + o
            info()
            sys.exit(2)
    np.random.seed(seed)   

    if not inputArff:
        print "\ninput file needed. (use --input=)"
        sys.exit(2)

    if not learner:
        print "\nlearner needed. (use --learner=)"
        sys.exit(2)

    (attributes, L, U, Ys,Test) = readInputFile(inputArff, labeledSize,testFrac,labelFrac)

    if not keepUnlabeled:
        U = []
    print "we are ready for the boosting, captain"
    # do boost stuff now
    Wl = []
    Wu = []

    Hs = []
    betas = []
    for i in xrange(len(L)):
        Wl.append(1)

    for i in xrange(len(U)):
        Wu.append(1)

    instanceMultiplier = 10
    
    # da boost loop
    for t in xrange(tboost):
        print "boost cycle", t,"/",tboost,
        (Wl, Wu) = normalize(Wl, Wu)
        Pu = EM(L, U, Wl, Wu, Ys, maxIter, thresh)
        if verbose:
            print "EM'd them instances"
        # learner part
        if learner in 'dt':
            # id3 decision tree, may need full instance counts (not weights)
            dataset = []

            for i in xrange(len(L)):
                instance = copy.deepcopy(L[i][0])
                instance[attributes[-1][0]] = L[i][1] # set class attribute to the labeled class

                # append instanceMultiplier to give the instance the same weight as the unlabeled fractions
                for j in xrange(instanceMultiplier):
                    dataset.append(instance)

            for i in xrange(len(U)):
                pos = int(Pu[i][0]*instanceMultiplier) # add frac of instanceMultiplier positive instances
                neg = instanceMultiplier - pos
                posinstance = copy.deepcopy(U[i][0])
                neginstance = copy.deepcopy(U[i][0])
                posinstance[attributes[-1][0]] = attributes[-1][1][0] # add in positve class value
                neginstance[attributes[-1][0]] = attributes[-1][1][1] # add in negative class value
                for j in xrange(pos):
                    dataset.append(posinstance)
                for j in xrange(neg):
                    dataset.append(neginstance)
            if verbose:
                print "dataset length: ", len(dataset), "actual length: ", len(L) + len(U)

            tree = id3(attributes[:len(attributes)-1], dataset, attributes, m=dtM)
            tree.maketree()
            Hs.append(tree)
            if verbose:
                print "\nGenerated ID3 Tree: " + tree.display()

        elif learner in 'bayes':
            # naive bayes
            print 'add in bayes code'
            # add model to Hs list

        # compute error value
        eps = 0.0

        for i in xrange(len(L)):
            if learner in 'dt':
                actual = L[i][1]
                predicted = Hs[-1].classify(L[i][0])
                #print actual, predicted

                if actual != predicted:
                    eps += Wl[i]

            elif learner in 'bayes':
                print 'todo'

        for i in xrange(len(U)):
            if learner in 'dt':
                # TODO ::: how to decide error for our unlabeled instances?
                eps += Wu[i]*(1.0 - Pu[i][Ys.index(Hs[-1].classify(U[i][0]))])
            elif learner in 'bayes':
                print 'todo'
        eps /= 2.0 #WE THINK THIS SHOULD HAPPEN BECAUSE THERE ARE 2 EPSILONS
        beta = eps / (1.0 - eps)
        print "| epsilon:", eps, "|  beta:", beta,

        numCorrect=0.0
        for datum in Test:
            actual = datum[1]
            predicted = Hs[-1].classify(datum[0])
                #print actual, predicted

            if actual == predicted:
                numCorrect+=1

        print "| H accuracy : "+str(numCorrect/len(Test)),
            
        if verbose:
            print ""
        else:
            print "      \t\t\r",
        sys.stdout.flush()

        if eps >=0.5:
            print "\nSTOPING BECAUSE EPS BAD"
            Hs = Hs[:-1]
            break
        betas.append(beta)


        # compute new weights
        # downweight correct examples
        for i in xrange(len(L)):
            if learner in 'dt':
                actual = L[i][1]
                predicted = Hs[-1].classify(L[i][0])

                if actual == predicted:
                    Wl[i] *= beta

            elif learner in 'bayes':
                print 'todo'

        for i in xrange(len(U)):
            if learner in 'dt':
                # TODO ::: how to reweight our unlabeled instances?
                Wu[i] *= beta*(1.0 - Pu[i][Ys.index(Hs[-1].classify(U[i][0]))])
            elif learner in 'bayes':
                print 'todo'


    

    numCorrect=0.0
    for datum in Test:
        yVals = []
        for yNdx in range(len(Ys)):
            yVal = 0.0
            for hNdx in range(len(Hs)):
                if Ys[yNdx] == Hs[hNdx].classify(datum[0]):
                    yVal += math.log(1/betas[hNdx])                 
            yVals.append(yVal)
        actual = datum[1]
        predicted = Ys[yVals.index(max(yVals))]
                #print actual, predicted
        if actual == predicted:
            numCorrect+=1

    print "\nTotal accuracy : "+str(numCorrect/len(Test))


#L=[[{1:1},-1.0],[{1:0},-1.0],[{1:2},1.0],[{1:3},1.0]]
#U=[[{1:1.5},None]]
#Wl=[.6,.2,.1,.1]
#Wu=[1]
#Ys=[-1.0,1.0]


#x=EM(L,U,Wl,Wu,Ys,-1,.0001)
#print x,"!"

if __name__ == "__main__":
    main()
