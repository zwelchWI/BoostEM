import sys
import subprocess


seeds = [1,2,3,4,5,6,7,8,9,10]

for seed in seeds:
    boostem='python EM.py --input=../data/data_banknote_authentication.arff --learner=dt --tboost=20 --labelFrac=0.2 --dtM=1000 --seed='+str(seed)
    print boostem
    subprocess.call(boostem, shell=True)
    fullBoost='python EM.py --input=../data/data_banknote_authentication.arff --learner=dt --tboost=20 --labelFrac=1.0 --dtM=1000 --seed='+str(seed)
    print fullBoost
    subprocess.call(fullBoost, shell=True)
    justBoost='python EM.py --input=../data/data_banknote_authentication.arff --learner=dt --tboost=20 --labelFrac=0.2 --dtM=1000 --loseUnlabeled  --seed='+str(seed)
    print justBoost
    subprocess.call(justBoost, shell=True)
    justEM='python EM.py --input=../data/data_banknote_authentication.arff --learner=dt --tboost=1 --labelFrac=0.2 --dtM=1000 --seed='+str(seed)
    print justEM
    subprocess.call(justEM, shell=True)
    FullOne='python EM.py --input=../data/data_banknote_authentication.arff --learner=dt --tboost=1 --labelFrac=1.0 --dtM=1000 --seed='+str(seed)
    print FullOne
    subprocess.call(FullOne, shell=True)
    justOne='python EM.py --input=../data/data_banknote_authentication.arff --learner=dt --tboost=1 --labelFrac=0.2 --dtM=1000 --loseUnlabeled --seed='+str(seed)
    print justOne
    subprocess.call(justOne, shell=True)



