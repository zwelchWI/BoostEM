#Because WEKA's python library is crap, we do it the command liney way
import sys
import os
import subprocess
import time
from datetime import datetime
import copy
class Weka:
    def __init__(self,cl,attribs,data,allAttribs,ndx):
    
        self.allAttr = allAttribs
        self.fileName = 'weka/weka.'+str(ndx)+'.arff'
        self.cl = cl
        self.ndx = ndx
        writeArff(self.allAttr,data,self.fileName)
        bldCmd='java -cp /usr/share/java/weka.jar:/usr/share/java/libsvm.jar '
        bldCmd=bldCmd+cl+' -t "'+self.fileName+'" -T "'+self.fileName+'" -d "weka/test'+str(ndx)+'" '#+self.cl
       # print bldCmd
        buildOut = subprocess.Popen(bldCmd,shell=True,stdout=subprocess.PIPE)
        buildOut.communicate()

    def classify(self,d):
        datum = copy.deepcopy(d)
        datum[self.allAttr[-1][0]]=self.allAttr[-1][1][0]
        data = [datum]
        writeArff(self.allAttr,data,'weka/wekaTeSt.arff')
        testCmd= 'java -cp /usr/share/java/weka.jar:/usr/share/java/libsvm.jar '+self.cl+' -p 0'
        testCmd=testCmd+'  -l "weka/test'+str(self.ndx)+'" -T "weka/wekaTeSt.arff"'
        #print testCmd
        testOut = subprocess.Popen(testCmd,shell=True,stdout=subprocess.PIPE)
        out,err=testOut.communicate()
 
        outlines = out.split('\n')
        for line in outlines[5:-2]:
            vals = line.split()
#should only be one
            return vals[2].split(':')[1]
         

def writeArff(attribs,data,filename):
    header = '% 1. Title : '+filename.rstrip('.arff')+'\n'
    header = header + '''%
%
% 2. Sources:
%    (a) Creator : Zach Welch
%    (b) Date 
@RELATION SOMETHING
'''

    for key,val in attribs:
        header = header + "@ATTRIBUTE "+key+" "
        if val[0] == 'numeric':
            header = header + 'numeric\n'
        else:
            header = header+'{'
            for v in val[:-1]:
                header = header+v+','
            header = header + val[-1]+'}\n'
        
    header = header + "@DATA \n"
    
    outfile= open(filename,'w')
    outfile.write(header)
    for datum in data[:]:
        #print datum
        for key,val in attribs[:-1]:
            outfile.write(str(datum[key])+',')
        outfile.write(str(datum[attribs[-1][0]])+'\n')
               
    outfile.close()



