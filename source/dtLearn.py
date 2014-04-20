import sys
import math
import random

class id3:
	""" id3 decision tree node """

	def __init__(self,attributes,dataset,allattributes,m=2):
		self.attributes = attributes
		self.dataset = dataset
		self.children = []
		self.attribute = None # {"what this node splits on": ["and a list", "of the values"]}
		self.splitval = -9000 # value of split for numeric attributes
		self.splitindex = -1 # index in sorted dataset to divide data by the splitval
		self.classification = None # this will have a value from the attribute "class" if this is a leaf node
		self.allattributes = allattributes
		self.m = m

	def display(self,level=0):
		s = ""
		# display my info
		if self.attribute == None:
			s += ": " + self.classification
		else:
			if len(self.attribute[1]) == 1:
				# we are a real-valued attribute
				s += "\n" + ("|\t" * level) + self.attribute[0] + " <= " + str(self.splitval) 
				s +=  " [" + self.instancesofreal(self.attribute[0], self.splitval, True) + "]"
				s += self.children[0].display(level+1)

				s += "\n" + ("|\t" * level) + self.attribute[0] + "  > " + str(self.splitval) 
				s +=  " [" + self.instancesofreal(self.attribute[0], self.splitval, False) + "]"
				s += self.children[1].display(level+1)
			else:
				# we are nominal
				counter = 0
				for child in self.children:
					s += "\n" + ("|\t" * level) + self.attribute[0] + " = " + self.attribute[1][counter]
					s +=  " [" + self.instancesof(self.attribute[0], self.attribute[1][counter]) + "]"
					counter += 1
					s += child.display(level+1)
		return s

	def instancesof(self,attributename,attributevalue):
		positives = 0
		negatives = 0
		for datapoint in self.dataset:
			if datapoint[attributename] == attributevalue:
				if datapoint["class"] == self.allattributes[-1][1][0]:
					negatives += 1
				else:
					positives += 1
		return str(negatives) + " " + str(positives)

	def instancesofreal(self,attributename,attributesplitpoint,lower=True):
		positives = 0
		negatives = 0
		if lower:
			for datapoint in self.dataset:
				if datapoint[attributename] <= attributesplitpoint:
					if datapoint["class"] == self.allattributes[-1][1][0]:
						negatives += 1
					else:
						positives += 1
		else:
			for datapoint in self.dataset:
				if datapoint[attributename] > attributesplitpoint:
					if datapoint["class"] == self.allattributes[-1][1][0]:
						negatives += 1
					else:
						positives += 1
		return str(negatives) + " " + str(positives)		


	def classify(self, datapoint):
		""" only call me on the root node of the tree externally! """
		if self.attribute == None:
			return self.classification

		splitattributename = self.attribute[0]
		datavalue = datapoint[splitattributename]

		# nominal data
		if len(self.attribute[1]) != 1:
			kidindex = self.attribute[1].index(datavalue)
		# real-valued data
		else:
			if datavalue <= self.splitval:
				kidindex = 0
			else:
				kidindex = 1

		return self.children[kidindex].classify(datapoint)


	def maxinfogain(self):
		""" given this node's data and list of attributes, return
			the attribute with the greatest info gain """
		maxattr = self.allattributes[0]
		maxgain = -9000
		splitval = -9000
		splitindex = -1

		myentropy = getentropy(self.dataset, self.allattributes)

		for attr in self.allattributes:
			thisgain = -9000
			# if this attribute's name is in this node's attributes (aka hasn't been split on before)
			if attr in self.attributes:
				# nominal data
				if len(attr[1]) != 1:
					# calculate info gain
					thisgain = myentropy

					for value in attr[1]: #attr[1] is a list of values this attr can be
						tmpset = []
						for datapoint in self.dataset:
							if datapoint[attr[0]] == value:
								tmpset.append(datapoint)
						thisgain -= (len(tmpset) / float(len(self.dataset))) * getentropy(tmpset)

				# real-valued data
				else:
					thisgain = -9000
					# make an ordered list of the values in our dataset for this attribute
					sorteddata = sorted(self.dataset, key = lambda k: k[attr[0]])

					for i in xrange(len(sorteddata)-1):
						realvaluemaxgain = -9000
						if sorteddata[i][attr[0]] != sorteddata[i+1][attr[0]]:
							testsplitval = (sorteddata[i+1][attr[0]] + sorteddata[i][attr[0]]) / float(2)
							set1 = sorteddata[:i+1]
							set2 = sorteddata[i+1:]

							realvaluemaxgain = myentropy
							realvaluemaxgain -= (len(set1) / float(len(self.dataset))) * getentropy(set1,self.allattributes)
							realvaluemaxgain -= (len(set2) / float(len(self.dataset))) * getentropy(set2,self.allattributes)
						# see if it's the best of this attr
						if realvaluemaxgain > thisgain:
							thisgain = realvaluemaxgain
							splitval = testsplitval
							splitindex = i+1

			# see if it's the very best
			if thisgain > maxgain:
				maxgain = thisgain
				maxattr = attr
				self.splitval = splitval
				self.splitindex = splitindex

		# reset splitval to flag -9000 if attribute selected is not real-valued
		if len(maxattr[1]) > 1:
			self.splitval = -9000

		return (maxattr, maxgain)

	def makeleaf(self):
		# simple majority rules
		positives = 0
		negatives = 0
		key = self.allattributes[-1][1][0] # get a positive or negative name of the class
		for data in self.dataset:
			if key in data["class"]:
				positives+=1
			else:
				negatives+=1
		if negatives > positives:
			self.classification = self.allattributes[-1][1][1]
		else:
			self.classification = self.allattributes[-1][1][0]

	def maketree(self):
		# check to see if we should be a leaf

		# 0) we have no data
		if len(self.dataset) == 0:
			self.makeleaf()
			return

		# 1) all instances are the same class
		firstclass = self.dataset[0]["class"]
		shouldbeleaf = True
		for datapoint in self.dataset:
			if datapoint["class"] != firstclass:
				shouldbeleaf = False
				break
		if shouldbeleaf:
			self.makeleaf()
			return

		# 2) we have less than m instances
		if len(self.dataset) < self.m:
			self.makeleaf()
			return

		# 3) no positive information gains
		(self.attribute, gain) = self.maxinfogain()

		if not gain > 0:
			self.makeleaf()
			return

		# 4) we have no more attributes
		if len(self.attributes) == 0:
			self.makeleaf()
			return

		# if we've gotten here, we should not be a leaf! hooray!

		if not len(self.attribute[1]) == 1:
			# we are nominal
			newattributes = []
			for attr in self.attributes:
				newattributes.append(attr)
			newattributes.remove(self.attribute)

			for value in self.attribute[1]:
				# get the data which are that value of the attr
				newdataset = []
				for datapoint in self.dataset:
					if datapoint[self.attribute[0]] == value:
						newdataset.append(datapoint)

				kid = id3(newattributes, newdataset)
				kid.maketree()
				self.children.append(kid)
		else:
			# we are real-valued
			newattributes = []
			for attr in self.attributes:
				newattributes.append(attr)

			sorteddata = sorted(self.dataset, key = lambda k: k[self.attribute[0]])
			set1 = sorteddata[:self.splitindex]
			set2 = sorteddata[self.splitindex:]
			
			kid = id3(newattributes, set1, self.allattributes)
			kid.maketree()
			self.children.append(kid)
			
			kid = id3(newattributes, set2, self.allattributes)
			kid.maketree()
			self.children.append(kid)

def readtrainingfile(trainfile):
	f = open(trainfile)

	# clear the relation line
	f.readline()

	# attributes list, each contains a pair (name, values)
	attributes = []

	# dataset
	dataset = []

	# read in the attributes
	while(True):
		line = f.readline()
		if "@data" in line:
			break
		line = line.strip()
		line = line.split('@attribute ')[1]
		line = line.replace(" ", "")
		line = line.replace("{", "")
		line = line.replace("}", "")
		line = line.split("'")

		attr = line[1]
		vals = line[2]
		vals = vals.split(",")
		attributes.append((attr,vals))

	# now get the dataset
	for line in f:
		dict = {}
		data = line.strip().replace(" ", "").split(",")
		for i in xrange(len(attributes)):
			if len(attributes[i][1]) == 1:
				# we have a real value
				dict[attributes[i][0]] = float(data[i])
			else:
				dict[attributes[i][0]] = data[i]
		dataset.append(dict)

	f.close()
	return (attributes,dataset)

def getentropy(dataset,allattributes):
	""" pass in a dataset (dictionary in form {attribute: value}) """
	positives = 0
	negatives = 0
	key = allattributes[-1][1][0] # get a positive or negative name of the class
	for data in dataset:
		if key in data["class"]:
			positives+=1
		else:
			negatives+=1

	sum = float(positives + negatives)
	if positives == sum or negatives == sum:
		return 0
	pplus = positives / sum
	pminus = negatives / sum

	return (-1 * pplus * math.log(pplus, 2)) - (pminus * math.log(pminus, 2))

def readtestfile(testfile):
	f = open(testfile)

	# clear the relation line
	f.readline()

	# dataset
	dataset = []

	# ignore the attributes
	while(True):
		line = f.readline()
		if "@data" in line:
			break

	for line in f:
		dict = {}
		data = line.strip().replace(" ", "").split(",")
		for i in xrange(len(attributes)):
			if len(attributes[i][1]) == 1:
				# we have a real value
				dict[attributes[i][0]] = float(data[i])
			else:
				dict[attributes[i][0]] = data[i]
		dataset.append(dict)

	f.close()
	return dataset

if __name__ == "__main__":
	train = sys.argv[1]
	test = sys.argv[2]
	m = float(sys.argv[3])

	(attributes, dataset) = readtrainingfile(train)

	tree = id3(attributes[:len(attributes)-1], dataset, attributes, m)

	tree.maketree()

	print "\nGenerated ID3 Tree: " + tree.display() + "\n\nTest Instances:"

	testset = readtestfile(test)

	countcorrect = 0

	for testdatapoint in testset:
		s = ""
		# display the datapoint's values
		for attr in attributes[:-1]:
			s += str(testdatapoint[attr[0]]) + " "

		actual = testdatapoint[attributes[-1][0]]
		s += " actual: " + actual

		predicted = tree.classify(testdatapoint)
		s += " predicted: " + predicted

		if actual == predicted:
			countcorrect += 1

		print s
	print "\nCorrectly classified " + str(countcorrect) + "/" + str(len(testset)) + " test instances. (" + "%.2f" % (100*countcorrect/(float(len(testset)))) + "%)"


	# from here on is for part 2 of the hw
	trainsizes = [25,50,100,200]

	# stratified data sets
	label = attributes[-1][1][0] # the first label of the "class" attribute
	positives = []
	negatives = []
	for trainingdatapoint in dataset:
		if trainingdatapoint[attributes[-1][0]] == label:
			negatives.append(trainingdatapoint)
		else:
			positives.append(trainingdatapoint)

	ratio = len(negatives) / float(len(positives) + len(negatives))

	print "\nEvaluation of different training set sizes:"

	for trainsize in trainsizes:
		numnegatives = int(ratio * trainsize)
		numpositives = trainsize - numnegatives
		averageacc = 0
		minacc = 101
		maxacc = -1

		for i in xrange(10):
			# draw 10 stratified training sets, make and train the tree, and evaluate
			trainset = []
			trainset += random.sample(negatives, numnegatives)
			trainset += random.sample(positives, numpositives)
			
			tree = id3(attributes[:len(attributes)-1], trainset)
			tree.maketree()

			countcorrect = 0

			for testdatapoint in testset:
				actual = testdatapoint[attributes[-1][0]]
				predicted = tree.classify(testdatapoint)

				if actual == predicted:
					countcorrect += 1

			percentage = (100*countcorrect/(float(len(testset))))

			if percentage > maxacc:
				maxacc = percentage
			if percentage < minacc:
				minacc = percentage
			averageacc += percentage

		averageacc /= 10

		print "Train Size:",trainsize,"MinAcc:",minacc,"AverageAcc:",averageacc,"MaxAcc:",maxacc









