import math

class DTL(object):

	def __init__(self,attrlist):
		#key = attr(num), value = value of attr(list)
		self.attrlist = attrlist

	def calcI(self,pos,neg):
		p_pn = pos/(pos+neg)
		n_pn = neg/(pos+neg)
		if p_pn != 0 and n_pn != 0:
			return - p_pn*math.log(p_pn,2) - n_pn*math.log(n_pn,2)
		elif p_pn == 0:
			return - n_pn*math.log(n_pn,2)
		elif n_pn == 0:
			return - p_pn*math.log(p_pn,2)

	def chooseAttr(self,examples,attrs):
		#choose the best
		attrlen = len(examples[0])
		attr_rank_dict = {}
		pos = 0
		neg = 0
		for example in examples:
			if example[attrlen-1] == 1 :
				pos+=1
			else :
				neg+=1
		IG = self.calcI(pos,neg)
		for attr in attrs:
			remainder = 0
			pos_i = {}
			neg_i = {}
			for value in self.attrlist[attr]:
				pos_i[value] = 0
				neg_i[value] = 0
			for example in examples:
				#print(self.attrlist[attr])
				#print(attr,example[attr])
				if example[attrlen-1] == 1 :
					pos_i[example[attr]] += 1
				else :
					neg_i[example[attr]] += 1
			for value in self.attrlist[attr]:
				if pos_i[value]+neg_i[value] != 0:
					remainder += self.calcI(pos_i[value],neg_i[value]) * (pos_i[value]+neg_i[value])/(pos+neg)
			attr_rank_dict[attr] = IG - remainder
		#print(attr_rank_dict)
		attrrank = sorted(attr_rank_dict.items(),key=lambda attr_rank_dict:attr_rank_dict[1],reverse=True)
		return attrrank[0][0]


	def testExample(self,examples):
		#if all value equals, then return true
		attrlen = len(examples[0])
		mark = examples[0][attrlen-1]
		for example in examples:
			if example[attrlen-1] != mark:
				return False,None
		return True,mark


	def run(self,examples,attrs,default):
		#didn't have examples, use the most common value in upper levels examples
		if examples == []:
			return default
		result,classification = self.testExample(examples)
		#if all are one value
		if result:
			return classification
		#if no attr to select,use the most common value
		if attrs == []:
			return self.mode(examples)
		#choose the best value
		best = self.chooseAttr(examples,attrs)
		#print(best)
		#generate a new tree 
		tree = DTL_tree(best)
		attrs_i = attrs
		attrs_i.remove(best)
		default_val = self.mode(examples)
		for value in self.attrlist[best]:
			example_i = []
			for example in examples:
				if example[best] == value:
					example_i.append(example)
			subtree = self.run(example_i,attrs_i,default_val)
			tree.addsubtree(subtree,value)
		return tree

	def mode(self,examples):
		#find the most common value
		attrlen = len(examples[0])
		true = 0
		false = 0
		for example in examples:
			if example[attrlen-1] == 1:
				true +=1
			else :
				false +=1
		if true > false:
			return 1
		else :
			return -1

	def predict(self,testset,tree):
		result = []
		for test in testset:
			result.append(tree.findleaf(test))
		return result


class DTL_tree(object):

	def __init__(self,best):
		self.key = best
		self.subtreelist = {}

	def addsubtree(self,subtree,value):
		self.subtreelist[value] = subtree

	def findleaf(self,test):
		if self.subtreelist[test[self.key]] is 1 or self.subtreelist[test[self.key]] is -1:
			return self.subtreelist[test[self.key]]
		else :
			return self.subtreelist[test[self.key]].findleaf(test)

	def treeprint(self):
		print(self.key)
		print(self.subtreelist)
		for value in self.subtreelist :
			if self.subtreelist[value] is 1 or self.subtreelist[value] is -1:
				continue
			self.subtreelist[value].treeprint()



def DTL_lab2(allset,labelset,trainlen,setlen,attrlist):
	examples = allset[0:trainlen]
	testset = allset[trainlen:setlen]
	attrs = []
	attrs_without = []
	for key in attrlist:
		attrs.append(key)
	for key in attrlist:
		if key != 30 and key != 31:
			attrs_without.append(key)
	for i in range(trainlen):
		examples[i].append(labelset[i])
	dtl = DTL(attrlist)
	tree = dtl.run(examples,attrs,1)
	tree.treeprint()
	result = dtl.predict(testset,tree)
	tree_without = dtl.run(examples,attrs_without,1)
	result_without = dtl.predict(testset,tree_without)
	return result,result_without
	
