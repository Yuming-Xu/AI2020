import numpy as np
import random
import numpy.linalg as la
from cvxopt import matrix,solvers


def selectJrand(i,m):
    j = i
    while j == i:
        j = int(random.uniform(0,m))
    return j
def clipAlpha(a_j,H,L):
    if a_j > H:
        a_j = H
    if L > a_j:
        a_j = L
    return a_j
def kernel_func(kernel,x,y) :
	if kernel['name'] == 'linear' :
		return np.inner(x, y)
	elif kernel['name'] == 'gaussian' :
		return np.exp(-np.sqrt(la.norm(np.array(x)-np.array(y)) ** 2 / (2 * kernel['sigma'] ** 2)))
	elif kernel['name'] == 'rbf' :
		return np.exp(-kernel['gamma']*la.norm(np.subtract(x, y)))
	elif kernel['name'] == 'poly':
		return (kernel['offset'] + np.inner(x, y)) ** kernel['dimension']

class PlattSMO:
    def __init__(self,dataMat,classlabels,C,toler,maxIter,kernel):
        self.x = np.array(dataMat)
        self.label = np.array(classlabels).transpose()
        #soft margin
        self.C = C
        #epsilon in smo
        self.toler = toler
        #the max count of iteration
        self.maxIter = maxIter
        #m = trainlen
        self.m = np.shape(dataMat)[0]
        #n = the dims of vector
        self.n = np.shape(dataMat)[1]
        self.alpha = np.array(np.zeros(self.m),dtype='float64')
        self.b = 0.0
        #EK in smo
        self.eCache = np.array(np.zeros((self.m,2)))
        #K(i,j)
        self.K = np.zeros((self.m,self.m),dtype='float64')
        #Kernel func, is a dict
        self.kwargs = kernel
        #svm vector
        self.SV = ()
        #svm vector in [0:trainlen]
        self.SVIndex = None
        #initial calculate
        for i in range(self.m):
            for j in range(self.m):
                self.K[i,j] = kernel_func(kernel,self.x[i,:],self.x[j,:])
    def calcEK(self,k):
        #calculate E of training sample k
        fxk = np.dot(self.alpha*self.label,self.K[:,k])+self.b
        Ek = fxk - float(self.label[k])
        return Ek
    def updateEK(self,k):
        #update E of training sample k
        Ek = self.calcEK(k)

        self.eCache[k] = [1 ,Ek]
    def selectJ(self,i,Ei):
        #select the second alpha, first search for nonzero,
        #if not then random select
        maxE = 0.0
        selectJ = 0
        Ej = 0.0
        validECacheList = np.nonzero(self.eCache[:,0])[0]
        if len(validECacheList) > 1:
            for k in validECacheList:
                if k == i:continue
                Ek = self.calcEK(k)
                deltaE = abs(Ei-Ek)
                if deltaE > maxE:
                    selectJ = k
                    maxE = deltaE
                    Ej = Ek
            return selectJ,Ej
        else:
            selectJ = selectJrand(i,self.m)
            Ej = self.calcEK(selectJ)
            return selectJ,Ej

    def innerL(self,i):
        Ei = self.calcEK(i)
        #can't keep for KKT
        if (self.label[i] * Ei < -self.toler and self.alpha[i] < self.C) or \
                (self.label[i] * Ei > self.toler and self.alpha[i] > 0):
            self.updateEK(i)
            #find j
            j,Ej = self.selectJ(i,Ei)
            #the old value
            alphaIOld = self.alpha[i].copy()
            alphaJOld = self.alpha[j].copy()
            #just as smo, if yi != yj, L = max(0,alphaj - alphai), H = min(C,C+alphaj-alphai)
            if self.label[i] != self.label[j]:
                L = max(0,self.alpha[j]-self.alpha[i])
                H = min(self.C,self.C + self.alpha[j]-self.alpha[i])
            else:
            #the same
                L = max(0,self.alpha[j]+self.alpha[i] - self.C)
                H = min(self.C,self.alpha[i]+self.alpha[j])
            if L == H:
                return 0
            #eta = K(x1,x1) K(x2,x2) - 2*K(x1,x2)
            eta = self.K[i,i] + self.K[j,j] - 2*self.K[i,j]
            if eta <= 0:
                #don't care about kernel function that can't keep for Mercer
                return 0
            self.alpha[j] += self.label[j]*(Ei-Ej)/eta
            #cut, or cliped
            self.alpha[j] = clipAlpha(self.alpha[j],H,L)
            self.updateEK(j)
            #change is small
            if abs(alphaJOld-self.alpha[j]) < 0.00001:
                return 0
            #update
            self.alpha[i] +=  self.label[i]*self.label[j]*(alphaJOld-self.alpha[j])
            self.updateEK(i)
            b1 = self.b - Ei - self.label[i] * self.K[i, i] * (self.alpha[i] - alphaIOld) - \
                 self.label[j] * self.K[i, j] * (self.alpha[j] - alphaJOld)
            b2 = self.b - Ej - self.label[i] * self.K[i, j] * (self.alpha[i] - alphaIOld) - \
                 self.label[j] * self.K[j, j] * (self.alpha[j] - alphaJOld)
            #calculate b
            if 0<self.alpha[i] and self.alpha[i] < self.C:
                self.b = b1
            elif 0 < self.alpha[j] and self.alpha[j] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) /2.0
            return 1
        else:
            return 0

    def smoP(self):
        iter = 0
        entrySet = True
        alphaPairChanged = 0
        while iter < self.maxIter and ((alphaPairChanged > 0) or (entrySet)):
            #print("current iter:",iter)
            alphaPairChanged = 0
            if entrySet:
                for i in range(self.m):
                    alphaPairChanged+=self.innerL(i)
                iter += 1
            else:
                nonBounds = np.nonzero((self.alpha > 0)*(self.alpha < self.C))[0]
                for i in nonBounds:
                    alphaPairChanged+=self.innerL(i)
                iter+=1
            if entrySet:
                entrySet = False
            elif alphaPairChanged == 0:
                entrySet = True
        self.SVIndex = np.nonzero(self.alpha)[0]
        self.SV = self.x[self.SVIndex]
        self.SVAlpha = self.alpha[self.SVIndex]
        self.SVLabel = self.label[self.SVIndex]
        self.x = None
        self.K = None
        self.label = None
        self.alpha = None
        self.eCache = None

    def calcw(self):
        for i in range(self.m):
            self.w += np.dot(self.alpha[i]*self.label[i],self.x[i,:])

    def predict(self,testData):
        test = np.array(testData)
        result = []
        m = np.shape(test)[0]
        for i in range(m):
            tmp = self.b
            for j in range(len(self.SVIndex)):
                tmp += self.SVAlpha[j] * self.SVLabel[j] * kernel_func(self.kwargs,self.SV[j],test[i,:])
            while tmp == 0:
                tmp = random.uniform(-1,1)
            if tmp > 0:
                tmp = 1
            else:
                tmp = -1
            result.append(tmp)
        return result


def SVM_lab2_no_cvxopt(allset,labelset,trainlen,alllen,C,maxiter,para) :
	trainlabel = labelset[0:trainlen]
	trainset = allset[0:trainlen]
	smo = PlattSMO(trainset,trainlabel,C,0.0001,maxiter,para)
	print("calculating with G1 G2")
	smo.smoP()
	print("predicting with G1 G2")
	result = smo.predict(allset[trainlen:alllen])
	trainset_without = [trainset[i][0:30] for i in range(trainlen)]
	#print(trainset[0])
	#print(trainset_without[0])
	smo = PlattSMO(trainset_without,trainlabel,C,0.0001,maxiter,para)
	print("calculating without G1 G2")
	smo.smoP()
	print("predicting without G1 G2")
	testset = [allset[i][0:30] for i in range(trainlen,alllen)]
	result_without = smo.predict(testset)
	return result,result_without

def SVM_lab2(allset,labelset,trainlen,alllen,C,**kernel) :
	trainlabel = labelset[0:trainlen]
	k = []
	k_without = []
	for i in range(trainlen) :
		temp = []
		for j in range(trainlen) :
			temp.append(kernel_func(kernel,allset[i],allset[j]))
		k.append(temp)
	for i in range(trainlen) :
		temp = []
		for j in range(trainlen) :
			temp.append(kernel_func(kernel,allset[i][0:29],allset[j][0:29]))
		k_without.append(temp)
	Q = matrix(-1 * np.ones(trainlen))
	#print(Q.size[0],Q.size[1])
	trainlabel_array = np.array(trainlabel)
	#print(trainlabel_array)
	P = matrix((np.outer(trainlabel,trainlabel)*np.array(k)).astype('float64'))
	P_without = matrix((np.outer(trainlabel,trainlabel)*np.array(k_without)).astype('float64'))
	tempA = []
	for i in trainlabel :
		tempA.append([1.0*i])
	#print(tempA)
	A = matrix(tempA)
	print(A.size[0],A.size[1])
	b = matrix(0.0)
	#print(b.size[0],b.size[1])
	G_0 = np.diag(-1*np.ones(trainlen))
	#print(G_0)
	h_0 = np.zeros(trainlen)
	G_C = np.diag(np.ones(trainlen))
	h_C = np.ones(trainlen)*C
	G = matrix(np.vstack((G_0,G_C)))
	#print(G.size[0],G.size[1])
	h = matrix(np.hstack((h_0,h_C)))
	#print(h.size[0],h.size[1])
	sol = solvers.qp(P,Q,G,h,A,b)
	alpha = np.ravel(sol['x'])
	alpha_list = list(alpha)
	alpha_nozero = {}
	for i in range(trainlen):
		if alpha_list[i] :
			alpha_nozero[i] = alpha_list[i]
	b_list = []
	for key in alpha_nozero :
		total = 0
		for i in alpha_nozero :
			total += alpha_nozero[i]*labelset[i]*k[key][i]
		b_list.append(labelset[key] - total)
	b_mean = np.mean(b_list)
	result = []
	for i in range(trainlen,alllen) :
		total = 0
		for key in alpha_nozero :
			total += alpha_nozero[key]*labelset[key]*kernel_func(kernel,allset[key],allset[i])
		result.append(np.sign(total+b_mean))
	sol = solvers.qp(P_without,Q,G,h,A,b)
	alpha_without = np.ravel(sol['x'])
	alpha_list = list(alpha_without)
	alpha_nozero = {}
	for i in range(trainlen):
		if alpha_list[i] :
			alpha_nozero[i] = alpha_list[i]
	b_list = []
	for key in alpha_nozero :
		total = 0
		for i in alpha_nozero :
			total += alpha_nozero[i]*labelset[i]*k[key][i]
		b_list.append(labelset[key] - total)
	b_mean = np.mean(b_list)
	result_without = []
	for i in range(trainlen,alllen) :
		total = 0
		for key in alpha_nozero :
			total += alpha_nozero[key]*labelset[key]*kernel_func(kernel,allset[key][0:29],allset[i][0:29])
		result_without.append(np.sign(total+b_mean))
	return result,result_without


