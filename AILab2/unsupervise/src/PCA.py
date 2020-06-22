import numpy as np
def PCA(data,threshold=0.9):
	cov = np.cov(data)
	#print(cov)
	eigVal,eigVec = np.linalg.eig(np.mat(cov))
	#rint(eigVal)
	#print(eigVec)
	eigVal_sum = np.sum(eigVal)
	eigVal_rank = np.argsort(eigVal)[::-1]
	#print(eigVal)
	m = 0
	lower_sum = 0
	while 1 :
		upper_sum = lower_sum + eigVal[eigVal_rank[m]]
		lower = lower_sum/eigVal_sum
		upper = upper_sum/eigVal_sum
		if threshold > lower and threshold <= upper:
			break
		#print("lower:",lower)
		#print("upper:",upper)
		m+=1
		lower_sum = upper_sum
	#print(m)	
	eigVec_firstm = eigVec[eigVal_rank[0:m+1]]
	#print(eigVec_firstm)
	#print(eigVec_firstm)
	data_pca = eigVec_firstm * data
	return data_pca
