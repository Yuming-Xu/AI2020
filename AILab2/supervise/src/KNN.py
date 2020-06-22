import math
def calculate_distance(lista,listb,len):
	#using eural distance
	total = 0;
	for i in range(0,len) :
		total += (lista[i] - listb[i])**2
	total = math.sqrt(total)
	return total
def KNN_lab2(allset,labelset,trainlen,alllen):
	count = trainlen;
	result = []
	result_without = []
	while count < alllen :
		rank = {}
		rank_without = {}
		for i in range(0,trainlen) :
			#calculate the difference of the count and the first [0,trainlen] examples
			#include with G1 G2 and without G1,G2
			dist = calculate_distance(allset[i],allset[count],len(allset[i]))
			dist_without = calculate_distance(allset[i],allset[count],len(allset[i])-2)
			rank[i] = dist
			rank_without[i] = dist_without
		#sort the dict with the distance
		ranksort = sorted(rank.items(),key=lambda rank:rank[1],reverse=False)
		ranksort_without = sorted(rank_without.items(),key=lambda rank_without:rank_without[1],reverse=False)
		temp = []
		#find the result with k from 1 to 20
		for k in range(1,21):
			###different result of k
			passlen = 0
			for m in range(0,k):
				if labelset[ranksort[m][0]] == 1 :
					passlen += 1
			if passlen > k//2 :
				temp.append(1)
			else :
				temp.append(-1)
		result.append(temp)
		temp = []
		for k in range(1,21):
			###different result of k
			passlen = 0
			for m in range(0,k):
				if labelset[ranksort_without[m][0]] == 1 :
					passlen += 1
			if passlen > k//2 :
				temp.append(1)
			else :
				temp.append(-1)
		result_without.append(temp)
		count += 1
	return result,result_without
