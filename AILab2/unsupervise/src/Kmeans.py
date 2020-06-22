import random
import numpy as np
def random_points(k,m):
	#k points,m examples
	randomP = []
	for i in range(k):
		point = int(random.uniform(0,m))
		while point in randomP:
			point = int(random.uniform(0,m))
		randomP.append(point)
	return randomP

def cal_dis(point_x,point_y):
	dis = 0
	for i in range(np.shape(point_x)[1]):
		dis += (point_x[0,i]-point_y[0,i])**2
	return dis**0.5

def Kmeans(k,data):
	data_mat = np.matrix(data)
	tot = np.shape(data)[0]
	col = np.shape(data)[1]
	currentP_index = random_points(k,len(data))
	currentP = data_mat[currentP_index]
	max_dis = 1
	#print(currentP)
	while max_dis >0.0001:
		#if dis <= 0.0001, regard them as one point
		#sort
		group = {}
		grouplabel = []
		for i in range(k):
			group[i] = []
		for example in data_mat:
			dis_list = []
			for center in currentP:
				dis = cal_dis(example,center)
				dis_list.append(dis)
			dis_rank = np.argsort(dis_list)
			group[dis_rank[0]].append(example)
			grouplabel.append(dis_rank[0])
		lastP = currentP
		currentP = []
		#calculate the difference
		for key in group:
			total = np.matrix(np.zeros(col))
			for example in group[key]:
				total += example
			if group[key] != []:
				total /= len(group[key])
			currentP.append(total)
		max_dis = 0
		for lastC,curC in zip(group.keys(),currentP):
			dis = cal_dis(lastP[lastC],curC)
			if dis>max_dis :
				max_dis = dis
	S = 0
	#calculate S
	for C in group.keys():
		dis_C_min = 100000
		for otherC in group.keys():
			if C is otherC:
				continue
			dis_C = cal_dis(currentP[C],currentP[otherC])
			if dis_C_min > dis_C :
				dis_C_min = dis_C
				nearC = otherC
		for Vec in group[C]:
			a = 0
			for otherVec in group[C]:
				if otherVec is Vec:
					continue
				a += cal_dis(Vec,otherVec)
			a /= len(group[C]) - 1
			b = 0
			for otherVec in group[nearC]:
				b += cal_dis(Vec,otherVec)
			b /= len(group[nearC]) - 1
			S += (b-a)/(max(a,b))
	S /= tot
	return (grouplabel,S)
