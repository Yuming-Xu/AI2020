import csv
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd
from KNN import *
from SVM import *
import time
import math
from other import *

appendlist = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
codelists = [["GP","MS"],
			["F","M"],
			[],
			["U","R"],
			["LE3","GT3"],
			["T","A"],
			[],
			[],
			["teacher","health","services","at_home","other"],
			["teacher","health","services","at_home","other"],
			["home","reputation","course","other"],
			["mother","father","other"],
			[],
			[],
			[],
			["yes","no"],
			["yes","no"],
			["yes","no"],
			["yes","no"],
			["yes","no"],
			["yes","no"],
			["yes","no"],
			["yes","no"],
			[],
			[],
			[],
			[],
			[],
			[],
			[],
			[],
			[],
			[]
			]
attrlist = {0:[0,1],
			1:[0,1],
			2:[15,16,17,18,19,20,21,22],
			3:[0,1],
			4:[0,1,2],
			5:[0,1],
			6:[0,1,2,3,4],
			7:[0,1,2,3,4],
			8:[0,1,2,3,4],
			9:[0,1,2,3,4],
			10:[0,1,2,3],
			11:[0,1,2],
			12:[1,2,3,4],
			13:[1,2,3,4],
			14:[0,1,2,3],
			15:[0,1],
			16:[0,1],
			17:[0,1],
			18:[0,1],
			19:[0,1],
			20:[0,1],
			21:[0,1],
			22:[0,1],
			23:[1,2,3,4,5],
			24:[1,2,3,4,5],
			25:[1,2,3,4,5],
			26:[1,2,3,4,5],
			27:[1,2,3,4,5],
			28:[1,2,3,4,5]
			}
Le_list = []

def Init():
	for codelist in codelists :
		le = preprocessing.LabelEncoder()
		if codelist :
			le = le.fit(codelist)
		else :
			le = None
		Le_list.append(le)
def Preprocessing(Path,train_factor):
	allset = []
	labelset = []
	with open(Path,encoding='utf-8')as dataset:
		data = csv.reader(dataset,delimiter=';')
		count = 0;
		print("preprocessing")
		for row in data :
			count+=1
			if count == 1 :
				continue
			#print(row)
			temp = []
			###preprocessing
			for appendnum in appendlist :
				if codelists[appendnum] :
					###need to transform
					temp_series = pd.Series(row[appendnum])
					#print(temp_series)
					temp.append(list(Le_list[appendnum].transform(temp_series))[0])
				else :
					###needn't
					temp.append(int(row[appendnum]))
			if int(row[32]) >= 10 :
				labelset.append(1)
			else :
				labelset.append(-1)
			#print(temp)	
			allset.append(temp)
		setlen = len(allset)
		trainlen = setlen//10*train_factor
	return allset,labelset,setlen,trainlen

def DTL_test(allset,labelset,trainlen,setlen,attrlist):
	result,result_without = DTL_lab2(allset,labelset,trainlen,setlen,attrlist)
	SVM_test(result,result_without,labelset,trainlen,setlen)


def KNN(allset,labelset,trainlen,setlen):
	print("calculating:KNN")
	result,result_without = KNN_lab2(allset,labelset,trainlen,setlen)
	TP_dist = {}
	FP_dist = {}
	FN_dist = {}
	F1 = []
	F1_without = []
	k_x = list(range(1,21))
	for k in range(1,21):
		TP_dist[k] = 0
		FP_dist[k] = 0
		FN_dist[k] = 0
	for testlen in range(trainlen,setlen):
		for k in range(1,21):
			if result[testlen-trainlen][k-1] == 1 and labelset[testlen] == 1 :
				TP_dist[k] += 1
			if result[testlen-trainlen][k-1] == 1 and labelset[testlen] == -1 :
				FP_dist[k] += 1
			if result[testlen-trainlen][k-1] == -1 and labelset[testlen] == 1 :
				FN_dist[k] += 1
	for k in range(1,21):
		P = TP_dist[k] / (TP_dist[k] + FP_dist[k])
		R = TP_dist[k] / (TP_dist[k] + FN_dist[k])
		F1.append((2*P*R/(P+R)))
	for k in range(1,21):
		TP_dist[k] = 0
		FP_dist[k] = 0
		FN_dist[k] = 0
	for testlen in range(trainlen,setlen):
		for k in range(1,21):
			if result_without[testlen-trainlen][k-1] == 1 and labelset[testlen] == 1 :
				TP_dist[k] += 1
			if result_without[testlen-trainlen][k-1] == 1 and labelset[testlen] == -1 :
				FP_dist[k] += 1
			if result_without[testlen-trainlen][k-1] == -1 and labelset[testlen] == 1 :
				FN_dist[k] += 1
	for k in range(1,21):
		P = TP_dist[k] / (TP_dist[k] + FP_dist[k])
		R = TP_dist[k] / (TP_dist[k] + FN_dist[k])
		F1_without.append((2*P*R/(P+R)))
	print("with G1,G2:")
	print(F1)
	print("without G1,G2:")
	print(F1_without)
	fig = plt.figure()
	x_major_locator=MultipleLocator(1)
	ax = plt.gca()
	ax.xaxis.set_major_locator(x_major_locator)
	plt.plot(k_x,F1,c='red',label = 'with G1,G2')
	plt.plot(k_x,F1_without,c='blue',label = 'without G1,G2')
	plt.legend(['with G1,G2','without G1,G2'])
	plt.savefig('F1_KNN.png')

def SVM_test(result,result_without,labelset,trainlen,setlen):
	TP = 0
	FP = 0
	FN = 0
	TN = 0
	for testlen in range(trainlen,setlen):
		if result[testlen-trainlen] == 1 and labelset[testlen] == 1 :
			TP += 1
		if result[testlen-trainlen] == -1 and labelset[testlen] == -1 :
			TN += 1
		if result[testlen-trainlen] == 1 and labelset[testlen] == -1 :
			FP += 1
		if result[testlen-trainlen] == -1 and labelset[testlen] == 1 :
			FN += 1
	if TP != 0 :
		P = TP/(TP+FP)
		R = TP/(TP+FN)
		F1 = 2*P*R/(P+R)
	else :
		F1 = 0
	print("with G1,G2:")
	print(F1)
	TP = 0
	FP = 0
	FN = 0
	TN = 0
	for testlen in range(trainlen,setlen):
		if result_without[testlen-trainlen] == 1 and labelset[testlen] == 1 :
			TP += 1
		if result_without[testlen-trainlen] == -1 and labelset[testlen] == -1 :
			TN += 1
		if result_without[testlen-trainlen] == 1 and labelset[testlen] == -1 :
			FP += 1
		if result_without[testlen-trainlen] == -1 and labelset[testlen] == 1 :
			FN += 1
	#print(TP,TN,FP,FN)
	if TP != 0 :
		P = TP/(TP+FP)
		R = TP/(TP+FN)
		F1_without = 2*P*R/(P+R)
	else :
		F1_without = 0
	print("without G1,G2:")
	print(F1_without)
	return F1,F1_without

def plotfig_SVM(runtime,F1,F1_without,para,Cs):
	print(F1)
	print(F1_without)
	color = ['blue','red','green','black','orange','purple']
	if para['name'] == 'rbf':
		logCs = [math.log(C,10) for C in Cs]
		labellist = []
		print(logCs)
		fig1 = plt.figure()
		x_major_locator=MultipleLocator(1)
		ax1 = plt.gca()
		ax1.xaxis.set_major_locator(x_major_locator)
		for i,Gamma in enumerate(para['gamma']):
			labellist.append('gamma='+str(Gamma)+' with G1,G2')
			plt.plot(logCs,F1[i],c=color[i],label = 'gamma='+str(Gamma)+' with G1,G2')
		plt.legend(labellist)
		plt.savefig('F1_SVM'+para['name']+'_with.png')
		fig2 = plt.figure()
		ax2 = plt.gca()
		ax2.xaxis.set_major_locator(x_major_locator)
		for i,Gamma in enumerate(para['gamma']):
			labellist.append('gamma='+str(Gamma)+' without G1,G2')
			plt.plot(logCs,F1_without[i],c=color[i],label = 'gamma='+str(Gamma)+' without G1,G2')
		plt.legend(labellist)
		plt.savefig('F1_SVM'+para['name']+'_without.png')
	elif para['name'] == 'linear':
		logCs = [math.log(C,10) for C in Cs]
		fig1 = plt.figure()
		x_major_locator=MultipleLocator(1)
		ax1 = plt.gca()
		ax1.xaxis.set_major_locator(x_major_locator)
		plt.plot(logCs,F1,c=color[0],label = 'linear with G1,G2')
		plt.plot(logCs,F1_without,c=color[1],label = 'linear without G1,G2')
		plt.legend(['linear with G1,G2','linear without G1,G2'])
		plt.savefig('F1_SVM'+para['name']+'.png')
	elif para['name'] == 'poly':
		logCs = [math.log(C,10) for C in Cs]
		labellist = []
		print(logCs)
		fig1 = plt.figure()
		x_major_locator=MultipleLocator(1)
		ax1 = plt.gca()
		ax1.xaxis.set_major_locator(x_major_locator)
		for i,Dimension in enumerate(para['dimension']):
			labellist.append('dimension='+str(Dimension)+' with G1,G2')
			plt.plot(logCs,F1[i],c=color[i],label = 'dimension='+str(Dimension)+' with G1,G2')
		plt.legend(labellist)
		plt.savefig('F1_SVM'+para['name']+'_with.png')
		fig2 = plt.figure()
		ax2 = plt.gca()
		ax2.xaxis.set_major_locator(x_major_locator)
		for i,Dimension in enumerate(para['dimension']):
			labellist.append('dimension='+str(Dimension)+' without G1,G2')
			plt.plot(logCs,F1_without[i],c=color[i],label = 'dimension='+str(Dimension)+' without G1,G2')
		plt.legend(labellist)
		plt.savefig('F1_SVM'+para['name']+'_without.png')
	elif para['name'] == 'gaussian':
		logCs = [math.log(C,10) for C in Cs]
		labellist = []
		print(logCs)
		fig1 = plt.figure()
		x_major_locator=MultipleLocator(1)
		ax1 = plt.gca()
		ax1.xaxis.set_major_locator(x_major_locator)
		for i,Sigma in enumerate(para['sigma']):
			labellist.append('sigma='+str(Sigma)+' with G1,G2')
			plt.plot(logCs,F1[i],c=color[i],label = 'sigma='+str(Sigma)+' with G1,G2')
		plt.legend(labellist)
		plt.savefig('F1_SVM'+para['name']+'_with.png')
		fig2 = plt.figure()
		ax2 = plt.gca()
		ax2.xaxis.set_major_locator(x_major_locator)
		for i,Sigma in enumerate(para['sigma']):
			labellist.append('sigma='+str(Sigma)+' without G1,G2')
			plt.plot(logCs,F1_without[i],c=color[i],label = 'sigma='+str(Sigma)+' without G1,G2')
		plt.legend(labellist)
		plt.savefig('F1_SVM'+para['name']+'_without.png')


def SVM(allset,labelset,trainlen,setlen):
	print("calculating:SVM")
	Cs = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
	Gammas = [0.001,0.01,0.1,1,10,100]
	Sigmas = [0.25,0.5,1,2,4,8]
	Dimensions = [1,2,3,4]
	runtime = []
	F1_list = []
	F1_without_list = []
	###use rbf
	"""
	para = {'name':'rbf'}
	for gamma in Gammas :
		para['gamma'] = gamma
		runtime_single_C = []
		F1_single_C = []
		F1_without_single_C = []
		for C in Cs :
			start = time.time()
			result,result_without = SVM_lab2_no_cvxopt(allset,labelset,trainlen,setlen,C,10000,para)
			F1,F1_without = SVM_test(result,result_without,labelset,trainlen,setlen)
			end = time.time()
			runtime_single_C.append(end-start)
			F1_single_C.append(F1)
			F1_without_single_C.append(F1_without)
		runtime.append(runtime_single_C)
		F1_list.append(F1_single_C)
		F1_without_list.append(F1_without_single_C)
	para['gamma'] = Gammas
	"""
	###use linear
	"""
	para = {'name':'linear'}
	for C in Cs:
		start = time.time()
		result,result_without = SVM_lab2_no_cvxopt(allset,labelset,trainlen,setlen,C,10000,para)
		F1,F1_without = SVM_test(result,result_without,labelset,trainlen,setlen)
		end = time.time()
		runtime.append(end-start)
		F1_list.append(F1)
		F1_without_list.append(F1_without)
	"""
	###use gaussian
	"""
	para = {'name':'gaussian'}
	for sigma in Sigmas :
		para['sigma'] = sigma
		runtime_single_C = []
		F1_single_C = []
		F1_without_single_C = []
		for C in Cs :
			start = time.time()
			result,result_without = SVM_lab2_no_cvxopt(allset,labelset,trainlen,setlen,C,10000,para)
			F1,F1_without = SVM_test(result,result_without,labelset,trainlen,setlen)
			end = time.time()
			runtime_single_C.append(end-start)
			F1_single_C.append(F1)
			F1_without_single_C.append(F1_without)
		runtime.append(runtime_single_C)
		F1_list.append(F1_single_C)
		F1_without_list.append(F1_without_single_C)
	para['sigma'] = Sigmas
	"""
	###use poly
	para = {'name':'poly','offset':1}
	for dimension in Dimensions :
		para['dimension'] = dimension
		runtime_single_C = []
		F1_single_C = []
		F1_without_single_C = []
		for C in Cs :
			start = time.time()
			result,result_without = SVM_lab2_no_cvxopt(allset,labelset,trainlen,setlen,C,10000,para)
			F1,F1_without = SVM_test(result,result_without,labelset,trainlen,setlen)
			end = time.time()
			runtime_single_C.append(end-start)
			F1_single_C.append(F1)
			F1_without_single_C.append(F1_without)
		runtime.append(runtime_single_C)
		F1_list.append(F1_single_C)
		F1_without_list.append(F1_without_single_C)
	para['dimension'] = Dismensions
	plotfig_SVM(runtime,F1_list,F1_without_list,para,Cs)

if __name__ == '__main__':
	train_factor = 7
	Init()
	allset,labelset,setlen,trainlen = Preprocessing("../data/student-mat.csv",train_factor)
	attrlist[29] = [i for i in range(94)]
	attrlist[30] = [i for i in range(21)]
	attrlist[31] = [i for i in range(21)]
	#DTL_test(allset,labelset,trainlen,setlen,attrlist)
	SVM(allset,labelset,trainlen,setlen)
	#KNN(allset,labelset,trainlen,setlen)