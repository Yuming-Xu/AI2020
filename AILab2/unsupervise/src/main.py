import numpy as np
import csv
from PCA import *
from Kmeans import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

def standardization(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return (x - mu) / sigma

def standaridzation_wrap(traindata):
	traindata_std = []
	for vector in traindata:
		traindata_std.append(list(standardization(np.array(vector,'float64'))))
	return np.array(traindata_std)

def Preprocessing(path):
	groupdata = []
	traindata = []
	with open(path,encoding='utf-8')as dataset:
		data = csv.reader(dataset)
		for row in data:
			groupdata.append(row[0])
			traindata.append(row[1:])
	return traindata,groupdata

def calLand(groupdata,group_after):
	#print(groupdata)
	#print(group_after)
	a,b,c,d = 0,0,0,0
	for i in range(len(groupdata)):
		for j in range(len(groupdata)):
			if i == j:
				continue
			if groupdata[i] == groupdata[j] and group_after[i] == group_after[j]:
				a += 1
			elif groupdata[i] == groupdata[j] and group_after[i] != group_after[j]:
				b += 1
			elif groupdata[i] != groupdata[j] and group_after[i] == group_after[j]:
				c += 1
			elif groupdata[i] != groupdata[j] and group_after[i] != group_after[j]:
				d += 1
	RI = (a+d)/(a+b+c+d)
	return RI

def showResult(data,label,dim,name):
	colors = ['red','blue','green','orange','purple']
	if dim == 2:
		fig = plt.figure()
		for i in range(len(data)):
			plt.scatter(float(data[i][0]), float(data[i][1]), c=colors[label[i]])
		plt.savefig('../output/'+name+'dim='+str(dim)+'.png')
	if dim == 3:
		fig = plt.figure()
		ax = Axes3D(fig)
		for i in range(len(data)):
			plt.scatter(float(data[i][0]), float(data[i][1]),float(data[i][2]), c=colors[label[i]])
		plt.savefig('../output/'+name+'dim='+str(dim)+'.png')

def showResult_privious(data,label,dim):
	colors = ['red','blue','green']
	if dim == 2:
		fig = plt.figure()
		for i in range(len(data)):
			plt.scatter(float(data[i][0]), float(data[i][1]), c=colors[int(label[i])-1])
		plt.savefig('../output/'+'privious_'+'dim='+str(dim)+'.png')
	if dim == 3:
		fig = plt.figure()
		ax = Axes3D(fig)
		for i in range(len(data)):
			plt.scatter(float(data[i][0]), float(data[i][1]),float(data[i][2]), c=colors[int(label[i])-1])
		plt.savefig('../output/'+'privious_'+'dim='+str(dim)+'.png')

def testThreshold(traindata):
	Thresholds = [0.7,0.8,0.9,0.99,0.999]
	dim_pca = []
	#print(np.shape(traindata))
	for threshold in Thresholds:
		pca = PCA(traindata,threshold)
		dim_pca.append(np.shape(pca)[0])
	print(dim_pca)
	plt.plot(Thresholds,dim_pca,c='red')
	plt.savefig('../output/threshold.png')

def Kmeans_warp(k,traindata,groupdata,traindata_origin):
	dim = np.shape(traindata)[0]
	group,S = Kmeans(k,traindata.T)
	showResult(traindata_origin,group,2,'k='+str(k)+'matrix_dim='+str(dim)+'_')
	showResult(traindata_origin,group,3,'k='+str(k)+'matrix_dim='+str(dim)+'_')
	RI = calLand(groupdata,group)
	with open("../output/result_k="+str(k)+"_dim="+str(dim)+""+".csv",'w',encoding='utf-8') as resultset:
		csv_writer = csv.writer(resultset)
		for (label,data) in zip(group,traindata_origin):
			row = []
			row.append(label)
			for item in data:
				row.append(float(item))
			csv_writer.writerow(row)
	return S,RI

def testKmeans(traindata,traindata_origin,groupdata):
	color = ['blue','red','green','black','purple']
	Thresholds = [0.7,0.8,0.9,0.99]
	Ks = [2,3,4,5]
	Ss = []
	RIs = []
	for threshold in Thresholds:
		S_single = []
		RI_single = []
		traindata_pca = PCA(traindata,threshold)
		for k in Ks:
			S,RI = Kmeans_warp(k,traindata_pca,groupdata,traindata_origin)
			S_single.append(S)
			RI_single.append(RI)
		Ss.append(S_single)
		RIs.append(RI_single)
	S_single = []
	RI_single = []
	for k in Ks:
		S,RI = Kmeans_warp(k,traindata,groupdata,traindata_origin)
		S_single.append(S)
		RI_single.append(RI)
	Ss.append(S_single)
	RIs.append(RI_single)
	labellist = []
	fig1 = plt.figure()
	x_major_locator=MultipleLocator(1)
	ax1 = plt.gca()
	ax1.xaxis.set_major_locator(x_major_locator)
	for i,threshold in enumerate(Thresholds):
		labellist.append('threshold='+str(threshold))
		plt.plot(Ks,Ss[i],c=color[i],label = 'threshold='+str(threshold))
	labellist.append('threshold=1')
	plt.plot(Ks,Ss[4],c=color[4],label = 'threshold=1')
	plt.legend(labellist)
	plt.savefig('../output/S_trend.png')
	labellist = []
	fig2 = plt.figure()
	ax2 = plt.gca()
	ax2.xaxis.set_major_locator(x_major_locator)
	for i,threshold in enumerate(Thresholds):
		labellist.append('threshold='+str(threshold))
		plt.plot(Ks,RIs[i],c=color[i],label = 'threshold='+str(threshold))
	labellist.append('threshold=1')
	plt.plot(Ks,RIs[4],c=color[4],label = 'threshold=1')
	plt.legend(labellist)
	plt.savefig('../output/RI_trend.png')
	print(Ss)
	print(RIs)

if __name__ == '__main__':
	traindata,groupdata = Preprocessing("../input/wine.data")
	#showResult_privious(traindata,groupdata,2)
	#showResult_privious(traindata,groupdata,3)
	# each row as a dim
	traindata_std = standaridzation_wrap(list(np.array(traindata).T))
	#print(traindata_std.T)
	#print(np.shape(traindata_std.T))
	#testKmeans(traindata_std,traindata,groupdata)
	#testThreshold(traindata_std)
	traindata_pca = PCA(traindata_std,threshold=0.5)
	print(traindata_pca.T[0])
	#group_std,S_std = Kmeans(3,traindata_std.T)
	group_pca,S_pca = Kmeans(3,traindata_pca.T)
	print(S_pca)
	RI = calLand(groupdata,group_pca)
	#showResult(traindata_std.T,group_std,2)
	print(RI)