```python
if __name__ == '__main__':
	train_factor = 7
	Init()
	allset,labelset,setlen,trainlen = Preprocessing("../data/student-por.csv",train_factor)
	attrlist[29] = [i for i in range(94)]
	attrlist[30] = [i for i in range(21)]
	attrlist[31] = [i for i in range(21)]
	DTL_test(allset,labelset,trainlen,setlen,attrlist)
	#SVM(allset,labelset,trainlen,setlen)
	#KNN(allset,labelset,trainlen,setlen)
```

将文件名**"../data/student-por.csv"**进行修改以进行不同的输入文件测试，train_factor用来设定数据集的划分，在这里我只是简单的将输入文件切成两份一份为测试集一份为训练集

### 测试KNN

将DTL_test、SVM函数注释掉，只调用KNN函数

### 测试SVM

将DTL_test、KNN函数注释掉，只调用SVM函数

#### 核函数设置:

在svm中将对于想用的核函数保持，其他注释掉

```python
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
	para = {'name':'linear'}
	for C in Cs:
		start = time.time()
		result,result_without = SVM_lab2_no_cvxopt(allset,labelset,trainlen,setlen,C,10000,para)
		F1,F1_without = SVM_test(result,result_without,labelset,trainlen,setlen)
		end = time.time()
		runtime.append(end-start)
		F1_list.append(F1)
		F1_without_list.append(F1_without)
```

#### 测试决策树

将KNN、SVM函数注释掉，只调用DTL_test函数