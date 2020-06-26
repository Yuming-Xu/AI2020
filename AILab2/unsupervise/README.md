### 文件输入:

```python
traindata,groupdata = Preprocessing("../input/wine.data")
```

修改文件名以修改输入

### PCA测试

这个函数将会测试threshold=[0.7,0.8,0.9,0.99,0.999]的结果，并将降维的维度结果输出为图

```python
testThreshold(traindata_std)
```

### Kmeans测试

这个函数将会测试k=[2,3,4,5]，threshold=[0.7,0.8,0.9,0.99,1]的结果，并将所有的结果输出为csv文件以及将轮廓系数和兰德系数制成图表输出

```python
testKmeans(traindata_std,traindata,groupdata)
```

