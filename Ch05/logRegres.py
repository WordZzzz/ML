# -*- coding: UTF-8 -*- #
'''
Created on Sep 15, 2017
Logistic Regression Working Module
@author: WordZzzz
'''
from numpy import *

def loadDataSet():
	"""
	Function：	加载数据集

	Input：		testSet.txt：数据集

	Output：	dataMat：数据矩阵100*3
				labelMat：类别标签矩阵1*100
	"""	
	#初始化数据列表和标签列表
	dataMat = []; labelMat = []
	#打开数据集
	fr = open('testSet.txt')
	#遍历每一行
	for line in fr.readlines():
		#删除空白符之后进行切分
		lineArr = line.strip().split()
		#数据加入数据列表
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		#标签加入数据列表
		labelMat.append(int(lineArr[2]))
	#返回数据列表和标签列表
	return dataMat, labelMat

def sigmoid(inX):
	"""
	Function：	sigmoid函数

	Input：		inX：矩阵运算结果100*1

	Output：	1.0/(1+exp(-inX))：计算结果
	"""	
	#返回计算结果
	return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
	"""
	Function：	logistic回归梯度上升函数

	Input：		dataMatIn：数据列表100*3
				classLabels：标签列表1*100

	Output：	weights：权重参数矩阵
	"""	
	#转换为numpy矩阵dataMatrix：100*3
	dataMatrix = mat(dataMatIn)
	#转换为numpy矩阵并转置为labelMat：100*1
	labelMat = mat(classLabels).transpose()
	#获得矩阵行列数
	m,n = shape(dataMatrix)
	#初始化移动步长
	alpha = 0.001
	#初始化地带次数
	maxCycles = 500
	#初始化权重参数矩阵，初始值都为1
	weights = ones((n,1))
	#开始迭代计算参数
	for k in range(maxCycles):
		#100*3 * 3*1 => 100*1
		h = sigmoid(dataMatrix * weights)
		#计算误差100*1
		error = (labelMat - h)
		#更新参数值
		weights = weights + alpha * dataMatrix.transpose() * error
	#返回权重参数矩阵
	return weights

def plotBestFit(weights):
	"""
	Function：	画出数据集和最佳拟合直线

	Input：		weights：权重参数矩阵

	Output：	包含数据集和拟合直线的图像
	"""	
	#加载matplotlib中的pyplot模块
	import matplotlib.pyplot as plt
	#导入数据
	dataMat, labelMat = loadDataSet()
	#创建数组
	dataArr =array(dataMat)
	#获取数组行数
	n = shape(dataArr)[0]
	#初始化坐标
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	#遍历每一行数据
	for i in range(n):
		#如果对应的类别标签对应数值1，就添加到xcord1，ycord1中
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
		#如果对应的类别标签对应数值0，就添加到xcord2，ycord2中
		else:
			xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
	#创建空图
	fig = plt.figure()
	#添加subplot，三种数据都画在一张图上
	ax = fig.add_subplot(111)
	#1类用红色标识，marker='s'形状为正方形
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
	#0类用绿色标识，弄认marker='o'为圆形
	ax.scatter(xcord2, ycord2, s=30, c='green')
	#设置x取值，arange支持浮点型
	x = arange(-3.0, 3.0, 0.1)
	#配计算y的值
	y = (-weights[0]-weights[1]*x)/weights[2]
	#画拟合直线
	ax.plot(x, y)
	#贴坐标表头
	plt.xlabel('X1'); plt.ylabel('X2')
	#显示结果
	plt.show()

def stocGradAscent0(dataMatrix, classLabels):
	"""
	Function：	随机梯度上升算法

	Input：		dataMatIn：数据列表100*3
				classLabels：标签列表1*100

	Output：	weights：权重参数矩阵
	"""	
	#获取数据列表大小
	m,n = shape(dataMatrix)
	#步长设置为0.01
	alpha = 0.01
	#初始化权重参数矩阵，初始值都为1
	weights = ones(n)
	#遍历每一行数据
	for i in range(m):
		#1*3 * 3*1
		h = sigmoid(sum(dataMatrix[i]*weights))
		#计算误差
		error = classLabels[i] - h
		#更新权重值
		weights = weights + alpha * error * dataMatrix[i]
	#返回权重参数矩阵
	return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
	"""
	Function：	改进的随机梯度上升算法

	Input：		dataMatIn：数据列表100*3
				classLabels：标签列表1*100
				numIter：迭代次数

	Output：	weights：权重参数矩阵
	"""	
	#获取数据列表大小
	m, n = shape(dataMatrix)
	#初始化权重参数矩阵，初始值都为1
	weights = ones(n)
	#开始迭代，迭代次数为numIter
	for j in range(numIter):
		#初始化index列表，这里要注意将range输出转换成list
		dataIndex = list(range(m))
		#遍历每一行数据，这里要注意将range输出转换成list
		for i in list(range(m)):
			#更新alpha值，缓解数据高频波动
			alpha = 4/(1.0+j+i)+0.0001
			#随机生成序列号，从而减少随机性的波动
			randIndex = int(random.uniform(0, len(dataIndex)))
			#序列号对应的元素与权重矩阵相乘，求和后再求sigmoid
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			#求误差，和之前一样的操作
			error = classLabels[randIndex] - h
			#更新权重矩阵
			weights = weights + alpha * error * dataMatrix[randIndex]
			#删除这次计算的数据
			del(dataIndex[randIndex])
	#返回权重参数矩阵
	return weights

def classifyVector(inX, weights):
	"""
	Function：	分类函数

	Input：		inX：计算得出的矩阵100*1
				weights：权重参数矩阵

	Output：	分类结果
	"""	
	#计算sigmoid值
	prob = sigmoid(sum(inX*weights))
	#返回分类结果
	if prob > 0.5:
		return 1.0
	else:
		return 0.0

def colicTest():
	"""
	Function：	训练和测试函数

	Input：		训练集和测试集文本文档

	Output：	分类错误率
	"""	
	#打开训练集
	frTrain = open('horseColicTraining.txt')
	#打开测试集
	frTest = open('horseColicTest.txt')
	#初始化训练集数据列表
	trainingSet = []
	#初始化训练集标签列表
	trainingLabels = []
	#遍历训练集数据
	for line in frTrain.readlines():
		#切分数据集
		currLine = line.strip().split('\t')
		#初始化临时列表
		lineArr = []
		#遍历21项数据重新生成列表，因为后面格式要求，这里必须重新生成一下。
		for i in range(21):
			lineArr.append(float(currLine[i]))
		#添加数据列表
		trainingSet.append(lineArr)
		#添加分类标签
		trainingLabels.append(float(currLine[21]))
	#获得权重参数矩阵
	trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
	#初始化错误分类计数
	errorCount = 0

	numTestVec = 0.0
	#遍历测试集数据
	for line in frTest.readlines():
		#
		numTestVec += 1.0
		#切分数据集
		currLine =line.strip().split('\t')
		#初始化临时列表
		lineArr = []
		#遍历21项数据重新生成列表，因为后面格式要求，这里必须重新生成一下。
		for i in range(21):
			lineArr.append(float(currLine[i]))
		#如果分类结果和分类标签不符，则错误计数+1
		if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
			errorCount += 1
	#计算分类错误率
	errorRate = (float(errorCount)/numTestVec)
	#打印分类错误率
	print("the error rate of this test is: %f" % errorRate)
	#返回分类错误率
	return errorRate

def multiTest():
	"""
	Function：	求均值函数

	Input：		无

	Output：	十次分类结果的平均值
	"""	
	#迭代次数
	numTests = 10
	#初始错误率和
	errorSum = 0.0
	#调用十次colicTest()，累加错误率
	for k in range(numTests):
		errorSum += colicTest()
	#打印平均分类结果
	print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))