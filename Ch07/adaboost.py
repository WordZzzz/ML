#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-10 16:04:14
# @Author  : WordZzzz (wordzzzz@foxmail.com)
# @Link    : http://blog.csdn.net/u011475210
# @Version : $Id$

from numpy import *

def loadSimpData():
	"""
	Function：	创建数据集

	Input：		NULL

	Output：	datMat：数据集
				classLabels：类别标签
	"""	
	#创建数据集
	datMat = matrix([[1. , 2.1],
		[2. , 1.1],
		[1.3, 1. ],
		[1. , 1. ],
		[2. , 1. ]])
	#创建类别标签
	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
	#返回数据集和标签
	return datMat, classLabels

def loadDataSet(fileName):
	"""
	Function：	自适应数据加载函数

	Input：		fileName：文件名称

	Output：	dataMat：数据集
				labelMat：类别标签
	"""	
	#自动获取特征个数，这是和之前不一样的地方
	numFeat = len(open(fileName).readline().split('\t'))
	#初始化数据集和标签列表
	dataMat = []; labelMat = []
	#打开文件
	fr = open(fileName)
	#遍历每一行
	for line in fr.readlines():
		#初始化列表，用来存储每一行的数据
		lineArr = []
		#切分文本
		curLine = line.strip().split('\t')
		#遍历每一个特征，某人最后一列为标签
		for i in range(numFeat-1):
			#将切分的文本全部加入行列表中
			lineArr.append(float(curLine[i]))
		#将每个行列表加入到数据集中
		dataMat.append(lineArr)
		#将每个标签加入标签列表中
		labelMat.append(float(curLine[-1]))
	#返回数据集和标签列表
	return dataMat, labelMat

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
	"""
	Function：	通过阈值比较对数据进行分类

	Input：		dataMatrix：数据集
				dimen：数据集列数
				threshVal：阈值
				threshIneq：比较方式：lt，gt

	Output：	retArray：分类结果
	"""	
	#新建一个数组用于存放分类结果，初始化都为1
	retArray = ones((shape(dataMatrix)[0],1))
	#lt：小于，gt；大于；根据阈值进行分类，并将分类结果存储到retArray
	if threshIneq == 'lt':
		retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:, dimen] > threshVal] = -1.0
	#返回分类结果
	return retArray

def buildStump(dataArr, classLabels, D):
	"""
	Function：	找到最低错误率的单层决策树

	Input：		dataArr：数据集
				classLabels：数据标签
				D：权重向量

	Output：	bestStump：分类结果
				minError：最小错误率
				bestClasEst：最佳单层决策树
	"""	
	#初始化数据集和数据标签
	dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
	#获取行列值
	m,n = shape(dataMatrix)
	#初始化步数，用于在特征的所有可能值上进行遍历
	numSteps = 10.0
	#初始化字典，用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
	bestStump = {}
	#初始化类别估计值
	bestClasEst = mat(zeros((m,1)))
	#将最小错误率设无穷大，之后用于寻找可能的最小错误率
	minError = inf
	#遍历数据集中每一个特征
	for i in range(n):
		#获取数据集的最大最小值
		rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()
		#根据步数求得步长
		stepSize = (rangeMax - rangeMin) / numSteps
		#遍历每个步长
		for j in range(-1, int(numSteps) + 1):
			#遍历每个不等号
			for inequal in ['lt', 'gt']:
				#设定阈值
				threshVal = (rangeMin + float(j) * stepSize)
				#通过阈值比较对数据进行分类
				predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
				#初始化错误计数向量
				errArr = mat(ones((m,1)))
				#如果预测结果和标签相同，则相应位置0
				errArr[predictedVals == labelMat] = 0
				#计算权值误差，这就是AdaBoost和分类器交互的地方
				weightedError = D.T * errArr
				#打印输出所有的值
				#print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
				#如果错误率低于minError，则将当前单层决策树设为最佳单层决策树，更新各项值
				if weightedError < minError:
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	#返回最佳单层决策树，最小错误率，类别估计值
	return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
	"""
	Function：	找到最低错误率的单层决策树

	Input：		dataArr：数据集
				classLabels：数据标签
				numIt：迭代次数

	Output：	weakClassArr：单层决策树列表
				aggClassEst：类别估计值
	"""	
	#初始化列表，用来存放单层决策树的信息
	weakClassArr = []
	#获取数据集行数
	m = shape(dataArr)[0]
	#初始化向量D每个值均为1/m，D包含每个数据点的权重
	D = mat(ones((m,1))/m)
	#初始化列向量，记录每个数据点的类别估计累计值
	aggClassEst = mat(zeros((m,1)))
	#开始迭代
	for i in range(numIt):
		#利用buildStump()函数找到最佳的单层决策树
		bestStump, error, classEst = buildStump(dataArr, classLabels, D)
		#print("D: ", D.T)
		#根据公式计算alpha的值，max(error, 1e-16)用来确保在没有错误时不会发生除零溢出
		alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
		#保存alpha的值
		bestStump['alpha'] = alpha
		#填入数据到列表
		weakClassArr.append(bestStump)
		#print("classEst: ", classEst.T)
		#为下一次迭代计算D
		expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
		D = multiply(D, exp(expon))
		D = D / D.sum()
		#累加类别估计值
		aggClassEst += alpha * classEst
		#print("aggClassEst: ", aggClassEst.T)
		#计算错误率，aggClassEst本身是浮点数，需要通过sign来得到二分类结果
		aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1)))
		errorRate = aggErrors.sum() / m
		print("total error: ", errorRate)
		#如果总错误率为0则跳出循环
		if errorRate == 0.0: break
	#返回单层决策树列表和累计错误率
	#return weakClassArr
	return weakClassArr, aggClassEst

def adaClassify(datToClass, classifierArr):
	"""
	Function：	AdaBoost分类函数

	Input：		datToClass：待分类样例
				classifierArr：多个弱分类器组成的数组

	Output：	sign(aggClassEst)：分类结果
	"""	
	#初始化数据集
	dataMatrix = mat(datToClass)
	#获得待分类样例个数
	m = shape(dataMatrix)[0]
	#构建一个初始化为0的列向量，记录每个数据点的类别估计累计值
	aggClassEst = mat(zeros((m,1)))
	#遍历每个弱分类器
	for i in range(len(classifierArr)):
		#基于stumpClassify得到类别估计值
		classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
		#累加类别估计值
		aggClassEst += classifierArr[i]['alpha']*classEst
		#打印aggClassEst，以便我们了解其变化情况
		#print(aggClassEst)
	#返回分类结果，aggClassEst大于0则返回+1，否则返回-1
	return sign(aggClassEst)

def plotROC(predStrengths, classLabels):
	"""
	Function：	ROC曲线的绘制及AUC计算函数

	Input：		predStrengths：分类器的预测强度
				classLabels：类别标签

	Output：	ySum * xStep：AUC计算结果
	"""	
	#导入pyplot
	import matplotlib.pyplot as plt
	#创建一个浮点数二元组并初始化为(1.0, 1.0),该元祖保留的是绘制光标的位置
	cur = (1.0, 1.0)
	#初始化ySum，用于计算AUC
	ySum = 0.0
	#通过数组过滤计算正例的数目，即y轴的步进数目
	numPosClas = sum(array(classLabels) == 1.0)
	#初始化x轴和y轴的步长
	yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels) - numPosClas)
	#得到排序索引，从<1.0,1.0>开始绘制，一直到<0,0>
	sortedIndicies = predStrengths.argsort()
	#用于构建画笔，熟悉的matlab绘图方式
	fig = plt.figure()
	fig.clf()
	ax = plt.subplot()
	#在所有排序值上遍历，因为python的迭代需要列表形式，所以调用tolist()方法
	for index in sortedIndicies.tolist()[0]:
		#每得到一个标签为1.0的类，则沿y轴下降一个步长
		if classLabels[index] == 1.0:
			delX = 0; delY = yStep
		#否则，则沿x轴下降一个步长
		else:
			delX = xStep; delY = 0
			#计数，用来计算所有高度的和，从而计算AUC
			ySum += cur[1]
		#下降步长的绘制，蓝色
		ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c = 'b')
		#绘制光标坐标值更新
		cur = (cur[0] - delX, cur[1] - delY)
	#蓝色双划线
	ax.plot([0,1], [0,1], 'b--')
	#坐标轴标签
	plt.xlabel('False positive rate'); plt.ylabel('Ture positive rate')
	#表头
	plt.title('ROC curve for AdaBoost horse colic detection system')
	#设定坐标范围
	ax.axis([0,1,0,1])
	#显示绘制结果
	plt.show()
	#打印AUC计算结果
	print("the Area Under the Curve is: ", ySum * xStep)