# -*- coding: UTF-8 -*-
"""
Created on Aug 18, 2017
kNN: k Nearest Neighbors
@author: wordzzzz
"""

from numpy import *
import operator
from os import listdir

def createDataSet():
	"""
	Function：	创建数据集和标签

	Args：		无

	Returns：	group：创建的数据集
				labels：创建的标签
	"""

	#创建数据集
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	#创建标签
	labels = ['A', 'A', 'B', 'B']
	#返回创建的数据集和标签						
	return group, labels

def classify0(inX, dataSet, labels, k):
	"""
	Function：	创建数据集和标签

	Args：		inX：用于分类的输入向量 (1xN)
            	dataSet：输入的训练样本集 (NxM)
            	labels：标签向量 (1xM vector)
            	k：用于比较的近邻数量 (should be an odd number)

	Returns：	sortedClassCount[0][0]：分类结果
	"""
	#dataSet.shape[0]：求dataSet矩阵的行数
	#dataSet.shape[1]：求dataSet矩阵的列数
	#dataSet.shape：元组形式输出矩阵行数、列数
	dataSetSize = dataSet.shape[0]
	#tile(A, B)：将A重复B次，其中B可以是int类型也可以是元组类型
	#这句话相当于向量inX与矩阵dataSet里面的每组数据做差
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	#对求差后的矩阵求平方
	sqDiffMat = diffMat**2
	#sqDiffMat.sum(axis=0)：对矩阵的每一列求和
	#sqDiffMat.sum(axis=1)：对矩阵的每一行求和
	#sqDiffMat.sum()：对整个矩阵求和
	sqDistances = sqDiffMat.sum(axis=1)
	#求平方根
	distances = sqDistances**0.5
	#对上式结果进行排序
	sortedDistIndicies = distances.argsort()
	#创建字典
	classCount = {}
	#给字典赋值
	for i in range(k):
		#字典的key
		voteIlabel = labels[sortedDistIndicies[i]]
		#classCount.get(voteIlabel,0)：如果字典键的值中有voteIlabel，则返回0（第二个参数的值）
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	#对classCount进行排序，sroted、items以及itermgetter随后讲解@1
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
	#返回分类结果
	return sortedClassCount[0][0]

def file2matrix(filename):
	"""
	Function：	从文本文件中解析数据

	Args：		filename：文件名称字符串

	Returns：	returnMat：训练样本矩阵
				classLabelVector：类标签向量
	"""	
	#打开文件
	fr = open(filename)
	#得到文件行数
	numberOFLines = len(fr.readlines())
	#创建返回的NumPy矩阵
	returnMat = zeros((numberOFLines, 3))
	#创建返回的向量列表
	classLabelVector = []
	fr = open(filename)
	index = 0
	for line in fr.readlines():
		#使用line.strip()截取掉多有的回车符
		line = line.strip()
		#使用tab字符将上一步得到的整行数据分割成一个元素列表
		listFromLine = line.split('\t')
		#选取前三个元素，存储到特征矩阵中
		returnMat[index, :] = listFromLine[0:3]
		#将列表最后一列存储到向量classLabelVector中
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	#返回训练样本矩阵和类标签向量
	return returnMat, classLabelVector

def autoNorm(dataSet):
	"""
	Function：	归一化特征值

	Args：		dataSet：训练验本矩阵

	Returns：	normDataSet：归一化矩阵
				ranges：每一列的差值
				minVals：每一列的最小值
	"""	
	#求取列的最小值
	minVals = dataSet.min(0)
	#求取列的最大值
	maxVals = dataSet.max(0)
	#最大值与最小值做差
	ranges = maxVals - minVals
	#创建输出矩阵normDataSet
	normDataSet = zeros(shape(dataSet))
	#m设定为矩阵dataSet的行数
	m = dataSet.shape[0]
	#对矩阵dataSet每个元素求差
	normDataSet = dataSet - tile(minVals, (m, 1))
	#对矩阵dataSet每个元素归一化
	normDataSet = normDataSet/tile(ranges, (m, 1))
	#返回归一化矩阵、差值向量和最小值向量
	return normDataSet, ranges, minVals

def datingClassTest():
	"""
	Function：	分类器测试代码

	Args：		无

	Returns：	classifierResult：分类器分类结果
				datingLabels[i]：真实结果
				errorCount：分类误差
	"""	
	#测试集比例设定：10%
	hoRatio = 0.1
	#从文本文件中解析数据
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	#归一化特征值
	normMat, ranges, minVals = autoNorm(datingDataMat)
	#计算normMat矩阵行数并赋值给m
	m = normMat.shape[0]
	#初始化测试向量个数
	numTestVecs = int(m*hoRatio)
	#初始化错误计数
	errorCount = 0.0
	#对测试集分类，返回分类结果并打印
	for i in range(numTestVecs):
		#传参给分类器进行分类，每个for循环改变的参数只有第一项的测试数据而已
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
		#打印当前测试数据的分类结果个真实结果
		print("the classfier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
		#如果分类结果不等于真是结果，错误计数加一
		if (classifierResult != datingLabels[i]): errorCount += 1.0
	#输出测试错误率
	print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
	#输出测试错误数
	print(errorCount)

def classifyPerson():
	"""
	Function：	约会网站测试函数

	Args：		percentTats：玩视频游戏消耗时间百分比
				ffMiles：每年获得的飞行常客里程数
				iceCream：每周消费的冰淇淋公升数

	Returns：	resultList：可交往程度
	"""
	#建立输出列表
	resultList = ['not at all', 'in small doses', 'in large doses']
	#读取键盘输入的数值
	percentTats = float(input("percentage of time spent playing video games?"))
	ffMiles = float(input("frequent flier miles earned per year?"))
	iceCream = float(input("liters of ice cream consumed per week?"))
	#从文本文件中解析数据
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	#归一化特征值
	normMat, ranges, minVals = autoNorm(datingDataMat)
	#将先前读取的键盘输入填入数组
	inArr = array([ffMiles, percentTats, iceCream])
	#分类：这里也对输入数据进行了归一化
	classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
	#打印分类信息
	print("You wil probably like this person: ", resultList[classifierResult - 1])

def img2vector(filename):
	"""
	Function：	32*32图像转换为1*1024向量

	Args：		filename：文件名称字符串

	Returns：	returnVect：转换之后的1*1024向量
	"""	
	#初始化要返回的1*1024向量
	returnVect = zeros((1, 1024))
	#打开文件
	fr = open(filename)
	#读取文件信息
	for i in range(32):
		#循环读取文件的前32行
		lineStr = fr.readline()
		for j in range(32):
			#将每行的头32个字符存储到要返回的向量中
			returnVect[0, 32*i+j] = int(lineStr[j])
	#返回要输出的1*1024向量
	return returnVect

def handwritingClassTest():
	"""
	Function：	手写数字测试程序

	Args：		无

	Returns：	returnVect：转换之后的1*1024向量
	"""	
	#初始化手写数字标签列表
	hwLabels = []
	#获取训练目录信息
	trainingFileList = listdir('trainingDigits')
	#获取训练文件数目
	m = len(trainingFileList)
	#初始化训练矩阵
	trainingMat = zeros((m,1024))
	#开始提取训练集
	for i in range(m):
		#从文件名解析出分类数字
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		#存储解析出的分类数字到标签中
		hwLabels.append(classNumStr)
		#载入图像
		trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
	#获取测试目录信息
	testFileList = listdir('testDigits')
	#初始化错误计数
	errorCount = 0.0
	#获取测试文件数目
	mTest = len(testFileList)
	#开始测试
	for i in range(mTest):
		#从文件名解析出分类数字
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		#载入图像
		vectorUnderTest = img2vector('trainingDigits/%s' % fileNameStr)
		#参数传入分类器进行分类
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		#打印输出分类结果和真实结果
		print("the classifier came back with: %d, the real answer is: %d" %(classifierResult, classNumStr))
		#如果分类结果不等于真实结果，错误计数加一
		if (classifierResult != classNumStr): errorCount += 1.0
	#输出错误技术
	print("\nthe total number of errors is: %d" % errorCount)
	#输出错误率
	print("\nthe total error rate is: %f" % (errorCount/float(mTest)))