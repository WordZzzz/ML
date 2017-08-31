# -*- coding: UTF-8 -*-
"""
Created on Aug 18, 2017
Decision Tree Source Code
@author: wordzzzz
"""
from math import log
import operator

def createDataSet():
	"""
	Function：	创建数据集

	Args：		无

	Returns：	dataSet：数据集
				labels：标签
	"""
	#创建数据集
	dataSet = [[1, 1, 'yes'],
				[1, 1, 'yes'],
				[1, 0, 'no'],
				[0, 1, 'no'],
				[0, 1, 'no']]
    #创建标签
	labels = ['no surfacing','flippers']
    #返回创建的数据集和标签
	return dataSet, labels

def calcShannonEnt(dataSet):
	"""
	Function：	计算给定数据集的香农熵

	Args：		dataSet：数据集

	Returns：	shannonEnt：香农熵
	"""
	#计算数据集中实例的总数
	numEntries = len(dataSet)
	#创建一个数据字典
	labelCounts = {}
	#为所有可能的分类创建字典
	for featVec in dataSet:
		#字典的键值等于最后一列的数值
		currentLabel = featVec[-1]
		#如果当前键值不存在，则扩展字典并将当前键值加入字典
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		#每个键值都记录下当前类别出现的次数
		labelCounts[currentLabel] += 1
	#初始化香农熵
	shannonEnt = 0.0
	#计算香农熵
	for key in labelCounts:
		#利用所有类别标签发生频率计算类别出现的概率
		prob = float(labelCounts[key])/numEntries
		#计算香农熵，log(prob, 2)是以2为底求prob的对数
		shannonEnt -=  prob * log(prob, 2)
	#返回香农熵计算结果
	return shannonEnt

def splitDataSet(dataSet, axis, value):
	"""
	Function：	按照给定特征划分数据集

	Args：		dataSet：待划分的数据集
				axis：划分数据集的特征
				value：需要返回的特征的值

	Returns：	retDataSet：符合特征的数据集
	"""	
	#创建新的list对象  
	retDataSet = []
	#抽取数据集
	for featVec in dataSet:
		#将符合特征的数据抽取出来
		if featVec[axis] == value:
			#截取列表中第axis+1个之前的数据
			reducedFeatVec = featVec[:axis]
			#将第axis+2之后的数据接入到上述数据集
			reducedFeatVec.extend(featVec[axis+1:])
			#将处理结果作为列表接入到返回数据集
			retDataSet.append(reducedFeatVec)
	#返回符合特征的数据集
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	"""
	Function：	选择最好的数据集划分方式

	Args：		dataSet：待划分的数据集

	Returns：	bestFeature：划分数据集最好的特征
	"""	
	#初始化特征数量
	numFeatures = len(dataSet[0]) - 1
	#计算原始香农熵
	baseEntropy = calcShannonEnt(dataSet)
	#初始化信息增益和最佳特征
	bestInfoGain = 0.0; bestFeature = -1
	#选出最好的划分数据集的特征
	for i in range(numFeatures):
		#创建唯一的分类标签列表
		featList = [example[i] for example in dataSet]
		#从列表中创建集合，以得到列表中唯一元素值
		uniqueVals = set(featList)
		#初始化香农熵
		newEntropy = 0.0
		#计算每种划分方式的信息熵
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		#得到信息增益
		infoGain = baseEntropy - newEntropy
		#计算最好的信息增益
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	#返回最好的特征
	return bestFeature

def majorityCnt(classList):
	"""
	Function：	决定叶子结点的分类

	Args：		classList：分类列表

	Returns：	sortedClassCount[0][0]：叶子结点分类结果
	"""		
	#创建字典
	classCount={}
	#给字典赋值
	for vote in classList:
		#如果字典中没有该键值，则创建
		if vote not in classCount.keys():
			classCount[vote] = 0
		#为每个键值计数
		classCount[vote] += 1
	#对classCount进行排序
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
	#返回叶子结点分类结果
	return sortedClassCount[0][0]

def createTree(dataSet, labels):
	"""
	Function：	创建树

	Args：		dataSet：数据集
				labels：标签列表

	Returns：	myTree：创建的树的信息
	"""	
	#创建分类列表
	classList = [example[-1] for example in dataSet]
	#类别完全相同则停止划分
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	#遍历完所有特征时返回出现次数最多的类别
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	#选取最好的分类特征
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	#创建字典存储树的信息
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	#得到列表包含的所有属性值
	featValues = [example[bestFeat] for example in dataSet]
	#从列表中创建集合
	uniqueVals = set(featValues)
	#遍历当前选择特征包含的所有属性值
	for value in uniqueVals:
		#复制类标签
		subLabels =labels[:]
		#递归调用函数createTree()，返回值将被插入到字典变量myTree中
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
	#返回字典变量myTree
	return myTree

def classify(inputTree, featLabels, testVec):
	"""
	Function：	使用决策树的分类函数

	Args：		inputTree：树信息
				featLabels：标签列表
				testVec：测试数据

	Returns：	classLabel：分类标签
	"""	
	#第一个关键字为第一次划分数据集的类别标签，附带的取值表示子节点的取值
	firstStr = list(inputTree.keys())[0]
	#新的树，相当于脱了一层皮
	secondDict = inputTree[firstStr]
	#将标签字符串转为索引
	featIndex = featLabels.index(firstStr)
	#遍历整棵树
	for key in secondDict.keys():
		#比较testVec变量中的值与树节点的值
		if testVec[featIndex] == key:
			#判断子节点是否为字典类型，进而得知是否到达叶子结点
			if type(secondDict[key]).__name__=='dict':
				#没到达叶子结点，则递归调用classify()
				classLabel = classify(secondDict[key], featLabels, testVec)
			else:
				#到达叶子结点，则分类结果为当前节点的分类标签
				classLabel = secondDict[key]
	#返回分类标签
	return classLabel

def  storeTree(inputTree, filename):
	"""
	Function：	存储决策树

	Args：		inputTree：树信息
				filename：文件名称

	Returns：	无
	"""	
	#导入模块
	import pickle
	#新建文件，一定要加b属性，否则可能报错：
	#TypeError: write() argument must be str, not bytes
	fw = open(filename, 'wb')
	#写入数据
	pickle.dump(inputTree, fw)
	#关闭文件
	fw.close()

def grabTree(filename):
	"""
	Function：	读取决策树

	Args：		filename：文件名称

	Returns：	pickle.load(fr)：树信息
	"""	
	#导入模块
	import pickle
	#打开文件，写入属性一致，否则可能报错：
	#UnicodeDecodeError: 'gbk' codec can't decode byte 0x80 in position 0: illegal multibyte sequence
	fr = open(filename, 'rb')
	#导出数据
	return pickle.load(fr)
