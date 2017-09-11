# -*- coding: UTF-8 -*-
"""
Created on Sep 18, 2017
kNN: k Nearest Neighbors
@author: wordzzzz
"""

from numpy import *
import codecs

def loadDataSet():
	"""
	Function：	创建实验样本

	Args：		无

	Returns：	postingList：词条切分后的文档集合
				classVec：类别标签的集合
	"""	
	#词条切分后的文档集合
	postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
				['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	#类别标签的集合
	classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
	#词条切分后的文档集合和类别标签结合
	return postingList, classVec

def createVocabList(dataSet):
	"""
	Function：	创建一个包含所有文档中出现的不重复词的列表

	Args：		dataSet：数据集

	Returns：	list(vocabSet)：返回一个包含所有文档中出现的不重复词的列表
	"""
	#创建一个空集
	vocabSet = set([])
	#将新词集合添加到创建的集合中
	for document in  dataSet:
		#操作符 | 用于求两个集合的并集
		vocabSet = vocabSet | set(document)
	#返回一个包含所有文档中出现的不重复词的列表
	return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
	"""
	Function：	词表到向量的转换

	Args：		vocabList：词汇表
				inputSet：某个文档

	Returns：	returnVec：文档向量
	"""
	#创建一个所含元素都为0的向量
	returnVec = [0]*len(vocabList)
	#遍历文档中词汇
	for word in inputSet:
		#如果文档中的单词在词汇表中，则相应向量位置置1
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		#否则输出打印信息
		else: print("the word: %s is not in my Vocablary!" % word)
	#向量的每一个元素为1或0，表示词汇表中的单词在文档中是否出现
	return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
	"""
	Function：	词袋到向量的转换

	Args：		vocabList：词袋
				inputSet：某个文档

	Returns：	returnVec：文档向量
	"""
	#创建一个所含元素都为0的向量
	returnVec = [0]*len(vocabList)
	#将新词集合添加到创建的集合中
	for word in inputSet:
		#如果文档中的单词在词汇表中，则相应向量位置加1
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
	#返回一个包含所有文档中出现的词的列表
	return returnVec

def trainNB0(trainMatrix, trainCategory):
	"""
	Function：	朴素贝叶斯分类器训练函数

	Args：		trainMatrix：文档矩阵
				trainCategory：类别标签向量

	Returns：	p0Vect：非侮辱性词汇概率向量
				p1Vect：侮辱性词汇概率向量
				pAbusive：侮辱性文档概率
	"""
	#获得训练集中文档个数
	numTrainDocs = len(trainMatrix)
	#获得训练集中单词个数
	numWords = len(trainMatrix[0])
	#计算文档属于侮辱性文档的概率
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	#初始化概率的分子变量，分母变量
	#p0Num = zeros(numWords); p1Num = zeros(numWords)
	#p0Denom = 0.0; p1Denom = 0.0
	#为了避免概率为0的出现影响最终结果，将所有词初始化为1，分母初始化为2
	#初始化概率的分子变量，分母变量
	p0Num = ones(numWords); p1Num = ones(numWords)
	p0Denom = 2.0; p1Denom = 2.0
	#遍历训练集trainMatrix中所有文档
	for i in range(numTrainDocs):
		#如果侮辱性词汇出现，则侮辱词汇计数加一，且文档的总词数加一
		if trainCategory[i] ==1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		#如果非侮辱性词汇出现，则非侮辱词汇计数加一，且文档的总词数加一
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	#对每个元素做除法求概率
	p1Vect = p1Num/p1Denom
	p0Vect = p0Num/p0Denom
	#对每个元素做除法求概率，为了避免下溢出的影响，对计算结果取自然对数
	p1Vect = log(p1Num/p1Denom)
	p0Vect = log(p0Num/p0Denom)
	#返回两个类别概率向量和一个概率
	return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	"""
	Function：	朴素贝叶斯分类函数

	Args：		vec2Classify：文档矩阵
				p0Vec：非侮辱性词汇概率向量
				p1Vec：侮辱性词汇概率向量
				pClass1：侮辱性文档概率

	Returns：	1：侮辱性文档
				0：非侮辱性文档
	"""
	#向量元素相乘后求和再加到类别的对数概率上，等价于概率相乘
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	#分类结果
	if p1 > p0:
		return 1
	else:
		return 0

def testingNB():
	"""
	Function：	朴素贝叶斯分类器测试函数

	Args：		无

	Returns：	testEntry：测试词汇列表
				classifyNB(thisDoc, p0V, p1V, pAb)：分类结果
	"""
	#从预先加载中调入数据
	listOPosts, listClasses = loadDataSet()
	#构建一个包含所有词的列表
	myVocabList = createVocabList(listOPosts)
	#初始化训练数据列表
	trainMat = []
	#填充训练数据列表
	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
	#训练
	p0V, p1V, pAb = trainNB0(trainMat, listClasses)
	#测试
	testEntry = ['love', 'my', 'dalmation']
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print(testEntry,'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
	#测试
	testEntry = ['stupid', 'garbage']
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print(testEntry,'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):
	"""
	Function：	切分文本

	Args：		bigString：输入字符串

	Returns：	[*]：切分后的字符串列表
	"""
	import re
	#利用正则表达式，来切分句子，其中分隔符是除单词、数字外的任意字符串

	listOfTokens = re.split(r'\W*', bigString)
	#返回切分后的字符串列表
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]
 
def spamTest():
	"""
	Function：	贝叶斯垃圾邮件分类器

	Args：		无

	Returns：	float(errorCount)/len(testSet)：错误率
				vocabList：词汇表
				fullText：文档中全部单词
	"""
	#初始化数据列表
	docList = []; classList = []; fullText = []
	#导入文本文件
	for i in range(1, 26):
		#切分文本
		wordList = textParse(open('email/spam/%d.txt' % i).read())
		#切分后的文本以原始列表形式加入文档列表
		docList.append(wordList)
		#切分后的文本直接合并到词汇列表
		fullText.extend(wordList)
		#标签列表更新
		classList.append(1)
		#切分文本
		#print('i = :', i)
		wordList = textParse(open('email/ham/%d.txt' % i).read())
		#切分后的文本以原始列表形式加入文档列表
		docList.append(wordList)
		#切分后的文本直接合并到词汇列表
		fullText.extend(wordList)
		#标签列表更新
		classList.append(0)
	#创建一个包含所有文档中出现的不重复词的列表
	vocabList = createVocabList(docList)
	#初始化训练集和测试集列表
	trainingSet = list(range(50)); testSet = []
	#随机构建测试集，随机选取十个样本作为测试样本，并从训练样本中剔除
	for i in range(10):
		#随机得到Index
		randIndex = int(random.uniform(0, len(trainingSet)))
		#将该样本加入测试集中
		testSet.append(trainingSet[randIndex])
		#同时将该样本从训练集中剔除
		del(trainingSet[randIndex])
	#初始化训练集数据列表和标签列表
	trainMat = []; trainClasses = []
	#遍历训练集
	for docIndex in trainingSet:
		#词表转换到向量，并加入到训练数据列表中
		trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
		#相应的标签也加入训练标签列表中
		trainClasses.append(classList[docIndex])
	#朴素贝叶斯分类器训练函数
	p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
	#初始化错误计数
	errorCount = 0
	#遍历测试集进行测试
	for docIndex in testSet:
		#词表转换到向量
		wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
		#判断分类结果与原标签是否一致
		if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
			#如果不一致则错误计数加1
			errorCount += 1
			#并且输出出错的文档
			print("classification error",docList[docIndex])
	#打印输出信息
	print('the erroe rate is: ', float(errorCount)/len(testSet))
	#返回词汇表和全部单词列表
	#return vocabList, fullText

def calcMostFreq(vocabList, fullText):
	"""
	Function：	计算出现频率

	Args：		vocabList：词汇表
				fullText：全部词汇

	Returns：	sortedFreq[:30]：出现频率最高的30个词
	"""
	import operator
	freqDict = {}
	for token in vocabList:
		freqDict[token] = fullText.count(token)
	sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
	return sortedFreq[:30]

def localWords(feed1, feed0):
	"""
	Function：	RSS源分类器

	Args：		feed1：RSS源
				feed0：RSS源

	Returns：	vocabList：词汇表
				p0V：类别概率向量
				p1V：类别概率向量
	"""
	import feedparser
	#初始化数据列表
	docList = []; classList = []; fullText = []
	minLen = min(len(feed1['entries']), len(feed0['entries']))
	#导入文本文件
	for i in range(minLen):
		#切分文本
		wordList = textParse(feed1['entries'][i]['summary'])
		#切分后的文本以原始列表形式加入文档列表
		docList.append(wordList)
		#切分后的文本直接合并到词汇列表
		fullText.extend(wordList)
		#标签列表更新
		classList.append(1)
		#切分文本
		wordList = textParse(feed0['entries'][i]['summary'])
		#切分后的文本以原始列表形式加入文档列表
		docList.append(wordList)
		#切分后的文本直接合并到词汇列表
		fullText.extend(wordList)
		#标签列表更新
		classList.append(0)
	#获得词汇表
	vocabList = createVocabList(docList)
	#获得30个频率最高的词汇
	top30Words = calcMostFreq(vocabList, fullText)
	#去掉出现次数最高的那些词
	for pairW in top30Words:
		if pairW[0] in vocabList: vocabList.remove(pairW[0])
	trainingSet = list(range(2*minLen)); testSet = []
	#随机构建测试集，随机选取二十个样本作为测试样本，并从训练样本中剔除
	for i in range(20):
		#随机得到Index
		randIndex = int(random.uniform(0, len(trainingSet)))
		#将该样本加入测试集中
		testSet.append(trainingSet[randIndex])
		#同时将该样本从训练集中剔除
		del(trainingSet[randIndex])
	#初始化训练集数据列表和标签列表
	trainMat = []; trainClasses = []
	#遍历训练集
	for docIndex in trainingSet:
		#词表转换到向量，并加入到训练数据列表中
		trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
		#相应的标签也加入训练标签列表中
		trainClasses.append(classList[docIndex])
	#朴素贝叶斯分类器训练函数
	p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
	#初始化错误计数
	errorCount = 0
	#遍历测试集进行测试
	for docIndex in testSet:
		#词表转换到向量
		wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
		#判断分类结果与原标签是否一致
		if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
			#如果不一致则错误计数加1
			errorCount += 1
	#打印输出信息
	print('the erroe rate is: ', float(errorCount)/len(testSet))
	#返回词汇表和两个类别概率向量
	return vocabList, p0V, p1V

def getTopWords(ny, sf):
	"""
	Function：	最具表征性的词汇显示函数

	Args：		ny：RSS源
				sf：RSS源

	Returns：	打印信息
	"""
	import operator
	#RSS源分类器返回概率
	vocabList, p0V, p1V=localWords(ny, sf)
	#初始化列表
	topNY = []; topSF = []
	#设定阈值，返回大于阈值的所有词，如果输出信息太多就提高阈值
	for i in range(len(p0V)):
		if p0V[i] > -4.5 : topSF.append((vocabList[i], p0V[i]))
		if p1V[i] > -4.5 : topNY.append((vocabList[i], p1V[i]))
	#排序
	sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
	print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
	#打印
	for item in sortedSF:
		print(item[0])
	#排序
	sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
	print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
	#打印
	for item in sortedNY:
		print(item[0])