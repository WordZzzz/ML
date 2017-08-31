# -*- coding: UTF-8 -*-
"""
Created on Aug 31, 2017
Test on the modules
@author: wordzzzz
"""
import trees
import treePlotter

def main():
	"""
	Function：	主函数

	Args：		无

	Returns：	无
	"""
	#打开文件
	fr = open('lenses.txt')
	#读取文件信息
	lenses = [inst.strip().split('\t') for inst in fr.readlines()]
	#定义标签
	lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
	#创建树
	lensesTree = trees.createTree(lenses, lensesLabels)
	#打印树信息
	print(lensesTree)
	#绘制树信息
	treePlotter.createPlot(lensesTree)

if __name__ == "__main__":
	main()
