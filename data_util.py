
# coding=utf-8

import os
import getConfig
import jieba
#结巴是国内的一个分词python库，分词效果非常不错。pip3 install jieba安装

gConfig = {}

gConfig=getConfig.get_config()

conv_path = gConfig['resource_data']
 
if not os.path.exists(conv_path):
	
	exit()

convs = []  # 用于存储对话的列表
with open(conv_path,encoding='utf-8') as f:
	one_conv = []        # 存储一次完整对话
	for line in f:
		line = line.strip('\n').replace('/', '')#去除换行符，并将原文件中已经分词的标记去掉，重新用结巴分词.
		if line == '':
			continue
		if line[0] == gConfig['e']:
			if one_conv:
				convs.append(one_conv)
			one_conv = []
		elif line[0] == gConfig['m']:
			one_conv.append(line.split(' ')[1])#将一次完整的对话存储下来

#1、初始化变量，ask response为List
#2、按照语句的顺序来分为问句和答句，根据行数的奇偶性来判断
#3、在存储语句的时候对语句使用结巴分词，jieba.cut

# 把对话分成问与答两个部分
seq = []        

for conv in convs:
	if len(conv) == 1:
		continue
	if len(conv) % 2 != 0:  # 因为默认是一问一答的，所以需要进行数据的粗裁剪，对话行数要是偶数的
		conv = conv[:-1]
	for i in range(len(conv)):
		if i % 2 == 0:
			conv[i]=" ".join(jieba.cut(conv[i]))
			conv[i+1]=" ".join(jieba.cut(conv[i+1]))
			seq.append(conv[i]+'\t'+conv[i+1])

seq_train = open(gConfig['seq_data'],'w') 

for i in range(len(seq)):
   seq_train.write(seq[i]+'\n')
 
   if i % 1000 == 0:
      print(len(range(len(seq))), '处理进度：', i)
 
seq_train.close()



