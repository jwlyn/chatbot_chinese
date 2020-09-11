# 计算前100、1k、1w项的BLEU值

import execute
import jieba
from nltk.translate.bleu_score import sentence_bleu
import time

count = 0
question = ''   # 问题
_answer = ''    # 参考答案
answer = ''     # 机器人的回答

reference = []  # BLEU参考内容
candidate = []  # 聊天机器人返回的内容

# 分别计算1-gram 2-gram 3-gram 4-gram
score_total1 = 0
score_total2 = 0
score_total3 = 0
score_total4 = 0
i = 0
with open('train_data/xiaohuangji50w_nofenci.conv', 'r') as f:
    start_time = time.time()
    
    # 更改判断条件即可选择测试的样本数
    while count < 1000:
        # 读取数据集
        f.readline()  # 读取E无效信息
        question = f.readline()  # 读取问题
        question = question[2:]  # 去掉问题的前缀
        _answer = f.readline()   # 读取参考答案
        _answer = _answer[2:]    # 去前缀

        # 分词
        question_fenci = ' '.join(jieba.cut(question))
        _answer_fenci = ' '.join(jieba.cut(_answer))

        # 与机器人聊天
        print('--------------------------------')
        print('question_fenci: ' + str(question_fenci))
        answer = execute.predict(question_fenci)

        # 答案分词
        answer_fenci = ' '.join(jieba.cut(answer))
        print('_answer_fenci: ' + str(_answer_fenci))
        print('answer_fenci: ' + str(answer_fenci))

        # 计算BLEU
        reference.append(_answer_fenci.split())
        candidate = (answer_fenci.split())
        score1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        score2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
        score3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
        score4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
        reference.clear()
        print('Cumulate 1-gram :%f' % score1)
        print('Cumulate 2-gram :%f' % score2)
        print('Cumulate 3-gram :%f' % score3)
        print('Cumulate 4-gram :%f' % score4)
        score_total1 += score1
        score_total2 += score2
        score_total3 += score3
        score_total4 += score4
        count += 1
        print('count：' + str(count) + ' score: ' + str(score1))
        print('count：' + str(count) + ' score: ' + str(score2))
        print('count：' + str(count) + ' score: ' + str(score3))
        print('count：' + str(count) + ' score: ' + str(score4))
        print('--------------------------------')

print('最终结果')
print('测试耗时：' + str(time.time() - start_time))
print('count: ' + str(count))
print('--------------------------------')
print('score_tatal1: ' + str(score_total1))
print('BLEU 1-gram: ' + str(score_total1 / count))
print('--------------------------------')
print('score_tatal2: ' + str(score_total2))
print('BLEU 2-gram: ' + str(score_total2 / count))
print('--------------------------------')
print('score_tatal3: ' + str(score_total3))
print('BLEU 3-gram: ' + str(score_total3 / count))
print('--------------------------------')
print('score_tatal4: ' + str(score_total4))
print('BLEU 4-gram: ' + str(score_total4 / count))

