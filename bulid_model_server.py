#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''This script is run on server'''

from gensim.models import word2vec
from gensim.models import fasttext
import jieba
import pdb
import os
from pprint import pprint
import time

# print(os.getcwd())

# stopwords = ['（', '）', '(', ')', ' ', '、', '·']

class GetSentences(object):
    def __init__(self, filepath):
        self.filepath = filepath
    
    # 避免全部加载到内存
    def __iter__(self):
        for line in open(self.filepath, 'r'):
            yield line.split('\n')  # 按照空格区分

def getStopwords(filepath):
    '''训练模型的时候不用加载停词文件表，直接使用只去除标点符号的停词'''
    with open(filepath, 'r', encoding='utf-8') as f:
        words = f.read()
        # print(type(words))  # str
        print(words.split('\n'))
        return words.split('\n')

# 分词作为模型训练输入
def parserCompanyName(name_generator, stopwords):
    train_sen = []
    with open('./parser_company_name_qw.txt', 'w') as f:
        for item in name_generator:
            parser_list = jieba.lcut(item[0])
            parser_list = [item for item in parser_list if item not in stopwords]
            train_sen.append(parser_list)
            tem = ''
            for i in parser_list:
                tem = tem + i + ' '
            tem = tem + '\n'
            f.write(tem)
    return train_sen

####  修改版-分词作为模型训练输入
def modifyParserCompanyName(name_generator, stopwords):
    train_sen = []
    # with open('./parser_company_name_qw.txt', 'w') as f:  # Do not write for improving efficiency
    for item in name_generator:
        # 考虑建模的时候就把stopwords先排除掉，就像run_model_server.main_gensim中一样
        sen = item[0]
        for stop in stopwords:
            sen = sen.replace(stop, '')
        parser_list = jieba.lcut(sen)
        parser_list = [item for item in parser_list if item not in stopwords]
        train_sen.append(parser_list)  # [ [], [], [], ... ]
        # tem = ''
        # for i in parser_list:
        #     tem = tem + i + ' '
        # tem = tem + '\n'
        # f.write(tem)
    return train_sen

# 训练模型
def trainModel(train_sen, model_output):      
    # sentences = GetSentences(file_input)
    # yield sentences
    # sentences = list(sentences)
    word2vec_model = word2vec.Word2Vec(train_sen, sg=SG, min_count = MIN_COUNT, workers = CPU_NUM, size = VEC_SIZE, window = CONTEXT_WINDOW)
    word2vec_model.save(model_output)

# 训练模型
def trainModel_fasttext(train_sen, model_output):
    # sentences = GetSentences(file_input)
    # yield sentences
    # sentences = list(sentences)
    word2vec_model = fasttext.FastText(train_sen, sg=SG, min_count = MIN_COUNT, workers = CPU_NUM, size = VEC_SIZE, window = CONTEXT_WINDOW)
    word2vec_model.save(model_output)

# 加载模型
def loadModel(model_path):
    return word2vec.Word2Vec.load(model_path)

# 测试模型
def testModel(model, test_sentence):
    similarity = model.most_similar(test_sentence)
    print(similarity)

def createGenerator(file_input):
    for line in open(file_input,'r'):
        yield line.split()

def readLineFromGenerator(gener):
    count = 1
    for i in gener:
        if count < 10:
            parser_word = jieba.lcut(i[0])
            print(i[0])
            print(parser_word)
            count += 1

# 使用这个函数进行建模
def trigger():
    # dirname = os.getcwd()
    # company_name_file = '\\company_name.txt

    # 加载公司名称文本
    # company_name_file = '../company_name/dw_list_train_qw.txt'
    # company_name_file = '../company_name/dw_list_train_all.txt'
    company_name_file = '../company_name/dw_list_train.txt'
    sentences = GetSentences(company_name_file)
    # readLineFromGenerator(sentences)

    stopwords_file = './stopwords_words.txt'
    stopwords = getStopwords(stopwords_file)

    # 分词 - 作为训练模型的输入
    ##### 初始版训练方法 #####
    # train_sen = parserCompanyName(sentences, stopwords)
    # print(train_sen)
    ##### 修改版训练方法 #####
    train_sen = modifyParserCompanyName(sentences, stopwords)

    # 训练并保存模型
    # word2vec model
    # model_name = './word2vec_model_skip_stopwords_qw_win2.model'
    # model_name = './word2vec_model_skip_stopwords_all_win2.model'
    # model_name = './word2vec_model_skip_stopwords_new.model'
    # model_name = './word2vec_model_skip_stopwords_all_win2_400.model'
    model_name = './word2vec_model_skip_stopwords_bw_win2_100d.model'
    model_output = model_name
    print('Training...')
    trainModel(train_sen, model_output)
    print('Successfully')

    # fasttext mdoel
    # model_name = './word2vec_model_skip_stopwords_fasttext_win3.model'
    # model_output = model_name
    # print('Training...')
    # trainModel_fasttext(train_sen, model_output)
    # print('Successfully')


def countWordFre(filepath):
    '''计算词频并找到Top N高频词'''
    word_fre = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        # print(type(lines))  # list
        # pdb.set_trace()
        for line in lines:
            # print(type(line))  # str
            for item in line.split():
                if item not in word_fre.keys():
                    word_fre[item] = 1
                else:
                    word_fre[item] += 1
            # pdb.set_trace()
        sorted_list = sorted(word_fre.items(), key=lambda items:items[1], reverse=True)
        with open('./high_fre_words.txt', 'w') as fw:
            for index in range(0,50):
                print(sorted_list[index])
                content = sorted_list[index][0] + '-' + str(sorted_list[index][1]) + '\n'
                fw.writelines(content)




if __name__ == '__main__':

    # 训练常量设置

    # # window = 2
    # MIN_COUNT = 0  # 忽略词频小于MIN_COUNT的  
    # CPU_NUM = 12  # CPU核心数
    # # VEC_SIZE = 200  # size - 特征向量维度
    # VEC_SIZE = 400  # size - 特征向量维度
    # CONTEXT_WINDOW = 2  # window - 上下文提取词的最大距离
    # SG = 1  # 1 -> skip-gram; Otherwise, 0: CBOW is used.

    MIN_COUNT = 0  # 忽略词频小于MIN_COUNT的  
    CPU_NUM = 12  # CPU核心数
    VEC_SIZE = 100  # size - 特征向量维度
    CONTEXT_WINDOW = 2  # window - 上下文提取词的最大距离
    SG = 1  # 1 -> skip-gram; Otherwise, CBOW is used.

    # # window = 3
    # MIN_COUNT = 0  # 忽略词频小于MIN_COUNT的  
    # CPU_NUM = 4  # CPU核心数
    # VEC_SIZE = 200  # size - 特征向量维度
    # CONTEXT_WINDOW = 3  # window - 上下文提取词的最大距离
    # SG = 1  # 1 -> skip-gram; Otherwise, CBOW is used.

    # # 停词表
    # stopwords_file = './stopwords.txt'
    # stopwords = getStopwords(stopwords_file)
    # pdb.set_trace()

    # # 词频统计
    # filepath = 'parser_company_name_10000V2.txt'
    # countWordFre(filepath)

    start_time = time.clock()
    # 建模
    trigger()
    end_time = time.clock()
    run_time = start_time - end_time
    print(run_time)
