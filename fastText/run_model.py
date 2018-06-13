#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models import fasttext
import jieba
import pdb
import os
from pprint import pprint
import numpy as np


# 加载停词
def getStopwords(filepath):
    '''训练模型的时候不用加载停词文件表，直接使用只去除标点符号的停词'''
    with open(filepath, 'r', encoding='utf-8') as f:
        words = f.read()
        # print(type(words))  # str
        print(words.split('\n'))
        return words.split('\n')

# 加载模型
def loadModel(model_path):
    return fasttext.FastText.load(model_path)

# 测试模型
def findWordVec(model, test_word):
    similarity = model.most_similar(test_word)
    # print(test_word + ':')
    # print(similarity)
    # word_vec = model.wv
    # print('-------------------')
    wordVec = model[test_word]
    # print(type(wordVec))  # numpy.ndarray
    # print(wordVec)
    # print(wordVec.sum(axis=0))
    # print(wordVec.shape)  # (100,) - 一维矩阵，显示的是矩阵的长度
    # print(wordVec.ndim)
    return wordVec

def parserSen(test_sentence, stopwords):
    '''分词'''
    parser_list = jieba.lcut(test_sentence)
    parser_list = [item for item in parser_list if item not in stopwords]
    return parser_list

def getSenVec(parser_list, model):
    '''由词得到句向量'''
    num_word = len(parser_list)
    temp = np.zeros([100,])
    for test_word in parser_list:
        try:
            wordVec_new = findWordVec(model, test_word)
            temp = np.vstack((wordVec_new, temp))
        except KeyError as e:
            wordVec_new = np.zeros((100,))
    # print(temp)
    # print(wordVec_new)
    # 所有词向量加起来，除以词数得到句向量
    temp_sum = temp.sum(axis=0)
    senVec = temp_sum / num_word
    # print('--------senVec--------')
    # print(senVec)
    # print(type(senVec))
    return senVec

def cosDist(senVec1, senVec2):
    '''计算cos距离'''
    cos = np.dot(senVec1.T, senVec2) / (np.linalg.norm(senVec1) * np.linalg.norm(senVec2))
    normalized_sim = 0.5 + 0.5 * cos
    print('------cosDist-------')
    print(normalized_sim)
    return normalized_sim

def euclideanDist(senVec1, senVec2):
    '''计算欧几里得距离'''
    euclidean = np.linalg.norm(senVec1 - senVec2)
    print('------euclideanDist-------')
    print(euclidean)
    # normalized_sim = 1.0 / (euclidean + 1.0)

# 总函数
def operation(model, test_sentence_1, test_sentence_2):
    stopwords_file = '../stopwords_words.txt'
    stopwords = getStopwords(stopwords_file)

    parser_list_1 = parserSen(test_sentence_1, stopwords)
    senVec_1 = getSenVec(parser_list_1, model)

    parser_list_2 = parserSen(test_sentence_2, stopwords)
    senVec_2 = getSenVec(parser_list_2, model)

    print('---------最终相似度-----------')
    # cosDist(senVec_1, senVec_2)
    euclideanDist(senVec_1, senVec_2)
    

if __name__ == '__main__':
    dirname = os.getcwd()
    # model_name = '\\word2vec_model.model'
    # model_name = '\\word2vec_model_10000.model'
    # model_name = '\\word2vec_model_10000V2.model'
    model_name = '\\word2vec_model_10000_fasttext_skip.model'
    model_output = dirname + model_name

    test_sentence_1 = '哈尔滨久久华康科技有限公司'
    # test_sentence_2 = '哈尔滨久久华康有限公司'  # 0.0
    # test_sentence_2 = '哈尔滨长久华康科技有限公司'  # 0.022203229313000852
    # test_sentence_2 = '哈尔滨长久康华科技有限公司'  # 0.05328034558352251
    # test_sentence_2 = '黑龙江长久康华科技有限公司'  # 0.042524020642892055
    test_sentence_2 = '黑龙江久久康华科技有限公司'  # 0.030177443023447763
    # test_sentence_2 = '北京市百融金服科技有限公司'  # 0.02750834860840494
    # test_sentence_2 = '北京市百融金服科技公司'  # 0.02750834860840494
    # test_sentence_2 = '四川省百融金服科技有限公司'  # 0.09338489938119292
    # test_sentence_2 = '四川省百融金服有限公司'  # 0.09338489938119292

    # 加载模型
    w2v_model = loadModel(model_output)

    operation(w2v_model, test_sentence_1, test_sentence_2)

    # 分词
    # test_sentence = '哈尔滨久久华康科技有限公司'
    # parser_list = parserSen(test_sentence)

    # # 测试模型
    # # test_word = ['哈尔滨久久华康科技有限公司']
    # test_word = '哈尔滨'
    # findWordVec(w2v_model, test_word)
    # test_word = '公司'  # -7.354347
    # test_word = '科技'  # -4.7045946
    # test_word = '哈尔滨'  # -1.788831
    # test_word = '康华'  # -0.049786568
    # test_word = '有限公司'  # -11.7632475
    # test_word = '北京'  # -6.831831
    # test_word = '四川'  # -3.8586986
    # test_word = '四川省'  # -0.9724991
    # test_word = '北京市'  # -1.9034648
    # findWordVec(w2v_model, test_word)
 