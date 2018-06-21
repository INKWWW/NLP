#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models import word2vec
import jieba
import pdb
import os
from pprint import pprint
import numpy as np
from decimal import Decimal



# 加载停词
def getStopwords(filepath):
    '''训练模型的时候不用加载停词文件表，直接使用只去除标点符号的停词'''
    with open(filepath, 'r', encoding='utf-8') as f:
        words = f.read()
        # print(type(words))  # str
        # print(words.split('\n'))
        return words.split('\n')

# 加载模型
def loadModel(model_path):
    return word2vec.Word2Vec.load(model_path)

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
    # temp = np.zeros([100,])
    temp = np.zeros([200,])
    for test_word in parser_list:
        try:
            wordVec_new = findWordVec(model, test_word)
            # temp = np.vstack((wordVec_new, temp))
        except KeyError as e:
            # print(test_word)
            wordVec_new = np.ones([200,])
            wordVec_new = wordVec_new*0.0001
        temp = np.vstack((wordVec_new, temp))

    # print(temp)
    # print(wordVec_new)
    # 所有词向量加起来，除以词数得到句向量
    temp_sum = temp.sum(axis=0)
    senVec = temp_sum / num_word
    sentence = ''.join(parser_list)
    # print('--------senVec {}--------'.format(sentence))
    # print(senVec)
    # print(type(senVec))
    return senVec

def cosDist(senVec1, senVec2):
    '''计算cos距离'''
    cos = np.dot(senVec1, senVec2) / (np.linalg.norm(senVec1) * np.linalg.norm(senVec2))
    # normalized_sim = 0.5 + 0.5 * cos
    # print(normalized_sim)
    # return normalized_sim
    # print('------cosDist-------')    
    # print(Decimal(cos))
    return cos

def euclideanDist(senVec1, senVec2):
    '''计算欧几里得距离'''
    euclidean = np.linalg.norm(senVec1 - senVec2)
    # print('------euclideanDist-------')
    # print(euclidean)
    normalized_sim = 1.0 / (euclidean + 1.0)
    # print(normalized_sim)
    return normalized_sim

# 总函数
def operation(model, test_sentence_1, test_sentence_2, distance_model):
    # stopwords_file = './stopwords_words.txt'
    stopwords_file = '/home/hanmo.wang/similarity/NLP/stopwords_words.txt'
    stopwords = getStopwords(stopwords_file)

    parser_list_1 = parserSen(test_sentence_1, stopwords)
    senVec_1 = getSenVec(parser_list_1, model)
    # print('-----------senVec_1-----------')
    # print(senVec_1)

    parser_list_2 = parserSen(test_sentence_2, stopwords)
    senVec_2 = getSenVec(parser_list_2, model)
    # print('-----------senVec2-----------')
    # print(senVec_2)

    # print('---------最终相似度-----------')
    if distance_model == 1:
        similarity = cosDist(senVec_1, senVec_2)
    if distance_model == 2:
        similarity = euclideanDist(senVec_1, senVec_2)
    # print('------------similarity----------')
    # print(similarity)
    return similarity
    

if __name__ == '__main__':

    model_output = './word2vec_model_skip_stopwords_win1.model'

    test_sentence_1 = '北京'
    test_sentence_2 = '北京市'
    
    # 加载模型
    w2v_model = loadModel(model_output)
    distance_model = 2
    operation(w2v_model, test_sentence_1, test_sentence_2, distance_model)
