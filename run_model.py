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
    # print('--------senVec--------')
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
    stopwords_file = 'C:\\Wang Hanmo\\projects\\similarity\\company_similarity\\stopwords_words.txt'
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

def main(model_output, test_sentence_1, test_sentence_2):
    stopwords_file = 'C:\\Wang Hanmo\\projects\\similarity\\company_similarity\\stopwords_words.txt'
    stopwords = getStopwords(stopwords_file)

    w2v_model = loadModel(model_output)
    split_sen = jieba.lcut(test_sentence_1)
    split_sen_1 = [item for item in split_sen if (item not in stopwords) and (item in w2v_model.wv.vocab)]
    split_sen = jieba.lcut(test_sentence_2)
    split_sen_2 = [item for item in split_sen if (item not in stopwords) and (item in w2v_model.wv.vocab)]

    similarity = w2v_model.wv.n_similarity(split_sen_1, split_sen_2)
    print(similarity)
    return similarity


if __name__ == '__main__':
    dirname = os.getcwd()
    # model_name = '\\word2vec_model.model'
    # model_name = '\\word2vec_model_10000.model'
    model_name = '\\word2vec_model_10000_skip_stopwords.model'
    # model_name = '\\word2vec_model_10000_skip.model'
    model_output = dirname + model_name

    test_sentence_1 = '中山市恒日传播策划有限公司'
    # test_sentence_2 = '中山市恒日传播有限公司'

    # test_sentence_1 = '哈尔滨久久华康科技有限公司'
    # test_sentence_2 = '哈尔滨久久华康有限公司'  # cos：0.9999999999999998  .0  enclidean：0.10381817865051336  V2:0.1393186701868827
    # test_sentence_2 = '哈尔滨长久华康科技有限公司'  # cos：1.0  0.0  enclidean：0.0  V2:0.0
    # test_sentence_2 = '北京市百融金服科技有限公司'  # cos：0.9999839620230886  0.043129391793413574  enclidean：0.027304886398964206  V2:0.030312174677792212
    # test_sentence_2 = '北京市百融金服科技公司'  # cos：0.9999839620230886  0.043129391793413574  enclidean：0.027304886398964206  V2:0.11785001169482552
    # test_sentence_2 = '四川省百融金服科技有限公司'  # cos：  0.07568612905089231  enclidean：0.0520855644828885  V2:0.12377131385837323  0.02897386663929932
    # test_sentence_2 = '四川省百融金服有限公司'  # cos：  0.07568612905089231  enclidean：0.16771822181146087  V2:0.2934710911310562  0.1832938640796101

    # # 第一组测试
    # test_sentence_1 = '哈尔滨久久华康科技有限公司'
    # test_sentence_2 = '哈尔滨久久华康有限公司'  # 0.0
    # test_sentence_2 = '哈尔滨长久华康科技有限公司'  # 0.022203229313000852   0.0
    # test_sentence_2 = '哈尔滨长久康华科技有限公司'  # 0.05328034558352251  0.007523175114716447  0.008065828246018075

    # test_sentence_2 = '黑龙江长久康华科技有限公司'  # 0.042524020642892055  0.07734101603986997
    # test_sentence_2 = '黑龙江久久康华科技有限公司'  # 0.030177443023447763  0.07734101603986997
    # test_sentence_2 = '黑龙江久久华康科技有限公司'  # 0.11105165879285421  0.10335089333917838

    # test_sentence_2 = '北京市百融金服科技有限公司'  # 0.02750834860840494  0.017958557949121177  0.013522223661126041
    # test_sentence_2 = '北京市百融金服科技公司'  # 0.02750834860840494  0.017958557949121177  
    # test_sentence_2 = '四川省百融金服科技有限公司'  # 0.09338489938119292  0.10301095735634219  0.014167722860191728
    # # test_sentence_2 = '四川省百融金服有限公司'  # 0.09338489938119292  0.10301095735634219  0.014167722860191728

    # 第二组测试
    # test_sentence_1 = '中兴通讯'
    # test_sentence_2 = '西安中兴新软件有限责任公司'
    # test_sentence_2 = '天津之海能源发展股份有限公司'
    # test_sentence_2 = '北京市红兴隆农垦民乐农业生产资料有限公司'  # 0.0
    # test_sentence_2 = '黑龙江红兴隆农垦民乐农业有限公司'  # 0.037899128553774  0.0
    # test_sentence_2 = '黑龙江红兴隆农垦民乐有限公司'  # 0.10465808393829548  0.15634690247603167  0.022769100045612813
    # test_sentence_2 = '黑龙江红兴隆农垦民乐生产资料有限公司'  # 0.12483684740017804  0.15634690247603167  0.022769100045612813
    # test_sentence_2 = '黑龙江红隆农垦民乐农业有限公司'  # 0.09307739644613543  0.08167516464292726  0.005218276287039758
    # test_sentence_2 = '黑龙江红兴隆农垦民乐农业有限公司'  # 0.0
    # test_sentence_2 = '黑龙江兴隆农垦民乐农业生产资料有限公司'

    # 第三组测试
    # test_sentence_1 = '蚌埠悠然酒店管理有限公司'
    # test_sentence_2 = '四川省蚌埠悠然酒店管理有限公司'
    # test_sentence_2 = '安徽省蚌埠华康悠然酒店管理公司'
    # test_sentence_2 = '安徽省蚌埠悠然酒店管理有限公司'  # 0.9777817058322253
    # test_sentence_2 = '安徽省蚌埠四凤悠然酒店管理有限公司'  # 0.011131538396269902
    # test_sentence_1 = '四川省'
    # test_sentence_2 = '安徽省美丽' # 0.3025576495178356
    # test_sentence_2 = '安徽省'  # 0.3025576495178356
    # test_sentence_1 = '有限公司'
    # test_sentence_2 = '有限公司'  # 
    # test_sentence_1 = '中医院'
    # test_sentence_2 = '中医医院'
    # test_sentence_1 = '黑龙江'
    # test_sentence_2 = '黑龙江省'
    # test_sentence_1 = '蚌埠'
    # test_sentence_2 = '安徽省'

    # 加载模型
    w2v_model = loadModel(model_output)
    # operation(w2v_model, test_sentence_1, test_sentence_2)
    main(model_output, test_sentence_1, test_sentence_2)

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
 