#!/usr/bin/env python
# -*- coding: utf-8 -*-

# fasttext版本

from gensim.models import fasttext
import jieba
import pdb
import os
from pprint import pprint

# print(os.getcwd())

stopwords = ['（', '）', '(', ')', ' ', '、']

class GetSentences(object):
    def __init__(self, filepath):
        self.filepath = filepath
    
    # 避免全部加载到内存
    def __iter__(self):
        for line in open(self.filepath, 'r'):
            yield line.split()  # 按照空格区分

def getStopwords(filepath):
    '''训练模型的时候不用加载停词文件表，直接使用只去除标点符号的停词'''
    with open(filepath, 'r', encoding='utf-8') as f:
        words = f.read()
        # print(type(words))  # str
        print(words.split('\n'))
        return words.split('\n')

# 分词作为模型训练输入
def parserCompanyName(name_generator):
    train_sen = []
    with open('./parser_company_name_10000V2.txt', 'w') as f:
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

# 训练模型
def trainModel(train_sen, model_output):         
    # sentences = GetSentences(file_input)
    # yield sentences
    # sentences = list(sentences)
    word2vec_model = fasttext.FastText(train_sen, sg=SG, min_count = MIN_COUNT, workers = CPU_NUM, size = VEC_SIZE, window = CONTEXT_WINDOW)
    word2vec_model.save(model_output)

# 加载模型
def loadModel(model_path):
    return fasttext.FastText.load(model_path)

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
    dirname = os.getcwd()
    # company_name_file = '\\company_name.txt

    # 加载公司名称文本
    # company_name_file = '..\\company_name_10000.txt'
    # filepath = dirname +  company_name_file
    filepath = '../company_name_10000.txt'
    sentences = GetSentences(filepath)
    # readLineFromGenerator(sentences)

    # 分词 - 作为训练模型的输入
    train_sen = parserCompanyName(sentences)
    # print(train_sen)

    # 训练并保存模型
    # model_name = '\\word2vec_model.model'  # V1.0
    # model_name = '\\word2vec_model_10000V2.model'  # V2.0 -- 1W词
    model_name = '\\word2vec_model_10000_fasttext_skip.model'  # skip-gram model
    model_output = dirname + model_name
    input_file = '\\parser_company_name.txt'
    file_input = dirname + input_file
    # with open(file_input, 'r') as f:
    #     fread = f.read()
    #     print(fread)  -- str
    trainModel(train_sen, model_output)
    # # 加载模型
    # w2v_model = loadModel(model_output)
    # # 测试模型
    # test_sentence = '哈尔滨久久康华科技有限责任公司'
    # testModel(w2v_model, test_sentence)

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
    # MIN_COUNT = 0  # 忽略词频小于MIN_COUNT的  
    # CPU_NUM = 3  # CPU核心数
    # VEC_SIZE = 100  # size - 特征向量维度
    # CONTEXT_WINDOW = 1  # window - 上下文提取词的最大距离
    # SG = 1  # 1 -> skip-gram; Otherwise, CBOW is used.

    MIN_COUNT = 0  # 忽略词频小于MIN_COUNT的  
    CPU_NUM = 3  # CPU核心数
    VEC_SIZE = 200  # size - 特征向量维度
    CONTEXT_WINDOW = 2  # window - 上下文提取词的最大距离
    SG = 1  # 1 -> skip-gram; Otherwise, CBOW is used.

    # # 停词表
    # stopwords_file = './stopwords.txt'
    # stopwords = getStopwords(stopwords_file)
    # pdb.set_trace()

    # # 词频统计
    # filepath = 'parser_company_name_10000V2.txt'
    # countWordFre(filepath)

    # 建模
    trigger()

    # dirname = os.getcwd()
    # # company_name_file = '\\company_name.txt'
    # company_name_file = '\\company_name_10000.txt'
    # filepath = dirname +  company_name_file

    # # 加载公司名称文本
    # sentences = GetSentences(filepath)
    # # readLineFromGenerator(sentences)
    # # 分词 - 作为训练模型的输入
    # train_sen = parserCompanyName(sentences)
    # # print(train_sen)

    # # 训练并保存模型
    # # model_name = '\\word2vec_model.model'
    # model_name = '\\word2vec_model_10000V2.model'
    # model_output = dirname + model_name
    # input_file = '\\parser_company_name.txt'
    # file_input = dirname + input_file
    # # with open(file_input, 'r') as f:
    # #     fread = f.read()
    # #     print(fread)  -- str
    # trainModel(train_sen, model_output)
    # # # 加载模型
    # # w2v_model = loadModel(model_output)
    # # # 测试模型
    # # test_sentence = '哈尔滨久久康华科技有限责任公司'
    # # testModel(w2v_model, test_sentence)