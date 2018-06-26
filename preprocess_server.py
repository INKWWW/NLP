#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''直接匹配算法'''

import os
import copy
import jieba
import csv

dirname = print(os.getcwd())

# 加载停词
def getStopwords(filepath):
    '''训练模型的时候不用加载停词文件表，直接使用只去除标点符号的停词'''
    with open(filepath, 'r', encoding='utf-8') as f:
        words = f.read()
        # print(type(words))  # str
        print(words.split('\n'))
        return words.split('\n')

def processSen(test_sentence_1, test_sentence_2, stopwords):
    sen_1_parser = jieba.lcut(test_sentence_1)
    sen_1_parser = [item for item in sen_1_parser if item not in stopwords]
    sen_2_parser = jieba.lcut(test_sentence_2)
    sen_2_parser = [item for item in sen_2_parser if item not in stopwords]
    if len(test_sentence_1) > len(test_sentence_2):
        longer_sen = sen_1_parser
        shorter_sen = sen_2_parser 
    elif len(test_sentence_1) < len(test_sentence_2):
        longer_sen = sen_2_parser
        shorter_sen = sen_1_parser
    else:
        longer_sen = sen_1_parser
        shorter_sen = sen_2_parser
    return (shorter_sen, longer_sen)

def compare(shorter_sen, longer_sen):
    shorter_sen_copy = copy.deepcopy(shorter_sen)
    longer_sen_copy = copy.deepcopy(longer_sen)
    for item in shorter_sen:
        if item in longer_sen:
            indexs_s = [i for i, v in enumerate(shorter_sen_copy) if v==item]
            indexs_l = [i for i, v in enumerate(longer_sen_copy) if v==item]
            for index in indexs_s:
                shorter_sen_copy.pop(index)
            for index in indexs_l:
                longer_sen_copy.pop(index)           
        else:
            continue
    if len(shorter_sen_copy) == 0:
        # print('@-----same-----')
        return True
    else:
        for item in shorter_sen_copy:
            for item_l in longer_sen_copy:
                if item in item_l:
                    indexs_s = [i for i, v in enumerate(shorter_sen_copy) if v in item]
                    indexs_l = [i for i, v in enumerate(longer_sen_copy) if v in item_l]
                    for index in indexs_s:
                        shorter_sen_copy.pop(index)
                    for index in indexs_l:
                        longer_sen_copy.pop(index)
    if len(shorter_sen_copy) == 0 or len(longer_sen_copy) == 0:
        # print('@@-----same-----')
        return True
    # elif len(shorter_sen_copy) == 1 or len(longer_sen_copy) == 1:
    #     if shorter_sen.index(shorter_sen_copy[0]) == 0 or longer_sen.index(longer_sen_copy[0]) == 0:
    #         # print('@@@-----same-----')
    #         return True
    else:
        # print('-----may not be the same-----')
        return False

def predict(input_file, out_file, stopwords):    
    base_name = []
    input_name = []
    base_result = []
    predict_result = []
    with open(input_file, 'r') as f:
        fread = f.read()
        lines = fread.split()
        for line in lines:
            line_split = line.split(',')
            base_name.append(line_split[0])
            input_name.append(line_split[1])
            base_result.append(line_split[2])       
            
    length = len(input_name)

    with open(out_file, 'w') as fw:
        for i in range(0, length):
            test_sentence_1 = base_name[i]
            test_sentence_2 = input_name[i]

            shorter_sen, longer_sen = processSen(test_sentence_1, test_sentence_2, stopwords)
            result = compare(shorter_sen, longer_sen)

            if result:
                predict_result[i].append('1')
            else:
                predict_result[i].append('0')
            
            content = test_sentence_1 + ',' + test_sentence_2 + ',' + base_result[i] + ',' + predict_result[i] + ',' + str(result) + '\n'
            fw.write(content)
    return base_result, predict_result         


def main():
    filepath = 'stopwords_words.txt'
    stopwords = getStopwords(filepath)

    # test_sentence_1 = '黑龙江红兴隆农垦民乐农业生产资料有限公司'
    # # test_sentence_2 = '黑龙江红兴隆农垦民乐农业生产资料公司'
    # # test_sentence_2 = '黑龙江兴隆农垦民乐生产资料有限公司'
    # test_sentence_2 = '四川省红兴隆农垦民乐生产资料有限公司'
    shorter_sen, longer_sen = processSen(test_sentence_1, test_sentence_2, stopwords)
    compare(shorter_sen, longer_sen)


if __name__ == '__main__':
    main()

   