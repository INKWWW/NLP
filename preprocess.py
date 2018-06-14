#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import copy
import jieba

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
        print('@-----same-----')
        return
    else:
        for item in shorter_sen_copy:
            for item_l in longer_sen_copy:
                if item in item_l:
                    indexs_s = [i for i, v in enumerate(shorter_sen_copy) if v==item]
                    indexs_l = [i for i, v in enumerate(longer_sen_copy) if v==item_l]
                    for index in indexs_s:
                        shorter_sen_copy.pop(index)
                    for index in indexs_l:
                        longer_sen_copy.pop(index)
    if len(shorter_sen_copy) == 0 or len(longer_sen_copy) == 0:
        print('@@-----same-----')
        return
    elif len(shorter_sen_copy) == 1 or len(longer_sen_copy) == 1:
        if shorter_sen.index(shorter_sen_copy[0]) == 0 or longer_sen.index(longer_sen_copy[0]) == 0:
            print('@@@-----same-----')
            return
    else:
        print('-----may not be the same-----')
        return

def evaluation():
    '''计算precision、recall、F1
    normal_pos, normal_neg, detected_pos, detected_neg
    '''
    # Ture Positive

    # Ture Negative

    # False Negative

    # Flase Positive

    # precision
    precision = tp / (tp + fp)

    # recall
    recall = tp / (tp + fn)

    # F1-score
    f1_score = 2 * ((precision * recall) / (precision + recall))

    return (precision, recall, f1_score)


def main():
    filepath = 'stopwords_words.txt'
    stopwords = getStopwords(filepath)

    test_sentence_1 = '黑龙江红兴隆农垦民乐农业生产资料有限公司'
    # test_sentence_2 = '黑龙江红兴隆农垦民乐农业生产资料公司'
    # test_sentence_2 = '黑龙江兴隆农垦民乐生产资料有限公司'
    test_sentence_2 = '四川省红兴隆农垦民乐生产资料有限公司'
    shorter_sen, longer_sen = processSen(test_sentence_1, test_sentence_2, stopwords)
    compare(shorter_sen, longer_sen)


if __name__ == '__main__':
    main()

    
