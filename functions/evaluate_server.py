#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pdb
from decimal import Decimal

# sys.path.append('C:\\Wang Hanmo\\projects\\similarity\\company_similarity')
sys.path.append("..")
sys.path.append("/home/hanmo.wang/similarity/NLP/") # 把这个路径添加进去，防止找不到./stopwords_words.txt
print(sys.path)
# pdb.set_trace()
import run_model_server
import preprocess_server


def loadModel():
    # model_output = '../word2vec_model_10000_skip.model'
    model_output = '../word2vec_model_skip_stopwords_win3.model'    
    return run_model_server.loadModel(model_output)


def predict_word2vec(filepath, model, resultpath, distance_model):
    base_name = []
    input_name = []
    base_result = []
    predict_result = []
    with open(filepath, 'r', encoding='gbk') as f:
        fread = f.read()
        lines = fread.split()
        for line in lines:
            line_split = line.split(',')
            base_name.append(line_split[1])
            input_name.append(line_split[0])
            base_result.append(line_split[2])      
            
    length = len(input_name)

    with open(resultpath, 'w') as fw:
        for i in range(0, length):
            test_sentence_1 = base_name[i]
            test_sentence_2 = input_name[i]
            # print('@@@: {},{}'.format(test_sentence_1, test_sentence_2))
            similarity = run_model_server.operation(model, test_sentence_1, test_sentence_2, distance_model)
            # similarity = Decimal(similarity)
            # print(similarity)
            # pdb.set_trace()

            ##### 针对euclidean Distance
            if similarity > 0.45:
                predict_result.append('1')
            else:
                predict_result.append('0')
            content = test_sentence_1 + ',' + test_sentence_2 + ',' + base_result[i] + ',' + predict_result[i] + ',' + str(similarity) + '\n'
            fw.write(content)
    return base_result, predict_result


def predict_strMatch(inputfile, outputfile, stopwords):
    base_name = []
    input_name = []
    base_result = []
    predict_result = []
    with open(inputfile, 'r', encoding='gbk') as f:
        fread = f.read()
        lines = fread.split()
        for line in lines:
            line_split = line.split(',')
            base_name.append(line_split[1])
            input_name.append(line_split[0])
            base_result.append(line_split[2])       
            
    length = len(input_name)

    with open(outputfile, 'w') as fw:
        for i in range(0, length):
            test_sentence_1 = base_name[i]
            test_sentence_2 = input_name[i]

            shorter_sen, longer_sen = preprocess_server.processSen(test_sentence_1, test_sentence_2, stopwords)
            result = preprocess_server.compare(shorter_sen, longer_sen)

            if result:
                predict_result.append('1')
            else:
                predict_result.append('0')
            
            content = test_sentence_1 + ',' + test_sentence_2 + ',' + base_result[i] + ',' + predict_result[i] + ',' + str(result) + '\n'
            fw.write(content)
    return base_result, predict_result


def evaluation(base_result, predict_result):
    '''计算precision、recall、F1
    normal_pos, normal_neg, detected_pos, detected_neg
    '''
    all_num = len(base_result)
    acc = 0
    for j in range(0, all_num):
        if base_result[j] == predict_result[j]:
            acc += 1
    accuracy = acc / all_num
    print('-------accuracy--------: {}'.format(accuracy))

    tp, fn, tn, fp = 0, 0, 0, 0
    length = len(base_result)
    for i in range(0, length):
        if base_result[i] == '0' and predict_result[i] == '0':
            tn += 1
        elif base_result[i] == '1' and predict_result[i] == '1':
            tp += 1
        elif base_result[i] == '1' and predict_result[i] == '0':
            fn += 1
        else:
            fp += 1
    # precision
    precision = tp / (tp + fp)
    print('-------precision--------: {}'.format(precision))
    # recall
    recall = tp / (tp + fn)
    print('-------recall--------: {}'.format(recall))
    # F1-score
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print('-------f1_score--------: {}'.format(f1_score))
    return (precision, recall, f1_score)

#######################################################################################
# 整合算法

def predict_agg(inputfile, outputfile, stopwords, model):
    '''首先直接匹配，不行的话使用word2vec再进行计算'''
    base_name = []
    input_name = []
    base_result = []
    predict_result = []
    with open(inputfile, 'r', encoding='gbk') as f:
        fread = f.read()
        lines = fread.split()
        for line in lines:
            line_split = line.split(',')
            base_name.append(line_split[1])
            input_name.append(line_split[0])
            base_result.append(line_split[2])      
            
    length = len(input_name)

    with open(outputfile, 'w') as fw:
        for i in range(0, length):
            test_sentence_1 = base_name[i]
            test_sentence_2 = input_name[i]

            shorter_sen, longer_sen = preprocess_server.processSen(test_sentence_1, test_sentence_2, stopwords)
            result = preprocess_server.compare(shorter_sen, longer_sen)

            if result:
                predict_result.append('1')
                content = test_sentence_1 + ',' + test_sentence_2 + ',' + base_result[i] + ',' + predict_result[i] + ',' + str(result) + '\n'
                fw.write(content)
            else:
                # 使用匹配模型判断不了，使用word2vec。默认使用distance_model = 2 -> 欧几里得距离
                distance_model = 2
                similarity = run_model_server.operation(model, test_sentence_1, test_sentence_2, distance_model)

                ##### 针对euclidean Distance
                if similarity > 0.5:
                    predict_result.append('1')
                else:
                    predict_result.append('0')
                content = test_sentence_1 + ',' + test_sentence_2 + ',' + base_result[i] + ',' + predict_result[i] + ',' + str(similarity) + '\n'
                fw.write(content)

    return base_result, predict_result   

#######################################################################################

def main_w2v(distance_model):
    '''main function for word2vec model

    :distance_mdoel -> 1:cos ;  2:euclidean
    '''
    # 测试word2vec
    w2v_model = loadModel()
    filepath = '../company_name_test.txt'
    resultpath = './test_result_w2c.txt'
    base_result, predict_result = predict_word2vec(filepath, w2v_model, resultpath, distance_model)
    if distance_model == 1:
        print('-----cos distance-----')
    else:
        print('-----euclidean distance-----')
    evaluation(base_result, predict_result)


def main_straight():
    '''main function for straight matching
    '''
    # 测试直接匹配
    filepath = '../stopwords_words.txt'
    stopwords = preprocess_server.getStopwords(filepath)
    inputfile = '../company_name_test.txt'
    outputfile = './test_result_match.txt'
    base_result, predict_result = predict_strMatch(inputfile, outputfile, stopwords)
    evaluation(base_result, predict_result)


def main_agg():
    w2v_model = loadModel()
    filepath = '../stopwords_words.txt'
    stopwords = preprocess_server.getStopwords(filepath)
    inputfile = '../company_name_test.txt'
    outputfile = './test_result_agg.txt'
    base_result, predict_result = predict_agg(inputfile, outputfile, stopwords, w2v_model)
    evaluation(base_result, predict_result)


if __name__ == '__main__':

    # word2vec匹配
    distance_model = 2  # distance_mdoel -> 1:cos ;  2:euclidean
    main_w2v(distance_model)

    # 直接匹配
    # main_straight()

    # 组合方法
    # main_agg()

    # w2v_model = loadModel()

    # # 测试word2vec
    # filepath = './test1.txt'
    # resultpath = './result1.txt'
    # base_result, predict_result = predict_word2vec(filepath, w2v_model, resultpath)
    # evaluation(base_result, predict_result)

    # # 测试直接匹配
    # filepath = '../stopwords_words.txt'
    # stopwords = preprocess_server.getStopwords(filepath)
    # inputfile = './test2.txt'
    # outputfile = './result_str2.txt'
    # base_result, predict_result = predict_strMatch(inputfile, outputfile, stopwords)
    # evaluation(base_result, predict_result)
