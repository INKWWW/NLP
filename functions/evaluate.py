#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pdb
from decimal import Decimal

# sys.path.append('C:\\Wang Hanmo\\projects\\similarity\\company_similarity')
sys.path.append("..")
print(sys.path)
# pdb.set_trace()
import run_model
import preprocess


def loadModel():
    # model_output = '../word2vec_model_10000_skip.model'
    model_output = '../word2vec_model_10000_skip_stopwords.model'    
    return run_model.loadModel(model_output)


def predict_word2vec(filepath, model, resultpath):
    base_name = []
    input_name = []
    base_result = []
    predict_result = []
    with open(filepath, 'r') as f:
        fread = f.read()
        lines = fread.split()
        for line in lines:
            line_split = line.split(',')
            base_name.append(line_split[0])
            input_name.append(line_split[1])
            base_result.append(line_split[2])       
            
    length = len(input_name)

    with open(resultpath, 'w') as fw:
        for i in range(0, length):
            test_sentence_1 = base_name[i]
            test_sentence_2 = input_name[i]
            # print('@@@: {},{}'.format(test_sentence_1, test_sentence_2))
            similarity = run_model.operation(model, test_sentence_1, test_sentence_2)
            # similarity = Decimal(similarity)
            # print(similarity)
            # pdb.set_trace()

            ##### 针对euclidean Distance
            if similarity > 0.999:
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
    with open(inputfile, 'r') as f:
        fread = f.read()
        lines = fread.split()
        for line in lines:
            line_split = line.split(',')
            base_name.append(line_split[0])
            input_name.append(line_split[1])
            base_result.append(line_split[2])       
            
    length = len(input_name)

    with open(outputfile, 'w') as fw:
        for i in range(0, length):
            test_sentence_1 = base_name[i]
            test_sentence_2 = input_name[i]

            shorter_sen, longer_sen = preprocess.processSen(test_sentence_1, test_sentence_2, stopwords)
            result = preprocess.compare(shorter_sen, longer_sen)

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

if __name__ == '__main__':

    w2v_model = loadModel()

    # 测试word2vec
    filepath = './test1.txt'
    resultpath = './result1.txt'
    base_result, predict_result = predict_word2vec(filepath, w2v_model, resultpath)
    evaluation(base_result, predict_result)

    # # 测试直接匹配
    # filepath = '../stopwords_words.txt'
    # stopwords = preprocess.getStopwords(filepath)
    # inputfile = './test2.txt'
    # outputfile = './result_str2.txt'
    # base_result, predict_result = predict_strMatch(inputfile, outputfile, stopwords)
    # evaluation(base_result, predict_result)