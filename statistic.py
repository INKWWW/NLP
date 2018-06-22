#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from pprint import pprint
from functions import evaluate
import numpy as np
import csv
import pdb


def handleXlsx(filepath):
    '''Preprocess the original .xlsx file and do some statistics on it 
    Arguments:
        filepath {[type]} -- [path of the original file]
    '''
    df = pd.read_excel(filepath)
    # df.to_csv('./公司名称提取.csv', index=False, sep=',', encoding='gbk')
    # print(df)
    df.rename(columns={'公司名称（客户输入）':'company_name', '匹配出来的公司结果':'match', '职业信息验证（画像匹配结果）.1':'match_result'\
        , '标注':'groundtruth'}, inplace=True)
    df.to_csv('./公司名称提取.csv', index=False, sep=',', encoding='gbk')
    # df.match_result = df.match_result.astype('int')
    df = pd.concat([df.company_name, df.match, df.match_result, df.groundtruth], axis=1)
    df = df[df.match != '\\N']  # delete rows with '\N' in column named 'match'
    df = df.dropna(axis=0)  # delete the rows with empty value in some column
    df = df.reset_index(drop=True)  # reset the index and prevent holding the old index
    df.match_result = df.match_result.astype('int')  # change the data type
    pprint(df)
    # output the new df to a csv
    df.to_csv('./company_name_match.csv', index=False, sep=',', encoding='gbk')
    print(df.shape)

    num_of_company = df.shape[0]
    accurate = df.loc[(df.match_result == df.groundtruth)].match_result.count()
    accuracy = accurate / num_of_company
    print('accuracy: {}'.format(accuracy))
    tp = df.loc[(df.match_result == 1) & (df.groundtruth == 1)].match_result.count() # count the num of true positive
    print('num of true positive: {}'.format(tp))
    fp = df.loc[(df.match_result == 1) & (df.groundtruth == 0)].match_result.count()  # false positive
    print('num of flase positive: {}'.format(fp))
    positive_in_groundtruth = df.loc[(df.groundtruth == 1)].groundtruth.count()
    print('num of positive in groundtruth: {}'.format(positive_in_groundtruth))
    precision = tp / (tp + fp)
    print('precision: {}'.format(precision))
    recall = tp / (positive_in_groundtruth)
    print('recall: {}'.format(recall))
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print('f1_score: {}'.format(f1_score))

    return df


def generate_txt(csv_file, output_file):
    '''生成的txt文件是两个地址都有的'''
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        with open(output_file, 'w') as fw:
            for row in reader:
                if reader.line_num == 1:
                    continue
                # print(row)
                new_row = ','.join([row[0], row[1], row[3]]) + '\n'
                fw.write(new_row)

def create_txt(inputfile, output_file):
    '''生成的txt文件含所有情况'''
    with open(inputfile, 'r') as f:
        reader = csv.reader(f)
        with open(output_file, 'w') as fw:
            for row in reader:
                if reader.line_num == 1:
                    continue
                if row[4] == '\\N' or row[4] == '':
                    row[4] = 'null'
                if row[5] != '1.0':
                    row[5] = '0'
                else:
                    row[5] = '1'
                new_row = ','.join([row[2], row[4], row[6]]) + '\n'
                fw.write(new_row)

def create_txt_2(inputfile, output_file):
    '''生成的txt文件含所有情况，含有四列的，拿来计算指标'''
    with open(inputfile, 'r') as f:
        reader = csv.reader(f)
        with open(output_file, 'w') as fw:
            for row in reader:
                if reader.line_num == 1:
                    continue
                if row[4] == '\\N' or row[4] == '':
                    row[4] = 'null'
                if row[5] != '1.0':
                    row[5] = '0'
                else:
                    row[5] = '1'
                new_row = ','.join([row[2], row[4], row[5], row[6]]) + '\n'
                fw.write(new_row)

def evaluation(inputfile):
    '''计算precision、recall、F1
    normal_pos, normal_neg, detected_pos, detected_neg

    base_result, predict_result
    '''
    base_result = []
    predict_result = []
    with open(inputfile, 'r') as f:
        lines = f.read()
        lines = lines.split('\n')
        # print(type(lines))
        for line in lines:
            words = line.split(',')
            # print(words)
            base_result.append(words[3])
            # print(type(words[3]))
            predict_result.append(words[2])
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
        if (base_result[i] == '0' or base_result[i] == '-1') and predict_result[i] == '0':
            tn += 1
        if base_result[i] == '1' and predict_result[i] == '1':
            tp += 1
        if base_result[i] == '1' and predict_result[i] == '0':
            fn += 1
        if (base_result[i] == '0' or base_result[i] == '-1') and predict_result[i] == '1':
            fp += 1
    # precision
    precision = tp / (tp + fp)
    print('-------precision--------: {}'.format(precision))
    # recall
    recall = tp / (tp + fn)
    print('-------recall--------: {}'.format(recall))
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print('-------f1_score--------: {}'.format(f1_score))
    return (precision, recall, f1_score)

   
if __name__ == '__main__':
    # filepath = './公司名称提取0619.xlsx'
    # handleXlsx(filepath)
    # csv_file = './company_name_match.csv'
    # output_file = './company_name_test.txt'
    # generate_txt(csv_file, output_file)

    # 生成全内容txt
    # inputfile = './公司名称提取.csv'
    # output_file = 'company_name_full_test.txt'
    # create_txt(inputfile, output_file)

    # 生成全内容txt,四列
    # inputfile = './公司名称提取.csv'
    output_file = 'company_name_full_calculate.txt'
    # create_txt_2(inputfile, output_file)
    evaluation(output_file)
