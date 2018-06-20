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
    df.rename(columns={'公司名称（客户输入）':'company_name', '匹配出来的公司结果':'match', '职业信息验证（画像匹配结果）.1':'match_result'\
        , '标注':'groundtruth'}, inplace=True)
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

    return df


def generate_txt(csv_file, output_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        with open(output_file, 'w') as fw:
            for row in reader:
                if reader.line_num == 1:
                    continue
                # print(row)
                new_row = ','.join([row[0], row[1], row[3]]) + '\n'
                fw.write(new_row)



if __name__ == '__main__':
    filepath = './公司名称提取0619.xlsx'
    handleXlsx(filepath)
    # csv_file = './company_name_match.csv'
    # output_file = './company_name_test.txt'
    # generate_txt(csv_file, output_file)
