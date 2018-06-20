#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from pprint import pprint

def handleXlsx(filepath):
    '''Handle xlsx file and do some preprocess on it    
    Arguments:
        filepath {[type]} -- [path of the original file]
    '''
    df = pd.read_excel(filepath)
    df.rename(columns={'公司名称（客户输入）':'company_name', '匹配出来的公司结果':'match', '职业信息验证（画像匹配结果）.1':'match_result'\
        , '标注':'groundtruth'}, inplace=True)
    # df.match_result = df.match_result.astype('int')
    df = pd.concat([df.company_name, df.match, df.match_result, df.groundtruth], axis=1)
    df = df[df.match != '\\N']
    df = df.dropna(axis=0)
    df = df.reset_index(drop=True)
    df.match_result = df.match_result.astype('int')
    pprint(df)
    print(df.shape)
    num_of_company = df.shape[0]
    tp = df.loc[(df.match_result == 1) & (df.groundtruth == 1)].match_result.count()
    print('num of true positive: {}'.format(tp))
    fp = df.loc[(df.match_result == 1) & (df.groundtruth == 0)].match_result.count()
    print('num of flase positive: {}'.format(fp))
    positive_in_groundtruth = df.loc[(df.groundtruth == 1)].groundtruth.count()
    print('num of positive in groundtruth: {}'.format(positive_in_groundtruth))
    precision = tp / (tp + fp)
    print('precision: {}'.format(precision))
    recall = tp / (positive_in_groundtruth)
    print('recall: {}'.format(recall))






if __name__ == '__main__':
    filepath = './公司名称提取0619.xlsx'
    handleXlsx(filepath)


