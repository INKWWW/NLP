#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pprint import pprint
import pandas as pd
import pdb

candidate1 = '黑龙江红兴隆农垦民乐农业生产资料有限公司'
candidate2 = '秦皇岛市京翰汽车销售有限公司'



def create(filepath):
    df = pd.read_table(filepath, encoding='gbk', header=None)
    print(df.at[0, 0])



if __name__ == '__main__':
    filepath = '../company_name_10000.txt'
    create(filepath)
