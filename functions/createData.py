#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pprint import pprint
import pandas as pd
import pdb

# def create(filepath):
#     df = pd.read_table(filepath, encoding='gbk', header=None)
#     print(df.at[0, 0])

def createData(filepath, outfile, candidate):
    with open(filepath, 'r') as f:
        fread = f.read()
        # print(fread)
        lines = fread.split()
        with open(outfile, 'w') as fw:
            for line in lines:
                if line != candidate:
                    new_line = candidate + ',' + line + ',' + '0' + '\n'
                    fw.write(new_line)
                else:
                    new_line = candidate + ',' + line + ',' + '1' + '\n'
                    fw.write(new_line)



if __name__ == '__main__':
    candidate1 = '黑龙江红兴隆农垦民乐农业生产资料有限公司'
    candidate2 = '秦皇岛市京翰汽车销售有限公司'

    filepath = '../company_name_10000.txt'
    outfile = './test2.txt'
    createData(filepath, outfile, candidate2)
