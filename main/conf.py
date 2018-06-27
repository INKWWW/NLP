#!/usr/bin/env python
# -*- coding: utf-8 -*-

param_list = {}
param_list['modelPath'] = {'model_path': './word2vec_model.wv'}
param_list['stopwords'] = {'stopword_path': './stopwords_words.txt'}


# 返回给我的json，里面的公司名称需要判断不为空，为了防止有些人乱输入，可以再设定一定的字符串长度检测（大于一定值就排除掉）
# param_list['com_name'] = {'input': input_name, 'query': query_name}  # input_name:用户输入的地址; query_name:库中查询所得地址