#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functions
from conf import param_list
from gensim.models import Word2Vec
import functions


class CompanyNameSim(object):
    """docstring for CompanyNameSim"""
    def __init__(self, input_name, query_name):
        self.input_name = input_name
        self.query_name = query_name


    @ classmethod
    def loadParam(cls, params):
        model = params['modelPath']
        stopword = params['stopwords']
        cls.w2v_model = cls.loadModel(model['model_path'])  # 创建类变量，后面非类函数可以试用self取用
        cls.stopwords = cls.loadStopWords(stopword['stopword_path'])    


    @ classmethod
    def loadModel(self, model_wv):
        '''load model with suffix '.wv'        
        Arguments:
            model_wv {[type]} -- [path of this model]
        '''
        Word2Vec.wv.load(model_wv)
        print('load model successfully')
        

    @ classmethod
    def loadStopWords(self, stopword_file):
        '''load stopword_file
        Arguments:
            stopword_file {[type]} -- [path of stopword_file file]
        '''
        with open(stopword_file, 'r', encoding='utf-8') as f:
            words = f.read()
            # print(words.split('\n'))
        return words.split('\n')


    def main(self):
        '''Main function to decide whether these two company names are similar

        '''
        function = functions.VariousMethods(self.input_name, self.query_name, self.stopwords, self.w2c_model)
        result = function.predict_agg()
        return result
        


if __name__ == '__main__':
    CompanyNameSim.loadParam(param_list)  # 加载参数列表
    comNameSim = CompanyNameSim(input_name, query_name)  # 实例化
    comNameSim.main()