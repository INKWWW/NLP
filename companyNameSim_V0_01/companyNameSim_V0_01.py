#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from conf import model_param
import companyNameSim
from abstractApi import AbstractModel


class FeatureDerive(AbstractModel):
    """docstring for FeatureDerive"""
    def __init__(self):
        pass


    def loadParam(self, params):
        '''load params from conf.params
        引擎启动时调用这个loadParam方法加载模型'''
        model = params['model_path']
        stopword = params['stopword_path']
        self.w2v_model = cls.loadModel(model['model_path'])  # 创建类变量，后面非类函数可以使用self取用
        self.stopwords = cls.loadStopWords(stopword['stopword_path'])    


    def predict(self, params):
        '''Main function to decide whether these two company names are similar
        系统会调用模型脚本predict()方法，传入所有需要的数据（params是引擎传入的参数）'''
        input_name = params['com_name']['input']
        query_name = params['com_name']['query']
        function = companyNameSim.VariousMethods(input_name, query_name, self.stopwords, self.w2v_model)
        result = function.predict_agg()
        return result


    @ classmethod
    def loadModel(self, model_wv):
        '''load model with suffix '.wv'        
        Arguments:
            model_wv {[type]} -- [path of this model]
        '''
        KeyedVectors.load(model_wv)
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
