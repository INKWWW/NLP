#!/usr/bin/env python
# -*- coding: utf-8 -*-

class VariousMethods(object):
    """docstring for Functions"""
    def __init__(self, input_name, query_name, stopwords, model):
        self.input_name = input_name
        self.query_name = query_name
        self.stopwords = stopwords
        self.model = model


    def tokenizer(self, sentence):
        '''tokenize sentence
        '''
        for stop in self.stopwords:
            sentence = sentence.replace(stop, '')
        sen = jieba.lcut(sentence)
        return sen


    def processSen(self):
        for stop in self.stopwords:
            self.input_name = self.input_name.replace(stop, '')
            self.query_name = self.query_name.replace(stop, '')     
        sen_1_parser = jieba.lcut(self.input_name)
        sen_1_parser = [item for item in sen_1_parser if item not in self.stopwords]
        sen_2_parser = jieba.lcut(self.query_name)
        sen_2_parser = [item for item in sen_2_parser if item not in self.stopwords]
        if len(self.input_name) > len(self.query_name):
            longer_sen = sen_1_parser
            shorter_sen = sen_2_parser 
        elif len(self.input_name) < len(self.query_name):
            longer_sen = sen_2_parser
            shorter_sen = sen_1_parser
        else:
            longer_sen = sen_1_parser
            shorter_sen = sen_2_parser
        return (shorter_sen, longer_sen)


    def compare(self, shorter_sen, longer_sen):
        shorter_sen_copy = copy.deepcopy(shorter_sen)
        longer_sen_copy = copy.deepcopy(longer_sen)
        for item in shorter_sen:
            if item in longer_sen:
                indexs_s = [i for i, v in enumerate(shorter_sen_copy) if v==item]
                indexs_l = [i for i, v in enumerate(longer_sen_copy) if v==item]
                for index in indexs_s:
                    shorter_sen_copy.pop(index)
                for index in indexs_l:
                    longer_sen_copy.pop(index)           
            else:
                continue
        if len(shorter_sen_copy) == 0:
            return True
        else:
            for item in shorter_sen_copy:
                for item_l in longer_sen_copy:
                    if item in item_l or item_l in item:
                        indexs_s = [i for i, v in enumerate(shorter_sen_copy) if v==item]
                        indexs_l = [i for i, v in enumerate(longer_sen_copy) if v==item_l]
                        for index in indexs_s:
                            shorter_sen_copy.pop(index)
                        for index in indexs_l:
                            longer_sen_copy.pop(index)
        if len(shorter_sen_copy) == 0 or len(longer_sen_copy) == 0:
            return True
        else:
            return False


    def match_straight(self):
        shorter_sen, longer_sen = self.processSen()
        result = self.compare(shorter_sen, longer_sen)
        return result
        

    def main_gensim(self):
        '''    
        Arguments:
            model {[type]} -- [description]
            self.query_name {[type]} -- 库中地址
            self.input_name {[type]} -- 用户输入地址
        '''
        if self.query_name == 'null' or len(self.input_name) > 50 or len(self.input_name) == 0:
            similarity = -1
        else:
            split_sen = self.tokenizer(self.query_name)
            split_sen_1 = [item for item in split_sen if (item not in stopwords) and (item in model.wv.vocab)]  # ensure words in the vocab
            split_sen = self.tokenizer(self.input_name)
            split_sen_2 = [item for item in split_sen if (item not in stopwords) and (item in model.wv.vocab)]
            if len(split_sen_1) == 0 or len(split_sen_2) == 0:
                similarity = 0
            else:
                similarity = self.model.n_similarity(split_sen_1, split_sen_2)
        return similarity


    def predict_agg(self):
        '''matching word'''        
        result = self.match_straight()
        if result:
            return 1
        else:
            similarity = self.main_gensim()
            if similarity > 0.95:
                return 1
            else:
                return 0