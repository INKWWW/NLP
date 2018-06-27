#!/usr/bin/env python
# -*- coding: utf-8 -*-

import companyNameSim
import json
from conf import param_list


def parseJson(jsonFile):
    '''Handle josn file    
    Arguments:
        jsonFile {[type]} -- [path of .json file] -{ "com_name": {"input": input_name, "query": query_name} }
    '''
    with open(jsonFile, 'r') as fr:
        load_f = json.load(fr)
        input_name = load_f['com_name']['input']
        query_name = load_f['com_name']['query']
    return input_name, query_name


def main(input_name, query_name):
    '''operation'''
    companyNameSim.CompanyNameSim.loadParam(param_list)
    comNameSim = companyNameSim.CompanyNameSim(input_name, query_name)
    result = comNameSim.main()
    return result


if __name__ == '__main__':
    jsonFile = './test.json'
    input_name, query_name = parseJson(jsonFile)

    result = main(input_name, query_name)
    print('result is {}'.format(result))