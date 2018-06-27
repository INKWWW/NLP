#!/usr/bin/env python
# -*- coding: utf-8 -*-

import companyNameSim
import json
from conf import param_list


def parseJson(jsonFile):
    '''{ "com_name": {"input": input_name, "query": query_name} }'''
    with open(jsonFile, 'r') as fr:
        load_f = json.load(fr)
        input_name = load_f['com_name']['input']
        query_name = load_f['com_name']['query']
    return input_name, query_name


def main(input_name, query_name):
    companyNameSim.CompanyNameSim.loadParam(param_list)
    comNameSim = companyNameSim.CompanyNameSim(input_name, query_name)
    comNameSim.main()


if __name__ == '__main__':
    jsonFile = './test.json'
    input_name, query_name = parseJson(jsonFile)

    result = main(input_name, query_name)
    print('result is {}'.format(result))