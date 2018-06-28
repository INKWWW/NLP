#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import neccessary packages
import copy
import jieba
import json
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# import personal functions
import functions
from conf import model_param
import companyNameSim
from abstractApi import AbstractModel