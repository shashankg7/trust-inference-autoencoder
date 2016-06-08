#! /usr/bin/env python


from __future__ import print_function
import numpy as np
from theano import Tensor as T
import theano
from scipy.sparse import coo_matrix
from data_handler import data


def AutoEncoder(object):

    def __init__(self, path, k):
        self.data = data(path)
        # hidden layer's dimension
        self.k = k


    def model(self):
        ''' define the AutoEncoder model '''



