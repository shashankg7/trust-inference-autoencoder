#! /usr/bin/env python

import numpy as np
import theano
from theano import tensor as T

w = np.asarray([[1,2,3],[4,5,6],[7,8,9]]).astype(np.float32)
W = theano.shared(w)

v = np.asarray([[1,2,3],[4,5,6],[7,8,9]]).astype(np.float32)
V = theano.shared(v)

indices = T.imatrix()
ratings= T.matrix()

def step(ind, rat, indices, ratings):
    # vector of index of non-zero entries in ind
    #ind_nz = T.gt(ind, 0).nonzero()
    rat_nz = T.gt(rat, 0).nonzero()
    hid = T.dot(V[:, rat_nz], rat[rat_nz])
    out = T.sum(hid)
    #out = T.tanh(T.dot(W[ind_nz, :], hid))
    return out

result, updates = theano.scan(fn=step, outputs_info=None, sequences=[indices, ratings], non_sequences=[W, V])

scan_test = theano.function([indices, ratings], result)

indice = np.array([[1,0,2], [1,0,0],[0,0,0]]).astype(np.int32)
rating = np.array([[1,0,1], [1,0,0],[0,0,0]]).astype(np.float32)

print scan_test(indice, rating)
