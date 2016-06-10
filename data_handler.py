#! /usr/bin/env python

import numpy as np
from scipy.sparse import coo_matrix
import pdb


class data(object):

    def __init__(self, path):
        self.path = path

    def load_data(self):
        f = open(self.path)
        # initializing row and column indices vector
        row = []
        col = []
        data = []
        for line in f:
            r, c, d = line.split("\t")
            #print r, c ,d
            row.append(int(r))
            col.append(int(c))
            data.append(int(d))
        #print row
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        Data = np.vstack([row, col, data]).T
        np.random.shuffle(Data)
        self.T = coo_matrix((Data[:, 2], (Data[:, 0], Data[:, 1])))
        #pdb.set_trace()


if __name__ == "__main__":
    data_handler = data('/home/shashank/data/SIEL/trust-aware-reco/Epinions_signed/soc-sign-epinions.txt')
    data_handler.load_data()
