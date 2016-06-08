
import numpy as np
from scipy.sparse import dok_matrix
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
        self.T = coo_matrix((data, (row, col)))
        pdb.set_trace()


if __name__ == "__main__":
    data_handler = data('/home/shashank/data/SIEL/trust-aware-reco/Epinions_signed/soc-sign-epinions.txt')
    data_handler.load_data()
