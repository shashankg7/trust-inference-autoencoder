#! /usr/bin/env python

''' Training module (mini-batch) '''

import numpy as np
import theano
from theano import tensor as T
from model import AutoEncoder
import pdb

class trainAE(object):

    def __init__(self, path, k, lr= 0.1, batch_size=1, loss='bce', n_epochs=100):
        '''
        Arguments:
            path : path to training data
            k : hidde unit's dimension
            lr : learning rate
            batch_size : batch_size for training, currently set to 1
            loss : loss function (bce, rmse) to train AutoEncoder
            n_epochs : number of epochs for training
        '''
        self.AE = AutoEncoder(path, k)
        # Definne the autoencoder model
        self.AE.model()
        self.epochs = n_epochs


    def train(self):
        T = self.AE.T
        # COnverting to csr format for indexing
        T = T.tocsr()
        #pdb.set_trace()
        nonzero_indices = T.nonzero()
        for epoch in xrange(self.epochs):
            print("Running epoch %d"%(epoch))
            for i in np.unique(nonzero_indices[0]):
                # get indices of observed values from the user 'i' 's vector
                indices = T[i, :].nonzero()[1]
                #print indices
                #indices = indices.reshape(indices.shape[0],)
                # Get correspoding ratings
                ratings = T[i, indices].toarray()
                #print ratings
                ratings = ratings.reshape(ratings.shape[1],)
                # Convert inputs to theano datatype
                indices = indices.astype(np.int32)
                ratings = ratings.astype(np.int32)
                #pdb.set_trace()
                loss = self.AE.ae(indices, ratings)
                print("Loss at epoch %d is %f"%(epoch, loss))


if __name__ == "__main__":
    autoencoder = trainAE('./soc-sign-epinions.txt', 100)
    autoencoder.train()




