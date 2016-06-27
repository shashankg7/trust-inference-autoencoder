#! /usr/bin/env python

''' Training module (mini-batch) '''

import numpy as np
import theano
from theano import tensor as T
from model import AutoEncoder
import pdb
import resource
from scipy.special import expit
import gc

#rsrc = resource.RLIMIT_DATA
#soft, hard = resource.getrlimit(rsrc)
#print 'Soft limit starts as :', soft

#resource.setrlimit(rsrc, (10000000, hard))

#soft, hard = resource.getrlimit(rsrc)
#print 'Soft limit changed to :', soft

f = open('log.txt', 'a')
class trainAE(object):

    def __init__(self, path, k, lr= 0.01, batch_size=1, loss='bce', n_epochs=500):
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
        #self.AE.model()
        self.AE.model_batch()
        self.epochs = n_epochs


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

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
            print("RMSE after one epoch is %f"%(self.RMSE()))

    # Batch training method
    def train_batch(self, batch_size):
        T = self.AE.T
        T = T.tocsr()
        nonzero_indices = T.nonzero()
        #pdb.set_trace()
        n_users = len(np.unique(nonzero_indices[0]))
        indices = np.unique(nonzero_indices[0])
        for epoch in xrange(self.epochs):
            l = []
            for ind, i in enumerate(xrange(0, n_users, batch_size)):
                # CHECK : SEEMS BUGGY. 
                #------------------------
                #ratings = T[indices[i:(i + batch_size)], :].toarray().astype(np.float32)
                ratings = T[indices[i:(i + batch_size)], :].toarray().astype(np.float32)

                #------------------------
                #print ratings
                #pdb.set_trace()
                loss = self.AE.ae_batch(ratings)
                #loss = self.AE.debug(ratings)
                #print loss
                #pdb.set_trace()
                l.append(loss)
            m = np.mean(np.array(l))
            print("mean Loss for epoch %d  batch %d is %f"%(epoch, ind, m))
            rmse = self.RMSE_sparse()
            print("RMSE after one epoch is %f"%(rmse))
            f.write(str(rmse) + '\n')

    def RMSE(self):
        W, V, b, mu = self.AE.get_params()
        print("testing process starts")
        test = self.AE.test_ind
        rat = []
        pred = []
        rmse = 0
        for i,j in test:
            Rt = self.AE.T[i, :].todense()
            Rt1 = np.zeros(Rt.shape[1])
            Rt1[:] = Rt[:]
            #pdb.set_trace()
            p = expit(np.dot(W, expit(np.dot(V, Rt1) + mu)) + b)
            #p = np.tanh(np.dot(W, np.tanh(np.dot(V, Rt1) + mu)) + b)
            p = p[j]
            pred.append(p)
            rat.append(self.AE.t[i, j])
        try:
            rat = np.array(rat)
            pred = np.array(pred)
            rmse = np.sqrt(np.mean((pred-rat)**2))
        except:
            print "exception"
            pdb.set_trace()
        np.save('test', test)
        np.save('W', W)
        np.save('V', V)
        np.save('mu', mu)
        np.save('b', b)
        return rmse
        #pdb.set_trace()

    def RMSE_sparse(self):
        W, V, b, mu = self.AE.get_params()
        #pdb.set_trace()
        mu = mu.reshape(100,1)
        print("testing process starts")
        test = self.AE.test_ind
        rat = []
        pred = []
        rmse = 0
        for i,j in test:
            #pdb.set_trace()
            Rt = np.array(self.AE.T[i, :].todense().tolist()).astype(np.float16)
            #pdb.set_trace()
            ind = np.where(Rt > 0)[1]
            #pdb.set_trace()
            Rt = Rt.T
            #pdb.set_trace()
            temp1 = V[:, ind]
            temp2 = Rt[ind]
            temp = expit(np.dot(temp1, temp2) + mu)
            #del temp1
            #del temp2
            p = expit(np.dot(W, temp) + b)
            #pdb.set_trace()
            p = p[0, j]
            pred.append(p)
            rat.append(self.AE.t[i, j])
            #gc.collect()
        try:
            rat = np.array(rat)
            pred = np.array(pred)
            rmse = np.sqrt(np.mean((pred-rat) ** 2))
        except:
            print "exception"
            pdb.set_trace()
        np.save('test', test)
        np.save('W', W)
        np.save('V', V)
        np.save('mu', mu)
        np.save('b', b)
        return rmse
        #pdb.set_trace()



if __name__ == "__main__":
    autoencoder = trainAE('../data/data.mat', 100)
    autoencoder.train_batch(32)
    autoencoder.RMSE_sparse()



