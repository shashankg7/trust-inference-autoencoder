#! /usr/bin/env python

from __future__ import print_function
import numpy as np
from theano import tensor as T
import theano
from scipy.sparse import coo_matrix
from data_handler import data
import pdb

class AutoEncoder(object):

    def __init__(self, path, k):
        self.data = data(path)
        # hidden layer's dimension
        self.k = k
        self.data.load_data()
        self.T = self.data.T
        self.n = self.T.shape[0]

    def model_batch(self, loss='bce'):
        ''' define the AutoEncoder model (mini-batch) '''
        # initializing network paramters
        w = np.random.uniform(low=-np.sqrt(6 / float(self.n + self.k)),
                              high=np.sqrt(6 / float(self.n + self.k)),
                              size=(self.n, self.k))
        v = np.random.uniform(low=-np.sqrt(6 / float(self.n + self.k)),
                              high=np.sqrt(6 / float(self.n + self.k)),
                              size=(self.k, self.n))
        MU = np.zeros((self.k, 1))
        B = np.zeros((self.n, 1))
        # Creating theano shared variables from these
        W = theano.shared(w, name='W', borrow=True)
        V = theano.shared(v, name='V', borrow=True)
        mu = theano.shared(MU, name='mu', borrow=True)
        b = theano.shared(B, name='b', borrow=True)
        self.param = [W, V, mu, b]
        # input variable (matrix) with each row corresponding to each user's
        # vector of indices of observed values
        # TO-DO : Check if this batch thing works and if not try out with vector
        #index = T.imatrix()
        # Ratings correspoding to the indices
        rating = T.imatrix()
        # Target for the network (ratings only)
        # TO-DO : Check for aliasing issue, if any
        target = rating
        ######################## Theano scan part ##############################
        def step(rat, W, V, mu, b):
            # Function applied at each step of scan
            # find all non-zero indices from index (observed values)
            #ind_nz = T.gt(ind, 0).nonzero()
            rat_nz = T.gt(rat, 0).nonzero()
            # target
            # Check for mem. aliasing
            tar = rat[rat_nz]
            hidden_activation = T.tanh(T.dot(V[:, rat_nz], rat[rat_nz]) \
                                       + mu)
            output_activation = T.tanh(T.dot(W[rat_nz, :], \
                                             hidden_activation) \
                                       + b[rat_nz])
            error = T.sum((tar - output_activation) ** 2)
            return error

        scan_res, scan_updates = theano.scan(fn=step, outputs_info=None, \
                                             sequences=[rating],
                                             non_sequences=[W, V, \
                                                            mu, b])

        self.loss = T.mean(scan_res)
        grads = T.grad(self.loss, self.param)
        updates = [(param, param - lr * grad) for (param, grad) in \
                   zip(self.param, grads)]

        self.ae_batch = theano.function([rating], self.loss, updates=updates)



    def model(self, lr = 0.4, loss='rmse'):
        ''' Define model on single example (batch_size = 1) '''
        W = np.random.uniform(low=-np.sqrt(6 / float(self.n + self.k)),
                              high=np.sqrt(6 / float(self.n + self.k)),
                              size=(self.n, self.k))
        V = np.random.uniform(low=-np.sqrt(6 / float(self.n + self.k)),
                              high=np.sqrt(6 / float(self.n + self.k)),
                              size=(self.k, self.n))
        #mu = np.zeros((self.k, 1))
        #b = np.zeros((self.n, 1))
        mu = np.zeros((self.k,))
        b = np.zeros((self.n,))
        # Creating theano shared variables from these
        self.W = theano.shared(W, name='W', borrow=True)
        self.V = theano.shared(V, name='V', borrow=True)
        self.mu = theano.shared(mu, name='mu', borrow=True)
        self.b = theano.shared(b, name='b', borrow=True)
        # Stack all parameters in a single array
        self.param = [self.W, self.V, self.mu, self.b]
        # input variable, vector corresponding to user i's trust
        # Vector of observed indices in the training data
        index = T.ivector()
        # Ratings correspoding to the observed indices
        # NOTE: Try with both (0,1) and (-1,1) thing
        rating = T.ivector()
        # Target for the network (target = input)
        # TO-DO : Check for aliasing issue, if any
        target = rating
        # Calculate hidden layer activations h =  (g(Vr + mu))
        hidden_activation = T.tanh(T.dot(self.V[:, index], rating)
                                   + self.mu)
        # Calculate output layer activations ( f(Wh + b))
        output_activation = T.tanh(T.dot(self.W[index, :], hidden_activation)
                                   + self.b[index])
        # Compute the loss
        if loss == 'bce':
            # binary cross-entropy loss
            self.loss = - T.sum(target * T.log(output_activation) + (1 - target)
                                * T.log(1 - output_activation))
        elif loss == 'rmse':
            # RMSE loss
            self.loss = T.sum((target - output_activation) ** 2) + \
                        0.001 * T.sum(self.W ** 2) + 0.001 * T.sum(self.V ** 2)

        # gradients w.r.t paramters
        grads = T.grad(self.loss, wrt=self.param)
        updates = [(param, param - lr * param_grad) for (param, param_grad)
                   in zip(self.param, grads)]
        self.ae = theano.function(inputs=[index, rating],
                                     outputs=self.loss, updates=updates)


if __name__ == "__main__":
    AE = AutoEncoder('/home/shashank/data/SIEL/trust-aware-reco/Epinions_signed/soc-sign-epinions.txt', 100)
    AE.model()
    #pdb.set_trace()
