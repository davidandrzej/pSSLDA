"""
Example script demonstrating usage of pSSLDA
"""
import pdb

import numpy as NP
import numpy.random as NPR

from pSSLDA import infer
import FastLDA 

# First, we create some synthetic data
#
# D = number of documents
# N = corpus length
# T = number of topics
# W = vocabulary size

# Synthetic topics
phi = NP.array([[0.45, 0.45, 0.05, 0.05],
               [0.05, 0.05, 0.85, 0.05],
                [0.05, 0.05, 0.05, 0.85]])
(T, W) = phi.shape
(D, doclen) = (1000, 100)
(wordvec, docvec, zvec) = ([], [], [])
# The theta for each doc will favor one topic
theta = NP.array([0.8, 0.1, 0.1])
# Generate each doc
for di in range(D):
    # Shuffle the topic weights for this doc
    theta_di = NPR.permutation(theta)
    # Sample for each word in the doc
    for wi in range(doclen):
        # Sample latent topic z
        topic = NPR.multinomial(1,theta_di).nonzero()[0][0]
        # Sample word from phi_z
        word = NPR.multinomial(1,phi[topic,:]).nonzero()[0][0]
        # Record sampled values
        docvec.append(di)
        wordvec.append(word)
        zvec.append(topic)

# pSSLDA input types
#
# w = Length N NumPy int array of word indices [0 <= w < W]
# d = Length N NumPy int array of doc indices [0 <= d < D]
# alpha = 1 x T NumPy float array of doc-topic Dirichlet hyperparameters
# beta = T x W NumPy float array of topic-word Dirichlet hyperparameters
#
# zlabels = Length N Python List (one entry per index in corpus)
#           If None
#              no z-label, treat this z normally (as in standard LDA)
#           Else
#              Length T NumPy float array (one entry per topic)
#              The probability of selecting each topic will
#              be multiplied by exp(this value)
#                

# Python Lists must be converted to NumPy arrays
#
(w, d) = (NP.array(wordvec, dtype = NP.int),
          NP.array(docvec, dtype = NP.int))

# Create parameters
alpha = NP.ones((1,T)) * 1
beta = NP.ones((T,W)) * 0.01

# How many parallel samplers do we wish to use?
P = 2

# Random number seed 
randseed = 194582

# Number of samples to take
numsamp = 500

# Do parallel inference
finalz = infer(w, d, alpha, beta, numsamp, randseed, P)

# Estimate phi and theta
(nw, nd) = FastLDA.countMatrices(w, W, d, D, finalz, T)
(estphi,esttheta) = FastLDA.estPhiTheta(nw, nd, alpha, beta)

print ''
print 'True topics'
print str(phi)
print 'Estimated topics'
print '\n'.join(['['+', '.join(['%.2f' % val for val in row]) + ']'
                 for row in estphi])
print ''

print 'Estimated topics should be similar to ground truth'
print '(up to a permutation of the rows)'
print 'enter \'c\' to continue...'
print ''
pdb.set_trace()

#
# Now, we add z-labels to *force* words 0 and 1 into separate topics
# (note that this is different than ground truth)
#
zlabels = []
for wi in w:
    if(wi == 0):
        zlabels.append(NP.array([5, 0, 0], dtype=NP.float))
    elif(wi == 1):
        zlabels.append(NP.array([0, 5, 0], dtype=NP.float))
    else:
        zlabels.append(None)
        
# Now inference will find topics with 0 and 1 in separate topics
finalz = infer(w, d, alpha, beta, numsamp, randseed, P,
               zlabels = zlabels)

# Re-estimate phi and theta
(nw, nd) = FastLDA.countMatrices(w, W, d, D, finalz, T)
(estphi,esttheta) = FastLDA.estPhiTheta(nw, nd, alpha, beta)

print ''
print 'z-label word 0 to topic 0'
print 'z-label word 1 to topic 1'
print ''
print 'True topics'
print str(phi)
print 'z-label estimated topics'
print '\n'.join(['['+', '.join(['%.2f' % val for val in row]) + ']'
                 for row in estphi])
print ''
print 'Note that learned topics now obey z-labels'
print '(even though that \"disagrees\" with data)'
print ''
