from __future__ import division
import numpy as np
import math, time, random

class Model(object):
    def __init__(self, data, maxIter, K, S, lr, cluster_centers=None):
        self.maxIter = maxIter
        self.data = data
        self.N = data.N
        self.V = data.V
        self.D = data.D
        self.docs = data.docs
        self.word_vector = data.vectors
        self.K = K
        self.S = S
        self.lr = lr
        self.eta = np.random.randn(self.N, K)
        self.theta = self.softmax(self.eta)
        if cluster_centers != None:
            self.topic_vector = cluster_centers
        else:
            self.topic_vector = np.random.randn(K, self.D)

    def train(self, bs):
        grad_topic = np.zeros((self.K, self.D))
        grad_eta = np.zeros(self.K)
        docIdxList = range(self.N)
        for it in xrange(self.maxIter):
            st = time.time()
            # random shuffle
            random.shuffle(docIdxList)
            for n in docIdxList:
                L = len(self.docs[n])
                bn = int(math.ceil(L/bs))
                for b in xrange(bn):
                    grad_topic *= 0
                    grad_eta *= 0
                    for w in self.docs[n][bs*b: bs*(b+1)]:
                        n_samples = self.negative_sampling(self.docs[n], self.S)
                        tmp_gt, tmp_ge = self.grad_one_sample(n, w, n_samples)
                        grad_topic += tmp_gt
                        grad_eta += tmp_ge
                    self.topic_vector += self.lr*grad_topic
                    self.eta[n] += self.lr*grad_eta
                    self.theta[n] = self.softmax(self.eta[n])
                #if n%100 == 0:
                #print n, " docs"
            LL = 0
            for n in xrange(self.N):
                doc_vec = self.theta[n].dot(self.topic_vector)
                doc_word = np.inner(doc_vec, self.word_vector)
                doc_word_prob = self.softmax(doc_word)
                for w in self.docs[n]:
                    LL += np.log(doc_word_prob[w])
            print it, LL
            print time.time()-st

    def save_matrix(self, fname, matrix):
        f = open(fname, "w")
        lines = [" ".join([str(e) for e in l]) for l in matrix]
        f.write("\n".join(lines))

    def save_model(self, n, model_prefix, tword_file):
        self.save_matrix(model_prefix+".eta.txt", self.eta)
        self.save_matrix(model_prefix+".theta.txt", self.theta)
        self.save_matrix(model_prefix+".topic_vector.txt", self.topic_vector)
        weight = np.dot(self.topic_vector, self.word_vector.T)
        prob = self.softmax(weight)
        lines = []
        for t in xrange(self.K):
            lines.append("Topic "+str(t))
            L = zip(prob[t], range(self.V))
            L.sort(key=lambda x:x[0], reverse=True)
            tmp = [self.data.wordDict[e[1]]+"\t"+str(e[0]) for e in L[:n]]
            lines += tmp
            lines.append("\n")
        f = open(tword_file, "w")
        f.write("\n".join(lines))
        f.close()

    def softmax(self, eta):
        theta = np.exp(eta)
        if len(eta.shape)==1:
            numerator = theta.sum()
            return theta/numerator
        else:
            numerator = theta.sum(axis=1)
            return theta/numerator.reshape(numerator.shape[0], 1)

    def grad_one_sample(self, n, i, neg_samples):
        prod_i = np.inner(self.word_vector[i], self.topic_vector)
        prod_neg = np.dot(self.word_vector[neg_samples], self.topic_vector.T)
        sig_i = 1-self.sigmoid(np.inner(self.theta[n], prod_i))
        sig_neg = 1-self.sigmoid(-np.inner(self.theta[n], prod_neg))
        grad_topic = (sig_i*self.word_vector[i] - np.sum(sig_neg.reshape(sig_neg.shape[0],1)*self.word_vector[neg_samples], axis=0))*self.theta[n].reshape(self.K, 1)
        weight = np.eye(self.K) - np.ones((self.K, self.K))*self.theta[n].reshape(self.K, 1)
        weight *= self.theta[n]
        grad_eta = sig_i*np.inner(weight, prod_i) - np.sum(sig_neg*np.dot(weight, prod_neg.T), axis=1)
        return grad_topic, grad_eta

    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))

    def negative_sampling(self, doc, S):
        samples = []
        l = 0
        while l < S:
            s = np.random.randint(0, self.V)
            if s not in doc:
                samples.append(s)
                l += 1
        return samples


