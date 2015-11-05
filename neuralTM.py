from __future import division
import numpy as np

class Model(object):
    def __init__(self, N, K, V, D, S, lr):
        self.docs
        self.N = N
        self.K = K
        self.V = V
        self.D = D
        self.S = S
        self.lr = lr
        self.theta = np.zeros((N, K))
        self.eta = np.zeros((N, K))
        self.topic_vector = np.zeros((K, D))
        self.word_vector = np.zeros((V, D))

    def train(self, bs):
        grad_topic = np.zeros((K, D))
        grad_eta = np.zeros((N, K))
        for n in xrange(self.N):
            L = len(self.docs[n])
            bn = int(math.ceil(L/bs))
            for b in xrange(bn):
                grad_topic *= 0
                grad_eta *= 0
                for w in self.docs[n][bs*b: bs*(b+1)]:
                    n_samples = self.negative_sampling(n, self.S)
                    tmp_gt, tmp_ge = self.grad_one_sample(n, w, n_samples)
                    grad_topic += tmp_gt
                    grad_eta += tmp_ge
                    self.topic_vector += self.lr*grad_topic
                    self.eta += self.lr*grad_eta
                    self.theta = softmax(self.eta)

    def softmax(self, eta):
        theta = np.exp(eta)
        numerator = theta.sum(axis=1)
        return theta/numerator.reshape(numerator.shape[0], 1)

    def grad_one_sample(self, n, i, neg_samples):
        prod_i = np.inner(self.word_vector[i], self.topic_vector)
        prod_neg = np.dot(self.word_vector[neg_samples], self.topic_vector.T)
        sig_i = self.sigmoid(np.inner(self.theta[n], prod_i))
        sig_neg = self.sigmoid(-np.inner(self.theta[n], prod_j))
        grad_topic = (sig_i*self.word_vector[i] - sig_j*np.sum(self.word_vector[neg_samples], axis=0))*self.theta[n].reshape(self.K, 1)
        weight = np.eye(self.K) - np.ones((self.K, self.K))*self.theta[n].reshape(self.K, 1)
        weight *= self.theta[n]
        grad_eta = sig_i*np.inner(weight, prod_i) - sig_j*np.sum(np.dot(weight, prod_neg.T), axis=1)
        return grad_topic, grad_eta

    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))

    def negative_sampling(self, n, doc, S):
        samples = []
        l = 0
        while l < S:
            s = random.randint(0, self.V)
            if s not in doc:
                samples.append(s)
                l += 1
        return samples


