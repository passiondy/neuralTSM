import numpy as np
import random

def checker(model, n, w, neg_samples):
    model.eta[n] = np.random.randn(model.K)
    model.theta[n] = model.softmax(model.eta[n])
    model.topic_vector = np.random.randn(*model.topic_vector.shape)
    eps = 1e-4
    grad_topic, grad_eta = model.grad_one_sample_no_ft(n, w, neg_samples)
    grad_topic_2, grad_eta_2, grad_vector_pos, grad_vector_neg = model.grad_one_sample_ft(n, w, neg_samples)
    for i in xrange(model.K):
        model.theta[n] = model.softmax(model.eta[n])
        model.eta[n][i] -= eps
        model.theta[n] = model.softmax(model.eta[n])
        doc_vector = np.dot(model.theta[n], model.topic_vector)
        f1 = np.inner(doc_vector, model.word_vector[w])
        f2 = np.inner(doc_vector, model.word_vector[neg_samples])
        fn = np.log(model.sigmoid(f1))+np.log(model.sigmoid(-f2)).sum()
        model.eta[n][i] += 2*eps
        model.theta[n] = model.softmax(model.eta[n])
        doc_vector = np.dot(model.theta[n], model.topic_vector)
        f1 = np.inner(doc_vector, model.word_vector[w])
        f2 = np.inner(doc_vector, model.word_vector[neg_samples])
        fp = np.log(model.sigmoid(f1))+np.log(model.sigmoid(-f2)).sum()
        grad = (fp-fn)/(2*eps)
        model.eta[n][i] -= eps
        if abs(grad-grad_eta[i]) > 1e-5:
            print "no ft gradient check fail", i, " @eta"
            print abs(grad-grad_eta[i])
        if abs(grad-grad_eta_2[i]) > 1e-5:
            print "ft gradient check fail", i, " @eta"
            print abs(grad-grad_eta_2[i])

    model.theta[n] = model.softmax(model.eta[n])
    for i in xrange(model.K):
        for j in xrange(model.D):
            #grad_topic, grad_eta = model.grad_one_sample_no_ft(n, w, neg_samples)
            #grad_topic_2, grad_eta_2, grad_vector_pos, grad_vector_neg = model.grad_one_sample_ft(n, w, neg_samples)
            model.topic_vector[i][j] -= eps
            doc_vector = np.dot(model.theta[n], model.topic_vector)
            f1 = np.inner(doc_vector, model.word_vector[w])
            f2 = np.inner(doc_vector, model.word_vector[neg_samples])
            fn = np.log(model.sigmoid(f1))+np.log(model.sigmoid(-f2)).sum()
            model.topic_vector[i][j] += 2*eps
            doc_vector = np.dot(model.theta[n], model.topic_vector)
            f1 = np.inner(doc_vector, model.word_vector[w])
            f2 = np.inner(doc_vector, model.word_vector[neg_samples])
            fp = np.log(model.sigmoid(f1))+np.log(model.sigmoid(-f2)).sum()
            grad = (fp-fn)/(2*eps)
            model.topic_vector[i][j] -= eps
            if abs(grad-grad_topic[i][j]) > 1e-5:
                print "no ft gradient check fail", i, j, " @topic_vector" 
                print abs(grad-grad_topic[i][j])
            if abs(grad-grad_topic_2[i][j]) > 1e-5:
                print "ft gradient check fail", i, j, " @topic_vector" 
                print abs(grad-grad_topic_2[i][j])

    for d in xrange(model.D):
        model.word_vector[w][d] -= eps
        doc_vector = np.dot(model.theta[n], model.topic_vector)
        f1 = np.inner(doc_vector, model.word_vector[w])
        f2 = np.inner(doc_vector, model.word_vector[neg_samples])
        fn = np.log(model.sigmoid(f1))+np.log(model.sigmoid(-f2)).sum()
        model.word_vector[w][d] += 2*eps
        doc_vector = np.dot(model.theta[n], model.topic_vector)
        f1 = np.inner(doc_vector, model.word_vector[w])
        f2 = np.inner(doc_vector, model.word_vector[neg_samples])
        fp = np.log(model.sigmoid(f1))+np.log(model.sigmoid(-f2)).sum()
        grad = (fp-fn)/(2*eps)
        model.word_vector[w][d] -= eps
        if abs(grad-grad_vector_pos[d]) > 1e-5:
            print "ft gradient check fail", w, d, " @word_vector"
            print abs(grad-grad_vector_pos[d])

    for i in range(len(neg_samples)):
        nw = neg_samples[i]
        for d in xrange(model.D):
            model.word_vector[nw][d] -= eps
            doc_vector = np.dot(model.theta[n], model.topic_vector)
            f1 = np.inner(doc_vector, model.word_vector[w])
            f2 = np.inner(doc_vector, model.word_vector[neg_samples])
            fn = np.log(model.sigmoid(f1))+np.log(model.sigmoid(-f2)).sum()
            model.word_vector[nw][d] += 2*eps
            doc_vector = np.dot(model.theta[n], model.topic_vector)
            f1 = np.inner(doc_vector, model.word_vector[w])
            f2 = np.inner(doc_vector, model.word_vector[neg_samples])
            fp = np.log(model.sigmoid(f1))+np.log(model.sigmoid(-f2)).sum()
            grad = (fp-fn)/(2*eps)
            model.word_vector[nw][d] -= eps
            if abs(grad-grad_vector_neg[i][d]) > 1e-5:
                print "ft neg gradient check fail", nw, d, " @word_vector"
                print abs(grad-grad_vector_neg[i][d])


