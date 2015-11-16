import numpy as np
class Data(object):
    # class for data, used as an input to model
    # three parts: documents, word vectors, ID-word mapping
    # meta info: V, N, D
    def __init__(self, doc_file, vec_file, wordDict_file):
        self.docs = self.read_doc(doc_file)
        print "docs read"
        self.vectors = self.read_vector(vec_file)
        print "vectors read"
        self.wordDict = self.read_wordDict(wordDict_file)
        print "dict read"
        self.N = len(self.docs)
        self.V = len(self.wordDict)
        self.D = len(self.vectors[0])

    def read_doc(self, fname):
        f = open(fname)
        lines = f.readlines()
        f.close()
        docs = [l.split() for l in lines]
        docs = [[int(e) for e in l] for l in docs]
        return docs

    def read_vector(self, fname):
        f = open(fname)
        lines = f.readlines()
        f.close()
        vectors = [l.split() for l in lines]
        vectors = [[float(e) for e in l] for l in vectors]
        return np.array(vectors)

    def read_wordDict(self, fname):
        f = open(fname)
        lines = f.readlines()
        f.close()
        lines = [l.split() for l in lines]
        items = [(int(l[1]), l[0]) for l in lines]
        return dict(items)

