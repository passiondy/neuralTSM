from gensim.models import Word2Vec

def transform(fname, vector_model, doc_file, vec_file, wordDict_file):
    f = open(fname)
    lines = f.readlines()
    f.close()
    word_dict = {}
    docs = []
    vectors = []
    for line in lines:
        tokens = line.split()
        doc = []
        for token in tokens:
            if token not in vector_model:
                continue
            if token not in word_dict:
                word_dict[token] = len(word_dict)
                vectors.append(vector_model[token])
            ID = word_dict[token]
            doc.append(ID)
        if len(doc) > 10:
            docs.append(doc)
    save_list(doc_file, docs)
    save_list(vec_file, vectors)
    save_dict(wordDict_file, word_dict)

def save_list(fname, data):
    lines = [" ".join([str(e) for e in d]) for d in data]
    f = open(fname, "w")
    f.write("\n".join(lines))
    f.close()

def save_dict(fname, Dict):
    items = Dict.items()
    items.sort()
    lines = [str(item[0])+" "+str(item[1]) for item in items]
    f = open(fname, "w")
    f.write("\n".join(lines))
    f.close()

if __name__ == "__main__":
    vector_model = Word2Vec.load_word2vec_format("../word_embedding_cluster/GoogleNews-vectors-negative300.bin.gz", binary=True)
