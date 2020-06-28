from numpy.linalg import norm
from gensim.models import KeyedVectors

import os
import numpy as np
import string
import time


'''
Word2Vec Agent logic

Uses W2V word embeddings for sentence transformation and to calculate similarity between them

Class constructor, calls load_w2v_model function to load W2V Model, processes AIA-BDE corpus witch mapaVars function, 
available in script directory, which creates and returns a dictionary. Final transformation is performed with mapa_vec function
which returns a new dictionary of corpus Question and its Bert embedding.
'''

class EmbeddingAgent:
    def __init__(self,configs):
        self.agentName = self.__class__.__name__
        self.import_modules = []
        self.agents_name = []
        self.w2v_model = self.load_w2v_model()
        self.mapa = self.mapaVars()
        self.mapa_vec = self.mapa_w2v(self.mapa.get("P"), self.w2v_model)

    def setAnswerAmount(self, n):
        self.answerAmount = n

    # requestAnswer is mandatory, recieves user Input and retuns lists of most similar awnsers
    def requestAnswer(self, userInput):
        answer = self.query_w2v(userInput, self.w2v_model, self.mapa_vec,5)

        ret_answer = []

        # Acessing awnsers text, r[1] is answer rank
        for r in answer:
            ret_answer.append(r[0])

        return ret_answer

    # Function to create a dict of AIA-BDE corpus
    def mapaVars(self):
        fich = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'AIA-BDE_v2.txt')
        mapa = {}
        with open(fich, "r") as corpus:
            p = None
            for line in corpus:
                dp = line.index(":")
                tag = line[:dp]
                content = line[dp + 1:].strip()
                # print(tag[0], tag, content)
                if tag == "P":
                    p = content
                    if tag not in mapa.keys():
                        mapa.setdefault(tag, [])
                    mapa.get(tag).append((content, p))
                elif tag[0] == 'V':
                    if tag not in mapa.keys():
                        mapa.setdefault(tag, [])
                    # print(p, content)
                    mapa.get(tag).append((content, p))
        return mapa

    # Create a dictionary where key is a question and the value its the question words average of W2V embedding.
    def mapa_w2v(self, lista, model):
        print("Criar mapa word2vec...")
        mapa_vec = {}
        for p in lista:
            vec = self.rep_w2v(p[0], model)
            # print(p[0], vec)
            mapa_vec.setdefault(p[0], vec)
        return mapa_vec

    # Function to represent a sentence to a Word embedding vector by calculating the embedding of each word and perform a average of all word vectors
    def rep_w2v(self, frase, model):
        tokens = self.tokenize(frase)
        # print(tokens)
        vec = np.zeros(len(model['palavra']))
        n = 0
        for t in tokens:
            if t in model.vocab:
                n += 1
                vec = [x + y for x, y in zip(vec, model[t])]
        if n > 0:
            vec = [i / n for i in vec]
        return vec

    #Auxiliary function to tokenize a sentence
    def tokenize(self,frase):
        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = frase.translate(table)
        # split into words by white space
        words = stripped.lower().split()
        return words

    # Function to enconde a query to bert embedding and find the most N similar sentences using cosine function
    def query_w2v(self, query, model, mapa, top):
        vec = self.rep_w2v(query, model)
        res = []
        for pp in mapa.keys():
            res.append((pp, self.cosine(vec, mapa.get(pp))))
        return sorted(res, key=lambda tup: tup[1], reverse=True)[:top]

    # Function to calculate cosine similarity between two vectors (v1 and v2)
    def cosine(self, v1, v2):
        if all(v == 0 for v in v1) or all(v == 0 for v in v2):
            return 0.0
        return np.dot(v1, v2) / (norm(v1) * norm(v2))

    # Loading w2v model funtion, must train a new model if now available
    def load_w2v_model(self):
        model_load_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nilc_cbow_s300_300k.txt')
        # model_load_path = os.path.join('models', 'word2vec', 'NILC', 'nilc_skip_s300.txt')
        start_time = time.time()
        print("Started loading the word2vec model")
        word2vec_model = KeyedVectors.load_word2vec_format(model_load_path)
        # word2vec_model = None
        print("Model loaded")
        print("--- %s seconds ---" % (time.time() - start_time))
        print('\a')

        return word2vec_model