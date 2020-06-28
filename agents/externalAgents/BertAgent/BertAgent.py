import os
import numpy as np
import time
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from sentence_transformers import models

'''
Bert Agent logic

Uses Portuguese Bert word embeddings for sentence transformation and to calculate similarity between them

Class constructor, calls load_bert_model function to load Bert Model, processes AIA-BDE corpus witch mapaVars function, 
available in script directory, which creates and returns a dictionary. Final transformation is performed with mapa_bert function
which returns a new dictionary of corpus Question and its Bert embedding.
'''

class BertAgent:
    def __init__(self,configs):
        self.agentName = self.__class__.__name__
        self.import_modules = []
        self.agents_name = []
        self.bert_model = self.load_bert_model()
        self.mapa = self.mapaVars()
        self.mapa_vec = self.mapa_bert(self.mapa.get("P"), self.bert_model)


    def setAnswerAmount(self, n):
        self.answerAmount = n

    #requestAnswer is mandatory, recieves user Input and retuns lists of most similar awnsers
    def requestAnswer(self, userInput):
        answer = self.query_bert(userInput,self.bert_model,self.mapa_vec,5)

        ret_answer = []

        #Acessing awnsers text, r[1] is answer rank
        for r in answer:
            ret_answer.append(r[0])

        return ret_answer

    #Create a dictionary where key is a question and the value its the question bert embedding.
    def mapa_bert(self, lista, bert):
        print("Criar mapa BERT...")
        mapa_vec = {}
        ps = [p[0] for p in lista]
        vec = bert.encode(ps)
        for i in range(len(lista)):
            # mapa_vec.setdefault(lista[i][0], vec[i][0]) #pooling_strategy=NONE, word embedding for `[CLS]`
            mapa_vec.setdefault(lista[i][0], vec[i])
        return mapa_vec

    #Function to enconde a query to bert embedding and find the most N similar sentences using cosine function
    def query_bert(self, query, bert, mapa, top):
        vec = bert.encode([query])[0]
        # vec = bert.encode([query])[0][0] #pooling_strategy=NONE, word embedding for `[CLS]`
        # print("vec=", vec)
        # print("mapa=", mapa)
        res = []
        for pp in mapa.keys():
            res.append((pp, self.cosine(vec, mapa.get(pp))))
        return sorted(res, key=lambda tup: tup[1], reverse=True)[:top]

    #Function to create a dict of AIA-BDE corpus
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

    #Function to calculate cosine similarity between two vectors (v1 and v2)
    def cosine(self, v1, v2):
        if all(v == 0 for v in v1) or all(v == 0 for v in v2):
            return 0.0
        return np.dot(v1, v2) / (norm(v1) * norm(v2))

    #Loading bert model funtion, will download to local memory if executing for the first time
    def load_bert_model(self):
        start_time = time.time()
        print("Started loading the BERT model")
        word_embedding_model = models.Transformer('neuralmind/bert-base-portuguese-cased')
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)

        bert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print("Model loaded")
        print("--- %s seconds ---" % (time.time() - start_time))
        print('\a')

        return bert_model