import os, os.path
from whoosh.index import create_in
import whoosh.index as index
from whoosh.fields import *
from whoosh.query import FuzzyTerm, Term
from whoosh.qparser import QueryParser, OrGroup


'''
Whoosh Agent logic

Uses Whoosh index for retrieving the most similar awnsers to a query

Class constructor, loads Whoosh index directory (defined in config.xml), processes AIA-BDE corpus available in this
directory to a dictionary fomrat and processes to creates Whoosh index in function indexa_whoosh. 
'''

class OshAgent:
    def __init__(self,configs):
        self.agentName = self.__class__.__name__
        self.import_modules = []
        self.agents_name = []
        self.index_dir = configs['indexDir']
        self.mapa = self.mapaPR()
        self.indexa_whoosh(self.mapa)

    def setAnswerAmount(self, n):
        self.answerAmount = n

    # requestAnswer is mandatory, recieves user Input and retuns lists of most similar awnsers
    def requestAnswer(self, userInput):
        answer = self.query_whoosh(userInput, 5)

        ret_answer = []

        for r in answer:
            ret_answer.append(r[0])

        return ret_answer

    # Function to query Whoosh index to retrieve the most N similar awnsers
    def query_whoosh(self, q, maxres=10, fuzzy=False):
        ix = index.open_dir(self.index_dir)
        with ix.searcher() as searcher:
            query = QueryParser("p", ix.schema, group=OrGroup, termclass=(FuzzyTerm if fuzzy else Term)).parse(q)
            results = searcher.search(query, limit=maxres)
            list_results = []
            for r in results:
                list_results.append((r.get("p"), r.get("r")))
            return list_results

    # Function to create a dict of AIA-BDE corpus
    def mapaPR(self):
        fich = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'AIA-BDE_v2.txt')
        mapa = {}
        with open(fich, "r") as corpus:
            s = None
            p = None
            for line in corpus:
                dp = line.index(":")
                tag = line[:dp]
                content = line[dp + 1:].strip()
                if tag == "S":
                    s = content
                elif tag == "P":
                    p = content
                elif tag == "R":
                    mapa[p] = (content, s)
        return mapa

    # Function to create whoosh index based on corpus input
    def indexa_whoosh(self, mapa):
        dir = self.index_dir
        if not os.path.exists(dir):
            os.mkdir(dir)

        # schema = Schema(s=TEXT(stored=True,phrase=False), p=TEXT(stored=True,phrase=False), r=TEXT(stored=True,phrase=False))
        # analyzer=LanguageAnalyzer("pt")
        schema = Schema(s=TEXT(stored=True), p=TEXT(stored=True), r=TEXT(stored=True))
        ix = create_in(dir, schema)
        writer = ix.writer()
        for p in mapa.keys():
            e = mapa.get(p)
            writer.add_document(s=e[1], p=p, r=e[0])
        writer.commit()
        print("Indexação concluída!")