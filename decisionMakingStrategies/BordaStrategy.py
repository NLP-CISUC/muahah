import sys, os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.append(current_dir)
from DecisionMethod import DecisionMethod


'''
Borda Count decision method. Method logic available in https://en.wikipedia.org/wiki/Borda_count
'''

class BordaStrategy(DecisionMethod):
    '''
    Borda count logic, accepts list of lists and returns a dict of candidates and its score ordered by score

    Input format: [ [Awnser X, Answer Y, Answer Z], [Answer Y, Anwser Z, Anwser Y]...]
    '''
    def borda_sort(self, votes):
        scores = {}
        for l in votes:
            for idx, elem in enumerate(reversed(l)):
                if not elem in scores:
                    scores[elem] = 0
                scores[elem] += idx
        print("BORDA SCORES")
        print(scores)
        return sorted(scores.keys(), key=lambda elem: scores[elem], reverse=True)

    #Borda count logic, function not used
    def borda_count(self, lista_ranks, top=5):
        respostas = []
        for i in range(len(lista_ranks[0])):
            map_tmp = {}
            for j in range(len(lista_ranks)):
                for k in range(min(len(lista_ranks[j][i]), top)):
                    r = lista_ranks[j][i][k]
                    pont = top - k
                    if r not in map_tmp.keys():
                        map_tmp[r] = pont
                    else:
                        map_tmp[r] += pont
            # print(i, map_tmp)
            list_tmp = [k for k, v in sorted(map_tmp.items(), key=lambda item: item[1], reverse=True)]
            respostas.append(list_tmp[0])
        return respostas

    #Mandatory function, accepts all answers returned by the agents and aplly them to borda logic, returns to user the winner
    def getAnswer(self, answers, query):
        responses = []
        for agent in answers.keys():
            responses.append(answers[agent])

        winner = self.borda_sort(responses)

        return winner[0]
