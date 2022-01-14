import numpy as np
from random import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

from .graph import random_graph

COOPERATE=0
DEFECT=1

class Agent(ABC):
    def __init__(self, agent_id):
        self.id = agent_id
        self.score = 0

    def __lt__(self, other):
        return self.score < other.score
        
    @abstractmethod 
    def __call__(self, state):
        raise NotImplementedError

class Linear(Agent):
    state_dim = 6
    def __init__(self, agent_id, **kwargs):
        super().__init__(agent_id)
        self.mean = kwargs.get("mean", 0)
        self.var = kwargs.get("var", 1.0)
        self.weights = np.random.normal(
            loc=self.mean, 
            scale=self.var,
            size=(2,Linear.state_dim),
        )

    def __call__(self, state):
        # features:
        # 0 % of opponent's cooperations
        # 1 % of opponent's defects
        # 2 relationship weight
        # 3..n+3 opponent's last n moves
        v = np.zeros(shape=(Linear.state_dim,))
        hist = state["history"]
        if len(hist)>0:
            v[0] = hist.count(0)/len(hist)
            v[1] = 1. - v[0]
        v[2] = state["weight"]        
        if len(hist) > 0:
            v[3] = hist[-1]
        if len(hist) > 1:
            v[4] = hist[-2]
        if len(hist) > 2:
            v[5] = hist[-3]
        
        p = np.exp( self.weights @ v )
        p /= p.sum()
        return COOPERATE if random() < p[0] else DEFECT

class Tit4Tat(Agent):
    def __init__(self, agent_id):
        super().__init__(agent_id)
    def __call__(self, state):
        if len(state["history"]) == 0:
            return COOPERATE 
        last = state["history"][-1]
        return last
    
class Tournament:
    def __init__(self, agents: List[Agent], graph: Dict[Tuple[int, int],float], **kwargs):
        self.payoff = [[3,0],
                       [5,1]]
        self.agents = agents
        self.graph = graph
    
    def play_match(self, id1: int, id2: int):
        if id1 > id2:
            id1, id2 = id2, id1
        a1 = self.agents[id1]
        a2 = self.agents[id2]
        
        match = self.graph[(id1,id2)]
        s1 = {"history": [], "weight": match}
        s2 = {"history": [], "weight": match}
        
        while random() <= match:   
            m1 = a1(s2)
            m2 = a2(s1)
            s1["history"].append(m1)
            s2["history"].append(m2)

            a1.score += self.payoff[m1][m2]
            a2.score += self.payoff[m2][m1]
        return s1["history"], s2["history"]

    def run(self):
        result = {}
        for e in self.graph:
            result[e] = self.play_match(e[0], e[1])
        return result
