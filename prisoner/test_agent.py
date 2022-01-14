from .agent import * 
from .graph import random_graph

def test_call():
    agent = Linear(0)
    state = {"weight": 0.5, "history": [1,0,1]}
    move = agent(state)
    assert move in (0,1) 

def test_tournament():
    agents = [Linear(i) for i in range(10)]
    graph = random_graph(10, 0.8)
    trn = Tournament(agents, graph)
    res = trn.run()
