#!/usr/bin/env python
from agent import * 
from graph import *
import numpy as np

N_AGENTS=150
GENS=3

if __name__ == "__main__":
    mean = np.zeros(shape=(Linear.state_dim,))
    var = 1.0
    graph = random_graph(N_AGENTS, 0.7)

    for i in range(GENS):
        agents = [Linear(i, mean=mean, var=var) for i in range(N_AGENTS)]
        trn = Tournament(agents, graph)
        res = trn.run()

        # get new params from agents with top scores
        elite = sorted(trn.agents)[-20:]
        weights = [a.weights for a in elite if isinstance(a, Linear)]
        mean = sum(weights)/len(weights) 
        var = sum([np.linalg.norm(w-mean) for w in weights])/len(weights)
        
