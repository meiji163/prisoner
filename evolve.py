#!/usr/bin/env python
import numpy as np

from prisoner.agent import * 
from prisoner.graph import *

N_AGENTS=100
GENS=20

if __name__ == "__main__":
    mean = np.zeros(shape=(Linear.state_dim,))
    var = 0.09 
    graph = random_graph(N_AGENTS, 0.8)

    print("======== PRISONER'S DILEMMA ========")
    print(f"{N_AGENTS} agents\n")
    print("\t\tcooperations\tdefections\texploits")

    for i in range(GENS):
        #agents = [Linear(i, mean=mean, var=var) for i in range(N_AGENTS)]
        agents = [Linear(i, mean=mean, var=var) for i in range(N_AGENTS//2)] + [Tit4Tat(i) for i in range(N_AGENTS//2,N_AGENTS)]

        trn = Tournament(agents, graph)
        res = trn.run()

        stats = metrics(N_AGENTS, res)
        exploits = sum([stats[i]["exploit"] for i in range(N_AGENTS)])
        coops = sum([stats[i]["cooperate"] for i in range(N_AGENTS)])
        defects = sum([stats[i]["defect"] for i in range(N_AGENTS)])
        print(f"GEN {i}/{GENS}:\t{coops:<12}\t{defects:<12}\t{exploits:<12}")

        # get new params from agents with top scores
        elite = sorted(trn.agents)[-40:]
        weights = [a.weights for a in elite if isinstance(a, Linear)]
        mean = sum(weights)/len(weights) 
        var = sum([np.linalg.norm(w-mean)**2 for w in weights])/len(weights)
        var = min(var, 1.0)
        print(var)

