from random import random

def random_graph(n, p, w=0.9):
    graph = {}
    for i in range(n):
        for j in range(i+1,n):
            if random() < p:
                graph[(i,j)] = w
    return graph

#def complex_network(n, p, e):

