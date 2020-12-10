from graph_tool.all import *
from graph_tool import generation as gt
import math
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import matplotlib


T = 300
b1 = 0.1 # infection rate
u1 = 0.2 # recovery rate
b2 = 0 # behavior change rate
u2 = 1 # rate of reverting behavioral change
u3 = 0.3 # media recovery rate
a1 = 0 # a1*b1 is the effective infection probability for adopters of protections
a2 = 1 # a2 * M[t] as the media effect on awareness spreading

def deg_sampler():
    accept = False
    while not accept:
        k = randint(2, 100)
        accept = random() < k**(-2.5)
    return k
    
    
g = gt.random_graph(1000, deg_sampler, directed = False)
pos = sfdp_layout(g)
#graph_draw(g, pos)
#pos = g.vp["pos"]  # layout positions



S = [1, 1, 1, 1]           # White color
I = [0, 0, 0, 1]           # Black color
N = 0 # protections not adopted
A = 1 # protections adopted

# Initialize all vertices to the S state
I_state = g.new_vertex_property("vector<double>")
for v in g.vertices():
    I_state[v] = S

#Initialize state of behavior
B_state = g.new_vertex_property("int")
for v in g.vertices():
    B_state[v] = N

for v in randint(0, g.num_vertices()-1, size = 10): 
    I_state[v] = I



I_t = np.zeros(T)
A_t = np.zeros(T)
M_t = np.zeros(T+1)

for t in range(T):
    vs = list(g.vertices())
    shuffle(vs)
    for v in vs:
        if I_state[v] == S:
            b_eff = (1 if B_state[v] == N else a1)*b1
            ns = [n for n in list(v.out_neighbors()) if I_state[n] == I]
            if random() < 1 - math.exp(-len(ns)*b_eff):
                I_state[v] = I
                B_state[v] = A
        elif I_state[v] == I:
            if random() < 1 - math.exp(-u1):
                I_state[v] = S
        
        if B_state[v] == N:
            ns = [n for n in list(v.out_neighbors()) if B_state[n] == A]
            if random() < max(1 - math.exp(-len(ns)*b2), a2*M_t[t]):
                B_state[v] = A
        elif B_state[v] == A:
            if random() < 1 - math.exp(-u2):
                B_state[v] = N
                
    I_t[t] = GraphView(g, vfilt=lambda v: I_state[v] == I).num_vertices()/g.num_vertices()
    M_t[t+1] = M_t[t] + (1-np.mean(M_t[max(0, t-30):t+1]))*I_t[t] - u3*M_t[t]
    A_t[t] = GraphView(g, vfilt=lambda v: B_state[v] == A).num_vertices()/g.num_vertices()


plt.plot(range(T), I_t)
plt.plot(range(T), A_t)
plt.plot(range(T+1), M_t)
plt.legend(["I", "A", "M"])
plt.show()

